use std::collections::HashMap;
use std::path::PathBuf;
use image::DynamicImage;
use spark_renderer::ash::vk;

pub struct ResourceManager {
    pub textures: HashMap<PathBuf, DynamicImage>,
    pub gpu_textures: Vec<spark_renderer::vulkan::texture::Texture>,
    pub scenes: HashMap<PathBuf, gltf::Document>,
    pub meshes: Vec<spark_renderer::Buffer>,
    pub all_vertices: Vec<spark_renderer::vertex::Vertex>,
    pub all_indices: Vec<u32>,
    pub needs_upload: bool,
}

impl ResourceManager {
    pub fn new() -> Self {
        Self {
            textures: HashMap::new(),
            gpu_textures: Vec::new(),
            scenes: HashMap::new(),
            meshes: Vec::new(),
            all_vertices: Vec::new(),
            all_indices: Vec::new(),
            needs_upload: false,
        }
    }

    pub fn upload_global_buffers(&mut self, renderer: &mut spark_renderer::Renderer) {
        use spark_renderer::ash::vk;
        if self.all_vertices.is_empty() || !self.needs_upload { return; }
        self.needs_upload = false;

        if let Some(vb) = renderer.global_vertex_buffer.take() {
            renderer.destroy_buffer(vb);
        }
        if let Some(ib) = renderer.global_index_buffer.take() {
            renderer.destroy_buffer(ib);
        }

        let v_sz = (self.all_vertices.len() * std::mem::size_of::<spark_renderer::vertex::Vertex>()) as u64;
        let vb = renderer.create_buffer(
            v_sz,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        // Stage and upload
        let staging_v = renderer.create_buffer(
            v_sz,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        renderer.upload_to_buffer(&staging_v, &self.all_vertices);
        self.copy_buffer(renderer, staging_v, vb);
        renderer.destroy_buffer(staging_v);

        let i_sz = (self.all_indices.len() * 4) as u64;
        let ib = renderer.create_buffer(
            i_sz,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let staging_i = renderer.create_buffer(
            i_sz,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        renderer.upload_to_buffer(&staging_i, &self.all_indices);
        self.copy_buffer(renderer, staging_i, ib);
        renderer.destroy_buffer(staging_i);

        renderer.set_global_buffers(vb, ib);
    }

    fn copy_buffer(&self, renderer: &spark_renderer::Renderer, src: spark_renderer::Buffer, dst: spark_renderer::Buffer) {
        use spark_renderer::ash::vk;
        let cb = renderer.begin_single_time_commands();
        unsafe {
            renderer.get_device().cmd_copy_buffer(
                cb,
                src.handle,
                dst.handle,
                &[vk::BufferCopy { src_offset: 0, dst_offset: 0, size: src.size }],
            );
        }
        renderer.end_single_time_commands(cb);
    }

    pub fn load_scene(
        &mut self,
        path: PathBuf,
        scene_tree: &mut crate::scene::Scene,
        renderer: &spark_renderer::Renderer,
    ) {
        self.all_vertices.clear();
        self.all_indices.clear();
        log::info!("Loading glTF scene: {:?}", path);
        let (doc, buffers, _) = gltf::import(path).expect("Failed to load glTF");

        for gltf_scene in doc.scenes() {
            for node in gltf_scene.nodes() {
                self.process_gltf_node(node, &buffers, scene_tree, scene_tree.root, renderer);
            }
        }
    }

    fn process_gltf_node(
        &mut self,
        node: gltf::Node,
        buffers: &[gltf::buffer::Data],
        scene_tree: &mut crate::scene::Scene,
        parent: crate::scene::NodeKey,
        renderer: &spark_renderer::Renderer,
    ) {
        use crate::scene::{Node, NodeData};
        use spark_math::{Mat4, Vec3, Quat};

        let (translation, rotation, scale) = node.transform().decomposed();
        let local_transform = Mat4::from_scale_rotation_translation(
            Vec3::from_array(scale),
            Quat::from_array(rotation),
            Vec3::from_array(translation),
        );

        let mut data = NodeData::None;

        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                use spark_renderer::vertex::Vertex;
                use spark_math::Vec2;
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                let positions = reader.read_positions().unwrap().collect::<Vec<_>>();
                let v_offset = self.all_vertices.len() as i32;
                let i_start = self.all_indices.len() as u32;

                let mut max_dist_sq = 0.0f32;
                let normals = reader.read_normals().map(|n| n.collect::<Vec<_>>());

                for i in 0..positions.len() {
                    let p = positions[i];
                    let dist_sq = p[0]*p[0] + p[1]*p[1] + p[2]*p[2];
                    if dist_sq > max_dist_sq {
                        max_dist_sq = dist_sq;
                    }

                    let n = if let Some(ref normals) = normals {
                        spark_math::Vec3::from_array(normals[i])
                    } else {
                        spark_math::Vec3::Y
                    };

                    self.all_vertices.push(Vertex {
                        pos: spark_math::Vec3::from_array(p),
                        normal: n,
                        color: spark_math::Vec3::ONE,
                        tex_coord: Vec2::ZERO,
                    });
                }

                let index_count = if let Some(indices) = reader.read_indices() {
                    let idxs: Vec<u32> = indices.into_u32().collect();
                    let count = idxs.len() as u32;
                    self.all_indices.extend(idxs);
                    count
                } else {
                    positions.len() as u32
                };

                let bounding_radius = max_dist_sq.sqrt();

                data = NodeData::Mesh {
                    vertex_count: positions.len() as u32,
                    index_count,
                    first_index: i_start,
                    vertex_offset: v_offset,
                    texture_id: None,
                    vertex_buffer_id: None,
                    bounding_radius
                };
                self.needs_upload = true;
                break;
            }
        }

        let spark_node = Node {
            name: node.name().unwrap_or("Unnamed Node").to_string(),
            local_transform,
            global_transform: Mat4::IDENTITY,
            parent: None,
            children: Vec::new(),
            data,
        };

        let key = scene_tree.add_node(parent, spark_node);

        for child in node.children() {
            self.process_gltf_node(child, buffers, scene_tree, key, renderer);
        }
    }

    pub fn load_texture(&mut self, path: PathBuf) -> &DynamicImage {
        self.textures.entry(path.clone()).or_insert_with(|| {
            log::info!("Loading texture: {:?}", path);
            image::open(path).expect("Failed to load texture")
        })
    }

    pub fn upload_texture(
        &mut self,
        path: PathBuf,
        renderer: &spark_renderer::Renderer,
    ) -> u32 {
        let img = self.load_texture(path);
        let texture = renderer.create_texture_from_image(img);
        self.gpu_textures.push(texture);
        (self.gpu_textures.len() - 1) as u32
    }
}
