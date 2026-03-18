use std::collections::HashMap;
use std::path::PathBuf;
use image::DynamicImage;
use spark_renderer::ash::vk;

pub struct ResourceManager {
    pub textures: HashMap<PathBuf, DynamicImage>,
    pub gpu_textures: Vec<spark_renderer::vulkan::texture::Texture>,
    pub scenes: HashMap<PathBuf, gltf::Document>,
    pub meshes: Vec<spark_renderer::Buffer>,
}

impl ResourceManager {
    pub fn new() -> Self {
        Self {
            textures: HashMap::new(),
            gpu_textures: Vec::new(),
            scenes: HashMap::new(),
            meshes: Vec::new(),
        }
    }

    pub fn load_scene(
        &mut self,
        path: PathBuf,
        scene_tree: &mut crate::scene::Scene,
        renderer: &spark_renderer::Renderer,
    ) {
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

        let data = if let Some(mesh) = node.mesh() {
            let mut mesh_id = None;
            let mut vertex_count = 0;
            let mut bounding_radius = 0.0f32;

            for primitive in mesh.primitives() {
                use spark_renderer::vertex::Vertex;
                use spark_math::Vec2;
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

                let positions = reader.read_positions().unwrap().collect::<Vec<_>>();

                let mut vertices = Vec::new();
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

                    vertices.push(Vertex {
                        pos: spark_math::Vec3::from_array(p),
                        normal: n,
                        color: spark_math::Vec3::ONE,
                        tex_coord: Vec2::ZERO,
                    });
                }

                vertex_count = vertices.len() as u32;
                bounding_radius = max_dist_sq.sqrt();
                let vb = renderer.create_buffer(
                    (std::mem::size_of::<Vertex>() * vertices.len()) as u64,
                    vk::BufferUsageFlags::VERTEX_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                );
                renderer.upload_to_buffer(&vb, &vertices);
                self.meshes.push(vb);
                mesh_id = Some((self.meshes.len() - 1) as u32);
            }
            NodeData::Mesh {
                vertex_count,
                index_count: 0, // Not used in this basic loader yet
                first_index: 0,
                vertex_offset: 0,
                texture_id: None,
                vertex_buffer_id: mesh_id,
                bounding_radius
            }
        } else {
            NodeData::None
        };

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
