use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::marker::PhantomData;
use image::DynamicImage;
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Handle<T> {
    id: u32,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self { *self }
}

impl<T> Copy for Handle<T> {}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool { self.id == other.id }
}

impl<T> Eq for Handle<T> {}

impl<T> std::hash::Hash for Handle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl<T> Handle<T> {
    pub fn new(id: u32) -> Self {
        Self { id, _marker: PhantomData }
    }
    pub fn id(&self) -> u32 { self.id }
}

pub struct AssetStorage<T> {
    assets: Vec<T>,
}

impl<T> AssetStorage<T> {
    pub fn new() -> Self { Self { assets: Vec::new() } }
    pub fn add(&mut self, asset: T) -> Handle<T> {
        let id = self.assets.len() as u32;
        self.assets.push(asset);
        Handle::new(id)
    }
    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        self.assets.get(handle.id as usize)
    }
}

pub struct ResourceManager {
    pub textures: AssetStorage<DynamicImage>,
    pub gpu_textures: AssetStorage<spark_renderer::vulkan::texture::Texture>,
    pub texture_path_map: HashMap<PathBuf, Handle<spark_renderer::vulkan::texture::Texture>>,
    pub all_vertices: Vec<spark_renderer::vertex::Vertex>,
    pub all_indices: Vec<u32>,
    pub all_materials: Vec<spark_renderer::MaterialDataSSBO>,
    pub needs_upload: bool,
}

impl ResourceManager {
    pub fn new() -> Self {
        Self {
            textures: AssetStorage::new(),
            gpu_textures: AssetStorage::new(),
            texture_path_map: HashMap::new(),
            all_vertices: Vec::new(),
            all_indices: Vec::new(),
            all_materials: Vec::new(),
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
        if let Some(mb) = renderer.global_material_buffer.take() {
            renderer.destroy_buffer(mb);
        }

        let v_sz = (self.all_vertices.len() * std::mem::size_of::<spark_renderer::vertex::Vertex>()) as u64;
        let vb = renderer.create_buffer(
            v_sz,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
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

        let m_sz = (self.all_materials.len() * std::mem::size_of::<spark_renderer::MaterialDataSSBO>()) as u64;
        let mb = renderer.create_buffer(
            m_sz,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let staging_m = renderer.create_buffer(
            m_sz,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        renderer.upload_to_buffer(&staging_m, &self.all_materials);
        self.copy_buffer(renderer, staging_m, mb);
        renderer.destroy_buffer(staging_m);

        renderer.set_global_buffers(vb, ib);
        renderer.set_material_buffer(mb);
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
        self.all_materials.clear();
        GltfLoader::load_scene(self, path, scene_tree, renderer);
    }

    pub fn upload_texture(
        &mut self,
        path: &Path,
        renderer: &spark_renderer::Renderer,
    ) -> Handle<spark_renderer::vulkan::texture::Texture> {
        if let Some(&handle) = self.texture_path_map.get(path) {
            return handle;
        }
        log::info!("Loading and uploading texture: {:?}", path);
        let img = image::open(path).unwrap_or_else(|e| {
            log::warn!("Failed to load texture {:?}: {}. Using fallback.", path, e);
            image::DynamicImage::ImageRgba8(image::RgbaImage::from_pixel(1, 1, image::Rgba([255, 0, 255, 255])))
        });
        let texture = renderer.create_texture_from_image(&img);
        let handle = self.gpu_textures.add(texture);
        self.texture_path_map.insert(path.to_path_buf(), handle);
        handle
    }
}

pub struct GltfLoader;

impl GltfLoader {
    pub fn load_scene(
        rm: &mut ResourceManager,
        path: PathBuf,
        scene_tree: &mut crate::scene::Scene,
        renderer: &spark_renderer::Renderer,
    ) {
        log::info!("Loading glTF scene: {:?}", path);
        let (doc, buffers, _) = gltf::import(&path).expect("Failed to load glTF");
        let parent_dir = path.parent().unwrap_or_else(|| Path::new("")).to_path_buf();
        let default_scene = doc.default_scene().or(doc.scenes().next());
        if let Some(scene) = default_scene {
            for node in scene.nodes() {
                Self::process_node(rm, node, &buffers, scene_tree, scene_tree.root, renderer, &parent_dir);
            }
        }
    }

    fn process_node(
        rm: &mut ResourceManager,
        node: gltf::Node,
        buffers: &[gltf::buffer::Data],
        scene_tree: &mut crate::scene::Scene,
        parent: crate::scene::NodeKey,
        renderer: &spark_renderer::Renderer,
        parent_dir: &Path,
    ) {
        use crate::scene::{Node, MeshComponent, Component};
        use spark_math::{Mat4, Vec3, Quat, Vec4};

        let (translation, rotation, scale) = node.transform().decomposed();
        let local_transform = Mat4::from_scale_rotation_translation(
            Vec3::from_array(scale),
            Quat::from_array(rotation),
            Vec3::from_array(translation),
        );

        let mut components: Vec<Box<dyn Component>> = Vec::new();

        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                use spark_renderer::vertex::Vertex;
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                let positions = match reader.read_positions() {
                    Some(p) => p.collect::<Vec<_>>(),
                    None => continue,
                };
                let v_offset = rm.all_vertices.len() as i32;
                let i_start = rm.all_indices.len() as u32;

                let mut max_dist_sq = 0.0f32;
                let normals = reader.read_normals().map(|n| n.collect::<Vec<_>>());
                let tex_coords = reader.read_tex_coords(0).map(|t| t.into_f32().collect::<Vec<_>>());

                for i in 0..positions.len() {
                    let p = positions[i];
                    let dist_sq = p[0]*p[0] + p[1]*p[1] + p[2]*p[2];
                    if dist_sq > max_dist_sq { max_dist_sq = dist_sq; }

                    let n = if let Some(ref normals) = normals {
                        spark_math::Vec3::from_array(normals[i])
                    } else {
                        spark_math::Vec3::Y
                    };

                    let tc = if let Some(ref tex_coords) = tex_coords {
                        spark_math::Vec2::from_array(tex_coords[i])
                    } else {
                        spark_math::Vec2::ZERO
                    };

                    rm.all_vertices.push(Vertex {
                        pos: spark_math::Vec3::from_array(p),
                        normal: n,
                        color: spark_math::Vec3::ONE,
                        tex_coord: tc,
                    });
                }

                let index_count = if let Some(indices) = reader.read_indices() {
                    let idxs: Vec<u32> = indices.into_u32().collect();
                    let count = idxs.len() as u32;
                    rm.all_indices.extend(idxs);
                    count
                } else {
                    positions.len() as u32
                };

                let gltf_mat = primitive.material();
                let pbr = gltf_mat.pbr_metallic_roughness();

                let mut mat_ssbo = spark_renderer::MaterialDataSSBO {
                    albedo_factor: Vec4::from_array(pbr.base_color_factor()),
                    emissive_factor: Vec4::from_array([
                        gltf_mat.emissive_factor()[0],
                        gltf_mat.emissive_factor()[1],
                        gltf_mat.emissive_factor()[2],
                        1.0
                    ]),
                    metallic_factor: pbr.metallic_factor(),
                    roughness_factor: pbr.roughness_factor(),
                    alpha_cutoff: gltf_mat.alpha_cutoff().unwrap_or(0.5),
                    flags: 0,
                    albedo_texture: -1,
                    normal_texture: -1,
                    metallic_roughness_texture: -1,
                    emissive_texture: -1,
                    occlusion_texture: -1,
                    padding: [0; 3],
                };

                if let Some(tex) = pbr.base_color_texture() {
                    if let gltf::image::Source::Uri { uri, .. } = tex.texture().source().source() {
                        let handle = rm.upload_texture(&parent_dir.join(uri), renderer);
                        if let Some(tex) = rm.gpu_textures.get(handle) {
                            mat_ssbo.albedo_texture = tex.bindless_index as i32;
                        }
                    }
                }
                if let Some(tex) = gltf_mat.normal_texture() {
                    if let gltf::image::Source::Uri { uri, .. } = tex.texture().source().source() {
                        let handle = rm.upload_texture(&parent_dir.join(uri), renderer);
                        if let Some(tex) = rm.gpu_textures.get(handle) {
                            mat_ssbo.normal_texture = tex.bindless_index as i32;
                        }
                    }
                }
                if let Some(tex) = pbr.metallic_roughness_texture() {
                    if let gltf::image::Source::Uri { uri, .. } = tex.texture().source().source() {
                        let handle = rm.upload_texture(&parent_dir.join(uri), renderer);
                        if let Some(tex) = rm.gpu_textures.get(handle) {
                            mat_ssbo.metallic_roughness_texture = tex.bindless_index as i32;
                        }
                    }
                }
                if let Some(tex) = gltf_mat.emissive_texture() {
                    if let gltf::image::Source::Uri { uri, .. } = tex.texture().source().source() {
                        let handle = rm.upload_texture(&parent_dir.join(uri), renderer);
                        if let Some(tex) = rm.gpu_textures.get(handle) {
                            mat_ssbo.emissive_texture = tex.bindless_index as i32;
                        }
                    }
                }
                if let Some(tex) = gltf_mat.occlusion_texture() {
                    if let gltf::image::Source::Uri { uri, .. } = tex.texture().source().source() {
                        let handle = rm.upload_texture(&parent_dir.join(uri), renderer);
                        if let Some(tex) = rm.gpu_textures.get(handle) {
                            mat_ssbo.occlusion_texture = tex.bindless_index as i32;
                        }
                    }
                }

                rm.all_materials.push(mat_ssbo);
                let mat_idx = (rm.all_materials.len() - 1) as u32;

                let mesh_comp = MeshComponent {
                    vertex_count: positions.len() as u32,
                    index_count,
                    first_index: i_start,
                    vertex_offset: v_offset,
                    texture_handle: None,
                    material_index: Some(mat_idx),
                    bounding_radius: max_dist_sq.sqrt()
                };
                components.push(Box::new(mesh_comp) as Box<dyn Component>);
                rm.needs_upload = true;
                break;
            }
        }

        let spark_node = Node {
            name: node.name().unwrap_or("Unnamed Node").to_string(),
            local_transform,
            global_transform: Mat4::IDENTITY,
            parent: None,
            children: Vec::new(),
            components,
        };

        let key = scene_tree.add_node(parent, spark_node);
        for child in node.children() {
            Self::process_node(rm, child, buffers, scene_tree, key, renderer, parent_dir);
        }
    }
}
