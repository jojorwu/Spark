pub use crate::asset::{AssetStorage, Handle};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub struct ResourceManager {
    pub gpu_textures: AssetStorage<spark_renderer::vulkan::texture::Texture>,
    pub all_vertices: Vec<spark_renderer::vertex::Vertex>,
    pub all_indices: Vec<u32>,
    pub all_materials_ssbo: Vec<spark_renderer::MaterialDataSSBO>,
    pub needs_upload: bool,
    pub registry: ResourceRegistry,
}

#[derive(Default)]
pub struct ResourceRegistry {
    pub active_meshes: HashMap<Handle<spark_renderer::vertex::Vertex>, u32>,
}

impl ResourceManager {
    pub fn new() -> Self {
        let mut rm = Self {
            gpu_textures: AssetStorage::new(),
            all_vertices: Vec::new(),
            all_indices: Vec::new(),
            all_materials_ssbo: Vec::new(),
            needs_upload: true,
            registry: ResourceRegistry::default(),
        };

        // Pre-generate primitive meshes
        let (v_cube, i_cube) = MeshGenerator::generate_cube();
        rm.all_vertices.extend(v_cube);
        rm.all_indices.extend(i_cube);

        let (v_plane, i_plane) = MeshGenerator::generate_plane(10.0);
        rm.all_vertices.extend(v_plane);
        rm.all_indices.extend(i_plane);

        let (v_sphere, i_sphere) = MeshGenerator::generate_sphere(0.5, 32);
        rm.all_vertices.extend(v_sphere);
        rm.all_indices.extend(i_sphere);

        rm
    }
}

impl Default for ResourceManager {
    fn default() -> Self {
        Self::new()
    }
}

impl spark_renderer::RenderableResourceManager for ResourceManager {}

pub struct MeshGenerator;

impl MeshGenerator {
    pub fn generate_cube() -> (Vec<spark_renderer::vertex::Vertex>, Vec<u32>) {
        use spark_math::{Vec2, Vec3};
        use spark_renderer::vertex::Vertex;

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let positions = [
            // Front
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
            // Back
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
        ];

        let normals = [
            [0.0, 0.0, 1.0],  // Front
            [0.0, 0.0, -1.0], // Back
            [0.0, 1.0, 0.0],  // Top
            [0.0, -1.0, 0.0], // Bottom
            [1.0, 0.0, 0.0],  // Right
            [-1.0, 0.0, 0.0], // Left
        ];

        let face_indices = [
            [0, 1, 2, 3], // Front
            [5, 4, 7, 6], // Back
            [3, 2, 6, 7], // Top
            [1, 0, 4, 5], // Bottom
            [1, 5, 6, 2], // Right
            [4, 0, 3, 7], // Left
        ];

        let uvs = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];

        for (i, face) in face_indices.iter().enumerate() {
            let n = Vec3::from_array(normals[i]);
            for (j, &idx) in face.iter().enumerate() {
                let p = Vec3::from_array(positions[idx]);
                let uv = Vec2::from_array(uvs[j]);
                vertices.push(Vertex::pack(p, n, uv, Vec3::ONE, Vec3::X));
            }
            let base = (i * 4) as u32;
            indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }

        (vertices, indices)
    }

    pub fn generate_plane(size: f32) -> (Vec<spark_renderer::vertex::Vertex>, Vec<u32>) {
        use spark_math::{Vec2, Vec3};
        use spark_renderer::vertex::Vertex;

        let h = size * 0.5;
        let vertices = vec![
            Vertex::pack(
                Vec3::new(-h, 0.0, h),
                Vec3::Y,
                Vec2::new(0.0, 1.0),
                Vec3::ONE,
                Vec3::X,
            ),
            Vertex::pack(
                Vec3::new(h, 0.0, h),
                Vec3::Y,
                Vec2::new(1.0, 1.0),
                Vec3::ONE,
                Vec3::X,
            ),
            Vertex::pack(
                Vec3::new(h, 0.0, -h),
                Vec3::Y,
                Vec2::new(1.0, 0.0),
                Vec3::ONE,
                Vec3::X,
            ),
            Vertex::pack(
                Vec3::new(-h, 0.0, -h),
                Vec3::Y,
                Vec2::new(0.0, 0.0),
                Vec3::ONE,
                Vec3::X,
            ),
        ];
        let indices = vec![0, 1, 2, 0, 2, 3];
        (vertices, indices)
    }

    pub fn generate_sphere(
        radius: f32,
        segments: u32,
    ) -> (Vec<spark_renderer::vertex::Vertex>, Vec<u32>) {
        use spark_math::{Vec2, Vec3};
        use spark_renderer::vertex::Vertex;
        use std::f32::consts::PI;

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for y in 0..=segments {
            for x in 0..=segments {
                let x_segment = x as f32 / segments as f32;
                let y_segment = y as f32 / segments as f32;
                let x_pos = (x_segment * 2.0 * PI).cos() * (y_segment * PI).sin();
                let y_pos = (y_segment * PI).cos();
                let z_pos = (x_segment * 2.0 * PI).sin() * (y_segment * PI).sin();

                let pos = Vec3::new(x_pos, y_pos, z_pos) * radius;
                let normal = Vec3::new(x_pos, y_pos, z_pos).normalize();
                let uv = Vec2::new(x_segment, y_segment);

                vertices.push(Vertex::pack(pos, normal, uv, Vec3::ONE, Vec3::X));
            }
        }

        for y in 0..segments {
            for x in 0..segments {
                let base = y * (segments + 1) + x;
                indices.push(base);
                indices.push(base + segments + 1);
                indices.push(base + segments + 2);

                indices.push(base);
                indices.push(base + segments + 2);
                indices.push(base + 1);
            }
        }

        (vertices, indices)
    }
}

impl ResourceManager {
    pub fn upload_global_buffers(&mut self, renderer: &mut spark_renderer::Renderer) {
        use spark_renderer::ash::vk;
        if self.all_vertices.is_empty() || !self.needs_upload {
            return;
        }
        self.needs_upload = false;

        if let Some(vb) = renderer.gpu_resource_manager.global_vertex_buffer.take() {
            renderer.destroy_buffer(vb);
        }
        if let Some(ib) = renderer.gpu_resource_manager.global_index_buffer.take() {
            renderer.destroy_buffer(ib);
        }
        if let Some(mb) = renderer.gpu_resource_manager.global_material_buffer.take() {
            renderer.destroy_buffer(mb);
        }

        let v_sz = (self.all_vertices.len() * std::mem::size_of::<spark_renderer::vertex::Vertex>())
            as u64;
        let vb = renderer.create_buffer(
            v_sz,
            vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let staging_v = renderer.create_buffer(
            v_sz,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        renderer.upload_to_buffer(&staging_v, &self.all_vertices);

        let i_sz = (self.all_indices.len() * 4) as u64;
        let ib = renderer.create_buffer(
            i_sz,
            vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let staging_i = renderer.create_buffer(
            i_sz,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        renderer.upload_to_buffer(&staging_i, &self.all_indices);

        let m_sz = (self.all_materials_ssbo.len()
            * std::mem::size_of::<spark_renderer::MaterialDataSSBO>()) as u64;
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
        renderer.upload_to_buffer(&staging_m, &self.all_materials_ssbo);

        // Batched copy
        let cb = renderer.begin_single_time_commands();
        unsafe {
            let device = renderer.get_device();
            device.cmd_copy_buffer(
                cb,
                staging_v.handle,
                vb.handle,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: staging_v.size,
                }],
            );
            device.cmd_copy_buffer(
                cb,
                staging_i.handle,
                ib.handle,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: staging_i.size,
                }],
            );
            device.cmd_copy_buffer(
                cb,
                staging_m.handle,
                mb.handle,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: staging_m.size,
                }],
            );
        }
        renderer.end_single_time_commands(cb);

        renderer.destroy_buffer(staging_v);
        renderer.destroy_buffer(staging_i);
        renderer.destroy_buffer(staging_m);

        renderer.set_global_buffers(vb, ib);
        renderer.set_material_buffer(mb);
    }

    pub fn load_scene(
        &mut self,
        path: PathBuf,
        scene_tree: &mut crate::scene::Scene,
        renderer: &mut spark_renderer::Renderer,
        asset_manager: &mut crate::asset::AssetManager,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.all_vertices.clear();
        self.all_indices.clear();
        self.all_materials_ssbo.clear();
        crate::gltf_loader::GltfLoader::load_scene(self, asset_manager, path, scene_tree, renderer)
    }

    /// Loads a texture from disk and uploads it to the GPU.
    /// Returns a handle to the newly created GPU resource.
    pub fn upload_texture(
        &mut self,
        path: &Path,
        renderer: &mut spark_renderer::Renderer,
        asset_manager: &mut crate::asset::AssetManager,
    ) -> Result<
        Handle<spark_renderer::vulkan::texture::Texture>,
        spark_renderer::error::RendererError,
    > {
        if let Some(&handle) = asset_manager.texture_path_map.get(path) {
            return Ok(handle);
        }

        log::debug!("Loading texture from disk: {:?}", path);
        let img = image::open(path).map_err(|e| {
            spark_renderer::error::RendererError::ResourceLoading(format!(
                "Failed to open texture {:?}: {}",
                path, e
            ))
        })?;

        let texture = renderer.create_texture_from_image(&img);
        let handle = self.gpu_textures.add(texture);
        asset_manager
            .texture_path_map
            .insert(path.to_path_buf(), handle);
        Ok(handle)
    }
}
