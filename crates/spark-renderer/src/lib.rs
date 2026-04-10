pub mod error;
pub mod factory;
pub mod graph;
pub mod passes;
pub mod pipeline;
pub mod resource;
pub mod ui;
pub mod vertex;
pub mod vulkan;

use crate::error::RendererError;
use crate::pipeline::Pipeline;
pub use crate::resource::{
    Attachment, Buffer, MaterialDataSSBO, ObjectDataSSBO, RenderFrame, RenderSettings,
    MAX_FRAMES_IN_FLIGHT,
};
use crate::ui::EguiRenderer;
use crate::vulkan::context::VulkanContext;
use crate::vulkan::device::VulkanDevice;
use crate::vulkan::gbuffer::GBuffer;
use crate::vulkan::swapchain::VulkanSwapchain;
use crate::vulkan::texture::Texture;
pub use ash;
use ash::vk;
use rayon::prelude::*;
use spark_math::Vec4Swizzles;
use winit::window::Window;

pub trait RenderableScene {
    fn get_active_camera_matrices(
        &self,
        extent: vk::Extent2D,
    ) -> Option<(spark_math::Mat4, spark_math::Mat4, f32, f32)>;
    fn collect_frame_packet(
        &self,
        frustum: Option<&spark_math::Frustum>,
        asset_manager: &dyn RenderableAssetManager,
    ) -> crate::resource::FramePacket;
}

pub trait RenderableResourceManager {}
pub trait RenderableAssetManager: Send + Sync {
    fn is_material_transparent(&self, material_index: u32) -> bool;
}

/// The main renderer of the Spark Engine, responsible for managing the Vulkan context,
/// swapchain, and executing the rendering pipeline through a modular pass system.
pub struct Renderer {
    pub context: VulkanContext,
    pub device: VulkanDevice,
    pub resource_tracker: crate::resource::ResourceTracker,
    pub gpu_resource_manager: crate::vulkan::resource_manager::GpuResourceManager,
    pub frame_manager: crate::vulkan::frame_manager::FrameManager,
    swapchain: VulkanSwapchain,
    pub global_descriptor_set_layout: vk::DescriptorSetLayout,
    pub light_count: u32,
    pub render_graph: crate::graph::RenderGraph,
    pub pipeline_cache: vk::PipelineCache,
    pub pipeline: Option<Pipeline>,
    pub default_descriptor_set: vk::DescriptorSet,
    pub scene_view_matrix_for_pos: spark_math::Mat4,
    pub prev_view_proj: spark_math::Mat4,
    pub common_sampler: vk::Sampler,
    pub common_shadow_view: vk::ImageView,
    pub hiz_view: vk::ImageView,
    pub as_manager:
        std::sync::Arc<std::sync::Mutex<crate::vulkan::as_manager::AccelerationStructureManager>>,
    egui_renderer: Option<EguiRenderer>,
    pub viewport_attachment: Option<Attachment>,
    pub ibl_maps: Option<crate::vulkan::ibl::IBLMaps>,
    /// Global settings for the renderer (Bloom, Shadows, Post-FX, etc.)
    pub settings: RenderSettings,
    pub main_light_view_proj: spark_math::Mat4,
    pub current_view_proj: spark_math::Mat4,
    pub last_object_count: u32,
    pub last_transparent_count: u32,
    pub last_draw_calls: std::sync::atomic::AtomicU32,
    pub last_triangle_count: std::sync::atomic::AtomicU32,
    pub current_image_index: u32,
    pub pass_descriptor_versions:
        Vec<std::sync::Arc<std::sync::Mutex<std::collections::HashMap<String, u64>>>>,
    pub dummy_buffer: Buffer,
    pub current_packet: Option<crate::resource::FramePacket>,
    pub current_znear: f32,
    pub current_zfar: f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct GlobalUBO {
    pub vp: spark_math::Mat4,
    pub lvp: [spark_math::Mat4; 4],
    pub inv_vp: spark_math::Mat4,
    pub camera_pos: [f32; 4],
    pub frustum: [spark_math::Vec4; 6],
    pub cascade_splits: spark_math::Vec4,
}

impl Renderer {
    /// Cascaded shadow map texture size. Default is 2048x2048.
    pub const SHADOW_MAP_CASCADE_SIZE: u32 = 2048;

    /// Returns the active view-projection matrix used in the current frame.
    pub fn get_current_view_proj(&self) -> spark_math::Mat4 {
        self.current_view_proj
    }

    /// Returns the current rendering settings.
    pub fn get_settings(&self) -> &RenderSettings {
        &self.settings
    }

    /// Updates the rendering settings.
    pub fn update_settings(&mut self, settings: RenderSettings) {
        self.settings = settings;
    }

    /// Creates a new Renderer instance.
    /// Initializes Vulkan context, device, swapchain, and core resource managers.
    pub fn new(
        window: &Window,
        ui_shaders: Option<(&[u32], &[u32])>,
    ) -> Result<Self, RendererError> {
        let context = VulkanContext::new(window)?;
        let device =
            VulkanDevice::new(&context.instance, &context.surface_loader, context.surface)?;
        let swapchain = VulkanSwapchain::new(
            &context.instance,
            &device.device,
            device.pdevice,
            &context.surface_loader,
            context.surface,
            window.inner_size().width,
            window.inner_size().height,
        )?;

        let gbuffer = GBuffer::new(
            &device,
            swapchain.extent,
            device.msaa_samples,
            device.depth_format,
        )?;

        let common_sampler = Self::create_common_sampler(&device.device)?;
        let dummy_buffer = Self::create_dummy_buffer(&device)?;
        let pipeline_cache = Self::create_pipeline_cache(&device.device)?;
        let global_descriptor_set_layout = Self::create_global_ds_layout(&device.device)?;

        let gpu_resource_manager =
            crate::vulkan::resource_manager::GpuResourceManager::new(&device)?;
        let mut frame_manager = crate::vulkan::frame_manager::FrameManager::new(&device)?;

        Self::allocate_global_descriptor_sets(
            &device,
            &gpu_resource_manager,
            &mut frame_manager,
            global_descriptor_set_layout,
        )?;

        let egui_renderer = ui_shaders.map(|(v, f)| {
            EguiRenderer::new(
                &device.device,
                v,
                f,
                swapchain.extent,
                vk::SampleCountFlags::TYPE_1,
            )
        });

        let mut renderer = Self {
            context,
            device,
            resource_tracker: crate::resource::ResourceTracker::new(),
            gpu_resource_manager,
            frame_manager,
            swapchain,
            global_descriptor_set_layout,
            light_count: 0,
            render_graph: crate::graph::RenderGraph::new(),
            pipeline_cache,
            pipeline: None,
            default_descriptor_set: vk::DescriptorSet::null(),
            scene_view_matrix_for_pos: spark_math::Mat4::IDENTITY,
            prev_view_proj: spark_math::Mat4::IDENTITY,
            common_sampler,
            common_shadow_view: vk::ImageView::null(),
            hiz_view: vk::ImageView::null(),
            egui_renderer,
            viewport_attachment: None,
            ibl_maps: None,
            settings: RenderSettings::default(),
            main_light_view_proj: spark_math::Mat4::IDENTITY,
            current_view_proj: spark_math::Mat4::IDENTITY,
            last_object_count: 0,
            last_transparent_count: 0,
            last_draw_calls: std::sync::atomic::AtomicU32::new(0),
            last_triangle_count: std::sync::atomic::AtomicU32::new(0),
            current_image_index: 0,
            pass_descriptor_versions: (0..MAX_FRAMES_IN_FLIGHT)
                .map(|_| {
                    std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()))
                })
                .collect(),
            dummy_buffer,
            as_manager: std::sync::Arc::new(std::sync::Mutex::new(
                crate::vulkan::as_manager::AccelerationStructureManager::new(),
            )),
            current_packet: None,
            current_znear: 0.1,
            current_zfar: 100.0,
        };

        renderer.init_gbuffer_attachments(gbuffer);
        renderer.init_default_resources();
        renderer.init_common_views();

        Ok(renderer)
    }

    fn allocate_global_descriptor_sets(
        device: &VulkanDevice,
        gpu_resource_manager: &crate::vulkan::resource_manager::GpuResourceManager,
        frame_manager: &mut crate::vulkan::frame_manager::FrameManager,
        layout: vk::DescriptorSetLayout,
    ) -> Result<(), RendererError> {
        let global_layouts = vec![layout; MAX_FRAMES_IN_FLIGHT];
        let global_descriptor_sets = unsafe {
            device.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(gpu_resource_manager.descriptor_pool)
                    .set_layouts(&global_layouts),
            )?
        };

        for (i, frame) in frame_manager.frames.iter_mut().enumerate() {
            frame.global_descriptor_set = global_descriptor_sets[i];
        }
        Ok(())
    }

    fn init_common_views(&mut self) {
        if let Some(ref tex) = self.gpu_resource_manager.default_texture {
            self.common_shadow_view = tex.view;
            self.hiz_view = tex.view;
        }
    }

    fn create_common_sampler(device: &ash::Device) -> Result<vk::Sampler, vk::Result> {
        unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR),
                None,
            )
        }
    }

    fn create_dummy_buffer(device: &VulkanDevice) -> Result<Buffer, RendererError> {
        device
            .create_buffer(
                64,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .map_err(|_| RendererError::NoSuitableDevice)
    }

    fn create_pipeline_cache(device: &ash::Device) -> Result<vk::PipelineCache, vk::Result> {
        unsafe { device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None) }
    }

    fn create_global_ds_layout(
        device: &ash::Device,
    ) -> Result<vk::DescriptorSetLayout, vk::Result> {
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX),
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(6)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(7)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];

        unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
    }

    fn init_gbuffer_attachments(&mut self, gbuffer: GBuffer) {
        self.render_graph
            .physical_attachments
            .insert("GBufferHDR".to_string(), gbuffer.hdr);
        self.render_graph
            .physical_attachments
            .insert("GBufferAlbedo".to_string(), gbuffer.albedo);
        self.render_graph
            .physical_attachments
            .insert("GBufferNormal".to_string(), gbuffer.normal);
        self.render_graph
            .physical_attachments
            .insert("GBufferPBR".to_string(), gbuffer.pbr);
        self.render_graph
            .physical_attachments
            .insert("GBufferVelocity".to_string(), gbuffer.velocity);
        self.render_graph
            .physical_attachments
            .insert("GBufferDepth".to_string(), gbuffer.depth);
    }

    pub fn set_global_buffers(&mut self, vertex: Buffer, index: Buffer) {
        self.gpu_resource_manager.global_vertex_buffer = Some(vertex);
        self.gpu_resource_manager.global_index_buffer = Some(index);
    }

    pub fn set_material_buffer(&mut self, buffer: Buffer) {
        self.gpu_resource_manager.global_material_buffer = Some(buffer);
    }

    pub fn set_pipeline(&mut self, pipeline: Pipeline) {
        if let Some(old) = &self.pipeline {
            if old.descriptor_set_layout != pipeline.descriptor_set_layout {
                self.gpu_resource_manager.texture_descriptor_sets.clear();
            }
        }
        self.pipeline = Some(pipeline);
        self.init_default_resources();
    }

    fn init_default_resources(&mut self) {
        if self.gpu_resource_manager.default_texture.is_none() {
            let white_pixel = [255u8, 255, 255, 255];
            let img = image::DynamicImage::ImageRgba8(
                image::RgbaImage::from_raw(1, 1, white_pixel.to_vec())
                    .expect("Failed to create 1x1 white texture"),
            );
            let tex = self.create_texture_from_image(&img);
            self.gpu_resource_manager.default_texture = Some(tex);
        }

        let (view, sampler) = if let Some(ref tex) = self.gpu_resource_manager.default_texture {
            (tex.view, tex.sampler)
        } else {
            return;
        };

        let pipeline_layout = if let Some(pipeline) = &self.pipeline {
            pipeline.descriptor_set_layout
        } else {
            return;
        };

        let device = &self.device.device;
        let ds = *self
            .gpu_resource_manager
            .texture_descriptor_sets
            .entry(view)
            .or_insert_with(|| unsafe {
                device
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(self.gpu_resource_manager.descriptor_pool)
                            .set_layouts(&[pipeline_layout]),
                    )
                    .expect("Failed to allocate texture descriptor set")[0]
            });

        let img_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(view)
            .sampler(sampler)];
        let writes = [vk::WriteDescriptorSet::default()
            .dst_set(ds)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&img_info)];
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }

        self.default_descriptor_set = ds;
    }

    pub fn update_all_descriptor_sets(&mut self) {
        for pass_node in &self.render_graph.passes {
            if pass_node.pass.is_enabled(self) {
                pass_node.pass.update_descriptor_sets(self);
            }
        }
        self.ensure_global_descriptor_set();
    }

    pub fn update_pass_descriptors_if_needed(&mut self, current_frame: usize) {
        let mut passes_to_update = Vec::new();
        for (i, pass_node) in self.render_graph.passes.iter().enumerate() {
            if pass_node.pass.is_enabled(self)
                && pass_node.pass.needs_descriptor_update(self, current_frame)
            {
                passes_to_update.push(i);
            }
        }

        for idx in passes_to_update {
            let pass_name = self.render_graph.passes[idx].pass.name().to_string();
            self.render_graph.passes[idx]
                .pass
                .update_descriptor_sets(self);

            // Note: This logic assumes that if update_descriptor_sets is called,
            // the pass is now up-to-date with whatever resource it tracked in needs_descriptor_update.
            self.pass_descriptor_versions[current_frame]
                .lock()
                .expect("Failed to lock pass descriptor versions")
                .insert(pass_name, 1);
        }
    }

    pub fn ensure_global_descriptor_set(&mut self) {
        let hiz_view = if self.hiz_view != vk::ImageView::null() {
            self.hiz_view
        } else {
            self.common_shadow_view
        };

        let grid_buf = self.get_resource_buffer("ClusteredPass", "light_grid", 0);
        let index_buf = self.get_resource_buffer("ClusteredPass", "index_list", 0);

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let frame = &self.frame_manager.frames[i];
            if let (Some(global_buffer), Some(obj_buf), Some(ind_buf), Some(cnt_buf)) = (
                frame.global_buffer.as_ref(),
                frame.object_data_buffer.as_ref(),
                frame.indirect_commands_buffer.as_ref(),
                frame.draw_count_buffer.as_ref(),
            ) {
                let ds = frame.global_descriptor_set;
                let buf_info = [vk::DescriptorBufferInfo::default()
                    .buffer(global_buffer.handle)
                    .range(global_buffer.size)];
                let obj_info = [vk::DescriptorBufferInfo::default()
                    .buffer(obj_buf.handle)
                    .range(obj_buf.size)];
                let ind_info = [vk::DescriptorBufferInfo::default()
                    .buffer(ind_buf.handle)
                    .range(ind_buf.size)];
                let cnt_info = [vk::DescriptorBufferInfo::default()
                    .buffer(cnt_buf.handle)
                    .range(cnt_buf.size)];

                let hiz_info = [vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(hiz_view)
                    .sampler(self.common_sampler)];

                let mut writes = vec![
                    vk::WriteDescriptorSet::default()
                        .dst_set(ds)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&buf_info),
                    vk::WriteDescriptorSet::default()
                        .dst_set(ds)
                        .dst_binding(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&obj_info),
                    vk::WriteDescriptorSet::default()
                        .dst_set(ds)
                        .dst_binding(2)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&ind_info),
                    vk::WriteDescriptorSet::default()
                        .dst_set(ds)
                        .dst_binding(3)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&cnt_info),
                    vk::WriteDescriptorSet::default()
                        .dst_set(ds)
                        .dst_binding(4)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&hiz_info),
                ];

                let mat_info;
                if let Some(ref mat_buf) = self.gpu_resource_manager.global_material_buffer {
                    mat_info = [vk::DescriptorBufferInfo::default()
                        .buffer(mat_buf.handle)
                        .range(mat_buf.size)];
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .dst_set(ds)
                            .dst_binding(5)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&mat_info),
                    );
                }

                let grid_info;
                if let Some(ref gb) = grid_buf {
                    grid_info = [vk::DescriptorBufferInfo::default()
                        .buffer(gb.handle)
                        .range(gb.size)];
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .dst_set(ds)
                            .dst_binding(6)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&grid_info),
                    );
                }

                let idx_info;
                if let Some(ref ib) = index_buf {
                    idx_info = [vk::DescriptorBufferInfo::default()
                        .buffer(ib.handle)
                        .range(ib.size)];
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .dst_set(ds)
                            .dst_binding(7)
                            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                            .buffer_info(&idx_info),
                    );
                }

                unsafe {
                    self.device.device.update_descriptor_sets(&writes, &[]);
                }
            }
        }
    }

    pub fn ensure_texture_descriptor(&mut self, texture: &Texture) {
        let pipeline_layout = if let Some(pipeline) = &self.pipeline {
            pipeline.descriptor_set_layout
        } else {
            return;
        };

        let device = &self.device.device;
        let ds = *self
            .gpu_resource_manager
            .texture_descriptor_sets
            .entry(texture.view)
            .or_insert_with(|| unsafe {
                device
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(self.gpu_resource_manager.descriptor_pool)
                            .set_layouts(&[pipeline_layout]),
                    )
                    .expect("Failed to allocate texture descriptor set")[0]
            });

        let img_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(texture.view)
            .sampler(texture.sampler)];
        let writes = [vk::WriteDescriptorSet::default()
            .dst_set(ds)
            .dst_binding(0)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&img_info)];
        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
    }

    pub fn update_lights_from_draw(&mut self, lights: &[crate::resource::LightDraw]) {
        let count = lights.len();
        if count == 0 {
            self.light_count = 0;
            return;
        }
        self.light_count = count as u32;
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct LD {
            pos: [f32; 4],
            dir: [f32; 4],
            col: [f32; 4],
            spot_angles: [f32; 4],
        }
        let ld: Vec<LD> = lights
            .iter()
            .map(|l| LD {
                pos: [l.position.x, l.position.y, l.position.z, l.range],
                dir: [
                    l.direction.x,
                    l.direction.y,
                    l.direction.z,
                    l.light_type as f32,
                ],
                col: [l.color.x, l.color.y, l.color.z, l.intensity],
                spot_angles: [l.spot_angles[0], l.spot_angles[1], 0.0, 0.0],
            })
            .collect();
        let sz = (ld.len() * std::mem::size_of::<LD>()) as u64;

        let mut needs_reupdate = false;
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let needs_new_buffer = {
                let frame = &self.frame_manager.frames[i];
                frame.light_buffer.as_ref().is_none_or(|b| b.size < sz)
            };

            if needs_new_buffer {
                let new_buffer = self.create_buffer(
                    sz,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                );

                let frame = &mut self.frame_manager.frames[i];
                if let Some(old) = frame.light_buffer.take() {
                    self.device.destroy_buffer(old);
                }
                frame.light_buffer = Some(new_buffer);
                needs_reupdate = true;
            }
        }

        if needs_reupdate {
            self.update_all_descriptor_sets();
        }

        let lb = self.frame_manager.frames[self.frame_manager.current_frame]
            .light_buffer
            .as_ref()
            .cloned()
            .expect("Light buffer was not initialized before upload");
        self.upload_to_buffer(&lb, &ld);
    }

    pub fn add_vertex_buffer(&mut self, buffer: Buffer) -> u32 {
        self.gpu_resource_manager.vertex_buffers.push(buffer);
        (self.gpu_resource_manager.vertex_buffers.len() - 1) as u32
    }

    pub fn add_instance_buffer(&mut self, buffer: Buffer) -> u32 {
        let frame = &mut self.frame_manager.frames[self.frame_manager.current_frame];
        frame.instance_pool.push(buffer);
        (frame.instance_pool.len() - 1) as u32
    }

    pub fn update_transparent_buffers(
        &mut self,
        commands: &[vk::DrawIndexedIndirectCommand],
        object_data: &[ObjectDataSSBO],
    ) {
        let frame_idx = self.frame_manager.current_frame;
        if commands.is_empty() {
            return;
        }

        let cmd_sz = std::mem::size_of_val(commands) as u64;

        let mut buffer = self.frame_manager.frames[frame_idx]
            .transparent_indirect_buffer
            .clone();
        if buffer.as_ref().is_none_or(|b| b.size < cmd_sz) {
            if let Some(old) = self.frame_manager.frames[frame_idx]
                .transparent_indirect_buffer
                .take()
            {
                self.device.destroy_buffer(old);
            }
            buffer = Some(self.create_buffer(
                cmd_sz.max(1024),
                vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ));
            self.frame_manager.frames[frame_idx].transparent_indirect_buffer = buffer.clone();
        }
        self.upload_to_buffer(
            &buffer.expect("Transparent indirect buffer missing"),
            commands,
        );

        let obj_sz = std::mem::size_of_val(object_data) as u64;
        let mut obj_buffer = self.frame_manager.frames[frame_idx]
            .transparent_object_buffer
            .clone();
        if obj_buffer.as_ref().is_none_or(|b| b.size < obj_sz) {
            if let Some(old) = self.frame_manager.frames[frame_idx]
                .transparent_object_buffer
                .take()
            {
                self.device.destroy_buffer(old);
            }
            obj_buffer = Some(self.create_buffer(
                obj_sz.max(1024),
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ));
            self.frame_manager.frames[frame_idx].transparent_object_buffer = obj_buffer.clone();
        }
        self.upload_to_buffer(
            &obj_buffer.expect("Transparent object buffer missing"),
            object_data,
        );
    }

    pub fn update_indirect_buffers(
        &mut self,
        commands: &[vk::DrawIndexedIndirectCommand],
        object_data: &[ObjectDataSSBO],
    ) {
        let frame_idx = self.frame_manager.current_frame;
        if commands.is_empty() {
            return;
        }

        let cmd_sz = std::mem::size_of_val(commands) as u64;

        let mut buffer = self.frame_manager.frames[frame_idx]
            .indirect_commands_buffer
            .clone();
        if buffer.as_ref().is_none_or(|b| b.size < cmd_sz) {
            if let Some(old) = self.frame_manager.frames[frame_idx]
                .indirect_commands_buffer
                .take()
            {
                self.device.destroy_buffer(old);
            }
            buffer = Some(self.create_buffer(
                cmd_sz.max(1024),
                vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ));
            self.frame_manager.frames[frame_idx].indirect_commands_buffer = buffer.clone();
        }
        self.upload_to_buffer(&buffer.expect("Indirect commands buffer missing"), commands);

        let obj_sz = std::mem::size_of_val(object_data) as u64;
        let mut obj_buffer = self.frame_manager.frames[frame_idx]
            .object_data_buffer
            .clone();
        if obj_buffer.as_ref().is_none_or(|b| b.size < obj_sz) {
            if let Some(old) = self.frame_manager.frames[frame_idx]
                .object_data_buffer
                .take()
            {
                self.device.destroy_buffer(old);
            }
            obj_buffer = Some(self.create_buffer(
                obj_sz.max(1024),
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ));
            self.frame_manager.frames[frame_idx].object_data_buffer = obj_buffer.clone();
        }
        self.upload_to_buffer(
            &obj_buffer.expect("Object data buffer missing"),
            object_data,
        );

        if self.frame_manager.frames[frame_idx]
            .draw_count_buffer
            .is_none()
        {
            self.frame_manager.frames[frame_idx].draw_count_buffer = Some(self.create_buffer(
                4,
                vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ));
        }
    }

    /// Gets a reusable instance buffer from the pool or creates a new one if necessary.
    pub fn get_or_create_instance_buffer(&mut self, sz: vk::DeviceSize) -> u32 {
        let frame_idx = self.frame_manager.current_frame;
        let idx = self.frame_manager.frames[frame_idx].instance_index;
        self.frame_manager.frames[frame_idx].instance_index += 1;

        if idx < self.frame_manager.frames[frame_idx].instance_pool.len() {
            if self.frame_manager.frames[frame_idx].instance_pool[idx].size >= sz {
                return idx as u32;
            }
            let old = self.frame_manager.frames[frame_idx].instance_pool[idx].clone();
            self.device.destroy_buffer(old);
        }

        let new_buffer = self
            .device
            .create_buffer(
                sz,
                vk::BufferUsageFlags::VERTEX_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .expect("Failed to create instance buffer");

        let frame = &mut self.frame_manager.frames[frame_idx];
        if idx < frame.instance_pool.len() {
            frame.instance_pool[idx] = new_buffer;
        } else {
            frame.instance_pool.push(new_buffer);
        }
        idx as u32
    }

    #[allow(clippy::too_many_arguments)]
    /// Executes the full rendering frame.
    /// Основная функция отрисовки кадра. Управляет синхронизацией, получением изображения
    /// из swapchain, записью команд и выводом на экран.
    ///
    /// # Arguments
    /// * `window` - The window to render to.
    /// * `egui_output` - Optional UI data to render on top of the scene.
    /// * `delta` - Time since last frame for temporal effects (TAA, Blur).
    pub fn draw_frame(
        &mut self,
        window: &Window,
        egui_output: Option<(egui::FullOutput, egui::Context)>,
        _object_count: u32,
        delta: f32,
    ) {
        let frame_idx = self.frame_manager.current_frame;
        let in_flight_fence = self.frame_manager.frames[frame_idx].in_flight;
        let image_available = self.frame_manager.frames[frame_idx].image_available;

        // 1. Wait for GPU and acquire next image
        self.wait_for_frame(in_flight_fence);
        let Some(image_index) = self.acquire_next_image(window, image_available) else {
            return;
        };

        // 2. Prepare frame resources (resets and pools)
        self.prepare_gpu_resources(frame_idx);

        // 3. Record and submit commands
        self.record_and_submit_frame(image_index, egui_output, delta);

        // 4. Present and advance
        self.present_frame(window, image_index);
        self.advance_frame();
    }

    fn wait_for_frame(&self, fence: vk::Fence) {
        unsafe {
            self.device
                .device
                .wait_for_fences(&[fence], true, u64::MAX)
                .expect("Failed to wait for fence");
        }
    }

    fn acquire_next_image(&mut self, window: &Window, semaphore: vk::Semaphore) -> Option<u32> {
        unsafe {
            let result = self.swapchain.loader.acquire_next_image(
                self.swapchain.handle,
                u64::MAX,
                semaphore,
                vk::Fence::null(),
            );

            match result {
                Ok((index, _)) => {
                    self.current_image_index = index;
                    Some(index)
                }
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    let _ = self.recreate_swapchain(window);
                    None
                }
                Err(e) => panic!("Failed to acquire swapchain image: {:?}", e),
            }
        }
    }

    fn prepare_gpu_resources(&self, frame_idx: usize) {
        unsafe {
            let frame = &self.frame_manager.frames[frame_idx];
            self.device
                .device
                .reset_fences(&[frame.in_flight])
                .expect("Failed to reset fence");

            self.device
                .device
                .reset_command_buffer(frame.command_buffer, vk::CommandBufferResetFlags::empty())
                .expect("Failed to reset command buffer");

            for pools in &self.device.thread_command_pools {
                let pool = pools[frame_idx];
                self.device
                    .device
                    .reset_command_pool(pool, vk::CommandPoolResetFlags::empty())
                    .expect("Failed to reset thread command pool");
            }
        }
    }

    fn record_and_submit_frame(
        &mut self,
        image_index: u32,
        egui_output: Option<(egui::FullOutput, egui::Context)>,
        delta: f32,
    ) {
        let frame_idx = self.frame_manager.current_frame;

        self.record_command_buffer(image_index, egui_output, delta);

        let frame = &self.frame_manager.frames[frame_idx];
        let wait_semaphores = [
            frame.image_available,
            self.frame_manager.culling_finished_semaphores[frame_idx],
        ];
        let wait_stages = [
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            vk::PipelineStageFlags::DRAW_INDIRECT,
        ];
        let signal_semaphores = [frame.render_finished];
        let command_buffers = [frame.command_buffer];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .signal_semaphores(&signal_semaphores);

        unsafe {
            self.device
                .device
                .queue_submit(self.device.graphics_queue, &[submit_info], frame.in_flight)
                .expect("Failed to submit frame to graphics queue");
        }
    }

    fn present_frame(&mut self, window: &Window, image_index: u32) {
        let frame_idx = self.frame_manager.current_frame;
        let signal_semaphores = [self.frame_manager.frames[frame_idx].render_finished];

        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(std::slice::from_ref(&self.swapchain.handle))
            .image_indices(std::slice::from_ref(&image_index));

        unsafe {
            let result = self
                .swapchain
                .loader
                .queue_present(self.device.graphics_queue, &present_info);

            match result {
                Ok(_) => {}
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                    let _ = self.recreate_swapchain(window);
                }
                Err(e) => panic!("Failed to present swapchain image: {:?}", e),
            }
        }
    }

    fn advance_frame(&mut self) {
        self.prev_view_proj = self.current_view_proj;
        self.frame_manager.frame_index += 1;
        self.frame_manager.current_frame =
            (self.frame_manager.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    pub fn set_common_shadow_view(&mut self, view: vk::ImageView) {
        self.common_shadow_view = view;
        self.update_all_descriptor_sets();
    }

    pub fn set_hiz_view(&mut self, view: vk::ImageView) {
        self.hiz_view = view;
        self.update_all_descriptor_sets();
    }

    fn record_command_buffer(
        &mut self,
        image_index: u32,
        egui_output: Option<(egui::FullOutput, egui::Context)>,
        delta: f32,
    ) {
        let command_buffer =
            self.frame_manager.frames[self.frame_manager.current_frame].command_buffer;
        let cf = self.frame_manager.current_frame;

        self.last_draw_calls
            .store(0, std::sync::atomic::Ordering::Relaxed);

        unsafe {
            self.device
                .device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
                .expect("Failed to begin command buffer");

            use crate::passes::RenderContext;

            let ctx = RenderContext {
                renderer: self,
                command_buffer,
                current_frame: cf,
                image_index,
                delta,
            };

            // 1. Execute RenderGraph
            self.render_graph.execute(&ctx, &[]);

            if let Some((output, egui_ctx)) = egui_output {
                let ext = self.swapchain.extent;
                if let Some(mut egui) = self.egui_renderer.take() {
                    egui.draw(
                        self,
                        command_buffer,
                        output,
                        [ext.width as f32, ext.height as f32],
                        &egui_ctx,
                    );
                    self.egui_renderer = Some(egui);
                }
            }

            self.device
                .device
                .end_command_buffer(command_buffer)
                .expect("Failed to end command buffer");
        }
    }

    pub fn create_viewport_attachment(&mut self, width: u32, height: u32) {
        if let Some(old) = self.viewport_attachment.take() {
            old.destroy(&self.device.device, &self.device.allocator);
        }

        let format = vk::Format::R16G16B16A16_SFLOAT;
        let attachment = Attachment::create_image_resource(
            &self.device,
            width,
            height,
            format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::SampleCountFlags::TYPE_1,
        )
        .expect("Failed to create viewport attachment");

        self.viewport_attachment = Some(attachment);
    }

    pub fn recreate_swapchain(&mut self, window: &Window) -> Result<(), RendererError> {
        unsafe {
            self.device.device.device_wait_idle()?;
            self.cleanup_swapchain();
            self.swapchain = VulkanSwapchain::new(
                &self.context.instance,
                &self.device.device,
                self.device.pdevice,
                &self.context.surface_loader,
                self.context.surface,
                window.inner_size().width,
                window.inner_size().height,
            )?;
            let extent = self.swapchain.extent;
            // Recreate G-Buffer attachments
            self.recreate_physical_attachment(
                "GBufferHDR",
                extent,
                vk::Format::R16G16B16A16_SFLOAT,
            )?;
            self.recreate_physical_attachment("GBufferAlbedo", extent, vk::Format::R8G8B8A8_UNORM)?;
            self.recreate_physical_attachment(
                "GBufferNormal",
                extent,
                vk::Format::A2B10G10R10_UNORM_PACK32,
            )?;
            self.recreate_physical_attachment("GBufferPBR", extent, vk::Format::R8G8B8A8_UNORM)?;
            self.recreate_physical_attachment(
                "GBufferVelocity",
                extent,
                vk::Format::R16G16_SFLOAT,
            )?;
            self.recreate_physical_attachment("GBufferDepth", extent, self.device.depth_format)?;

            if self.viewport_attachment.is_some() {
                self.create_viewport_attachment(extent.width, extent.height);
            }

            let mut passes = std::mem::take(&mut self.render_graph.passes);
            for node in &mut passes {
                node.pass.on_resize(self, extent);
            }
            self.render_graph.passes = passes;

            self.update_all_descriptor_sets();
        }
        Ok(())
    }

    fn cleanup_swapchain(&mut self) {
        unsafe {
            for (_, attachments) in self.render_graph.physical_attachments.iter_mut() {
                for a in attachments.drain(..) {
                    a.destroy(&self.device.device, &self.device.allocator);
                }
            }

            self.swapchain
                .loader
                .destroy_swapchain(self.swapchain.handle, None);
        }
    }

    /// Creates a GPU-resident texture from a CPU-side image.
    ///
    /// This handles automatic mipmap generation and bindless descriptor registration.
    pub fn create_texture_from_image(&mut self, img: &image::DynamicImage) -> Texture {
        let (w, h) = (img.width(), img.height());
        let mip = (((w.max(h) as f32).log2().floor()) as u32) + 1;
        let rgba = img.to_rgba8();
        let pixels = rgba.as_raw();
        let size = pixels.len() as u64;

        let frame_idx = self.frame_manager.current_frame;
        self.ensure_staging_buffer_capacity(frame_idx, size);

        let staging = self.frame_manager.frames[frame_idx]
            .texture_staging_buffer
            .as_ref()
            .expect("Staging buffer missing after ensure");
        unsafe {
            std::ptr::copy_nonoverlapping(pixels.as_ptr(), staging.ptr as *mut u8, pixels.len());
        }

        let (image, allocation) =
            self.create_image_basic(&crate::vulkan::device::ImageCreateParams {
                width: w,
                height: h,
                mip_levels: mip,
                format: vk::Format::R8G8B8A8_SRGB,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED,
                properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
                samples: vk::SampleCountFlags::TYPE_1,
            });

        self.transition_image_layout_basic(
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            mip,
        );

        let cb = self.begin_single_time_commands();
        let region = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width: w,
                height: h,
                depth: 1,
            });

        unsafe {
            self.device.device.cmd_copy_buffer_to_image(
                cb,
                staging.handle,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }
        self.end_single_time_commands(cb);

        self.generate_mipmaps(image, vk::Format::R8G8B8A8_SRGB, w, h, mip);
        let view = self.create_image_view_basic(image, vk::Format::R8G8B8A8_SRGB, mip);
        let sampler = self.create_texture_sampler(mip);

        let bindless_index = self.gpu_resource_manager.bindless.allocate_index();
        self.gpu_resource_manager.bindless.update_texture(
            &self.device.device,
            bindless_index,
            view,
            sampler,
        );

        Texture {
            image,
            allocation: Some(allocation),
            view,
            sampler,
            mip_levels: mip,
            bindless_index,
        }
    }

    fn ensure_staging_buffer_capacity(&mut self, frame_idx: usize, sz: u64) {
        let needs_new = {
            let f = &self.frame_manager.frames[frame_idx];
            f.texture_staging_buffer
                .as_ref()
                .is_none_or(|b| b.size < sz)
        };

        if needs_new {
            if let Some(old) = self.frame_manager.frames[frame_idx]
                .texture_staging_buffer
                .take()
            {
                self.device.destroy_buffer(old);
            }
            let b = self.create_buffer(
                sz.max(1024 * 1024),
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            self.frame_manager.frames[frame_idx].texture_staging_buffer = Some(b);
        }
    }

    pub fn destroy_texture(&mut self, mut t: Texture) {
        self.gpu_resource_manager
            .texture_descriptor_sets
            .remove(&t.view);

        self.gpu_resource_manager
            .bindless
            .deallocate_index(t.bindless_index);

        unsafe {
            self.device.device.destroy_sampler(t.sampler, None);
            self.device.device.destroy_image_view(t.view, None);
            self.device.device.destroy_image(t.image, None);
            if let Some(alloc) = t.allocation.take() {
                self.device
                    .allocator
                    .lock()
                    .expect("Failed to lock allocator")
                    .free(alloc)
                    .expect("Failed to free texture allocation");
            }
        }
    }

    pub fn create_image_basic(
        &self,
        params: &crate::vulkan::device::ImageCreateParams,
    ) -> (vk::Image, gpu_allocator::vulkan::Allocation) {
        self.device
            .create_image(params)
            .expect("Failed to create image")
    }

    pub fn create_texture_sampler(&self, mip: u32) -> vk::Sampler {
        let props = unsafe {
            self.context
                .instance
                .get_physical_device_properties(self.device.pdevice)
        };
        unsafe {
            self.device
                .device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR)
                        .address_mode_u(vk::SamplerAddressMode::REPEAT)
                        .address_mode_v(vk::SamplerAddressMode::REPEAT)
                        .address_mode_w(vk::SamplerAddressMode::REPEAT)
                        .anisotropy_enable(true)
                        .max_anisotropy(props.limits.max_sampler_anisotropy)
                        .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
                        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                        .min_lod(0.0)
                        .max_lod(mip as f32),
                    None,
                )
                .expect("Failed to create texture sampler")
        }
    }

    fn generate_mipmaps(&self, i: vk::Image, _f: vk::Format, w: u32, h: u32, mip: u32) {
        let props = unsafe {
            self.context
                .instance
                .get_physical_device_format_properties(self.device.pdevice, _f)
        };
        if !props
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
        {
            panic!("Linear filtering not supported");
        }
        let cb = self.begin_single_time_commands();
        let mut bar = vk::ImageMemoryBarrier::default()
            .image(i)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_array_layer: 0,
                layer_count: 1,
                level_count: 1,
                ..Default::default()
            });
        let (mut mw, mut mh) = (w as i32, h as i32);
        for j in 1..mip {
            bar.subresource_range.base_mip_level = j - 1;
            bar.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            bar.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            bar.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            bar.dst_access_mask = vk::AccessFlags::TRANSFER_READ;
            unsafe {
                self.device.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[bar],
                );
            }
            let blit = vk::ImageBlit::default()
                .src_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D { x: mw, y: mh, z: 1 },
                ])
                .src_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: j - 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .dst_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: (mw / 2).max(1),
                        y: (mh / 2).max(1),
                        z: 1,
                    },
                ])
                .dst_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: j,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            unsafe {
                self.device.device.cmd_blit_image(
                    cb,
                    i,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    i,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[blit],
                    vk::Filter::LINEAR,
                );
            }
            bar.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            bar.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            bar.src_access_mask = vk::AccessFlags::TRANSFER_READ;
            bar.dst_access_mask = vk::AccessFlags::SHADER_READ;
            unsafe {
                self.device.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[bar],
                );
            }
            mw = (mw / 2).max(1);
            mh = (mh / 2).max(1);
        }
        bar.subresource_range.base_mip_level = mip - 1;
        bar.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        bar.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        bar.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        bar.dst_access_mask = vk::AccessFlags::SHADER_READ;
        unsafe {
            self.device.device.cmd_pipeline_barrier(
                cb,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[bar],
            );
        }
        self.end_single_time_commands(cb);
    }

    pub fn get_buffer(&self, id: u32) -> Option<&Buffer> {
        self.gpu_resource_manager.vertex_buffers.get(id as usize)
    }
    pub fn get_instance_buffer(&self, id: u32) -> Option<&Buffer> {
        self.frame_manager.frames[self.frame_manager.current_frame]
            .instance_pool
            .get(id as usize)
    }
    /// Resets the instance buffer index for the current frame to allow reuse in the next cycle.
    pub fn clear_instance_buffers(&mut self) {
        self.frame_manager.frames[self.frame_manager.current_frame].instance_index = 0;
    }
    /// Returns the graphics queue handle.
    pub fn get_graphics_queue(&self) -> vk::Queue {
        self.device.graphics_queue
    }
    /// Returns the configured MSAA sample count.
    pub fn get_msaa_samples(&self) -> vk::SampleCountFlags {
        self.device.msaa_samples
    }

    pub fn get_pass_resource_view(
        &self,
        _pass_name: &str,
        resource_name: &str,
        frame_index: usize,
    ) -> Option<vk::ImageView> {
        let resource_name = self
            .render_graph
            .aliased_resources
            .get(resource_name)
            .map(|s| s.as_str())
            .unwrap_or(resource_name);

        // Check graph-managed physical attachments
        if let Some(attachments) = self.render_graph.physical_attachments.get(resource_name) {
            return Some(attachments[frame_index].view);
        }
        // Check graph-managed transient attachments
        if let Some(attachments) = self.render_graph.transient_attachments.get(resource_name) {
            return Some(attachments[frame_index].view);
        }

        for pass_node in &self.render_graph.passes {
            if let Some(view) = pass_node.pass.get_resource_view(resource_name, frame_index) {
                return Some(view);
            }
        }
        None
    }

    pub fn get_resource_buffer(
        &self,
        _pass_name: &str,
        resource_name: &str,
        frame_index: usize,
    ) -> Option<Buffer> {
        if resource_name == "light_buffer" || resource_name == "Lights" {
            return self.frame_manager.frames[frame_index].light_buffer.clone();
        }
        if resource_name == "object_data_buffer" || resource_name == "MeshData" {
            return self.frame_manager.frames[frame_index]
                .object_data_buffer
                .clone();
        }
        if resource_name == "Vertices" {
            return self.gpu_resource_manager.global_vertex_buffer.clone();
        }
        if resource_name == "Indices" {
            return self.gpu_resource_manager.global_index_buffer.clone();
        }
        if resource_name == "Materials" {
            return self.gpu_resource_manager.global_material_buffer.clone();
        }

        for pass_node in &self.render_graph.passes {
            if let Some(buffer) = pass_node
                .pass
                .get_resource_buffer(resource_name, frame_index)
            {
                return Some(buffer);
            }
        }
        None
    }
    /// Returns the current swapchain extent.
    pub fn get_extent(&self) -> vk::Extent2D {
        self.swapchain.extent
    }

    pub fn register_egui_texture(
        &mut self,
        view: vk::ImageView,
        sampler: vk::Sampler,
    ) -> egui::TextureId {
        let mut egui_renderer = self
            .egui_renderer
            .take()
            .expect("EguiRenderer not initialized");
        let id = egui_renderer.register_native_texture(self, view, sampler);
        self.egui_renderer = Some(egui_renderer);
        id
    }

    /// Calculates the projection jitter for temporal anti-aliasing.
    pub fn get_jitter(&self) -> [f32; 2] {
        let x =
            crate::vulkan::utils::halton((self.frame_manager.frame_index % 16) as u32 + 1, 2) - 0.5;
        let y =
            crate::vulkan::utils::halton((self.frame_manager.frame_index % 16) as u32 + 1, 3) - 0.5;
        [
            x / self.swapchain.extent.width as f32,
            y / self.swapchain.extent.height as f32,
        ]
    }

    pub fn add_render_pass<P: crate::passes::RenderPass + 'static>(
        &mut self,
        pass: P,
        inputs: &[&str],
        outputs: &[&str],
    ) {
        self.render_graph.add_pass(pass, inputs, outputs);
    }

    pub fn compile_render_graph(&mut self) {
        let mut graph = std::mem::take(&mut self.render_graph);
        graph.compile(self);
        self.render_graph = graph;
    }

    /// Calculates the six frustum planes from a view-projection matrix.
    /// Used for basic GPU-side culling and visibility testing.
    fn calculate_frustum_planes(view_proj: spark_math::Mat4) -> [spark_math::Vec4; 6] {
        let mut frustum = [spark_math::Vec4::ZERO; 6];
        let m = view_proj.transpose();
        frustum[0] = m.w_axis + m.x_axis;
        frustum[1] = m.w_axis - m.x_axis;
        frustum[2] = m.w_axis + m.y_axis;
        frustum[3] = m.w_axis - m.y_axis;
        frustum[4] = m.w_axis + m.z_axis;
        frustum[5] = m.w_axis - m.z_axis;

        for plane in &mut frustum {
            let len = spark_math::Vec3::new(plane.x, plane.y, plane.z).length();
            *plane /= len;
        }
        frustum
    }

    /// Prepares the GPU for a new frame by updating buffers and sorting meshes.
    ///
    /// # Arguments
    /// * `packet` - The frame data collected from the scene.
    ///
    /// # Returns
    /// * The total number of opaque objects to be rendered.
    ///
    /// Prepares the engine for rendering the current frame by collecting scene data,
    /// updating GPU buffers, and performing frustum culling.
    pub fn prepare_frame(
        &mut self,
        scene: &impl RenderableScene,
        _resource_manager: &impl RenderableResourceManager,
        asset_manager: &impl RenderableAssetManager,
    ) -> u32 {
        let extent = self.get_extent();
        let (_view_matrix, projection_matrix, frustum) =
            self.calculate_view_constants(scene, extent);

        let mut packet = scene.collect_frame_packet(frustum.as_ref(), asset_manager);
        packet.projection_matrix = projection_matrix;
        self.sort_transparent_meshes(&mut packet);

        self.scene_view_matrix_for_pos = packet.view_matrix;
        self.update_statistics(&packet);

        let total_objects = self.prepare_mesh_data(&packet);
        self.update_lights_from_draw(&packet.lights);
        self.update_global_ubo(&packet);

        self.current_packet = Some(packet);
        self.prepare_passes();

        self.last_object_count = total_objects;
        total_objects
    }

    fn calculate_view_constants(
        &mut self,
        scene: &impl RenderableScene,
        extent: vk::Extent2D,
    ) -> (
        spark_math::Mat4,
        spark_math::Mat4,
        Option<spark_math::Frustum>,
    ) {
        if let Some((view, proj, near, far)) = scene.get_active_camera_matrices(extent) {
            self.current_znear = near;
            self.current_zfar = far;
            let camera_matrix = proj * view;
            return (
                view,
                proj,
                Some(spark_math::Frustum::from_matrix(camera_matrix)),
            );
        }
        (spark_math::Mat4::IDENTITY, spark_math::Mat4::IDENTITY, None)
    }

    fn sort_transparent_meshes(&self, packet: &mut crate::resource::FramePacket) {
        let view_pos = packet.view_matrix.inverse().w_axis.xyz();
        packet.transparent_meshes.par_sort_by(|a, b| {
            let dist_a = (a.model.w_axis.xyz() - view_pos).length_squared();
            let dist_b = (b.model.w_axis.xyz() - view_pos).length_squared();
            dist_b
                .partial_cmp(&dist_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    fn update_statistics(&self, packet: &crate::resource::FramePacket) {
        let triangle_count: u32 = packet
            .opaque_meshes
            .iter()
            .map(|m| m.index_count / 3)
            .sum::<u32>()
            + packet
                .transparent_meshes
                .iter()
                .map(|m| m.index_count / 3)
                .sum::<u32>();
        self.last_triangle_count
            .store(triangle_count, std::sync::atomic::Ordering::Relaxed);
    }

    fn prepare_passes(&mut self) {
        let cf = self.frame_manager.current_frame;
        for i in 0..self.render_graph.passes.len() {
            if self.render_graph.passes[i].pass.is_enabled(self) {
                // We use a little unsafe here because we know prepare doesn't mutate the pass list itself
                let pass_ptr =
                    &self.render_graph.passes[i].pass as *const Box<dyn crate::passes::RenderPass>;
                unsafe {
                    (*pass_ptr).prepare(self, cf);
                }
            }
        }
    }

    /// Prepares mesh-related buffers (indirect commands, SSBOs).
    fn prepare_mesh_data(&mut self, packet: &crate::resource::FramePacket) -> u32 {
        let frame_idx = self.frame_manager.current_frame;

        // 1. Opaque meshes.
        let opaque_count = packet.opaque_meshes.len() as u32;
        if opaque_count > 0 {
            let cmd_sz = (opaque_count as usize
                * std::mem::size_of::<vk::DrawIndexedIndirectCommand>())
                as u64;
            let obj_sz = (opaque_count as usize * std::mem::size_of::<ObjectDataSSBO>()) as u64;

            self.ensure_indirect_buffers_capacity(frame_idx, cmd_sz, obj_sz);

            let (icb, odb) = {
                let frame = &self.frame_manager.frames[frame_idx];
                (
                    frame
                        .indirect_commands_buffer
                        .clone()
                        .expect("Indirect commands buffer missing"),
                    frame
                        .object_data_buffer
                        .clone()
                        .expect("Object data buffer missing"),
                )
            };

            Self::fill_mesh_data_buffers(&packet.opaque_meshes, &icb, &odb);
        }

        // 2. Transparent meshes.
        let trans_count = packet.transparent_meshes.len() as u32;
        if trans_count > 0 {
            let cmd_sz = (trans_count as usize
                * std::mem::size_of::<vk::DrawIndexedIndirectCommand>())
                as u64;
            let obj_sz = (trans_count as usize * std::mem::size_of::<ObjectDataSSBO>()) as u64;

            self.ensure_transparent_buffers_capacity(frame_idx, cmd_sz, obj_sz);

            let (ticb, todb) = {
                let frame = &self.frame_manager.frames[frame_idx];
                (
                    frame
                        .transparent_indirect_buffer
                        .clone()
                        .expect("Transparent indirect buffer missing"),
                    frame
                        .transparent_object_buffer
                        .clone()
                        .expect("Transparent object buffer missing"),
                )
            };

            Self::fill_mesh_data_buffers(&packet.transparent_meshes, &ticb, &todb);
        }
        self.last_transparent_count = trans_count;

        opaque_count
    }

    /// Fills the indirect command and object data buffers for a list of meshes.
    fn fill_mesh_data_buffers(
        meshes: &[crate::resource::MeshDraw],
        indirect_buffer: &Buffer,
        object_buffer: &Buffer,
    ) {
        use rayon::prelude::*;

        let cmd_ptr = indirect_buffer.ptr as usize;
        let obj_ptr = object_buffer.ptr as usize;

        meshes.par_iter().enumerate().for_each(|(i, mesh)| {
            let m = &mesh.model;
            unsafe {
                let obj_ptr = obj_ptr as *mut ObjectDataSSBO;
                let cmd_ptr = cmd_ptr as *mut vk::DrawIndexedIndirectCommand;
                *obj_ptr.add(i) = ObjectDataSSBO {
                    model_row0: m.col(0),
                    model_row1: m.col(1),
                    model_row2: m.col(2),
                    sphere: spark_math::Vec4::new(0.0, 0.0, 0.0, mesh.bounding_radius),
                    index_count: mesh.index_count,
                    first_index: mesh.first_index,
                    vertex_offset: mesh.vertex_offset,
                    material_index: mesh.material_index,
                };
                *cmd_ptr.add(i) = vk::DrawIndexedIndirectCommand {
                    index_count: mesh.index_count,
                    instance_count: 1,
                    first_index: mesh.first_index,
                    vertex_offset: mesh.vertex_offset,
                    first_instance: i as u32,
                };
            }
        });

        indirect_buffer
            .version
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        object_buffer
            .version
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    fn ensure_indirect_buffers_capacity(&mut self, frame_idx: usize, cmd_sz: u64, obj_sz: u64) {
        let (has_ic, has_ob, has_dc, ic_sz, ob_sz) = {
            let f = &self.frame_manager.frames[frame_idx];
            (
                f.indirect_commands_buffer.is_some(),
                f.object_data_buffer.is_some(),
                f.draw_count_buffer.is_some(),
                f.indirect_commands_buffer
                    .as_ref()
                    .map(|b| b.size)
                    .unwrap_or(0),
                f.object_data_buffer.as_ref().map(|b| b.size).unwrap_or(0),
            )
        };

        if !has_ic || ic_sz < cmd_sz {
            if let Some(old) = self.frame_manager.frames[frame_idx]
                .indirect_commands_buffer
                .take()
            {
                self.device.destroy_buffer(old);
            }
            let b = self.create_buffer(
                cmd_sz.max(1024),
                vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            self.frame_manager.frames[frame_idx].indirect_commands_buffer = Some(b);
        }
        if !has_ob || ob_sz < obj_sz {
            if let Some(old) = self.frame_manager.frames[frame_idx]
                .object_data_buffer
                .take()
            {
                self.device.destroy_buffer(old);
            }
            let b = self.create_buffer(
                obj_sz.max(1024),
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            self.frame_manager.frames[frame_idx].object_data_buffer = Some(b);
        }
        if !has_dc {
            let b = self.create_buffer(
                4,
                vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );
            self.frame_manager.frames[frame_idx].draw_count_buffer = Some(b);
        }
    }

    fn ensure_transparent_buffers_capacity(&mut self, frame_idx: usize, cmd_sz: u64, obj_sz: u64) {
        let (has_tic, has_tob, tic_sz, tob_sz) = {
            let f = &self.frame_manager.frames[frame_idx];
            (
                f.transparent_indirect_buffer.is_some(),
                f.transparent_object_buffer.is_some(),
                f.transparent_indirect_buffer
                    .as_ref()
                    .map(|b| b.size)
                    .unwrap_or(0),
                f.transparent_object_buffer
                    .as_ref()
                    .map(|b| b.size)
                    .unwrap_or(0),
            )
        };

        if !has_tic || tic_sz < cmd_sz {
            if let Some(old) = self.frame_manager.frames[frame_idx]
                .transparent_indirect_buffer
                .take()
            {
                self.device.destroy_buffer(old);
            }
            let b = self.create_buffer(
                cmd_sz.max(1024),
                vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            self.frame_manager.frames[frame_idx].transparent_indirect_buffer = Some(b);
        }
        if !has_tob || tob_sz < obj_sz {
            if let Some(old) = self.frame_manager.frames[frame_idx]
                .transparent_object_buffer
                .take()
            {
                self.device.destroy_buffer(old);
            }
            let b = self.create_buffer(
                obj_sz.max(1024),
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            self.frame_manager.frames[frame_idx].transparent_object_buffer = Some(b);
        }
    }

    /// Updates the global Uniform Buffer, calculates CSM matrices, and handles projection jitter.
    fn update_global_ubo(&mut self, packet: &crate::resource::FramePacket) {
        let extent = self.get_extent();
        let jitter = self.get_jitter();

        let mut projection = if packet.projection_matrix != spark_math::Mat4::IDENTITY {
            packet.projection_matrix
        } else {
            spark_math::Mat4::perspective_rh(
                45.0f32.to_radians(),
                extent.width as f32 / extent.height as f32,
                0.1,
                100.0,
            )
        };
        projection.col_mut(2).x += jitter[0] * projection.col(0).x;
        projection.col_mut(2).y += jitter[1] * projection.col(1).y;
        let view_proj = projection * packet.view_matrix;
        self.current_view_proj = view_proj;

        let cascade_splits = [10.0, 25.0, 50.0, 100.0];
        let mut light_view_projs = [spark_math::Mat4::IDENTITY; 4];

        let light_dir = spark_math::Vec3::new(0.5, -1.0, 0.5).normalize();
        let light_view = spark_math::Mat4::look_at_rh(
            -light_dir * 50.0,
            spark_math::Vec3::ZERO,
            spark_math::Vec3::Y,
        );

        let inv_vp = view_proj.inverse();
        for proj in &mut light_view_projs {
            let mut frustum_corners = [spark_math::Vec3::ZERO; 8];
            let mut idx = 0;
            for x in &[-1.0, 1.0] {
                for y in &[-1.0, 1.0] {
                    for z in &[0.0, 1.0] {
                        let pt = inv_vp * spark_math::Vec4::new(*x, *y, *z, 1.0);
                        frustum_corners[idx] = pt.xyz() / pt.w;
                        idx += 1;
                    }
                }
            }

            let mut min_p = spark_math::Vec3::new(f32::MAX, f32::MAX, f32::MAX);
            let mut max_p = spark_math::Vec3::new(f32::MIN, f32::MIN, f32::MIN);
            for corner in &frustum_corners {
                let light_space_p = light_view * corner.extend(1.0);
                min_p = min_p.min(light_space_p.xyz());
                max_p = max_p.max(light_space_p.xyz());
            }

            let shadow_res = Self::SHADOW_MAP_CASCADE_SIZE as f32;
            let world_units_per_texel = (max_p - min_p) / shadow_res;
            min_p = (min_p / world_units_per_texel).floor() * world_units_per_texel;
            max_p = (max_p / world_units_per_texel).floor() * world_units_per_texel;

            let light_proj = spark_math::Mat4::orthographic_rh(
                min_p.x,
                max_p.x,
                min_p.y,
                max_p.y,
                -max_p.z - 50.0,
                -min_p.z,
            );
            *proj = light_proj * light_view;
        }

        let current_frame = self.frame_manager.current_frame;
        self.frame_manager.frames[current_frame].light_view_projs = light_view_projs;
        self.main_light_view_proj = light_view_projs[0];

        let inv_v = packet.view_matrix.inverse();
        let camera_pos = [inv_v.w_axis.x, inv_v.w_axis.y, inv_v.w_axis.z, 1.0];
        let frustum = Self::calculate_frustum_planes(view_proj);

        let ubo = GlobalUBO {
            vp: view_proj,
            lvp: light_view_projs,
            inv_vp: view_proj.inverse(),
            camera_pos,
            frustum,
            cascade_splits: spark_math::Vec4::new(
                cascade_splits[0],
                cascade_splits[1],
                cascade_splits[2],
                cascade_splits[3],
            ),
        };

        if self.frame_manager.frames[0].global_buffer.is_none() {
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                let new_buffer = self.create_buffer(
                    std::mem::size_of::<GlobalUBO>() as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                );
                self.frame_manager.frames[i].global_buffer = Some(new_buffer);
            }
            self.ensure_global_descriptor_set();
        }
        let gb = self.frame_manager.frames[self.frame_manager.current_frame]
            .global_buffer
            .as_ref()
            .cloned()
            .expect("Global UBO buffer was not initialized before upload");
        self.upload_to_buffer(&gb, &[ubo]);
    }

    /// Returns the raw ash::Device.
    pub fn get_device(&self) -> &ash::Device {
        &self.device.device
    }

    pub fn get_thread_command_pool(&self) -> vk::CommandPool {
        let thread_idx = rayon::current_thread_index().unwrap_or(0);
        let pools =
            &self.device.thread_command_pools[thread_idx % self.device.thread_command_pools.len()];
        let current_frame = self.frame_manager.current_frame;
        pools[current_frame]
    }

    pub fn allocate_secondary_command_buffer(&self) -> vk::CommandBuffer {
        let pool = self.get_thread_command_pool();
        self.device
            .create_command_buffer(pool, vk::CommandBufferLevel::SECONDARY)
    }

    /// Uploads data to a GPU buffer.
    ///
    /// # Safety
    /// The user must ensure the buffer is not being used by the GPU when calling this,
    /// or that the buffer is host-coherent and visible.
    pub fn upload_to_buffer<T: Copy>(&self, b: &Buffer, data: &[T]) {
        self.device.upload_to_buffer(b, data);
    }
    pub fn create_buffer(
        &self,
        sz: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Buffer {
        self.device
            .create_buffer(sz, usage, properties)
            .expect("Failed to create buffer")
    }
    pub fn destroy_buffer(&self, buffer: Buffer) {
        self.device.destroy_buffer(buffer);
    }
    pub fn create_image_view_basic(&self, i: vk::Image, f: vk::Format, mip: u32) -> vk::ImageView {
        self.device.create_image_view(i, f, mip)
    }
    pub fn transition_image_layout_basic(
        &self,
        i: vk::Image,
        ol: vk::ImageLayout,
        nl: vk::ImageLayout,
        mip: u32,
    ) {
        self.device.transition_image_layout(i, ol, nl, mip)
    }
    pub fn copy_buffer_to_image_basic(
        &self,
        buffer: vk::Buffer,
        image: vk::Image,
        width: u32,
        height: u32,
    ) {
        let command_buffer = self.begin_single_time_commands();
        let region = vk::BufferImageCopy::default()
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });
        unsafe {
            self.device.device.cmd_copy_buffer_to_image(
                command_buffer,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );
        }
        self.end_single_time_commands(command_buffer);
    }
    pub fn begin_single_time_commands(&self) -> vk::CommandBuffer {
        self.device.begin_single_time_commands()
    }
    pub fn end_single_time_commands(&self, cb: vk::CommandBuffer) {
        self.device.end_single_time_commands(cb);
    }

    fn recreate_physical_attachment(
        &mut self,
        name: &str,
        extent: vk::Extent2D,
        format: vk::Format,
    ) -> Result<(), RendererError> {
        if let Some(attachments) = self.render_graph.physical_attachments.get_mut(name) {
            for a in attachments {
                a.recreate(&self.device, extent.width, extent.height, format)?;
            }
        }
        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.device.device.device_wait_idle().ok();

            for mut pass_node in std::mem::take(&mut self.render_graph.passes) {
                pass_node.pass.destroy(self);
            }

            self.render_graph
                .destroy_resources(&self.device.device, &self.device.allocator);

            self.cleanup_swapchain();
            self.frame_manager.destroy(&self.device);

            if let Some(p) = self.pipeline.take() {
                self.device
                    .device
                    .destroy_pipeline(p.graphics_pipeline, None);
                self.device.device.destroy_pipeline_layout(p.layout, None);
                self.device
                    .device
                    .destroy_descriptor_set_layout(p.descriptor_set_layout, None);
            }

            if let Some(mut e) = self.egui_renderer.take() {
                e.destroy(self);
            }
            if let Some(t) = self.gpu_resource_manager.default_texture.take() {
                self.destroy_texture(t);
            }

            self.gpu_resource_manager.destroy(&self.device);

            self.device
                .device
                .destroy_descriptor_set_layout(self.global_descriptor_set_layout, None);

            self.as_manager
                .lock()
                .expect("Failed to lock AS manager during cleanup")
                .cleanup(&self.device);

            self.device
                .device
                .destroy_sampler(self.common_sampler, None);
            self.device.destroy_buffer(std::mem::replace(
                &mut self.dummy_buffer,
                Buffer {
                    handle: vk::Buffer::null(),
                    allocation: std::sync::Arc::new(std::sync::Mutex::new(None)),
                    size: 0,
                    ptr: std::ptr::null_mut(),
                    address: 0,
                    version: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
                },
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn test_render_settings_default() {
        let settings = RenderSettings::default();
        assert!(settings.enable_shadows);
        assert_eq!(settings.gamma, 2.2);
    }

    #[test]
    fn test_buffer_struct() {
        let buffer = Buffer {
            handle: vk::Buffer::null(),
            allocation: Arc::new(Mutex::new(None)),
            size: 1024,
            ptr: std::ptr::null_mut(),
            address: 0,
            version: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        };
        assert_eq!(buffer.size, 1024);
        assert_eq!(buffer.handle, vk::Buffer::null());
    }

    #[test]
    fn test_light_data_layout() {
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct LD {
            pos: [f32; 4],
            dir: [f32; 4],
            col: [f32; 4],
            spot_angles: [f32; 4],
        }
        assert_eq!(std::mem::size_of::<LD>(), 64);
    }
}
