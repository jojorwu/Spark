pub mod error;
pub mod pipeline;
pub mod passes;
pub mod resource;
pub mod factory;
pub mod ui;
pub mod vertex;
pub mod vulkan;

use crate::error::RendererError;
use crate::pipeline::Pipeline;
pub use crate::resource::{Attachment, Buffer, RenderFrame, MAX_FRAMES_IN_FLIGHT, ObjectDataSSBO, MaterialDataSSBO};
use crate::ui::EguiRenderer;
use crate::vulkan::context::VulkanContext;
use crate::vulkan::device::VulkanDevice;
use crate::vulkan::gbuffer::GBuffer;
use crate::vulkan::swapchain::VulkanSwapchain;
use crate::vulkan::texture::Texture;
pub use ash;
use ash::vk;
use winit::window::Window;

/// The main renderer of the Spark Engine, responsible for managing the Vulkan context,
/// swapchain, and executing the rendering pipeline through a modular pass system.
pub struct Renderer {
    pub context: VulkanContext,
    pub device: VulkanDevice,
    swapchain: VulkanSwapchain,
    pub gbuffer: GBuffer,
    pub global_descriptor_set_layout: vk::DescriptorSetLayout,
    pub light_count: u32,
    pub render_passes: Vec<Box<dyn crate::passes::RenderPass>>,
    pub frames: [RenderFrame; MAX_FRAMES_IN_FLIGHT],
    pub pipeline_cache: vk::PipelineCache,
    current_frame: usize,
    pub pipeline: Option<Pipeline>,
    pub vertex_buffers: Vec<Buffer>,
    pub index_buffer: Option<Buffer>,
    pub global_vertex_buffer: Option<Buffer>,
    pub global_index_buffer: Option<Buffer>,
    pub global_material_buffer: Option<Buffer>,
    pub descriptor_pool: vk::DescriptorPool,
    pub texture_descriptor_sets: std::collections::HashMap<vk::ImageView, vk::DescriptorSet>,
    pub default_texture: Option<Texture>,
    pub default_descriptor_set: vk::DescriptorSet,
    pub scene_view_matrix_for_pos: spark_math::Mat4,
    pub prev_view_proj: spark_math::Mat4,
    pub frame_index: u64,
    pub common_sampler: vk::Sampler,
    pub common_shadow_view: vk::ImageView,
    pub hiz_view: vk::ImageView,
    egui_renderer: Option<EguiRenderer>,
    pub bindless_descriptor_set_layout: vk::DescriptorSetLayout,
    pub bindless_descriptor_set: vk::DescriptorSet,
    pub next_bindless_index: std::sync::atomic::AtomicU32,
    pub viewport_attachment: Option<Attachment>,
    pub ibl_maps: Option<crate::vulkan::ibl::IBLMaps>,
    pub exposure: f32,
    pub gamma: f32,
    pub enable_ssao: bool,
    pub enable_taa: bool,
    pub enable_shadows: bool,
    pub enable_volumetric: bool,
    pub enable_grid: bool,
    pub enable_ibl: bool,
    pub enable_bloom: bool,
    pub main_light_view_proj: spark_math::Mat4,
    pub current_view_proj: spark_math::Mat4,
    pub last_object_count: u32,
    pub current_image_index: u32,
    pub pass_descriptor_versions: Vec<std::collections::HashMap<String, u64>>,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct GlobalUBO {
    pub vp: spark_math::Mat4,
    pub lvp: [spark_math::Mat4; 4],
    pub inv_vp: spark_math::Mat4,
    pub camera_pos: [f32; 4],
    pub frustum: [spark_math::Vec4; 6],
    pub cascade_splits: [f32; 4],
}

impl Renderer {
    /// Cascaded shadow map texture size.
    pub const SHADOW_MAP_CASCADE_SIZE: u32 = 2048;

    /// Creates a new Renderer instance.
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

        let common_sampler = unsafe {
            device.device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR),
                None,
            )?
        };

        let (av, fi, in_f) = Self::create_sync_objects_impl(&device.device);
        let pipeline_cache = unsafe {
            device.device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)?
        };

        let global_descriptor_set_layout = unsafe {
            device.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[
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
                ]),
                None,
            )?
        };

        let descriptor_pool = Self::create_descriptor_pool_impl(&device.device);

        let bindless_descriptor_set_layout = unsafe {
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(10000)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
            let flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND];
            let mut binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                .binding_flags(&flags);
            device.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(&bindings)
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                    .push_next(&mut binding_flags),
                None,
            )?
        };

        let bindless_descriptor_set = unsafe {
            device.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&[bindless_descriptor_set_layout]),
            )? [0]
        };

        let global_descriptor_sets = unsafe {
            device.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&[global_descriptor_set_layout, global_descriptor_set_layout]),
            )?
        };

        let frames = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|i| {
                let alloc_info = vk::CommandBufferAllocateInfo::default()
                    .command_pool(device.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                let cb = unsafe { device.device.allocate_command_buffers(&alloc_info).unwrap()[0] };

                RenderFrame {
                    command_buffer: cb,
                    image_available: av[i],
                    render_finished: fi[i],
                    in_flight: in_f[i],
                    global_buffer: None,
                    light_buffer: None,
                    global_descriptor_set: global_descriptor_sets[i],
                    instance_pool: Vec::new(),
                    instance_index: 0,
                    indirect_commands_buffer: None,
                    object_data_buffer: None,
                    draw_count_buffer: None,
                }
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let egui_renderer = ui_shaders.map(|(v, f)| {
            EguiRenderer::new(
                &device.device,
                v,
                f,
                swapchain.extent,
                vk::SampleCountFlags::TYPE_1,
            )
        });

        Ok(Self {
            context,
            device,
            swapchain,
            gbuffer,
            global_descriptor_set_layout,
            light_count: 0,
            render_passes: Vec::new(),
            frames,
            pipeline_cache,
            current_frame: 0,
            pipeline: None,
            vertex_buffers: Vec::new(),
            index_buffer: None,
            global_vertex_buffer: None,
            global_index_buffer: None,
            global_material_buffer: None,
            descriptor_pool,
            texture_descriptor_sets: std::collections::HashMap::new(),
            default_texture: None,
            default_descriptor_set: vk::DescriptorSet::null(),
            scene_view_matrix_for_pos: spark_math::Mat4::IDENTITY,
            prev_view_proj: spark_math::Mat4::IDENTITY,
            frame_index: 0,
            common_sampler,
            common_shadow_view: vk::ImageView::null(),
            hiz_view: vk::ImageView::null(),
            egui_renderer,
            bindless_descriptor_set_layout,
            bindless_descriptor_set,
            next_bindless_index: std::sync::atomic::AtomicU32::new(0),
            viewport_attachment: None,
            ibl_maps: None,
            exposure: 1.0,
            gamma: 2.2,
            enable_ssao: true,
            enable_taa: true,
            enable_shadows: true,
            enable_volumetric: true,
            enable_grid: true,
            enable_ibl: true,
            enable_bloom: true,
            main_light_view_proj: spark_math::Mat4::IDENTITY,
            current_view_proj: spark_math::Mat4::IDENTITY,
            last_object_count: 0,
            current_image_index: 0,
            pass_descriptor_versions: (0..MAX_FRAMES_IN_FLIGHT).map(|_| std::collections::HashMap::new()).collect(),
        })
    }

    pub fn set_global_buffers(&mut self, vertex: Buffer, index: Buffer) {
        self.global_vertex_buffer = Some(vertex);
        self.global_index_buffer = Some(index);
    }

    pub fn set_material_buffer(&mut self, buffer: Buffer) {
        self.global_material_buffer = Some(buffer);
    }

    pub fn set_pipeline(&mut self, pipeline: Pipeline) {
        if let Some(old) = &self.pipeline {
            if old.descriptor_set_layout != pipeline.descriptor_set_layout {
                self.texture_descriptor_sets.clear();
            }
        }
        self.pipeline = Some(pipeline);
        self.init_default_resources();
    }

    fn init_default_resources(&mut self) {
        if self.default_texture.is_none() {
            let white_pixel = [255u8, 255, 255, 255];
            let img = image::DynamicImage::ImageRgba8(
                image::RgbaImage::from_raw(1, 1, white_pixel.to_vec()).unwrap(),
            );
            let tex = self.create_texture_from_image(&img);
            self.default_texture = Some(tex);
        }

        let view = self.default_texture.as_ref().unwrap().view;
        let sampler = self.default_texture.as_ref().unwrap().sampler;

        let pipeline_layout = if let Some(pipeline) = &self.pipeline {
            pipeline.descriptor_set_layout
        } else {
            return;
        };

        let device = &self.device.device;
        let ds = *self
            .texture_descriptor_sets
            .entry(view)
            .or_insert_with(|| unsafe {
                device
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(self.descriptor_pool)
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
        for pass in &self.render_passes {
            pass.update_descriptor_sets(self);
        }
        self.ensure_global_descriptor_set();
    }

    pub fn update_pass_descriptors_if_needed(&mut self, current_frame: usize) {
        // Collect passes that need update to avoid borrow issues during iteration
        let mut passes_to_update = Vec::new();
        for (i, pass) in self.render_passes.iter().enumerate() {
            if pass.needs_descriptor_update(self, current_frame) {
                passes_to_update.push(i);
            }
        }

        for idx in passes_to_update {
            self.render_passes[idx].update_descriptor_sets(self);
        }
    }

    pub fn ensure_global_descriptor_set(&mut self) {
        let hiz_view = if self.hiz_view != vk::ImageView::null() {
            self.hiz_view
        } else {
            self.common_shadow_view
        };

        let (grid_buf, index_buf) = {
            let mut g = None;
            let mut idx = None;
            for pass in &self.render_passes {
                if let Some(b) = pass.get_resource_buffer("light_grid") {
                    g = Some(b);
                }
                if let Some(b) = pass.get_resource_buffer("index_list") {
                    idx = Some(b);
                }
            }
            (g, idx)
        };

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let frame = &self.frames[i];
            if let (Some(global_buffer), Some(obj_buf), Some(ind_buf), Some(cnt_buf)) =
                (frame.global_buffer.as_ref(), frame.object_data_buffer.as_ref(), frame.indirect_commands_buffer.as_ref(), frame.draw_count_buffer.as_ref()) {
                let ds = frame.global_descriptor_set;
                let buf_info = [vk::DescriptorBufferInfo::default().buffer(global_buffer.handle).range(global_buffer.size)];
                let obj_info = [vk::DescriptorBufferInfo::default().buffer(obj_buf.handle).range(obj_buf.size)];
                let ind_info = [vk::DescriptorBufferInfo::default().buffer(ind_buf.handle).range(ind_buf.size)];
                let cnt_info = [vk::DescriptorBufferInfo::default().buffer(cnt_buf.handle).range(cnt_buf.size)];

                let hiz_info = [vk::DescriptorImageInfo::default()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(hiz_view)
                    .sampler(self.common_sampler)];

                let mut writes = vec![
                    vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(0).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&buf_info),
                    vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&obj_info),
                    vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&ind_info),
                    vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(3).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&cnt_info),
                    vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(4).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&hiz_info),
                ];

                let mat_info;
                if let Some(ref mat_buf) = self.global_material_buffer {
                    mat_info = [vk::DescriptorBufferInfo::default().buffer(mat_buf.handle).range(mat_buf.size)];
                    writes.push(vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(5).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&mat_info));
                }

                let grid_info;
                if let Some(ref gb) = grid_buf {
                    grid_info = [vk::DescriptorBufferInfo::default().buffer(gb.handle).range(gb.size)];
                    writes.push(vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(6).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&grid_info));
                }

                let idx_info;
                if let Some(ref ib) = index_buf {
                    idx_info = [vk::DescriptorBufferInfo::default().buffer(ib.handle).range(ib.size)];
                    writes.push(vk::WriteDescriptorSet::default().dst_set(ds).dst_binding(7).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&idx_info));
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
            .texture_descriptor_sets
            .entry(texture.view)
            .or_insert_with(|| unsafe {
                device
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(self.descriptor_pool)
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
            return;
        }
        self.light_count = count as u32;
        #[repr(C)]
        #[derive(Copy, Clone)]
        struct LD {
            pos: [f32; 4],
            col: [f32; 4],
        }
        let ld: Vec<LD> = lights
            .iter()
            .map(|l| LD {
                pos: [l.position.x, l.position.y, l.position.z, 1.0],
                col: [l.color.x, l.color.y, l.color.z, l.intensity],
            })
            .collect();
        let sz = (ld.len() * std::mem::size_of::<LD>()) as u64;

        let mut needs_reupdate = false;
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let needs_new_buffer = {
                let frame = &self.frames[i];
                frame.light_buffer.is_none() || frame.light_buffer.as_ref().unwrap().size < sz
            };

            if needs_new_buffer {
                let new_buffer = self.create_buffer(
                    sz,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                );

                let frame = &mut self.frames[i];
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

        let lb = self.frames[self.current_frame].light_buffer.as_ref().cloned().unwrap();
        self.upload_to_buffer(&lb, &ld);
    }

    pub fn add_vertex_buffer(&mut self, buffer: Buffer) -> u32 {
        self.vertex_buffers.push(buffer);
        (self.vertex_buffers.len() - 1) as u32
    }

    pub fn add_instance_buffer(&mut self, buffer: Buffer) -> u32 {
        let frame = &mut self.frames[self.current_frame];
        frame.instance_pool.push(buffer);
        (frame.instance_pool.len() - 1) as u32
    }

    pub fn update_indirect_buffers(
        &mut self,
        commands: &[vk::DrawIndexedIndirectCommand],
        object_data: &[ObjectDataSSBO],
    ) {
        let frame_idx = self.current_frame;
        let cmd_sz = std::mem::size_of_val(commands) as u64;

        let mut buffer = self.frames[frame_idx].indirect_commands_buffer.clone();
        if buffer.is_none() || buffer.as_ref().unwrap().size < cmd_sz {
            if let Some(old) = self.frames[frame_idx].indirect_commands_buffer.take() {
                self.device.destroy_buffer(old);
            }
            buffer = Some(self.create_buffer(
                cmd_sz.max(1024),
                vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ));
            self.frames[frame_idx].indirect_commands_buffer = buffer.clone();
        }
        self.upload_to_buffer(&buffer.unwrap(), commands);

        let obj_sz = std::mem::size_of_val(object_data) as u64;
        let mut obj_buffer = self.frames[frame_idx].object_data_buffer.clone();
        if obj_buffer.is_none() || obj_buffer.as_ref().unwrap().size < obj_sz {
            if let Some(old) = self.frames[frame_idx].object_data_buffer.take() {
                self.device.destroy_buffer(old);
            }
            obj_buffer = Some(self.create_buffer(
                obj_sz.max(1024),
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ));
            self.frames[frame_idx].object_data_buffer = obj_buffer.clone();
        }
        self.upload_to_buffer(&obj_buffer.unwrap(), object_data);

        if self.frames[frame_idx].draw_count_buffer.is_none() {
            self.frames[frame_idx].draw_count_buffer = Some(self.create_buffer(
                4,
                vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ));
        }
    }

    /// Gets a reusable instance buffer from the pool or creates a new one if necessary.
    pub fn get_or_create_instance_buffer(&mut self, sz: vk::DeviceSize) -> u32 {
        let frame_idx = self.current_frame;
        let idx = self.frames[frame_idx].instance_index;
        self.frames[frame_idx].instance_index += 1;

        if idx < self.frames[frame_idx].instance_pool.len() {
            if self.frames[frame_idx].instance_pool[idx].size >= sz {
                return idx as u32;
            }
            let old = self.frames[frame_idx].instance_pool[idx].clone();
            self.device.destroy_buffer(old);
        }

        let new_buffer = self.device.create_buffer(
            sz,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ).expect("Failed to create instance buffer");

        let frame = &mut self.frames[frame_idx];
        if idx < frame.instance_pool.len() {
            frame.instance_pool[idx] = new_buffer;
        } else {
            frame.instance_pool.push(new_buffer);
        }
        idx as u32
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw_frame(
        &mut self,
        window: &Window,
        egui_output: Option<(egui::FullOutput, egui::Context)>,
        _object_count: u32,
    ) {
        // Retrieve view_proj and light_view_proj from the current frame's global buffer
        // For simplicity, we'll keep them as parameters or fetch from UBO.
        // Since prepare_frame just uploaded them, we can use them from there or just pass them.
        // Let's modify prepare_frame to store them in the Renderer.
        let (image_available, in_flight, command_buffer, render_finished) = {
            let frame = &self.frames[self.current_frame];
            (frame.image_available, frame.in_flight, frame.command_buffer, frame.render_finished)
        };

        unsafe {
            self.device
                .device
                .wait_for_fences(&[in_flight], true, u64::MAX)
                .expect("Failed to wait for fence");
            let result = self.swapchain.loader.acquire_next_image(
                self.swapchain.handle,
                u64::MAX,
                image_available,
                vk::Fence::null(),
            );
            let image_index = match result {
                Ok((index, _)) => {
                    self.current_image_index = index;
                    index
                },
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    let _ = self.recreate_swapchain(window);
                    return;
                }
                Err(e) => panic!("Failed to acquire swapchain image: {:?}", e),
            };
            self.device
                .device
                .reset_fences(&[in_flight])
                .expect("Failed to reset fence");
            self.device
                .device
                .reset_command_buffer(
                    command_buffer,
                    vk::CommandBufferResetFlags::empty(),
                )
                .expect("Failed to reset command buffer");

            // Clean up secondary command buffers from previous frame
            let thread_pools = self.device.thread_command_pools.clone();
            for pool in thread_pools {
                self.device.device.reset_command_pool(pool, vk::CommandPoolResetFlags::empty()).unwrap();
            }
            self.record_command_buffer(
                image_index,
                egui_output,
            );
            let s_available = [image_available];
            let s_finished = [render_finished];
            let w_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let c_buffers = [command_buffer];
            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&s_available)
                .wait_dst_stage_mask(&w_stages)
                .command_buffers(&c_buffers)
                .signal_semaphores(&s_finished);
            self.device
                .device
                .queue_submit(
                    self.device.graphics_queue,
                    &[submit_info],
                    in_flight,
                )
                .unwrap();
            let result = self.swapchain.loader.queue_present(
                self.device.graphics_queue,
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(&s_finished)
                    .swapchains(&[self.swapchain.handle])
                    .image_indices(&[image_index]),
            );
            match result {
                Ok(_) => {}
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Err(vk::Result::SUBOPTIMAL_KHR) => {
                    let _ = self.recreate_swapchain(window);
                }
                Err(e) => panic!("Failed to present swapchain image: {:?}", e),
            }
            let cvp = self.current_view_proj;
            self.prev_view_proj = cvp;
            self.frame_index += 1;
            self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        }
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
    ) {
        let command_buffer = self.frames[self.current_frame].command_buffer;
        let cf = self.current_frame;

        unsafe {
            self.device
                .device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
                .unwrap();

            use crate::passes::RenderContext;

            // Record passes sequentially for now to avoid complex borrowing/Vulkan state issues
            // while maintaining the modular architecture.
            let ctx = RenderContext {
                renderer: self,
                command_buffer,
                current_frame: cf,
                image_index,
            };

            for pass in &self.render_passes {
                pass.record_commands(&ctx);
            }

            if let Some((output, egui_ctx)) = egui_output {
                let ext = self.swapchain.extent;
                if let Some(mut egui) = self.egui_renderer.take() {
                    egui.draw(self, command_buffer, output, [ext.width as f32, ext.height as f32], &egui_ctx);
                    self.egui_renderer = Some(egui);
                }
            }

            self.device.device.end_command_buffer(command_buffer).unwrap();
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
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::SampleCountFlags::TYPE_1,
        ).expect("Failed to create viewport attachment");

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
            self.gbuffer.recreate(
                &self.device,
                self.swapchain.extent,
                self.device.msaa_samples,
                self.device.depth_format,
            )?;
            self.update_all_descriptor_sets();
        }
        Ok(())
    }

    fn cleanup_swapchain(&mut self) {
        unsafe {
            self.gbuffer.destroy(&self.device.device, &self.device.allocator);
            self.swapchain
                .loader
                .destroy_swapchain(self.swapchain.handle, None);
        }
    }

    pub fn create_texture_from_image(&self, img: &image::DynamicImage) -> Texture {
        let (w, h) = (img.width(), img.height());
        let mip = (((w.max(h) as f32).log2().floor()) as u32) + 1;
        let rgba = img.to_rgba8();
        let pix = rgba.as_raw();
        let sz = pix.len() as u64;
        let st = self.create_buffer(
            sz,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        self.upload_to_buffer(&st, pix);
        let (i, m) = self.create_image_basic(
            &crate::vulkan::device::ImageCreateParams {
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
            }
        );
        self.transition_image_layout_basic(
            i,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            mip,
        );
        let cb = self.begin_single_time_commands();
        unsafe {
            self.device.device.cmd_copy_buffer_to_image(
                cb,
                st.handle,
                i,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
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
                    })],
            );
        }
        self.end_single_time_commands(cb);
        self.generate_mipmaps(i, vk::Format::R8G8B8A8_SRGB, w, h, mip);
        let v = self.create_image_view_basic(i, vk::Format::R8G8B8A8_SRGB, mip);
        let s = self.create_texture_sampler(mip);
        self.destroy_buffer(st);

        let bindless_index = self.next_bindless_index.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let img_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(v)
            .sampler(s)];
        let writes = [vk::WriteDescriptorSet::default()
            .dst_set(self.bindless_descriptor_set)
            .dst_binding(0)
            .dst_array_element(bindless_index)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&img_info)];
        unsafe {
            self.device.device.update_descriptor_sets(&writes, &[]);
        }

        Texture {
            image: i,
            allocation: Some(m),
            view: v,
            sampler: s,
            mip_levels: mip,
            bindless_index,
        }
    }

    pub fn destroy_texture(&mut self, mut t: Texture) {
        self.texture_descriptor_sets.remove(&t.view);
        unsafe {
            self.device.device.destroy_sampler(t.sampler, None);
            self.device.device.destroy_image_view(t.view, None);
            self.device.device.destroy_image(t.image, None);
            if let Some(alloc) = t.allocation.take() {
                self.device.allocator.lock().unwrap().free(alloc).unwrap();
            }
        }
    }

    pub fn create_image_basic(
        &self,
        params: &crate::vulkan::device::ImageCreateParams,
    ) -> (vk::Image, gpu_allocator::vulkan::Allocation) {
        self.device.create_image(params).expect("Failed to create image")
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
                .unwrap()
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
        self.vertex_buffers.get(id as usize)
    }
    pub fn get_instance_buffer(&self, id: u32) -> Option<&Buffer> {
        self.frames[self.current_frame].instance_pool.get(id as usize)
    }
    /// Resets the instance buffer index for the current frame to allow reuse in the next cycle.
    pub fn clear_instance_buffers(&mut self) {
        self.frames[self.current_frame].instance_index = 0;
    }
    /// Returns the graphics queue handle.
    pub fn get_graphics_queue(&self) -> vk::Queue {
        self.device.graphics_queue
    }
    /// Returns the configured MSAA sample count.
    pub fn get_msaa_samples(&self) -> vk::SampleCountFlags {
        self.device.msaa_samples
    }
    /// Returns the current swapchain extent.
    pub fn get_extent(&self) -> vk::Extent2D {
        self.swapchain.extent
    }

    pub fn register_egui_texture(&mut self, view: vk::ImageView, sampler: vk::Sampler) -> egui::TextureId {
        let mut egui_renderer = self.egui_renderer.take().expect("EguiRenderer not initialized");
        let id = egui_renderer.register_native_texture(self, view, sampler);
        self.egui_renderer = Some(egui_renderer);
        id
    }

    /// Calculates the projection jitter for temporal anti-aliasing.
    pub fn get_jitter(&self) -> [f32; 2] {
        let x = crate::vulkan::utils::halton((self.frame_index % 16) as u32 + 1, 2) - 0.5;
        let y = crate::vulkan::utils::halton((self.frame_index % 16) as u32 + 1, 3) - 0.5;
        [x / self.swapchain.extent.width as f32, y / self.swapchain.extent.height as f32]
    }

    pub fn add_render_pass<P: crate::passes::RenderPass + 'static>(&mut self, pass: P) {
        self.render_passes.push(Box::new(pass));
    }

    /// Calculates the six frustum planes from a view-projection matrix.
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

    pub fn prepare_frame(
        &mut self,
        packet: crate::resource::FramePacket,
    ) -> u32 {
        use rayon::prelude::*;
        self.scene_view_matrix_for_pos = packet.view_matrix;

        // 1. Prepare GPU Indirect and Object buffers in parallel
        let (object_ssbos, indirect_commands): (Vec<_>, Vec<_>) = packet.meshes.par_iter().enumerate().map(|(i, mesh)| {
             let m = mesh.model.transpose();
             let ssbo = ObjectDataSSBO {
                model_row0: m.row(0),
                model_row1: m.row(1),
                model_row2: m.row(2),
                sphere: spark_math::Vec4::new(0.0, 0.0, 0.0, mesh.bounding_radius),
                index_count: mesh.index_count,
                first_index: mesh.first_index,
                vertex_offset: mesh.vertex_offset,
                material_index: mesh.material_index,
            };
            let cmd = vk::DrawIndexedIndirectCommand {
                index_count: mesh.index_count,
                instance_count: 1,
                first_index: mesh.first_index,
                vertex_offset: mesh.vertex_offset,
                first_instance: i as u32,
            };
            (ssbo, cmd)
        }).unzip();

        self.update_indirect_buffers(&indirect_commands, &object_ssbos);
        let total_objects = object_ssbos.len() as u32;

        // 2. Update Lights
        self.update_lights_from_draw(&packet.lights);

        // 3. Update Global UBO
        let extent = self.get_extent();
        let jitter = self.get_jitter();
        let mut projection = spark_math::Mat4::perspective_rh(
            45.0f32.to_radians(),
            extent.width as f32 / extent.height as f32,
            0.1,
            100.0,
        );
        projection.col_mut(2).x += jitter[0] * projection.col(0).x;
        projection.col_mut(2).y += jitter[1] * projection.col(1).y;
        let view_proj = projection * packet.view_matrix;
        self.current_view_proj = view_proj;

        let light_pos = spark_math::Vec3::new(10.0, 10.0, 10.0);
        let light_view = spark_math::Mat4::look_at_rh(
            light_pos,
            spark_math::Vec3::ZERO,
            spark_math::Vec3::Y,
        );
        let light_proj = spark_math::Mat4::orthographic_rh(-20.0, 20.0, -20.0, 20.0, 0.1, 100.0);
        self.main_light_view_proj = light_proj * light_view;

        let inv_v = packet.view_matrix.inverse();
        let camera_pos = [inv_v.w_axis.x, inv_v.w_axis.y, inv_v.w_axis.z, 1.0];
        let frustum = Self::calculate_frustum_planes(view_proj);

        let ubo = GlobalUBO {
            vp: view_proj,
            lvp: [self.main_light_view_proj; 4],
            inv_vp: view_proj.inverse(),
            camera_pos,
            frustum,
            cascade_splits: [0.1, 0.2, 0.5, 1.0],
        };

        if self.frames[0].global_buffer.is_none() {
            for i in 0..MAX_FRAMES_IN_FLIGHT {
                let new_buffer = self.create_buffer(
                    std::mem::size_of::<GlobalUBO>() as u64,
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                );
                self.frames[i].global_buffer = Some(new_buffer);
            }
            self.ensure_global_descriptor_set();
        }
        let gb = self.frames[self.current_frame].global_buffer.as_ref().cloned().unwrap();
        self.upload_to_buffer(&gb, &[ubo]);

        // 4. Prepare Passes
        let cf = self.current_frame;

        for pass in &self.render_passes {
            pass.prepare(self, cf);
        }

        self.last_object_count = total_objects;
        total_objects
    }

    /// Returns the raw ash::Device.
    pub fn get_device(&self) -> &ash::Device {
        &self.device.device
    }

    pub fn get_thread_command_pool(&self, thread_idx: usize) -> vk::CommandPool {
        self.device.thread_command_pools[thread_idx % self.device.thread_command_pools.len()]
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
        self.device.create_buffer(sz, usage, properties).expect("Failed to create buffer")
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

    fn create_descriptor_pool_impl(device: &ash::Device) -> vk::DescriptorPool {
        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(20000),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::INPUT_ATTACHMENT)
                .descriptor_count(100),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(100),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(100),
        ];
        unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .pool_sizes(&sizes)
                        .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
                        .max_sets(2000),
                    None,
                )
                .expect("Failed to create descriptor pool")
        }
    }

    fn create_sync_objects_impl(
        device: &ash::Device,
    ) -> (Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>) {
        let mut av = Vec::new();
        let mut fi = Vec::new();
        let mut in_f = Vec::new();
        let s_info = vk::SemaphoreCreateInfo::default();
        let f_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            unsafe {
                av.push(device.create_semaphore(&s_info, None).unwrap());
                fi.push(device.create_semaphore(&s_info, None).unwrap());
                in_f.push(device.create_fence(&f_info, None).unwrap());
            }
        }
        (av, fi, in_f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

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
            col: [f32; 4],
        }
        assert_eq!(std::mem::size_of::<LD>(), 32);
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.device.device.device_wait_idle().ok();

            for mut pass in std::mem::take(&mut self.render_passes) {
                pass.destroy(self);
            }

            self.cleanup_swapchain();

            for frame in &mut self.frames {
                if let Some(lb) = frame.light_buffer.take() {
                    self.device.destroy_buffer(lb);
                }
                if let Some(gb) = frame.global_buffer.take() {
                    self.device.destroy_buffer(gb);
                }
                for ib in frame.instance_pool.drain(..) {
                    self.device.destroy_buffer(ib);
                }
                if let Some(ib) = frame.indirect_commands_buffer.take() {
                    self.device.destroy_buffer(ib);
                }
                if let Some(ob) = frame.object_data_buffer.take() {
                    self.device.destroy_buffer(ob);
                }
                if let Some(dc) = frame.draw_count_buffer.take() {
                    self.device.destroy_buffer(dc);
                }
                self.device.device.destroy_semaphore(frame.image_available, None);
                self.device.device.destroy_semaphore(frame.render_finished, None);
                self.device.device.destroy_fence(frame.in_flight, None);
            }

            if let Some(p) = self.pipeline.take() {
                self.device
                    .device
                    .destroy_pipeline(p.graphics_pipeline, None);
                self.device.device.destroy_pipeline_layout(p.layout, None);
                self.device
                    .device
                    .destroy_descriptor_set_layout(p.descriptor_set_layout, None);
            }
            self.texture_descriptor_sets.clear();
            self.device
                .device
                .destroy_descriptor_set_layout(self.global_descriptor_set_layout, None);
            self.device
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            if let Some(mut e) = self.egui_renderer.take() {
                e.destroy(self);
            }
            if let Some(t) = self.default_texture.take() {
                self.destroy_texture(t);
            }
            let vbs = std::mem::take(&mut self.vertex_buffers);
            for vb in vbs {
                self.device.destroy_buffer(vb);
            }
            if let Some(ib) = self.index_buffer.take() {
                self.device.destroy_buffer(ib);
            }
        }
    }
}
