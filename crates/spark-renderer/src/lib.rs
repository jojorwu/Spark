pub mod error;
pub mod pipeline;
pub mod ui;
pub mod vertex;
pub mod vulkan;

use crate::error::RendererError;
use crate::pipeline::Pipeline;
use crate::ui::EguiRenderer;
use crate::vulkan::context::VulkanContext;
use crate::vulkan::device::VulkanDevice;
use crate::vulkan::swapchain::VulkanSwapchain;
use crate::vulkan::texture::Texture;
pub use ash;
use ash::vk;
use winit::window::Window;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

#[derive(Copy, Clone)]
pub struct Buffer {
    pub handle: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
}

pub struct Renderer {
    context: VulkanContext,
    pub device: VulkanDevice,
    swapchain: VulkanSwapchain,
    pub render_pass: vk::RenderPass,
    pub hdr_images: Vec<vk::Image>,
    pub hdr_memories: Vec<vk::DeviceMemory>,
    pub hdr_views: Vec<vk::ImageView>,
    pub gbuffer_albedo_images: Vec<vk::Image>,
    pub gbuffer_albedo_memories: Vec<vk::DeviceMemory>,
    pub gbuffer_albedo_views: Vec<vk::ImageView>,
    pub gbuffer_normal_images: Vec<vk::Image>,
    pub gbuffer_normal_memories: Vec<vk::DeviceMemory>,
    pub gbuffer_normal_views: Vec<vk::ImageView>,
    pub gbuffer_position_images: Vec<vk::Image>,
    pub gbuffer_position_memories: Vec<vk::DeviceMemory>,
    pub gbuffer_position_views: Vec<vk::ImageView>,
    pub gbuffer_pbr_images: Vec<vk::Image>,
    pub gbuffer_pbr_memories: Vec<vk::DeviceMemory>,
    pub gbuffer_pbr_views: Vec<vk::ImageView>,
    pub depth_images: Vec<vk::Image>,
    pub depth_memories: Vec<vk::DeviceMemory>,
    pub depth_views: Vec<vk::ImageView>,
    pub light_buffers: Vec<Buffer>,
    pub light_count: u32,
    pub shadow_image: vk::Image,
    pub shadow_memory: vk::DeviceMemory,
    pub shadow_view: vk::ImageView,
    pub shadow_sampler: vk::Sampler,
    pub shadow_render_pass: vk::RenderPass,
    pub shadow_framebuffer: vk::Framebuffer,
    pub shadow_pipeline: Option<vk::Pipeline>,
    pub shadow_pipeline_layout: vk::PipelineLayout,
    pub post_process_pipeline: Option<vk::Pipeline>,
    pub post_process_layout: vk::PipelineLayout,
    pub bloom_pipeline: Option<vk::Pipeline>,
    pub post_process_descriptor_pool: vk::DescriptorPool,
    pub post_process_descriptor_sets: Vec<vk::DescriptorSet>,
    pub post_process_render_pass: vk::RenderPass,
    pub post_process_framebuffers: Vec<vk::Framebuffer>,
    pub bloom_images: Vec<vk::Image>,
    pub bloom_memories: Vec<vk::DeviceMemory>,
    pub bloom_views: Vec<vk::ImageView>,
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,
    pub pipeline: Option<Pipeline>,
    pub deferred_pipeline: Option<vk::Pipeline>,
    pub deferred_layout: vk::PipelineLayout,
    pub deferred_descriptor_sets: Vec<vk::DescriptorSet>,
    pub vertex_buffers: Vec<Buffer>,
    pub instance_buffers: Vec<Buffer>,
    pub index_buffer: Option<Buffer>,
    pub descriptor_pool: vk::DescriptorPool,
    pub texture_descriptor_sets: std::collections::HashMap<vk::ImageView, vk::DescriptorSet>,
    pub default_texture: Option<Texture>,
    pub default_descriptor_set: vk::DescriptorSet,
    egui_renderer: Option<EguiRenderer>,
}

impl Renderer {
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
        );

        let (sh_i, sh_m, sh_v, sh_s) =
            Self::create_shadow_resources_impl(&device.device, device.pdevice, &context.instance);
        let shadow_render_pass = Self::create_shadow_render_pass_impl(&device.device);
        let shadow_framebuffer = unsafe {
            device
                .device
                .create_framebuffer(
                    &vk::FramebufferCreateInfo::default()
                        .render_pass(shadow_render_pass)
                        .attachments(&[sh_v])
                        .width(Self::SHADOW_MAP_CASCADE_SIZE)
                        .height(Self::SHADOW_MAP_CASCADE_SIZE)
                        .layers(1),
                    None,
                )
                .unwrap()
        };
        let shadow_pipeline_layout = unsafe {
            device
                .device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&[
                        vk::PushConstantRange::default()
                            .stage_flags(vk::ShaderStageFlags::VERTEX)
                            .offset(0)
                            .size(128),
                    ]),
                    None,
                )
                .unwrap()
        };

        let mut hdr_i = Vec::new();
        let mut hdr_m = Vec::new();
        let mut hdr_v = Vec::new();
        let mut g_alb_i = Vec::new();
        let mut g_alb_m = Vec::new();
        let mut g_alb_v = Vec::new();
        let mut g_norm_i = Vec::new();
        let mut g_norm_m = Vec::new();
        let mut g_norm_v = Vec::new();
        let mut g_pos_i = Vec::new();
        let mut g_pos_m = Vec::new();
        let mut g_pos_v = Vec::new();
        let mut g_pbr_i = Vec::new();
        let mut g_pbr_m = Vec::new();
        let mut g_pbr_v = Vec::new();
        let mut depth_i = Vec::new();
        let mut depth_m = Vec::new();
        let mut depth_v = Vec::new();

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let (i, m, v) = Self::create_hdr_resources_impl(
                &device.device,
                device.pdevice,
                &context.instance,
                swapchain.extent,
                vk::SampleCountFlags::TYPE_1,
            );
            hdr_i.push(i);
            hdr_m.push(m);
            hdr_v.push(v);

            let (i, m, v) = Self::create_image_resource_impl(
                &device.device,
                device.pdevice,
                &context.instance,
                swapchain.extent,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT,
                device.msaa_samples,
            );
            g_alb_i.push(i);
            g_alb_m.push(m);
            g_alb_v.push(v);

            let (i, m, v) = Self::create_image_resource_impl(
                &device.device,
                device.pdevice,
                &context.instance,
                swapchain.extent,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT,
                device.msaa_samples,
            );
            g_norm_i.push(i);
            g_norm_m.push(m);
            g_norm_v.push(v);

            let (i, m, v) = Self::create_image_resource_impl(
                &device.device,
                device.pdevice,
                &context.instance,
                swapchain.extent,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT,
                device.msaa_samples,
            );
            g_pos_i.push(i);
            g_pos_m.push(m);
            g_pos_v.push(v);

            let (i, m, v) = Self::create_image_resource_impl(
                &device.device,
                device.pdevice,
                &context.instance,
                swapchain.extent,
                vk::Format::R8G8B8A8_UNORM,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT,
                device.msaa_samples,
            );
            g_pbr_i.push(i);
            g_pbr_m.push(m);
            g_pbr_v.push(v);

            let (i, m, v) = Self::create_depth_resources_impl(
                &device.device,
                device.pdevice,
                &context.instance,
                swapchain.extent,
                device.msaa_samples,
                device.depth_format,
            );
            depth_i.push(i);
            depth_m.push(m);
            depth_v.push(v);

        }

        let render_pass = Self::create_render_pass_impl(
            &device.device,
            swapchain.format,
            device.msaa_samples,
            device.depth_format,
        );
        let framebuffers = Self::create_framebuffers_impl(
            &device.device,
            render_pass,
            &hdr_v,
            &g_alb_v,
            &g_norm_v,
            &g_pos_v,
            &g_pbr_v,
            &depth_v,
            swapchain.extent,
        );
        let post_process_render_pass =
            Self::create_post_process_render_pass_impl(&device.device, swapchain.format);
        let post_process_framebuffers = Self::create_post_process_framebuffers_impl(
            &device.device,
            post_process_render_pass,
            &swapchain.views,
            swapchain.extent,
        );
        let command_pool = Self::create_command_pool_impl(&device.device, device.graphics_family);
        let command_buffers = Self::create_command_buffers_impl(&device.device, command_pool);
        let (av, fi, in_f) = Self::create_sync_objects_impl(&device.device);
        let descriptor_pool = Self::create_descriptor_pool_impl(&device.device);
        let post_process_descriptor_pool = Self::create_descriptor_pool_impl(&device.device);
        let mut bloom_images = Vec::new();
        let mut bloom_memories = Vec::new();
        let mut bloom_views = Vec::new();
        for i in 1..6 {
            let (im, me, vi) = Self::create_hdr_resources_impl(
                &device.device,
                device.pdevice,
                &context.instance,
                vk::Extent2D {
                    width: (swapchain.extent.width >> i).max(1),
                    height: (swapchain.extent.height >> i).max(1),
                },
                vk::SampleCountFlags::TYPE_1,
            );
            bloom_images.push(im);
            bloom_memories.push(me);
            bloom_views.push(vi);
        }
        let egui_renderer = ui_shaders.map(|(v, f)| {
            EguiRenderer::new(
                &device.device,
                render_pass,
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
            render_pass,
            hdr_images: hdr_i,
            hdr_memories: hdr_m,
            hdr_views: hdr_v,
            gbuffer_albedo_images: g_alb_i,
            gbuffer_albedo_memories: g_alb_m,
            gbuffer_albedo_views: g_alb_v,
            gbuffer_normal_images: g_norm_i,
            gbuffer_normal_memories: g_norm_m,
            gbuffer_normal_views: g_norm_v,
            gbuffer_position_images: g_pos_i,
            gbuffer_position_memories: g_pos_m,
            gbuffer_position_views: g_pos_v,
            gbuffer_pbr_images: g_pbr_i,
            gbuffer_pbr_memories: g_pbr_m,
            gbuffer_pbr_views: g_pbr_v,
            depth_images: depth_i,
            depth_memories: depth_m,
            depth_views: depth_v,
            light_buffers: Vec::new(),
            light_count: 0,
            shadow_image: sh_i,
            shadow_memory: sh_m,
            shadow_view: sh_v,
            shadow_sampler: sh_s,
            shadow_render_pass,
            shadow_framebuffer,
            shadow_pipeline: None,
            shadow_pipeline_layout,
            post_process_render_pass,
            post_process_framebuffers,
            post_process_pipeline: None,
            post_process_layout: vk::PipelineLayout::null(),
            bloom_pipeline: None,
            bloom_images,
            bloom_memories,
            bloom_views,
            post_process_descriptor_pool,
            post_process_descriptor_sets: Vec::new(),
            framebuffers,
            command_pool,
            command_buffers,
            image_available_semaphores: av,
            render_finished_semaphores: fi,
            in_flight_fences: in_f,
            current_frame: 0,
            pipeline: None,
            deferred_pipeline: None,
            deferred_layout: vk::PipelineLayout::null(),
            deferred_descriptor_sets: Vec::new(),
            vertex_buffers: Vec::new(),
            instance_buffers: Vec::new(),
            index_buffer: None,
            descriptor_pool,
            texture_descriptor_sets: std::collections::HashMap::new(),
            default_texture: None,
            default_descriptor_set: vk::DescriptorSet::null(),
            egui_renderer,
        })
    }

    pub fn set_deferred_pipeline(
        &mut self,
        pipeline: vk::Pipeline,
        layout: vk::PipelineLayout,
        ds_layout: vk::DescriptorSetLayout,
    ) {
        self.deferred_pipeline = Some(pipeline);
        self.deferred_layout = layout;
        self.deferred_descriptor_sets = unsafe {
            self.device
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.descriptor_pool)
                        .set_layouts(&[ds_layout, ds_layout]),
                )
                .unwrap()
        };
        self.update_deferred_descriptor_sets();
    }

    fn update_deferred_descriptor_sets(&self) {
        if self.deferred_descriptor_sets.is_empty() {
            return;
        }
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let alb_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(self.gbuffer_albedo_views[i])
                .sampler(vk::Sampler::null())];
            let norm_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(self.gbuffer_normal_views[i])
                .sampler(vk::Sampler::null())];
            let pos_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(self.gbuffer_position_views[i])
                .sampler(vk::Sampler::null())];
            let pbr_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(self.gbuffer_pbr_views[i])
                .sampler(vk::Sampler::null())];
            let depth_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(self.depth_views[i])
                .sampler(vk::Sampler::null())];

            let shadow_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(self.shadow_view)
                .sampler(self.shadow_sampler)];

            let mut writes = vec![
                vk::WriteDescriptorSet::default()
                    .dst_set(self.deferred_descriptor_sets[i])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                    .image_info(&alb_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.deferred_descriptor_sets[i])
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                    .image_info(&norm_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.deferred_descriptor_sets[i])
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                    .image_info(&pos_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.deferred_descriptor_sets[i])
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                    .image_info(&pbr_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.deferred_descriptor_sets[i])
                    .dst_binding(4)
                    .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                    .image_info(&depth_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.deferred_descriptor_sets[i])
                    .dst_binding(5)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&shadow_info),
            ];

            let mut buf_info = Vec::new();
            if let Some(lb) = self.light_buffers.get(i) {
                buf_info.push(
                    vk::DescriptorBufferInfo::default()
                        .buffer(lb.handle)
                        .offset(0)
                        .range(lb.size),
                );
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.deferred_descriptor_sets[i])
                        .dst_binding(6)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&buf_info),
                );
            }

            unsafe {
                self.device.device.update_descriptor_sets(&writes, &[]);
            }
        }
    }

    pub fn set_pipeline(&mut self, pipeline: Pipeline) {
        if self.pipeline.is_some() {
            // New pipeline might have a different descriptor set layout.
            // Since we don't support individual descriptor set freeing,
            // we clear the cache and will re-allocate from the pool.
            // Ideally we should reset the pool here if we change pipelines often.
            self.texture_descriptor_sets.clear();
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
    pub fn set_shadow_pipeline(&mut self, pipeline: vk::Pipeline) {
        self.shadow_pipeline = Some(pipeline);
    }
    pub fn set_post_process_pipeline(
        &mut self,
        pipeline: vk::Pipeline,
        layout: vk::PipelineLayout,
        ds_layout: vk::DescriptorSetLayout,
        _bloom_pipeline: vk::Pipeline,
    ) {
        self.post_process_pipeline = Some(pipeline);
        self.post_process_layout = layout;
        self.post_process_descriptor_sets = unsafe {
            self.device
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.post_process_descriptor_pool)
                        .set_layouts(&[ds_layout, ds_layout]),
                )
                .unwrap()
        };
        self.update_post_process_descriptor_sets();
    }

    fn update_post_process_descriptor_sets(&self) {
        if self.post_process_descriptor_sets.is_empty() {
            return;
        }
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let img_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(self.hdr_views[i])
                .sampler(self.shadow_sampler)];
            let blm_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(self.bloom_views[0])
                .sampler(self.shadow_sampler)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(self.post_process_descriptor_sets[i])
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&img_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.post_process_descriptor_sets[i])
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&blm_info),
            ];
            unsafe {
                self.device.device.update_descriptor_sets(&writes, &[]);
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
    pub fn update_lights(&mut self, lights: &[(spark_math::Vec3, spark_math::Vec3, f32)]) {
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
            .map(|(p, c, i)| LD {
                pos: [p.x, p.y, p.z, 1.0],
                col: [c.x, c.y, c.z, *i],
            })
            .collect();
        let sz = (ld.len() * std::mem::size_of::<LD>()) as u64;

        if self.light_buffers.is_empty() {
            for _ in 0..MAX_FRAMES_IN_FLIGHT {
                self.light_buffers.push(self.create_buffer(
                    sz,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                ));
            }
            self.update_deferred_descriptor_sets();
        }

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            if self.light_buffers[i].size < sz {
                let old = self.light_buffers[i];
                unsafe {
                    self.device.device.device_wait_idle().ok();
                    self.device.device.destroy_buffer(old.handle, None);
                    self.device.device.free_memory(old.memory, None);
                }
                self.light_buffers[i] = self.create_buffer(
                    sz,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                );
                if i == MAX_FRAMES_IN_FLIGHT - 1 {
                    self.update_deferred_descriptor_sets();
                }
            }
        }

        self.upload_to_buffer(&self.light_buffers[self.current_frame], &ld);
    }

    pub fn add_vertex_buffer(&mut self, buffer: Buffer) -> u32 {
        self.vertex_buffers.push(buffer);
        (self.vertex_buffers.len() - 1) as u32
    }

    pub fn add_instance_buffer(&mut self, buffer: Buffer) -> u32 {
        self.instance_buffers.push(buffer);
        (self.instance_buffers.len() - 1) as u32
    }

    pub fn draw_frame(
        &mut self,
        renderables: &[(spark_math::Mat4, u32, Option<vk::ImageView>, Option<u32>)],
        instanced_renderables: &[(u32, u32, u32, Option<vk::ImageView>)],
        view_proj: spark_math::Mat4,
        light_view_proj: spark_math::Mat4,
        window: &Window,
        egui_output: Option<(egui::FullOutput, egui::Context)>,
    ) {
        unsafe {
            self.device
                .device
                .wait_for_fences(&[self.in_flight_fences[self.current_frame]], true, u64::MAX)
                .expect("Failed to wait for fence");
            let result = self.swapchain.loader.acquire_next_image(
                self.swapchain.handle,
                u64::MAX,
                self.image_available_semaphores[self.current_frame],
                vk::Fence::null(),
            );
            let image_index = match result {
                Ok((index, _)) => index,
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                    self.recreate_swapchain(window);
                    return;
                }
                Err(e) => panic!("Failed to acquire swapchain image: {:?}", e),
            };
            self.device
                .device
                .reset_fences(&[self.in_flight_fences[self.current_frame]])
                .expect("Failed to reset fence");
            self.device
                .device
                .reset_command_buffer(
                    self.command_buffers[self.current_frame],
                    vk::CommandBufferResetFlags::empty(),
                )
                .expect("Failed to reset command buffer");
            self.record_command_buffer(
                image_index,
                renderables,
                instanced_renderables,
                view_proj,
                light_view_proj,
                egui_output,
            );
            let s_available = [self.image_available_semaphores[self.current_frame]];
            let s_finished = [self.render_finished_semaphores[self.current_frame]];
            let w_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let c_buffers = [self.command_buffers[self.current_frame]];
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
                    self.in_flight_fences[self.current_frame],
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
                    self.recreate_swapchain(window);
                }
                Err(e) => panic!("Failed to present swapchain image: {:?}", e),
            }
            self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        }
    }
    fn record_command_buffer(
        &mut self,
        image_index: u32,
        renderables: &[(spark_math::Mat4, u32, Option<vk::ImageView>, Option<u32>)],
        instanced_renderables: &[(u32, u32, u32, Option<vk::ImageView>)],
        view_proj: spark_math::Mat4,
        light_view_proj: spark_math::Mat4,
        egui_output: Option<(egui::FullOutput, egui::Context)>,
    ) {
        let command_buffer = self.command_buffers[self.current_frame];
        unsafe {
            self.device
                .device
                .begin_command_buffer(command_buffer, &vk::CommandBufferBeginInfo::default())
                .unwrap();
            if let Some(shadow_pipeline) = self.shadow_pipeline {
                let clear = [vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                }];
                self.device.device.cmd_begin_render_pass(
                    command_buffer,
                    &vk::RenderPassBeginInfo::default()
                        .render_pass(self.shadow_render_pass)
                        .framebuffer(self.shadow_framebuffer)
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: vk::Extent2D {
                                width: Self::SHADOW_MAP_CASCADE_SIZE,
                                height: Self::SHADOW_MAP_CASCADE_SIZE,
                            },
                        })
                        .clear_values(&clear),
                    vk::SubpassContents::INLINE,
                );
                self.device.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    shadow_pipeline,
                );
                let shadow_viewport = vk::Viewport::default()
                    .x(0.0)
                    .y(0.0)
                    .width(Self::SHADOW_MAP_CASCADE_SIZE as f32)
                    .height(Self::SHADOW_MAP_CASCADE_SIZE as f32)
                    .min_depth(0.0)
                    .max_depth(1.0);
                let shadow_scissor = vk::Rect2D::default().extent(vk::Extent2D {
                    width: Self::SHADOW_MAP_CASCADE_SIZE,
                    height: Self::SHADOW_MAP_CASCADE_SIZE,
                });
                self.device.device.cmd_set_viewport(command_buffer, 0, &[shadow_viewport]);
                self.device.device.cmd_set_scissor(command_buffer, 0, &[shadow_scissor]);
                let lvp_bytes =
                    std::slice::from_raw_parts(&light_view_proj as *const _ as *const u8, 64);
                for (vb_id, ib_id, count, _) in instanced_renderables {
                    if let (Some(vb), Some(ib)) =
                        (self.get_buffer(*vb_id), self.get_instance_buffer(*ib_id))
                    {
                        self.device.device.cmd_bind_vertex_buffers(
                            command_buffer,
                            0,
                            &[vb.handle, ib.handle],
                            &[0, 0],
                        );
                        self.device.device.cmd_push_constants(
                            command_buffer,
                            self.shadow_pipeline_layout,
                            vk::ShaderStageFlags::VERTEX,
                            0,
                            lvp_bytes,
                        );
                        if let Some(ib) = self.index_buffer {
                            self.device.device.cmd_bind_index_buffer(
                                command_buffer,
                                ib.handle,
                                0,
                                vk::IndexType::UINT32,
                            );
                            self.device.device.cmd_draw_indexed(
                                command_buffer,
                                (ib.size / 4) as u32,
                                *count,
                                0,
                                0,
                                0,
                            );
                        } else {
                            self.device.device.cmd_draw(
                                command_buffer,
                                (vb.size / std::mem::size_of::<crate::vertex::Vertex>() as u64) as u32,
                                *count,
                                0,
                                0,
                            );
                        }
                    }
                }
                // Handle non-instanced renderables in shadow pass (though currently they might be empty or used differently)
                for (model, vb_id, _, _) in renderables {
                    if let Some(vb) = self.get_buffer(*vb_id) {
                        self.device.device.cmd_bind_vertex_buffers(
                            command_buffer,
                            0,
                            &[vb.handle],
                            &[0],
                        );
                        // For non-instanced, we'd need to send both light_view_proj and model matrix.
                        // Currently shadow.vert expects light_view_proj * instanceModel.
                        // We can reuse the same push constant layout if we send light_view_proj * model.
                        let mvp = light_view_proj * (*model);
                        let mvp_bytes = std::slice::from_raw_parts(&mvp as *const _ as *const u8, 64);

                        self.device.device.cmd_push_constants(
                            command_buffer,
                            self.shadow_pipeline_layout,
                            vk::ShaderStageFlags::VERTEX,
                            0,
                            mvp_bytes,
                        );
                        if let Some(ib) = self.index_buffer {
                            self.device.device.cmd_bind_index_buffer(
                                command_buffer,
                                ib.handle,
                                0,
                                vk::IndexType::UINT32,
                            );
                            self.device.device.cmd_draw_indexed(
                                command_buffer,
                                (ib.size / 4) as u32,
                                1,
                                0,
                                0,
                                0,
                            );
                        } else {
                            self.device.device.cmd_draw(
                                command_buffer,
                                (vb.size / std::mem::size_of::<crate::vertex::Vertex>() as u64) as u32,
                                1,
                                0,
                                0,
                            );
                        }
                    }
                }
                self.device.device.cmd_end_render_pass(command_buffer);
            }
            let clear = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                },
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                },
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0],
                    },
                },
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0], // PBR
                    },
                },
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.1, 0.1, 0.1, 1.0],
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];
            self.device.device.cmd_begin_render_pass(
                command_buffer,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(self.framebuffers[self.current_frame])
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: self.swapchain.extent,
                    })
                    .clear_values(&clear),
                vk::SubpassContents::INLINE,
            );
            #[repr(C)]
            #[repr(C)]
            struct PC {
                vp: spark_math::Mat4,
                lvp: spark_math::Mat4,
                count: u32,
                metallic: f32,
                roughness: f32,
                pad: u32,
            }
            let pc = PC {
                vp: view_proj,
                lvp: light_view_proj,
                count: self.light_count,
                metallic: 0.5, // Default for now
                roughness: 0.5, // Default for now
                pad: 0,
            };
            let pc_bytes = std::slice::from_raw_parts(&pc as *const _ as *const u8, 144);
            if let Some(pipeline) = &self.pipeline {
                self.device.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.graphics_pipeline,
                );
                let viewport = vk::Viewport::default()
                    .x(0.0)
                    .y(self.swapchain.extent.height as f32)
                    .width(self.swapchain.extent.width as f32)
                    .height(-(self.swapchain.extent.height as f32))
                    .min_depth(0.0)
                    .max_depth(1.0);
                let scissor = vk::Rect2D::default().extent(self.swapchain.extent);
                self.device.device.cmd_set_viewport(command_buffer, 0, &[viewport]);
                self.device.device.cmd_set_scissor(command_buffer, 0, &[scissor]);

                self.device.device.cmd_push_constants(
                    command_buffer,
                    pipeline.layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    pc_bytes,
                );
                for (vb_id, ib_id, count, tex_view) in instanced_renderables {
                    if let (Some(vb), Some(ib)) =
                        (self.get_buffer(*vb_id), self.get_instance_buffer(*ib_id))
                    {
                        let ds = tex_view.and_then(|v| self.texture_descriptor_sets.get(&v))
                            .unwrap_or(&self.default_descriptor_set);

                        self.device.device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline.layout,
                            0,
                            &[*ds],
                            &[],
                        );

                        self.device.device.cmd_bind_vertex_buffers(
                            command_buffer,
                            0,
                            &[vb.handle, ib.handle],
                            &[0, 0],
                        );
                        if let Some(ib) = self.index_buffer {
                            self.device.device.cmd_bind_index_buffer(
                                command_buffer,
                                ib.handle,
                                0,
                                vk::IndexType::UINT32,
                            );
                            self.device.device.cmd_draw_indexed(
                                command_buffer,
                                (ib.size / 4) as u32,
                                *count,
                                0,
                                0,
                                0,
                            );
                        } else {
                            self.device.device.cmd_draw(
                                command_buffer,
                                (vb.size / std::mem::size_of::<crate::vertex::Vertex>() as u64) as u32,
                                *count,
                                0,
                                0,
                            );
                        }
                    }
                }
                for (_model, vb_id, tex_view, _) in renderables {
                    if let Some(vb) = self.get_buffer(*vb_id) {
                        let ds = tex_view.and_then(|v| self.texture_descriptor_sets.get(&v))
                            .unwrap_or(&self.default_descriptor_set);

                        self.device.device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline.layout,
                            0,
                            &[*ds],
                            &[],
                        );

                        // For non-instanced, we'd need to send the model matrix via push constants.
                        // Currently our push constant struct in lib.rs only has vp and lvp.
                        // For simplicity, let's just use vp for now, but in a real case we'd need model.
                        // Re-using pc_bytes but with a single model matrix could work if we adjust the shader.
                        // But let's stay consistent with what we have.

                        self.device.device.cmd_bind_vertex_buffers(
                            command_buffer,
                            0,
                            &[vb.handle],
                            &[0],
                        );
                        if let Some(ib) = self.index_buffer {
                            self.device.device.cmd_bind_index_buffer(
                                command_buffer,
                                ib.handle,
                                0,
                                vk::IndexType::UINT32,
                            );
                            self.device.device.cmd_draw_indexed(
                                command_buffer,
                                (ib.size / 4) as u32,
                                1,
                                0,
                                0,
                                0,
                            );
                        } else {
                            self.device.device.cmd_draw(
                                command_buffer,
                                (vb.size / std::mem::size_of::<crate::vertex::Vertex>() as u64) as u32,
                                1,
                                0,
                                0,
                            );
                        }
                    }
                }
            }
            self.device
                .device
                .cmd_next_subpass(command_buffer, vk::SubpassContents::INLINE);
            if let Some(deferred_pipe) = self.deferred_pipeline {
                self.device.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    deferred_pipe,
                );
                let viewport = vk::Viewport::default()
                    .x(0.0)
                    .y(0.0)
                    .width(self.swapchain.extent.width as f32)
                    .height(self.swapchain.extent.height as f32)
                    .min_depth(0.0)
                    .max_depth(1.0);
                let scissor = vk::Rect2D::default().extent(self.swapchain.extent);
                self.device.device.cmd_set_viewport(command_buffer, 0, &[viewport]);
                self.device.device.cmd_set_scissor(command_buffer, 0, &[scissor]);
                if !self.deferred_descriptor_sets.is_empty() {
                    self.device.device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.deferred_layout,
                        0,
                        &[self.deferred_descriptor_sets[self.current_frame]],
                        &[],
                    );
                }
                self.device.device.cmd_push_constants(
                    command_buffer,
                    self.deferred_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    pc_bytes,
                );
                self.device.device.cmd_draw(command_buffer, 3, 1, 0, 0);
            }
            if let Some((output, ctx)) = egui_output {
                let ext = self.swapchain.extent;
                if let Some(mut egui) = self.egui_renderer.take() {
                    egui.draw(
                        self,
                        command_buffer,
                        output,
                        [ext.width as f32, ext.height as f32],
                        &ctx,
                    );
                    self.egui_renderer = Some(egui);
                }
            }
            self.device.device.cmd_end_render_pass(command_buffer);
            if let Some(post_pipeline) = self.post_process_pipeline {
                let barrier = vk::ImageMemoryBarrier::default()
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .image(self.swapchain.images[image_index as usize])
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                self.device.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
                self.device.device.cmd_begin_render_pass(
                    command_buffer,
                    &vk::RenderPassBeginInfo::default()
                        .render_pass(self.post_process_render_pass)
                        .framebuffer(self.post_process_framebuffers[image_index as usize])
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: self.swapchain.extent,
                        })
                        .clear_values(&[vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 1.0],
                            },
                        }]),
                    vk::SubpassContents::INLINE,
                );
                self.device.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    post_pipeline,
                );
                let viewport = vk::Viewport::default()
                    .x(0.0)
                    .y(0.0)
                    .width(self.swapchain.extent.width as f32)
                    .height(self.swapchain.extent.height as f32)
                    .min_depth(0.0)
                    .max_depth(1.0);
                let scissor = vk::Rect2D::default().extent(self.swapchain.extent);
                self.device.device.cmd_set_viewport(command_buffer, 0, &[viewport]);
                self.device.device.cmd_set_scissor(command_buffer, 0, &[scissor]);
                if !self.post_process_descriptor_sets.is_empty() {
                    self.device.device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.post_process_layout,
                        0,
                        &[self.post_process_descriptor_sets[self.current_frame]],
                        &[],
                    );
                }
                self.device.device.cmd_draw(command_buffer, 3, 1, 0, 0);
                self.device.device.cmd_end_render_pass(command_buffer);
            }
            self.device
                .device
                .end_command_buffer(command_buffer)
                .unwrap();
        }
    }

    pub fn recreate_swapchain(&mut self, window: &Window) {
        unsafe {
            self.device.device.device_wait_idle().unwrap();
            self.cleanup_swapchain();
            self.swapchain = VulkanSwapchain::new(
                &self.context.instance,
                &self.device.device,
                self.device.pdevice,
                &self.context.surface_loader,
                self.context.surface,
                window.inner_size().width,
                window.inner_size().height,
            );
            let mut hdr_i = Vec::new();
            let mut hdr_m = Vec::new();
            let mut hdr_v = Vec::new();
            let mut g_alb_i = Vec::new();
            let mut g_alb_m = Vec::new();
            let mut g_alb_v = Vec::new();
            let mut g_norm_i = Vec::new();
            let mut g_norm_m = Vec::new();
            let mut g_norm_v = Vec::new();
            let mut g_pos_i = Vec::new();
            let mut g_pos_m = Vec::new();
            let mut g_pos_v = Vec::new();
            let mut depth_i = Vec::new();
            let mut depth_m = Vec::new();
            let mut depth_v = Vec::new();
            for _ in 0..MAX_FRAMES_IN_FLIGHT {
                let (i, m, v) = Self::create_hdr_resources_impl(
                    &self.device.device,
                    self.device.pdevice,
                    &self.context.instance,
                    self.swapchain.extent,
                    vk::SampleCountFlags::TYPE_1,
                );
                hdr_i.push(i);
                hdr_m.push(m);
                hdr_v.push(v);
                let (i, m, v) = Self::create_image_resource_impl(
                    &self.device.device,
                    self.device.pdevice,
                    &self.context.instance,
                    self.swapchain.extent,
                    vk::Format::R8G8B8A8_UNORM,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT,
                    self.device.msaa_samples,
                );
                g_alb_i.push(i);
                g_alb_m.push(m);
                g_alb_v.push(v);
                let (i, m, v) = Self::create_image_resource_impl(
                    &self.device.device,
                    self.device.pdevice,
                    &self.context.instance,
                    self.swapchain.extent,
                    vk::Format::R16G16B16A16_SFLOAT,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT,
                    self.device.msaa_samples,
                );
                g_norm_i.push(i);
                g_norm_m.push(m);
                g_norm_v.push(v);
                let (i, m, v) = Self::create_image_resource_impl(
                    &self.device.device,
                    self.device.pdevice,
                    &self.context.instance,
                    self.swapchain.extent,
                    vk::Format::R16G16B16A16_SFLOAT,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT,
                    self.device.msaa_samples,
                );
                g_pos_i.push(i);
                g_pos_m.push(m);
                g_pos_v.push(v);
                let (i, m, v) = Self::create_depth_resources_impl(
                    &self.device.device,
                    self.device.pdevice,
                    &self.context.instance,
                    self.swapchain.extent,
                    self.device.msaa_samples,
                    self.device.depth_format,
                );
                depth_i.push(i);
                depth_m.push(m);
                depth_v.push(v);
            }
            self.hdr_images = hdr_i;
            self.hdr_memories = hdr_m;
            self.hdr_views = hdr_v;
            self.gbuffer_albedo_images = g_alb_i;
            self.gbuffer_albedo_memories = g_alb_m;
            self.gbuffer_albedo_views = g_alb_v;
            self.gbuffer_normal_images = g_norm_i;
            self.gbuffer_normal_memories = g_norm_m;
            self.gbuffer_normal_views = g_norm_v;
            self.gbuffer_position_images = g_pos_i;
            self.gbuffer_position_memories = g_pos_m;
            self.gbuffer_position_views = g_pos_v;
            self.depth_images = depth_i;
            self.depth_memories = depth_m;
            self.depth_views = depth_v;
            self.framebuffers = Self::create_framebuffers_impl(
                &self.device.device,
                self.render_pass,
                &self.hdr_views,
                &self.gbuffer_albedo_views,
                &self.gbuffer_normal_views,
                &self.gbuffer_position_views,
                &self.gbuffer_pbr_views,
                &self.depth_views,
                self.swapchain.extent,
            );
            self.post_process_framebuffers = Self::create_post_process_framebuffers_impl(
                &self.device.device,
                self.post_process_render_pass,
                &self.swapchain.views,
                self.swapchain.extent,
            );
            self.update_deferred_descriptor_sets();
            self.update_post_process_descriptor_sets();
        }
    }

    fn cleanup_swapchain(&mut self) {
        unsafe {
            for &f in &self.framebuffers {
                self.device.device.destroy_framebuffer(f, None);
            }
            for &f in &self.post_process_framebuffers {
                self.device.device.destroy_framebuffer(f, None);
            }
            for &v in &self.hdr_views {
                self.device.device.destroy_image_view(v, None);
            }
            for &i in &self.hdr_images {
                self.device.device.destroy_image(i, None);
            }
            for &m in &self.hdr_memories {
                self.device.device.free_memory(m, None);
            }
            for &v in &self.gbuffer_albedo_views {
                self.device.device.destroy_image_view(v, None);
            }
            for &i in &self.gbuffer_albedo_images {
                self.device.device.destroy_image(i, None);
            }
            for &m in &self.gbuffer_albedo_memories {
                self.device.device.free_memory(m, None);
            }
            for &v in &self.gbuffer_normal_views {
                self.device.device.destroy_image_view(v, None);
            }
            for &i in &self.gbuffer_normal_images {
                self.device.device.destroy_image(i, None);
            }
            for &m in &self.gbuffer_normal_memories {
                self.device.device.free_memory(m, None);
            }
            for &v in &self.gbuffer_position_views {
                self.device.device.destroy_image_view(v, None);
            }
            for &i in &self.gbuffer_position_images {
                self.device.device.destroy_image(i, None);
            }
            for &m in &self.gbuffer_position_memories {
                self.device.device.free_memory(m, None);
            }
            for &v in &self.gbuffer_pbr_views {
                self.device.device.destroy_image_view(v, None);
            }
            for &i in &self.gbuffer_pbr_images {
                self.device.device.destroy_image(i, None);
            }
            for &m in &self.gbuffer_pbr_memories {
                self.device.device.free_memory(m, None);
            }
            for &v in &self.depth_views {
                self.device.device.destroy_image_view(v, None);
            }
            for &i in &self.depth_images {
                self.device.device.destroy_image(i, None);
            }
            for &m in &self.depth_memories {
                self.device.device.free_memory(m, None);
            }
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
            w,
            h,
            mip,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
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
        unsafe {
            self.device.device.destroy_buffer(st.handle, None);
            self.device.device.free_memory(st.memory, None);
        }
        Texture {
            image: i,
            memory: m,
            view: v,
            sampler: s,
            mip_levels: mip,
        }
    }

    pub fn destroy_texture(&mut self, t: Texture) {
        self.texture_descriptor_sets.remove(&t.view);
        unsafe {
            self.device.device.destroy_sampler(t.sampler, None);
            self.device.device.destroy_image_view(t.view, None);
            self.device.device.destroy_image(t.image, None);
            self.device.device.free_memory(t.memory, None);
        }
    }

    pub fn create_image_basic(
        &self,
        w: u32,
        h: u32,
        mip: u32,
        f: vk::Format,
        t: vk::ImageTiling,
        u: vk::ImageUsageFlags,
        p: vk::MemoryPropertyFlags,
    ) -> (vk::Image, vk::DeviceMemory) {
        let i = unsafe {
            self.device
                .device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .extent(vk::Extent3D {
                            width: w,
                            height: h,
                            depth: 1,
                        })
                        .mip_levels(mip)
                        .array_layers(1)
                        .format(f)
                        .tiling(t)
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .usage(u)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .unwrap()
        };
        let reqs = unsafe { self.device.device.get_image_memory_requirements(i) };
        let m = unsafe {
            self.device
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(reqs.size)
                        .memory_type_index(self.find_memory_type(reqs.memory_type_bits, p)),
                    None,
                )
                .unwrap()
        };
        unsafe {
            self.device.device.bind_image_memory(i, m, 0).unwrap();
        }
        (i, m)
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

    fn find_memory_type(&self, filter: u32, props: vk::MemoryPropertyFlags) -> u32 {
        let p = unsafe {
            self.context
                .instance
                .get_physical_device_memory_properties(self.device.pdevice)
        };
        for i in 0..p.memory_type_count {
            if (filter & (1 << i)) != 0
                && (p.memory_types[i as usize].property_flags & props) == props
            {
                return i;
            }
        }
        panic!("Memory type not found")
    }

    pub fn get_buffer(&self, id: u32) -> Option<&Buffer> {
        self.vertex_buffers.get(id as usize)
    }
    pub fn get_instance_buffer(&self, id: u32) -> Option<&Buffer> {
        self.instance_buffers.get(id as usize)
    }
    pub fn clear_instance_buffers(&mut self) {
        // In a better implementation, we would keep these buffers for a few frames
        // using a ring buffer or similar, to avoid wait_idle.
        // For now, we'll keep the wait_idle to ensure safety, but we can potentially
        // optimize this by not clearing every frame if we manage lifecycles better.
        unsafe {
            self.device.device.device_wait_idle().ok();
            let buffers = std::mem::take(&mut self.instance_buffers);
            for b in buffers {
                self.device.device.destroy_buffer(b.handle, None);
                self.device.device.free_memory(b.memory, None);
            }
        }
    }
    pub fn get_graphics_queue(&self) -> vk::Queue {
        self.device.graphics_queue
    }
    pub fn get_msaa_samples(&self) -> vk::SampleCountFlags {
        self.device.msaa_samples
    }
    pub fn get_extent(&self) -> vk::Extent2D {
        self.swapchain.extent
    }
    pub fn get_device(&self) -> &ash::Device {
        &self.device.device
    }

    pub fn upload_to_buffer<T: Copy>(&self, b: &Buffer, data: &[T]) {
        unsafe {
            let p = self
                .device
                .device
                .map_memory(b.memory, 0, b.size, vk::MemoryMapFlags::empty())
                .unwrap();
            std::ptr::copy_nonoverlapping(data.as_ptr(), p as *mut T, data.len());
            self.device.device.unmap_memory(b.memory);
        }
    }
    pub fn create_buffer(
        &self,
        sz: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Buffer {
        let h = unsafe {
            self.device
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::default()
                        .size(sz)
                        .usage(usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .unwrap()
        };
        let reqs = unsafe { self.device.device.get_buffer_memory_requirements(h) };
        let m = unsafe {
            self.device
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(reqs.size)
                        .memory_type_index(self.find_memory_type(reqs.memory_type_bits, properties)),
                    None,
                )
                .unwrap()
        };
        unsafe {
            self.device.device.bind_buffer_memory(h, m, 0).unwrap();
        }
        Buffer {
            handle: h,
            memory: m,
            size: sz,
        }
    }
    pub fn destroy_buffer(&self, buffer: Buffer) {
        unsafe {
            self.device.device.destroy_buffer(buffer.handle, None);
            self.device.device.free_memory(buffer.memory, None);
        }
    }
    pub fn create_image_view_basic(&self, i: vk::Image, f: vk::Format, mip: u32) -> vk::ImageView {
        unsafe {
            self.device
                .device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(i)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(f)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: mip,
                            base_array_layer: 0,
                            layer_count: 1,
                        }),
                    None,
                )
                .unwrap()
        }
    }
    pub fn transition_image_layout_basic(
        &self,
        i: vk::Image,
        ol: vk::ImageLayout,
        nl: vk::ImageLayout,
        mip: u32,
    ) {
        let cb = self.begin_single_time_commands();
        let bar = vk::ImageMemoryBarrier::default()
            .old_layout(ol)
            .new_layout(nl)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(i)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mip,
                base_array_layer: 0,
                layer_count: 1,
            });
        let (ss, ds) = match (ol, nl) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            ),
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            _ => (
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
            ),
        };
        unsafe {
            self.device.device.cmd_pipeline_barrier(
                cb,
                ss,
                ds,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[bar],
            );
        }
        self.end_single_time_commands(cb);
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
        let cb = unsafe {
            self.device
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_pool(self.command_pool)
                        .command_buffer_count(1),
                )
                .unwrap()[0]
        };
        unsafe {
            self.device
                .device
                .begin_command_buffer(
                    cb,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();
        }
        cb
    }
    pub fn end_single_time_commands(&self, cb: vk::CommandBuffer) {
        unsafe {
            self.device.device.end_command_buffer(cb).unwrap();
            self.device
                .device
                .queue_submit(
                    self.device.graphics_queue,
                    &[vk::SubmitInfo::default().command_buffers(&[cb])],
                    vk::Fence::null(),
                )
                .unwrap();
            self.device
                .device
                .queue_wait_idle(self.device.graphics_queue)
                .unwrap();
            self.device
                .device
                .free_command_buffers(self.command_pool, &[cb]);
        }
    }

    fn create_hdr_resources_impl(
        device: &ash::Device,
        pdevice: vk::PhysicalDevice,
        instance: &ash::Instance,
        extent: vk::Extent2D,
        samples: vk::SampleCountFlags,
    ) -> (vk::Image, vk::DeviceMemory, vk::ImageView) {
        let mut usage = vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::INPUT_ATTACHMENT;
        if samples != vk::SampleCountFlags::TYPE_1 {
            usage |= vk::ImageUsageFlags::TRANSIENT_ATTACHMENT;
        }
        Self::create_image_resource_impl(
            device,
            pdevice,
            instance,
            extent,
            vk::Format::R16G16B16A16_SFLOAT,
            usage,
            samples,
        )
    }
    fn create_depth_resources_impl(
        device: &ash::Device,
        pdevice: vk::PhysicalDevice,
        instance: &ash::Instance,
        extent: vk::Extent2D,
        samples: vk::SampleCountFlags,
        format: vk::Format,
    ) -> (vk::Image, vk::DeviceMemory, vk::ImageView) {
        let mut usage =
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT;
        if samples != vk::SampleCountFlags::TYPE_1 {
            usage |= vk::ImageUsageFlags::TRANSIENT_ATTACHMENT;
        }
        Self::create_image_resource_impl(device, pdevice, instance, extent, format, usage, samples)
    }
    fn create_image_resource_impl(
        device: &ash::Device,
        pdevice: vk::PhysicalDevice,
        instance: &ash::Instance,
        extent: vk::Extent2D,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        samples: vk::SampleCountFlags,
    ) -> (vk::Image, vk::DeviceMemory, vk::ImageView) {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .samples(samples)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let img = unsafe { device.create_image(&info, None).unwrap() };
        let reqs = unsafe { device.get_image_memory_requirements(img) };
        let props = unsafe { instance.get_physical_device_memory_properties(pdevice) };
        let mut type_idx = 0;
        for i in 0..props.memory_type_count {
            if (reqs.memory_type_bits & (1 << i)) != 0
                && (props.memory_types[i as usize].property_flags
                    & vk::MemoryPropertyFlags::DEVICE_LOCAL)
                    == vk::MemoryPropertyFlags::DEVICE_LOCAL
            {
                type_idx = i;
                break;
            }
        }
        let mem = unsafe {
            device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(reqs.size)
                        .memory_type_index(type_idx),
                    None,
                )
                .unwrap()
        };
        unsafe {
            device.bind_image_memory(img, mem, 0).unwrap();
        }
        let aspect = if format == vk::Format::D32_SFLOAT
            || format == vk::Format::D32_SFLOAT_S8_UINT
            || format == vk::Format::D24_UNORM_S8_UINT
            || format == vk::Format::D16_UNORM
        {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        let v_info = vk::ImageViewCreateInfo::default()
            .image(img)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });
        let view = unsafe { device.create_image_view(&v_info, None).unwrap() };
        (img, mem, view)
    }
    const SHADOW_MAP_CASCADE_SIZE: u32 = 2048;
    fn create_shadow_resources_impl(
        device: &ash::Device,
        pdevice: vk::PhysicalDevice,
        instance: &ash::Instance,
    ) -> (vk::Image, vk::DeviceMemory, vk::ImageView, vk::Sampler) {
        let (img, mem, view) = Self::create_image_resource_impl(
            device,
            pdevice,
            instance,
            vk::Extent2D {
                width: Self::SHADOW_MAP_CASCADE_SIZE,
                height: Self::SHADOW_MAP_CASCADE_SIZE,
            },
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::SampleCountFlags::TYPE_1,
        );
        let s = unsafe {
            device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR)
                        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                        .anisotropy_enable(false)
                        .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
                        .compare_enable(false)
                        .mipmap_mode(vk::SamplerMipmapMode::LINEAR),
                    None,
                )
                .unwrap()
        };
        (img, mem, view, s)
    }

    fn create_render_pass_impl(
        device: &ash::Device,
        _format: vk::Format,
        msaa_samples: vk::SampleCountFlags,
        depth_format: vk::Format,
    ) -> vk::RenderPass {
        let albedo_att = vk::AttachmentDescription::default()
            .format(vk::Format::R8G8B8A8_UNORM)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let normal_att = vk::AttachmentDescription::default()
            .format(vk::Format::R16G16B16A16_SFLOAT)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let position_att = vk::AttachmentDescription::default()
            .format(vk::Format::R16G16B16A16_SFLOAT)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let pbr_att = vk::AttachmentDescription::default()
            .format(vk::Format::R8G8B8A8_UNORM)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let hdr_att = vk::AttachmentDescription::default()
            .format(vk::Format::R16G16B16A16_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let depth_att = vk::AttachmentDescription::default()
            .format(depth_format)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let albedo_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let normal_ref = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let position_ref = vk::AttachmentReference::default()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let pbr_ref = vk::AttachmentReference::default()
            .attachment(3)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let hdr_ref = vk::AttachmentReference::default()
            .attachment(4)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let depth_ref = vk::AttachmentReference::default()
            .attachment(5)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let subpass0_color_attachments = [albedo_ref, normal_ref, position_ref, pbr_ref];
        let subpass0 = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&subpass0_color_attachments)
            .depth_stencil_attachment(&depth_ref);

        let subpass1_input_attachments = [
            vk::AttachmentReference::default()
                .attachment(0)
                .layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            vk::AttachmentReference::default()
                .attachment(1)
                .layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            vk::AttachmentReference::default()
                .attachment(2)
                .layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            vk::AttachmentReference::default()
                .attachment(3)
                .layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            vk::AttachmentReference::default()
                .attachment(5)
                .layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        ];
        let subpass1 = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&hdr_ref))
            .input_attachments(&subpass1_input_attachments);

        let deps = [
            vk::SubpassDependency::default()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(vk::PipelineStageFlags::BOTTOM_OF_PIPE)
                .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags::MEMORY_READ)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
            vk::SubpassDependency::default()
                .src_subpass(0)
                .dst_subpass(1)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ),
            vk::SubpassDependency::default()
                .src_subpass(1)
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ),
        ];
        let attachments = [albedo_att, normal_att, position_att, pbr_att, hdr_att, depth_att];
        let subpasses = [subpass0, subpass1];
        unsafe {
            device
                .create_render_pass(
                    &vk::RenderPassCreateInfo::default()
                        .attachments(&attachments)
                        .subpasses(&subpasses)
                        .dependencies(&deps),
                    None,
                )
                .expect("Failed to create render pass")
        }
    }
    fn create_framebuffers_impl(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        hdr_views: &[vk::ImageView],
        albedo_views: &[vk::ImageView],
        normal_views: &[vk::ImageView],
        position_views: &[vk::ImageView],
        pbr_views: &[vk::ImageView],
        depth_views: &[vk::ImageView],
        extent: vk::Extent2D,
    ) -> Vec<vk::Framebuffer> {
        (0..MAX_FRAMES_IN_FLIGHT)
            .map(|i| {
                let attachments = [
                    albedo_views[i],
                    normal_views[i],
                    position_views[i],
                    pbr_views[i],
                    hdr_views[i],
                    depth_views[i],
                ];
                unsafe {
                    device
                        .create_framebuffer(
                            &vk::FramebufferCreateInfo::default()
                                .render_pass(render_pass)
                                .attachments(&attachments)
                                .width(extent.width)
                                .height(extent.height)
                                .layers(1),
                            None,
                        )
                        .expect("Failed to create framebuffer")
                }
            })
            .collect()
    }
    fn create_post_process_render_pass_impl(
        device: &ash::Device,
        format: vk::Format,
    ) -> vk::RenderPass {
        let color_attachment = vk::AttachmentDescription::default()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
        let color_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_ref));
        unsafe {
            device
                .create_render_pass(
                    &vk::RenderPassCreateInfo::default()
                        .attachments(std::slice::from_ref(&color_attachment))
                        .subpasses(std::slice::from_ref(&subpass)),
                    None,
                )
                .expect("Failed to create post-process render pass")
        }
    }
    fn create_post_process_framebuffers_impl(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        views: &[vk::ImageView],
        extent: vk::Extent2D,
    ) -> Vec<vk::Framebuffer> {
        views
            .iter()
            .map(|&v| {
                let attachments = [v];
                unsafe {
                    device
                        .create_framebuffer(
                            &vk::FramebufferCreateInfo::default()
                                .render_pass(render_pass)
                                .attachments(&attachments)
                                .width(extent.width)
                                .height(extent.height)
                                .layers(1),
                            None,
                        )
                        .expect("Failed to create post-process framebuffer")
                }
            })
            .collect()
    }
    fn create_command_pool_impl(device: &ash::Device, family: u32) -> vk::CommandPool {
        unsafe {
            device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .expect("Failed to create command pool")
        }
    }
    fn create_command_buffers_impl(
        device: &ash::Device,
        pool: vk::CommandPool,
    ) -> Vec<vk::CommandBuffer> {
        unsafe {
            device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32),
                )
                .expect("Failed to allocate command buffers")
        }
    }
    fn create_descriptor_pool_impl(device: &ash::Device) -> vk::DescriptorPool {
        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1000),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::INPUT_ATTACHMENT)
                .descriptor_count(100),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(100),
        ];
        unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .pool_sizes(&sizes)
                        .max_sets(1000),
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
    fn create_shadow_render_pass_impl(device: &ash::Device) -> vk::RenderPass {
        let att = vk::AttachmentDescription::default()
            .format(vk::Format::D32_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let r = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        let sub = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .depth_stencil_attachment(&r);
        let dep = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .src_access_mask(vk::AccessFlags::SHADER_READ)
            .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE);
        unsafe {
            device
                .create_render_pass(
                    &vk::RenderPassCreateInfo::default()
                        .attachments(std::slice::from_ref(&att))
                        .subpasses(std::slice::from_ref(&sub))
                        .dependencies(std::slice::from_ref(&dep)),
                    None,
                )
                .unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_struct() {
        let buffer = Buffer {
            handle: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            size: 1024,
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
            self.cleanup_swapchain();
            for lb in self.light_buffers.drain(..) {
                self.device.device.destroy_buffer(lb.handle, None);
                self.device.device.free_memory(lb.memory, None);
            }
            self.device
                .device
                .destroy_sampler(self.shadow_sampler, None);
            self.device
                .device
                .destroy_image_view(self.shadow_view, None);
            self.device.device.destroy_image(self.shadow_image, None);
            self.device.device.free_memory(self.shadow_memory, None);
            self.device
                .device
                .destroy_framebuffer(self.shadow_framebuffer, None);
            self.device
                .device
                .destroy_render_pass(self.shadow_render_pass, None);
            self.device
                .device
                .destroy_render_pass(self.post_process_render_pass, None);
            self.device
                .device
                .destroy_pipeline_layout(self.shadow_pipeline_layout, None);
            if let Some(p) = self.shadow_pipeline {
                self.device.device.destroy_pipeline(p, None);
            }
            for &s in &self.image_available_semaphores {
                self.device.device.destroy_semaphore(s, None);
            }
            for &s in &self.render_finished_semaphores {
                self.device.device.destroy_semaphore(s, None);
            }
            for &f in &self.in_flight_fences {
                self.device.device.destroy_fence(f, None);
            }
            self.device
                .device
                .destroy_command_pool(self.command_pool, None);
            if let Some(p) = self.pipeline.take() {
                self.device
                    .device
                    .destroy_pipeline(p.graphics_pipeline, None);
                self.device.device.destroy_pipeline_layout(p.layout, None);
                self.device
                    .device
                    .destroy_descriptor_set_layout(p.descriptor_set_layout, None);
            }
            if let Some(p) = self.deferred_pipeline {
                self.device.device.destroy_pipeline(p, None);
            }
            self.texture_descriptor_sets.clear();
            self.device
                .device
                .destroy_pipeline_layout(self.deferred_layout, None);
            self.device
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .device
                .destroy_descriptor_pool(self.post_process_descriptor_pool, None);
            for &v in &self.gbuffer_pbr_views {
                self.device.device.destroy_image_view(v, None);
            }
            for &i in &self.gbuffer_pbr_images {
                self.device.device.destroy_image(i, None);
            }
            for &m in &self.gbuffer_pbr_memories {
                self.device.device.free_memory(m, None);
            }
            for (img, (mem, view)) in self.bloom_images.drain(..).zip(
                self.bloom_memories
                    .drain(..)
                    .zip(self.bloom_views.drain(..)),
            ) {
                self.device.device.destroy_image_view(view, None);
                self.device.device.destroy_image(img, None);
                self.device.device.free_memory(mem, None);
            }
            if let Some(mut e) = self.egui_renderer.take() {
                e.destroy(self);
            }
            if let Some(t) = self.default_texture.take() {
                self.destroy_texture(t);
            }
            let vbs = std::mem::take(&mut self.vertex_buffers);
            for vb in vbs {
                self.device.device.destroy_buffer(vb.handle, None);
                self.device.device.free_memory(vb.memory, None);
            }
            let ibs = std::mem::take(&mut self.instance_buffers);
            for ib in ibs {
                self.device.device.destroy_buffer(ib.handle, None);
                self.device.device.free_memory(ib.memory, None);
            }
            if let Some(ib) = self.index_buffer.take() {
                self.device.device.destroy_buffer(ib.handle, None);
                self.device.device.free_memory(ib.memory, None);
            }
        }
    }
}
