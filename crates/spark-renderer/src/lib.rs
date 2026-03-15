pub mod pipeline;
pub mod vertex;
pub mod vulkan;
pub mod error;
pub mod ui;

use ash::vk;
use winit::window::Window;
use crate::pipeline::Pipeline;
use crate::vulkan::context::VulkanContext;
use crate::vulkan::device::VulkanDevice;
use crate::vulkan::swapchain::VulkanSwapchain;
use crate::vulkan::texture::Texture;
pub use ash;
use crate::error::RendererError;
use crate::ui::EguiRenderer;

#[allow(dead_code)]
pub struct Renderer {
    context: VulkanContext,
    device: VulkanDevice,
    swapchain: VulkanSwapchain,
    pub render_pass: vk::RenderPass,
    pub hdr_image: vk::Image,
    pub hdr_memory: vk::DeviceMemory,
    pub hdr_view: vk::ImageView,
    pub depth_image: vk::Image,
    pub depth_memory: vk::DeviceMemory,
    pub depth_view: vk::ImageView,
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
    pipeline: Option<Pipeline>,
    pub vertex_buffers: Vec<Buffer>,
    pub instance_buffers: Vec<Buffer>,
    index_buffer: Option<Buffer>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    egui_renderer: Option<EguiRenderer>,
}

const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct Buffer {
    pub handle: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
}

impl Renderer {
    pub fn new(
        window: &Window,
        ui_shaders: Option<(&[u32], &[u32])>
    ) -> Result<Self, RendererError> {
        let context = VulkanContext::new(window)?;
        let device = VulkanDevice::new(&context.instance, &context.surface_loader, context.surface)?;
        let swapchain = VulkanSwapchain::new(
            &context.instance,
            &device.device,
            device.pdevice,
            &context.surface_loader,
            context.surface,
            window.inner_size().width,
            window.inner_size().height,
        );

        let (hdr_image, hdr_memory, hdr_view) = Self::create_hdr_resources(
            &device.device,
            device.pdevice,
            &context.instance,
            swapchain.extent,
        );

        let (depth_image, depth_memory, depth_view) = Self::create_depth_resources(
            &device.device,
            device.pdevice,
            &context.instance,
            swapchain.extent,
        );

        let (shadow_image, shadow_memory, shadow_view, shadow_sampler) =
            Self::create_shadow_resources(&device.device, device.pdevice, &context.instance);
        let shadow_render_pass = Self::create_shadow_render_pass(&device.device);
        let shadow_framebuffer = {
            let attachments = [shadow_view];
            let info = vk::FramebufferCreateInfo::default()
                .render_pass(shadow_render_pass)
                .attachments(&attachments)
                .width(Self::SHADOW_MAP_CASCADE_SIZE)
                .height(Self::SHADOW_MAP_CASCADE_SIZE)
                .layers(1);
            unsafe { device.device.create_framebuffer(&info, None).unwrap() }
        };

        let shadow_pipeline_layout = {
            let push_constant_ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::VERTEX)
                .offset(0)
                .size((std::mem::size_of::<spark_math::Mat4>() * 2) as u32)];
            let info = vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_constant_ranges);
            unsafe { device.device.create_pipeline_layout(&info, None).unwrap() }
        };

        let render_pass = Self::create_render_pass(&device.device, swapchain.format);
        let framebuffers = Self::create_framebuffers(
            &device.device,
            render_pass,
            hdr_view,
            depth_view,
            swapchain.extent,
        );

        let post_process_render_pass = Self::create_post_process_render_pass(&device.device, swapchain.format);
        let post_process_framebuffers = Self::create_post_process_framebuffers(
            &device.device,
            post_process_render_pass,
            &swapchain.views,
            swapchain.extent,
        );

        let command_pool = Self::create_command_pool(&device.device, device.graphics_family);
        let command_buffers = Self::create_command_buffers(&device.device, command_pool);

        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) =
            Self::create_sync_objects(&device.device);

        let descriptor_pool = Self::create_descriptor_pool(&device.device);
        let post_process_descriptor_pool = Self::create_descriptor_pool(&device.device);
        let mut bloom_images = Vec::new();
        let mut bloom_memories = Vec::new();
        let mut bloom_views = Vec::new();

        for i in 1..6 { // 5 levels of bloom downsampling
            let (img, mem, view) = Self::create_hdr_resources(
                &device.device,
                device.pdevice,
                &context.instance,
                vk::Extent2D {
                    width: (swapchain.extent.width >> i).max(1),
                    height: (swapchain.extent.height >> i).max(1),
                },
            );
            bloom_images.push(img);
            bloom_memories.push(mem);
            bloom_views.push(view);
        }
        let egui_renderer = ui_shaders.map(|(v, f)| {
            EguiRenderer::new(&device.device, render_pass, v, f, swapchain.extent)
        });

        Ok(Self {
            context,
            device,
            swapchain,
            render_pass,
            hdr_image,
            hdr_memory,
            hdr_view,
            depth_image,
            depth_memory,
            depth_view,
            post_process_render_pass,
            post_process_framebuffers,
            shadow_image,
            shadow_memory,
            shadow_view,
            shadow_sampler,
            shadow_render_pass,
            shadow_framebuffer,
            shadow_pipeline: None,
            shadow_pipeline_layout,
            post_process_pipeline: None,
            post_process_layout: vk::PipelineLayout::null(),
            bloom_pipeline: None,
            post_process_descriptor_pool,
            post_process_descriptor_sets: Vec::new(),
            bloom_images,
            bloom_memories,
            bloom_views,
            framebuffers,
            command_pool,
            command_buffers,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            current_frame: 0,
            pipeline: None,
            vertex_buffers: Vec::new(),
            instance_buffers: Vec::new(),
            index_buffer: None,
            descriptor_pool,
            descriptor_sets: Vec::new(),
            egui_renderer,
        })
    }

    pub fn set_pipeline(&mut self, pipeline: Pipeline) {
        self.pipeline = Some(pipeline);
    }

    pub fn set_shadow_pipeline(&mut self, pipeline: vk::Pipeline) {
        self.shadow_pipeline = Some(pipeline);
    }

    pub fn set_post_process_pipeline(&mut self, pipeline: vk::Pipeline, layout: vk::PipelineLayout, ds_layout: vk::DescriptorSetLayout, bloom_pipeline: vk::Pipeline) {
        self.post_process_pipeline = Some(pipeline);
        self.post_process_layout = layout;
        self.bloom_pipeline = Some(bloom_pipeline);

        let layouts = [ds_layout, ds_layout];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.post_process_descriptor_pool)
            .set_layouts(&layouts);

        let sets = unsafe {
            self.device.device
                .allocate_descriptor_sets(&alloc_info)
                .expect("Failed to allocate post-process descriptor sets")
        };

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let image_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(self.hdr_view)
                .sampler(self.shadow_sampler)];

            let bloom_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(self.bloom_views[0]) // Simplification: just one level
                .sampler(self.shadow_sampler)];

            let descriptor_writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(sets[i])
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&image_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(sets[i])
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&bloom_info),
            ];

            unsafe {
                self.device.device.update_descriptor_sets(&descriptor_writes, &[]);
            }
        }

        self.post_process_descriptor_sets = sets;
    }

    pub fn set_texture(&mut self, texture: &Texture) {
        if let Some(pipeline) = &self.pipeline {
            let layouts = [pipeline.descriptor_set_layout];
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(self.descriptor_pool)
                .set_layouts(&layouts);

            let sets = unsafe {
                self.device.device
                    .allocate_descriptor_sets(&alloc_info)
                    .expect("Failed to allocate descriptor sets")
            };

            let image_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(texture.view)
                .sampler(texture.sampler)];

            let shadow_image_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(self.shadow_view)
                .sampler(self.shadow_sampler)];

            let descriptor_writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(sets[0])
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&image_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(sets[0])
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&shadow_image_info),
            ];

            unsafe {
                self.device.device.update_descriptor_sets(&descriptor_writes, &[]);
            }

            self.descriptor_sets = sets;
        }
    }

    pub fn add_vertex_buffer(&mut self, buffer: Buffer) -> u32 {
        self.vertex_buffers.push(buffer);
        (self.vertex_buffers.len() - 1) as u32
    }

    pub fn add_instance_buffer(&mut self, buffer: Buffer) -> u32 {
        self.instance_buffers.push(buffer);
        (self.instance_buffers.len() - 1) as u32
    }

    pub fn set_index_buffer(&mut self, buffer: Buffer) {
        self.index_buffer = Some(buffer);
    }

    pub fn draw_frame(
        &mut self,
        renderables: &[(spark_math::Mat4, u32, Option<String>, Option<u32>)],
        instanced_renderables: &[(u32, u32, u32)], // (vertex_buffer_id, instance_buffer_id, instance_count)
        view_proj: spark_math::Mat4,
        light_view_proj: spark_math::Mat4,
        window: &Window,
        egui_output: Option<(egui::FullOutput, egui::Context)>,
    ) {
        unsafe {
            self.device.device
                .wait_for_fences(
                    &[self.in_flight_fences[self.current_frame]],
                    true,
                    u64::MAX,
                )
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

            self.device.device
                .reset_fences(&[self.in_flight_fences[self.current_frame]])
                .expect("Failed to reset fence");

            self.device.device
                .reset_command_buffer(
                    self.command_buffers[self.current_frame],
                    vk::CommandBufferResetFlags::empty(),
                )
                .expect("Failed to reset command buffer");

            self.record_command_buffer(image_index, renderables, instanced_renderables, view_proj, light_view_proj, egui_output);

            let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let command_buffers = [self.command_buffers[self.current_frame]];
            let signal_semaphores = [self.render_finished_semaphores[self.current_frame]];

            let submit_info = vk::SubmitInfo::default()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_stages)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores);

            self.device.device
                .queue_submit(
                    self.device.graphics_queue,
                    &[submit_info],
                    self.in_flight_fences[self.current_frame],
                )
                .expect("Failed to submit draw command");

            let swapchains = [self.swapchain.handle];
            let image_indices = [image_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            let result = self.swapchain.loader.queue_present(self.device.graphics_queue, &present_info);
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
        _renderables: &[(spark_math::Mat4, u32, Option<String>, Option<u32>)],
        instanced_renderables: &[(u32, u32, u32)],
        view_proj: spark_math::Mat4,
        light_view_proj: spark_math::Mat4,
        egui_output: Option<(egui::FullOutput, egui::Context)>,
    ) {
        let command_buffer = self.command_buffers[self.current_frame];

        let begin_info = vk::CommandBufferBeginInfo::default();

        unsafe {
            self.device.device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Failed to begin recording command buffer");

            // --- Shadow Pass ---
            if let Some(shadow_pipeline) = self.shadow_pipeline {
                let shadow_clear_values = [vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                }];

                let shadow_render_pass_info = vk::RenderPassBeginInfo::default()
                    .render_pass(self.shadow_render_pass)
                    .framebuffer(self.shadow_framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D { width: Self::SHADOW_MAP_CASCADE_SIZE, height: Self::SHADOW_MAP_CASCADE_SIZE },
                    })
                    .clear_values(&shadow_clear_values);

                self.device.device.cmd_begin_render_pass(
                    command_buffer,
                    &shadow_render_pass_info,
                    vk::SubpassContents::INLINE,
                );

                self.device.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    shadow_pipeline,
                );

                let light_view_proj_bytes = std::slice::from_raw_parts(
                    &light_view_proj as *const _ as *const u8,
                    std::mem::size_of::<spark_math::Mat4>(),
                );

                // We use push constants to send light_view_proj.
                // We need to offset the model matrix push constant or use a separate range.
                // In shadow pass, we need model and light_view_proj.

                for (vb_id, instance_buffer_id, instance_count) in instanced_renderables {
                    if let (Some(vb), Some(ib)) = (self.get_buffer(*vb_id), self.get_instance_buffer(*instance_buffer_id)) {
                        self.device.device.cmd_bind_vertex_buffers(
                            command_buffer,
                            0,
                            &[vb.handle, ib.handle],
                            &[0, 0]
                        );

                        self.device.device.cmd_push_constants(
                            command_buffer,
                            self.shadow_pipeline_layout,
                            vk::ShaderStageFlags::VERTEX,
                            0, // Offset 0 for light_view_proj
                            light_view_proj_bytes,
                        );

                        let vertex_count = (vb.size / std::mem::size_of::<crate::vertex::Vertex>() as u64) as u32;
                        self.device.device.cmd_draw(command_buffer, vertex_count, *instance_count, 0, 0);
                    }
                }

                self.device.device.cmd_end_render_pass(command_buffer);
            }

            let clear_values = [
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

            let render_pass_info = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.framebuffers[self.current_frame])
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.swapchain.extent,
                })
                .clear_values(&clear_values);

            self.device.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );

            if let Some(pipeline) = &self.pipeline {
                self.device.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.graphics_pipeline,
                );

                if !self.descriptor_sets.is_empty() {
                    self.device.device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.layout,
                        0,
                        &self.descriptor_sets,
                        &[],
                    );
                }

                let mut view_proj_constants = [spark_math::Mat4::IDENTITY; 2];
                view_proj_constants[0] = view_proj;
                view_proj_constants[1] = light_view_proj;

                let view_proj_bytes = std::slice::from_raw_parts(
                    view_proj_constants.as_ptr() as *const u8,
                    std::mem::size_of::<spark_math::Mat4>() * 2,
                );
                self.device.device.cmd_push_constants(
                    command_buffer,
                    pipeline.layout,
                    vk::ShaderStageFlags::VERTEX,
                    0,
                    view_proj_bytes,
                );

                // Instanced draw calls
                for (vb_id, instance_buffer_id, instance_count) in instanced_renderables {
                    if let (Some(vb), Some(ib)) = (self.get_buffer(*vb_id), self.get_instance_buffer(*instance_buffer_id)) {
                        self.device.device.cmd_bind_vertex_buffers(
                            command_buffer,
                            0,
                            &[vb.handle, ib.handle],
                            &[0, 0]
                        );

                        let vertex_count = (vb.size / std::mem::size_of::<crate::vertex::Vertex>() as u64) as u32;
                        self.device.device.cmd_draw(command_buffer, vertex_count, *instance_count, 0, 0);
                    }
                }
            }

            if let Some((output, ctx)) = egui_output {
                let extent = self.get_extent();
                let screen_size = [extent.width as f32, extent.height as f32];

                if let Some(mut egui) = self.egui_renderer.take() {
                    egui.draw(self, command_buffer, output, screen_size, &ctx);
                    self.egui_renderer = Some(egui);
                }
            }

            self.device.device.cmd_end_render_pass(command_buffer);

            // --- Post Process Pass ---
            if let Some(post_pipeline) = self.post_process_pipeline {
                // Bloom Filter Pass (Extract bright areas)
                if let Some(bloom_pipe) = self.bloom_pipeline {
                     // We would need a dedicated render pass and framebuffer for the bloom filter.
                     // For this high-level skeleton, we'll focus on the Tone Mapping dispatch.
                }

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

                let post_clear_values = [vk::ClearValue {
                    color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] },
                }];

                let post_pass_info = vk::RenderPassBeginInfo::default()
                    .render_pass(self.post_process_render_pass)
                    .framebuffer(self.post_process_framebuffers[image_index as usize])
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: self.swapchain.extent,
                    })
                    .clear_values(&post_clear_values);

                self.device.device.cmd_begin_render_pass(
                    command_buffer,
                    &post_pass_info,
                    vk::SubpassContents::INLINE,
                );

                self.device.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    post_pipeline,
                );

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

            self.device.device
                .end_command_buffer(command_buffer)
                .expect("Failed to record command buffer");
        }
    }

    fn create_post_process_render_pass(device: &ash::Device, format: vk::Format) -> vk::RenderPass {
        let color_attachment = vk::AttachmentDescription::default()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_attachment_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_attachment_ref));

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(std::slice::from_ref(&color_attachment))
            .subpasses(std::slice::from_ref(&subpass));

        unsafe {
            device
                .create_render_pass(&render_pass_info, None)
                .expect("Failed to create post-process render pass")
        }
    }

    fn create_render_pass(device: &ash::Device, _format: vk::Format) -> vk::RenderPass {
        let color_attachment = vk::AttachmentDescription::default()
            .format(vk::Format::R16G16B16A16_SFLOAT) // Intermediate HDR format
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let color_attachment_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let depth_attachment = vk::AttachmentDescription::default()
            .format(vk::Format::D32_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let depth_attachment_ref = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_attachment_ref))
            .depth_stencil_attachment(&depth_attachment_ref);

        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE);

        let attachments = [color_attachment, depth_attachment];
        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(std::slice::from_ref(&dependency));

        unsafe {
            device
                .create_render_pass(&render_pass_info, None)
                .expect("Failed to create render pass")
        }
    }

    fn create_post_process_framebuffers(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        views: &[vk::ImageView],
        extent: vk::Extent2D,
    ) -> Vec<vk::Framebuffer> {
        views
            .iter()
            .map(|&view| {
                let attachments = [view];
                let framebuffer_info = vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1);

                unsafe {
                    device
                        .create_framebuffer(&framebuffer_info, None)
                        .expect("Failed to create post-process framebuffer")
                }
            })
            .collect()
    }

    fn create_framebuffers(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        hdr_view: vk::ImageView,
        depth_view: vk::ImageView,
        extent: vk::Extent2D,
    ) -> Vec<vk::Framebuffer> {
        (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| {
                let attachments = [hdr_view, depth_view];
                let framebuffer_info = vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&attachments)
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1);

                unsafe {
                    device
                        .create_framebuffer(&framebuffer_info, None)
                        .expect("Failed to create framebuffer")
                }
            })
            .collect()
    }

    fn create_command_pool(device: &ash::Device, graphics_family: u32) -> vk::CommandPool {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(graphics_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        unsafe {
            device
                .create_command_pool(&pool_info, None)
                .expect("Failed to create command pool")
        }
    }

    fn create_command_buffers(device: &ash::Device, pool: vk::CommandPool) -> Vec<vk::CommandBuffer> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32);

        unsafe {
            device
                .allocate_command_buffers(&alloc_info)
                .expect("Failed to allocate command buffers")
        }
    }

    fn create_descriptor_pool(device: &ash::Device) -> vk::DescriptorPool {
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(100)]; // Increased capacity

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(100);

        unsafe {
            device
                .create_descriptor_pool(&pool_info, None)
                .expect("Failed to create descriptor pool")
        }
    }

    fn create_sync_objects(
        device: &ash::Device,
    ) -> (Vec<vk::Semaphore>, Vec<vk::Semaphore>, Vec<vk::Fence>) {
        let mut image_available_semaphores = Vec::new();
        let mut render_finished_semaphores = Vec::new();
        let mut in_flight_fences = Vec::new();

        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            unsafe {
                image_available_semaphores.push(
                    device
                        .create_semaphore(&semaphore_info, None)
                        .expect("Failed to create semaphore"),
                );
                render_finished_semaphores.push(
                    device
                        .create_semaphore(&semaphore_info, None)
                        .expect("Failed to create semaphore"),
                );
                in_flight_fences.push(
                    device
                        .create_fence(&fence_info, None)
                        .expect("Failed to create fence"),
                );
            }
        }

        (
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
        )
    }

    pub fn get_extent(&self) -> vk::Extent2D {
        self.swapchain.extent
    }

    pub fn get_device(&self) -> &ash::Device {
        &self.device.device
    }

    pub fn get_buffer(&self, id: u32) -> Option<&Buffer> {
        self.vertex_buffers.get(id as usize)
    }

    pub fn get_instance_buffer(&self, id: u32) -> Option<&Buffer> {
        self.instance_buffers.get(id as usize)
    }

    pub fn clear_instance_buffers(&mut self) {
        unsafe {
            // In a real engine, we'd wait for the specific fence of the frame
            // using these buffers, but for now we wait for idle.
            self.device.device.device_wait_idle().ok();
            let buffers = std::mem::take(&mut self.instance_buffers);
            for buffer in buffers {
                self.destroy_buffer(buffer);
            }
        }
    }

    pub fn get_graphics_queue(&self) -> vk::Queue {
        self.device.graphics_queue
    }

    fn create_shadow_render_pass(device: &ash::Device) -> vk::RenderPass {
        let shadow_attachment = vk::AttachmentDescription::default()
            .format(vk::Format::D32_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        let shadow_attachment_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .depth_stencil_attachment(&shadow_attachment_ref);

        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .src_access_mask(vk::AccessFlags::SHADER_READ)
            .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .dependency_flags(vk::DependencyFlags::BY_REGION);

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(std::slice::from_ref(&shadow_attachment))
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(std::slice::from_ref(&dependency));

        unsafe {
            device
                .create_render_pass(&render_pass_info, None)
                .expect("Failed to create shadow render pass")
        }
    }

    fn create_hdr_resources(
        device: &ash::Device,
        pdevice: vk::PhysicalDevice,
        instance: &ash::Instance,
        extent: vk::Extent2D,
    ) -> (vk::Image, vk::DeviceMemory, vk::ImageView) {
        let format = vk::Format::R16G16B16A16_SFLOAT;

        let image_info = vk::ImageCreateInfo::default()
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
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
            .samples(vk::SampleCountFlags::TYPE_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image = unsafe {
            device
                .create_image(&image_info, None)
                .expect("Failed to create HDR image")
        };

        let mem_requirements = unsafe { device.get_image_memory_requirements(image) };
        let mem_properties = unsafe { instance.get_physical_device_memory_properties(pdevice) };
        let mut mem_type_index = 0;
        let properties = vk::MemoryPropertyFlags::DEVICE_LOCAL;

        for i in 0..mem_properties.memory_type_count {
            if (mem_requirements.memory_type_bits & (1 << i)) != 0
                && (mem_properties.memory_types[i as usize].property_flags & properties) == properties
            {
                mem_type_index = i;
                break;
            }
        }

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index);

        let memory = unsafe {
            device
                .allocate_memory(&alloc_info, None)
                .expect("Failed to allocate HDR image memory")
        };

        unsafe {
            device
                .bind_image_memory(image, memory, 0)
                .expect("Failed to bind HDR image memory");
        }

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = unsafe {
            device
                .create_image_view(&view_info, None)
                .expect("Failed to create HDR image view")
        };

        (image, memory, view)
    }

    fn create_depth_resources(
        device: &ash::Device,
        pdevice: vk::PhysicalDevice,
        instance: &ash::Instance,
        extent: vk::Extent2D,
    ) -> (vk::Image, vk::DeviceMemory, vk::ImageView) {
        let format = vk::Format::D32_SFLOAT;

        let image_info = vk::ImageCreateInfo::default()
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
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image = unsafe {
            device
                .create_image(&image_info, None)
                .expect("Failed to create depth image")
        };

        let mem_requirements = unsafe { device.get_image_memory_requirements(image) };
        let mem_properties = unsafe { instance.get_physical_device_memory_properties(pdevice) };
        let mut mem_type_index = 0;
        let properties = vk::MemoryPropertyFlags::DEVICE_LOCAL;

        for i in 0..mem_properties.memory_type_count {
            if (mem_requirements.memory_type_bits & (1 << i)) != 0
                && (mem_properties.memory_types[i as usize].property_flags & properties) == properties
            {
                mem_type_index = i;
                break;
            }
        }

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index);

        let memory = unsafe {
            device
                .allocate_memory(&alloc_info, None)
                .expect("Failed to allocate depth image memory")
        };

        unsafe {
            device
                .bind_image_memory(image, memory, 0)
                .expect("Failed to bind depth image memory");
        }

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = unsafe {
            device
                .create_image_view(&view_info, None)
                .expect("Failed to create depth image view")
        };

        (image, memory, view)
    }

    const SHADOW_MAP_CASCADE_SIZE: u32 = 2048;

    fn create_shadow_resources(
        device: &ash::Device,
        pdevice: vk::PhysicalDevice,
        instance: &ash::Instance,
    ) -> (vk::Image, vk::DeviceMemory, vk::ImageView, vk::Sampler) {
        let extent = vk::Extent2D { width: Self::SHADOW_MAP_CASCADE_SIZE, height: Self::SHADOW_MAP_CASCADE_SIZE };
        let format = vk::Format::D32_SFLOAT;

        let image_info = vk::ImageCreateInfo::default()
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
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
            .samples(vk::SampleCountFlags::TYPE_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image = unsafe {
            device
                .create_image(&image_info, None)
                .expect("Failed to create shadow image")
        };

        let mem_requirements = unsafe { device.get_image_memory_requirements(image) };
        let mem_properties = unsafe { instance.get_physical_device_memory_properties(pdevice) };
        let mut mem_type_index = 0;
        let properties = vk::MemoryPropertyFlags::DEVICE_LOCAL;

        for i in 0..mem_properties.memory_type_count {
            if (mem_requirements.memory_type_bits & (1 << i)) != 0
                && (mem_properties.memory_types[i as usize].property_flags & properties) == properties
            {
                mem_type_index = i;
                break;
            }
        }

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index);

        let memory = unsafe {
            device
                .allocate_memory(&alloc_info, None)
                .expect("Failed to allocate shadow image memory")
        };

        unsafe {
            device
                .bind_image_memory(image, memory, 0)
                .expect("Failed to bind shadow image memory");
        }

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::DEPTH,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let view = unsafe {
            device
                .create_image_view(&view_info, None)
                .expect("Failed to create shadow image view")
        };

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
            .anisotropy_enable(false)
            .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
            .unnormalized_coordinates(false)
            .compare_enable(true)
            .compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .min_lod(0.0)
            .max_lod(1.0);

        let sampler = unsafe {
            device
                .create_sampler(&sampler_info, None)
                .expect("Failed to create shadow sampler")
        };

        (image, memory, view, sampler)
    }

    pub fn recreate_swapchain(&mut self, window: &Window) {
        unsafe {
            self.device.device.device_wait_idle().unwrap();
            // Don't cleanup shadow resources here, they don't depend on window size
            self.cleanup_swapchain();

            let swapchain = VulkanSwapchain::new(
                &self.context.instance,
                &self.device.device,
                self.device.pdevice,
                &self.context.surface_loader,
                self.context.surface,
                window.inner_size().width,
                window.inner_size().height,
            );

            self.swapchain = swapchain;

            let (depth_image, depth_memory, depth_view) = Self::create_depth_resources(
                &self.device.device,
                self.device.pdevice,
                &self.context.instance,
                self.swapchain.extent,
            );
            self.depth_image = depth_image;
            self.depth_memory = depth_memory;
            self.depth_view = depth_view;

            let (hdr_image, hdr_memory, hdr_view) = Self::create_hdr_resources(
                &self.device.device,
                self.device.pdevice,
                &self.context.instance,
                self.swapchain.extent,
            );
            self.hdr_image = hdr_image;
            self.hdr_memory = hdr_memory;
            self.hdr_view = hdr_view;

            self.render_pass = Self::create_render_pass(&self.device.device, self.swapchain.format);
            self.framebuffers = Self::create_framebuffers(
                &self.device.device,
                self.render_pass,
                self.hdr_view,
                self.depth_view,
                self.swapchain.extent,
            );
        }
    }

    fn cleanup_swapchain(&mut self) {
        unsafe {
            for &framebuffer in &self.framebuffers {
                self.device.device.destroy_framebuffer(framebuffer, None);
            }
            self.device.device.destroy_render_pass(self.render_pass, None);
            for &view in &self.swapchain.views {
                self.device.device.destroy_image_view(view, None);
            }
            self.device.device.destroy_image_view(self.hdr_view, None);
            self.device.device.destroy_image(self.hdr_image, None);
            self.device.device.free_memory(self.hdr_memory, None);
            self.device.device.destroy_image_view(self.depth_view, None);
            self.device.device.destroy_image(self.depth_image, None);
            self.device.device.free_memory(self.depth_memory, None);
            self.swapchain.loader
                .destroy_swapchain(self.swapchain.handle, None);
        }
    }

    pub fn create_texture_from_image(&self, img: &image::DynamicImage) -> Texture {
        let (width, height) = (img.width(), img.height());
        let mip_levels = (((width.max(height) as f32).log2().floor()) as u32) + 1;

        let rgba = img.to_rgba8();
        let pixels = rgba.as_raw();
        let image_size = (pixels.len()) as u64;

        let staging_buffer = self.create_buffer(
            image_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        self.upload_to_buffer(&staging_buffer, pixels);

        let (image, memory) = self.create_image(
            width,
            height,
            mip_levels,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        self.transition_image_layout(
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            mip_levels,
        );
        self.copy_buffer_to_image(staging_buffer.handle, image, width, height);

        self.generate_mipmaps(image, vk::Format::R8G8B8A8_SRGB, width, height, mip_levels);

        let view = self.create_image_view(image, vk::Format::R8G8B8A8_SRGB, mip_levels);
        let sampler = self.create_texture_sampler(mip_levels);

        self.destroy_buffer(staging_buffer);

        Texture {
            image,
            memory,
            view,
            sampler,
            mip_levels,
        }
    }

    fn create_image(
        &self,
        width: u32,
        height: u32,
        mip_levels: u32,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> (vk::Image, vk::DeviceMemory) {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D { width, height, depth: 1 })
            .mip_levels(mip_levels)
            .array_layers(1)
            .format(format)
            .tiling(tiling)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .samples(vk::SampleCountFlags::TYPE_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let image = unsafe {
            self.device.device
                .create_image(&image_info, None)
                .expect("Failed to create image")
        };

        let mem_requirements = unsafe { self.device.device.get_image_memory_requirements(image) };
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(self.find_memory_type(mem_requirements.memory_type_bits, properties));

        let memory = unsafe {
            self.device.device
                .allocate_memory(&alloc_info, None)
                .expect("Failed to allocate image memory")
        };

        unsafe {
            self.device.device
                .bind_image_memory(image, memory, 0)
                .expect("Failed to bind image memory");
        }

        (image, memory)
    }

    fn create_image_view(&self, image: vk::Image, format: vk::Format, mip_levels: u32) -> vk::ImageView {
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            });

        unsafe {
            self.device.device
                .create_image_view(&view_info, None)
                .expect("Failed to create image view")
        }
    }

    fn create_texture_sampler(&self, mip_levels: u32) -> vk::Sampler {
        let properties = unsafe {
            self.context.instance.get_physical_device_properties(self.device.pdevice)
        };

        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(properties.limits.max_sampler_anisotropy)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .min_lod(0.0)
            .max_lod(mip_levels as f32)
            .mip_lod_bias(0.0);

        unsafe {
            self.device.device
                .create_sampler(&sampler_info, None)
                .expect("Failed to create texture sampler")
        }
    }

    fn generate_mipmaps(
        &self,
        image: vk::Image,
        _format: vk::Format,
        width: u32,
        height: u32,
        mip_levels: u32,
    ) {
        // Check if image format supports linear filtering
        let properties = unsafe {
            self.context.instance.get_physical_device_format_properties(self.device.pdevice, _format)
        };

        if !properties.optimal_tiling_features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR) {
            panic!("Texture image format does not support linear filtering!");
        }

        let command_buffer = self.begin_single_time_commands();

        let mut barrier = vk::ImageMemoryBarrier::default()
            .image(image)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_array_layer: 0,
                layer_count: 1,
                level_count: 1,
                ..Default::default()
            });

        let mut mip_width = width as i32;
        let mut mip_height = height as i32;

        for i in 1..mip_levels {
            barrier.subresource_range.base_mip_level = i - 1;
            barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

            unsafe {
                self.device.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }

            let blit = vk::ImageBlit::default()
                .src_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D { x: mip_width, y: mip_height, z: 1 },
                ])
                .src_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i - 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .dst_offsets([
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: if mip_width > 1 { mip_width / 2 } else { 1 },
                        y: if mip_height > 1 { mip_height / 2 } else { 1 },
                        z: 1,
                    },
                ])
                .dst_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i,
                    base_array_layer: 0,
                    layer_count: 1,
                });

            unsafe {
                self.device.device.cmd_blit_image(
                    command_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[blit],
                    vk::Filter::LINEAR,
                );
            }

            barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
            barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

            unsafe {
                self.device.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }

            if mip_width > 1 { mip_width /= 2; }
            if mip_height > 1 { mip_height /= 2; }
        }

        barrier.subresource_range.base_mip_level = mip_levels - 1;
        barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

        unsafe {
            self.device.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }

        self.end_single_time_commands(command_buffer);
    }

    pub fn create_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Buffer {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let handle = unsafe {
            self.device.device
                .create_buffer(&buffer_info, None)
                .expect("Failed to create buffer")
        };

        let mem_requirements = unsafe { self.device.device.get_buffer_memory_requirements(handle) };
        let mem_type_index = self.find_memory_type(mem_requirements.memory_type_bits, properties);

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index);

        let memory = unsafe {
            self.device.device
                .allocate_memory(&alloc_info, None)
                .expect("Failed to allocate buffer memory")
        };

        unsafe {
            self.device.device
                .bind_buffer_memory(handle, memory, 0)
                .expect("Failed to bind buffer memory");
        }

        Buffer { handle, memory, size }
    }

    fn transition_image_layout(
        &self,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        mip_levels: u32,
    ) {
        let command_buffer = self.begin_single_time_commands();

        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            });

        let (src_stage, dst_stage) = match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
            ),
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            _ => panic!("Unsupported layout transition"),
        };

        unsafe {
            self.device.device.cmd_pipeline_barrier(
                command_buffer,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }

        self.end_single_time_commands(command_buffer);
    }

    fn copy_buffer_to_image(&self, buffer: vk::Buffer, image: vk::Image, width: u32, height: u32) {
        let command_buffer = self.begin_single_time_commands();

        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            })
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D { width, height, depth: 1 });

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

    fn begin_single_time_commands(&self) -> vk::CommandBuffer {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(self.command_pool)
            .command_buffer_count(1);

        let command_buffer = unsafe {
            self.device.device
                .allocate_command_buffers(&alloc_info)
                .unwrap()[0]
        };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device.device
                .begin_command_buffer(command_buffer, &begin_info)
                .unwrap();
        }

        command_buffer
    }

    fn end_single_time_commands(&self, command_buffer: vk::CommandBuffer) {
        unsafe {
            self.device.device.end_command_buffer(command_buffer).unwrap();

            let command_buffers = [command_buffer];
            let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);

            let submits = [submit_info];
            self.device.device
                .queue_submit(self.device.graphics_queue, &submits, vk::Fence::null())
                .unwrap();
            self.device.device.queue_wait_idle(self.device.graphics_queue).unwrap();

            self.device.device
                .free_command_buffers(self.command_pool, &[command_buffer]);
        }
    }

    pub fn upload_to_buffer<T: Copy>(&self, buffer: &Buffer, data: &[T]) {
        unsafe {
            let ptr = self.device.device
                .map_memory(buffer.memory, 0, buffer.size, vk::MemoryMapFlags::empty())
                .expect("Failed to map memory");
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut T, data.len());
            self.device.device.unmap_memory(buffer.memory);
        }
    }

    pub fn destroy_buffer(&self, buffer: Buffer) {
        unsafe {
            self.device.device.destroy_buffer(buffer.handle, None);
            self.device.device.free_memory(buffer.memory, None);
        }
    }

    pub fn destroy_texture(&self, texture: Texture) {
        unsafe {
            self.device.device.destroy_sampler(texture.sampler, None);
            self.device.device.destroy_image_view(texture.view, None);
            self.device.device.destroy_image(texture.image, None);
            self.device.device.free_memory(texture.memory, None);
        }
    }

    fn find_memory_type(&self, type_filter: u32, properties: vk::MemoryPropertyFlags) -> u32 {
        let mem_properties = unsafe {
            self.context.instance
                .get_physical_device_memory_properties(self.device.pdevice)
        };

        for i in 0..mem_properties.memory_type_count {
            if (type_filter & (1 << i)) != 0
                && (mem_properties.memory_types[i as usize].property_flags & properties) == properties
            {
                return i;
            }
        }

        panic!("Failed to find suitable memory type")
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.device.device.device_wait_idle().ok();

            self.device.device.destroy_sampler(self.shadow_sampler, None);
            self.device.device.destroy_image_view(self.shadow_view, None);
            self.device.device.destroy_image(self.shadow_image, None);
            self.device.device.free_memory(self.shadow_memory, None);
            self.device.device.destroy_framebuffer(self.shadow_framebuffer, None);
            self.device.device.destroy_render_pass(self.shadow_render_pass, None);
            self.device.device.destroy_pipeline_layout(self.shadow_pipeline_layout, None);
            if let Some(p) = self.shadow_pipeline {
                self.device.device.destroy_pipeline(p, None);
            }

            for &semaphore in &self.image_available_semaphores {
                self.device.device.destroy_semaphore(semaphore, None);
            }
            for &semaphore in &self.render_finished_semaphores {
                self.device.device.destroy_semaphore(semaphore, None);
            }
            for &fence in &self.in_flight_fences {
                self.device.device.destroy_fence(fence, None);
            }

            self.device.device.destroy_command_pool(self.command_pool, None);

            self.cleanup_swapchain();

            if let Some(pipeline) = self.pipeline.take() {
                self.device.device.destroy_pipeline(pipeline.graphics_pipeline, None);
                self.device.device.destroy_pipeline_layout(pipeline.layout, None);
                self.device.device.destroy_descriptor_set_layout(pipeline.descriptor_set_layout, None);
            }

            self.device.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.device.destroy_descriptor_pool(self.post_process_descriptor_pool, None);

            for (img, (mem, view)) in self.bloom_images.drain(..).zip(self.bloom_memories.drain(..).zip(self.bloom_views.drain(..))) {
                self.device.device.destroy_image_view(view, None);
                self.device.device.destroy_image(img, None);
                self.device.device.free_memory(mem, None);
            }

            if let Some(mut egui) = self.egui_renderer.take() {
                egui.destroy(self);
            }

            let vertex_buffers = std::mem::take(&mut self.vertex_buffers);
            for vb in vertex_buffers {
                self.destroy_buffer(vb);
            }

            let instance_buffers = std::mem::take(&mut self.instance_buffers);
            for ib in instance_buffers {
                self.destroy_buffer(ib);
            }

            if let Some(ib) = self.index_buffer.take() {
                self.destroy_buffer(ib);
            }
        }
    }
}
