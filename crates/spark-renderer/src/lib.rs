pub mod pipeline;
pub mod vertex;
pub mod vulkan;
pub mod error;

use ash::vk;
use winit::window::Window;
use crate::pipeline::Pipeline;
use crate::vulkan::context::VulkanContext;
use crate::vulkan::device::VulkanDevice;
use crate::vulkan::swapchain::VulkanSwapchain;
use crate::vulkan::texture::Texture;
use crate::error::RendererError;

#[allow(dead_code)]
pub struct Renderer {
    context: VulkanContext,
    device: VulkanDevice,
    swapchain: VulkanSwapchain,
    pub render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,
    pipeline: Option<Pipeline>,
    vertex_buffer: Option<Buffer>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct Buffer {
    pub handle: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
}

impl Renderer {
    pub fn new(window: &Window) -> Result<Self, RendererError> {
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

        let render_pass = Self::create_render_pass(&device.device, swapchain.format);
        let framebuffers = Self::create_framebuffers(&device.device, render_pass, &swapchain.views, swapchain.extent);

        let command_pool = Self::create_command_pool(&device.device, device.graphics_family);
        let command_buffers = Self::create_command_buffers(&device.device, command_pool);

        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) =
            Self::create_sync_objects(&device.device);

        let descriptor_pool = Self::create_descriptor_pool(&device.device);

        Ok(Self {
            context,
            device,
            swapchain,
            render_pass,
            framebuffers,
            command_pool,
            command_buffers,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            current_frame: 0,
            pipeline: None,
            vertex_buffer: None,
            descriptor_pool,
            descriptor_sets: Vec::new(),
        })
    }

    pub fn set_pipeline(&mut self, pipeline: Pipeline) {
        self.pipeline = Some(pipeline);
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

            let descriptor_writes = [vk::WriteDescriptorSet::default()
                .dst_set(sets[0])
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&image_info)];

            unsafe {
                self.device.device.update_descriptor_sets(&descriptor_writes, &[]);
            }

            self.descriptor_sets = sets;
        }
    }

    pub fn set_vertex_buffer(&mut self, buffer: Buffer) {
        self.vertex_buffer = Some(buffer);
    }

    pub fn draw_frame(
        &mut self,
        renderables: &[(spark_math::Mat4, u32, Option<String>, Option<u32>)],
        view_proj: spark_math::Mat4,
        window: &Window,
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

            self.record_command_buffer(image_index, renderables, view_proj);

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
        &self,
        image_index: u32,
        renderables: &[(spark_math::Mat4, u32, Option<String>, Option<u32>)],
        view_proj: spark_math::Mat4,
    ) {
        let command_buffer = self.command_buffers[self.current_frame];

        let begin_info = vk::CommandBufferBeginInfo::default();

        unsafe {
            self.device.device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("Failed to begin recording command buffer");

            let clear_values = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.1, 0.1, 0.1, 1.0],
                },
            }];

            let render_pass_info = vk::RenderPassBeginInfo::default()
                .render_pass(self.render_pass)
                .framebuffer(self.framebuffers[image_index as usize])
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

                for (model, vertex_count, _texture_id, vertex_buffer_id) in renderables {
                    if let Some(_id) = vertex_buffer_id {
                        if let Some(vb) = &self.vertex_buffer {
                             self.device.device.cmd_bind_vertex_buffers(command_buffer, 0, &[vb.handle], &[0]);
                        }
                    }

                    let mut constants = [spark_math::Mat4::IDENTITY; 2];
                    constants[0] = *model;
                    constants[1] = view_proj;

                    let bytes = std::slice::from_raw_parts(
                        constants.as_ptr() as *const u8,
                        std::mem::size_of::<spark_math::Mat4>() * 2,
                    );
                    self.device.device.cmd_push_constants(
                        command_buffer,
                        pipeline.layout,
                        vk::ShaderStageFlags::VERTEX,
                        0,
                        bytes,
                    );
                    self.device.device.cmd_draw(command_buffer, *vertex_count, 1, 0, 0);
                }
            }

            self.device.device.cmd_end_render_pass(command_buffer);

            self.device.device
                .end_command_buffer(command_buffer)
                .expect("Failed to record command buffer");
        }
    }

    fn create_render_pass(device: &ash::Device, format: vk::Format) -> vk::RenderPass {
        let color_attachment = vk::AttachmentDescription::default()
            .format(format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_attachment_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&color_attachment_ref));

        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(std::slice::from_ref(&color_attachment))
            .subpasses(std::slice::from_ref(&subpass))
            .dependencies(std::slice::from_ref(&dependency));

        unsafe {
            device
                .create_render_pass(&render_pass_info, None)
                .expect("Failed to create render pass")
        }
    }

    fn create_framebuffers(
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
            .descriptor_count(10)]; // Arbitrary large enough for now

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(10);

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

    pub fn recreate_swapchain(&mut self, window: &Window) {
        unsafe {
            self.device.device.device_wait_idle().unwrap();
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
            self.render_pass = Self::create_render_pass(&self.device.device, self.swapchain.format);
            self.framebuffers = Self::create_framebuffers(
                &self.device.device,
                self.render_pass,
                &self.swapchain.views,
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
            self.swapchain.loader
                .destroy_swapchain(self.swapchain.handle, None);
        }
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
                level_count: 1,
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

            if let Some(vb) = self.vertex_buffer.take() {
                self.destroy_buffer(vb);
            }
        }
    }
}
