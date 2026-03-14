pub mod pipeline;
pub mod vertex;

use ash::{vk, Entry, Instance, Device, khr::surface::Instance as Surface, khr::swapchain::Device as Swapchain};
use std::ffi::CString;
use winit::window::Window;
use crate::pipeline::Pipeline;

#[allow(dead_code)]
pub struct Renderer {
    entry: Entry,
    instance: Instance,
    pdevice: vk::PhysicalDevice,
    device: Device,
    surface_loader: Surface,
    surface: vk::SurfaceKHR,
    swapchain_loader: Swapchain,
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    pub render_pass: vk::RenderPass,
    framebuffers: Vec<vk::Framebuffer>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,
    pipeline: Option<Pipeline>,
}

const MAX_FRAMES_IN_FLIGHT: usize = 2;

pub struct Buffer {
    pub handle: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
}

impl Renderer {
    pub fn new(window: &Window) -> Self {
        use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

        let entry = unsafe { Entry::load().expect("Failed to load Vulkan") };

        let app_name = CString::new("Spark Engine").unwrap();
        let engine_name = CString::new("Spark").unwrap();

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        let display_handle = window.display_handle().unwrap().as_raw();
        let window_handle = window.window_handle().unwrap().as_raw();

        let extensions = ash_window::enumerate_required_extensions(display_handle)
            .expect("Failed to enumerate required extensions");

        let instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(extensions);

        let instance = unsafe {
            entry
                .create_instance(&instance_create_info, None)
                .expect("Failed to create Vulkan instance")
        };

        let surface = unsafe {
            ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)
                .expect("Failed to create surface")
        };
        let surface_loader = Surface::new(&entry, &instance);

        let pdevices = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("Failed to enumerate physical devices")
        };

        let pdevice = pdevices[0];

        let queue_priorities = [1.0];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(0)
            .queue_priorities(&queue_priorities);

        let device_extension_names_raw = [ash::khr::swapchain::NAME.as_ptr()];

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extension_names_raw);

        let device = unsafe {
            instance
                .create_device(pdevice, &device_create_info, None)
                .expect("Failed to create logical device")
        };

        let swapchain_loader = Swapchain::new(&instance, &device);

        let surface_format = unsafe {
            surface_loader
                .get_physical_device_surface_formats(pdevice, surface)
                .unwrap()[0]
        };

        let surface_caps = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(pdevice, surface)
                .unwrap()
        };

        let mut extent = surface_caps.current_extent;
        if extent.width == u32::MAX {
            extent.width = window.inner_size().width;
            extent.height = window.inner_size().height;
        }

        let mut image_count = surface_caps.min_image_count + 1;
        if surface_caps.max_image_count > 0 && image_count > surface_caps.max_image_count {
            image_count = surface_caps.max_image_count;
        }

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_color_space(surface_format.color_space)
            .image_format(surface_format.format)
            .image_extent(extent)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(surface_caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .image_array_layers(1);

        let swapchain = unsafe {
            swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .expect("Failed to create swapchain")
        };

        let images = unsafe {
            swapchain_loader
                .get_swapchain_images(swapchain)
                .expect("Failed to get swapchain images")
        };

        let views: Vec<vk::ImageView> = images
            .iter()
            .map(|&image| {
                let create_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                unsafe {
                    device
                        .create_image_view(&create_info, None)
                        .expect("Failed to create image view")
                }
            })
            .collect();

        let render_pass = Self::create_render_pass(&device, surface_format.format);
        let framebuffers = Self::create_framebuffers(&device, render_pass, &views, extent);

        let command_pool = Self::create_command_pool(&device);
        let command_buffers = Self::create_command_buffers(&device, command_pool);

        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) =
            Self::create_sync_objects(&device);

        Self {
            entry,
            instance,
            pdevice,
            device,
            surface_loader,
            surface,
            swapchain_loader,
            swapchain,
            swapchain_images: images,
            swapchain_image_views: views,
            swapchain_format: surface_format.format,
            swapchain_extent: extent,
            render_pass,
            framebuffers,
            command_pool,
            command_buffers,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            current_frame: 0,
            pipeline: None,
        }
    }

    fn create_swapchain_internal(
        instance: &Instance,
        pdevice: vk::PhysicalDevice,
        device: &Device,
        surface_loader: &Surface,
        surface: vk::SurfaceKHR,
        swapchain_loader: &Swapchain,
        width: u32,
        height: u32,
    ) -> (
        vk::SwapchainKHR,
        Vec<vk::Image>,
        Vec<vk::ImageView>,
        vk::Format,
        vk::Extent2D,
    ) {
        let surface_format = unsafe {
            surface_loader
                .get_physical_device_surface_formats(pdevice, surface)
                .unwrap()[0]
        };

        let surface_caps = unsafe {
            surface_loader
                .get_physical_device_surface_capabilities(pdevice, surface)
                .unwrap()
        };

        let mut extent = surface_caps.current_extent;
        if extent.width == u32::MAX {
            extent.width = width;
            extent.height = height;
        }

        let mut image_count = surface_caps.min_image_count + 1;
        if surface_caps.max_image_count > 0 && image_count > surface_caps.max_image_count {
            image_count = surface_caps.max_image_count;
        }

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface)
            .min_image_count(image_count)
            .image_color_space(surface_format.color_space)
            .image_format(surface_format.format)
            .image_extent(extent)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(surface_caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .image_array_layers(1);

        let swapchain = unsafe {
            swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .expect("Failed to create swapchain")
        };

        let images = unsafe {
            swapchain_loader
                .get_swapchain_images(swapchain)
                .expect("Failed to get swapchain images")
        };

        let views: Vec<vk::ImageView> = images
            .iter()
            .map(|&image| {
                let create_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(surface_format.format)
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                unsafe {
                    device
                        .create_image_view(&create_info, None)
                        .expect("Failed to create image view")
                }
            })
            .collect();

        (swapchain, images, views, surface_format.format, extent)
    }

    pub fn set_pipeline(&mut self, pipeline: Pipeline) {
        self.pipeline = Some(pipeline);
    }

    pub fn draw_frame(
        &mut self,
        renderables: &[(spark_math::Mat4, u32)],
        view_proj: spark_math::Mat4,
        window: &Window,
    ) {
        unsafe {
            self.device
                .wait_for_fences(
                    &[self.in_flight_fences[self.current_frame]],
                    true,
                    u64::MAX,
                )
                .expect("Failed to wait for fence");

            let result = self.swapchain_loader.acquire_next_image(
                self.swapchain,
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
                .reset_fences(&[self.in_flight_fences[self.current_frame]])
                .expect("Failed to reset fence");

            self.device
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

            let graphics_queue = self.device.get_device_queue(0, 0);

            self.device
                .queue_submit(
                    graphics_queue,
                    &[submit_info],
                    self.in_flight_fences[self.current_frame],
                )
                .expect("Failed to submit draw command");

            let swapchains = [self.swapchain];
            let image_indices = [image_index];
            let present_info = vk::PresentInfoKHR::default()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            let result = self.swapchain_loader.queue_present(graphics_queue, &present_info);
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
        renderables: &[(spark_math::Mat4, u32)],
        view_proj: spark_math::Mat4,
    ) {
        let command_buffer = self.command_buffers[self.current_frame];

        let begin_info = vk::CommandBufferBeginInfo::default();

        unsafe {
            self.device
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
                    extent: self.swapchain_extent,
                })
                .clear_values(&clear_values);

            self.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );

            if let Some(pipeline) = &self.pipeline {
                self.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.graphics_pipeline,
                );

                for (model, vertex_count) in renderables {
                    let mut constants = [spark_math::Mat4::IDENTITY; 2];
                    constants[0] = *model;
                    constants[1] = view_proj;

                    let bytes = std::slice::from_raw_parts(
                        constants.as_ptr() as *const u8,
                        std::mem::size_of::<spark_math::Mat4>() * 2,
                    );
                    self.device.cmd_push_constants(
                        command_buffer,
                        pipeline.layout,
                        vk::ShaderStageFlags::VERTEX,
                        0,
                        bytes,
                    );
                    self.device.cmd_draw(command_buffer, *vertex_count, 1, 0, 0);
                }
            }

            self.device.cmd_end_render_pass(command_buffer);

            self.device
                .end_command_buffer(command_buffer)
                .expect("Failed to record command buffer");
        }
    }

    fn create_render_pass(device: &Device, format: vk::Format) -> vk::RenderPass {
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
        device: &Device,
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

    fn create_command_pool(device: &Device) -> vk::CommandPool {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(0)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        unsafe {
            device
                .create_command_pool(&pool_info, None)
                .expect("Failed to create command pool")
        }
    }

    fn create_command_buffers(device: &Device, pool: vk::CommandPool) -> Vec<vk::CommandBuffer> {
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

    fn create_sync_objects(
        device: &Device,
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
        self.swapchain_extent
    }

    pub fn get_device(&self) -> &Device {
        &self.device
    }

    pub fn recreate_swapchain(&mut self, window: &Window) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.cleanup_swapchain();

            let (swapchain, images, views, format, extent) = Self::create_swapchain_internal(
                &self.instance,
                self.pdevice,
                &self.device,
                &self.surface_loader,
                self.surface,
                &self.swapchain_loader,
                window.inner_size().width,
                window.inner_size().height,
            );

            self.swapchain = swapchain;
            self.swapchain_images = images;
            self.swapchain_image_views = views;
            self.swapchain_format = format;
            self.swapchain_extent = extent;

            self.render_pass = Self::create_render_pass(&self.device, self.swapchain_format);
            self.framebuffers = Self::create_framebuffers(
                &self.device,
                self.render_pass,
                &self.swapchain_image_views,
                self.swapchain_extent,
            );
        }
    }

    fn cleanup_swapchain(&mut self) {
        unsafe {
            for &framebuffer in &self.framebuffers {
                self.device.destroy_framebuffer(framebuffer, None);
            }
            self.device.destroy_render_pass(self.render_pass, None);
            for &view in &self.swapchain_image_views {
                self.device.destroy_image_view(view, None);
            }
            self.swapchain_loader
                .destroy_swapchain(self.swapchain, None);
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
            self.device
                .create_buffer(&buffer_info, None)
                .expect("Failed to create buffer")
        };

        let mem_requirements = unsafe { self.device.get_buffer_memory_requirements(handle) };
        let mem_type_index = self.find_memory_type(mem_requirements.memory_type_bits, properties);

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(mem_type_index);

        let memory = unsafe {
            self.device
                .allocate_memory(&alloc_info, None)
                .expect("Failed to allocate buffer memory")
        };

        unsafe {
            self.device
                .bind_buffer_memory(handle, memory, 0)
                .expect("Failed to bind buffer memory");
        }

        Buffer { handle, memory, size }
    }

    pub fn upload_to_buffer<T: Copy>(&self, buffer: &Buffer, data: &[T]) {
        unsafe {
            let ptr = self.device
                .map_memory(buffer.memory, 0, buffer.size, vk::MemoryMapFlags::empty())
                .expect("Failed to map memory");
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut T, data.len());
            self.device.unmap_memory(buffer.memory);
        }
    }

    pub fn destroy_buffer(&self, buffer: Buffer) {
        unsafe {
            self.device.destroy_buffer(buffer.handle, None);
            self.device.free_memory(buffer.memory, None);
        }
    }

    fn find_memory_type(&self, type_filter: u32, properties: vk::MemoryPropertyFlags) -> u32 {
        let mem_properties = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.pdevice)
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
            self.device.device_wait_idle().ok();

            for &semaphore in &self.image_available_semaphores {
                self.device.destroy_semaphore(semaphore, None);
            }
            for &semaphore in &self.render_finished_semaphores {
                self.device.destroy_semaphore(semaphore, None);
            }
            for &fence in &self.in_flight_fences {
                self.device.destroy_fence(fence, None);
            }

            self.device.destroy_command_pool(self.command_pool, None);

            for &framebuffer in &self.framebuffers {
                self.device.destroy_framebuffer(framebuffer, None);
            }

            if let Some(pipeline) = self.pipeline.take() {
                self.device.destroy_pipeline(pipeline.graphics_pipeline, None);
                self.device.destroy_pipeline_layout(pipeline.layout, None);
            }

            self.device.destroy_render_pass(self.render_pass, None);

            for &view in &self.swapchain_image_views {
                self.device.destroy_image_view(view, None);
            }

            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}
