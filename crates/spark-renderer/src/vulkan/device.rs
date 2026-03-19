use crate::error::RendererError;
use crate::resource::Buffer;
use ash::{khr::surface::Instance as Surface, vk, Device, Instance};

pub struct VulkanDevice {
    pub pdevice: vk::PhysicalDevice,
    pub device: Device,
    pub graphics_queue: vk::Queue,
    pub graphics_family: u32,
    pub msaa_samples: vk::SampleCountFlags,
    pub depth_format: vk::Format,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub command_pool: vk::CommandPool,
    pub thread_command_pools: Vec<vk::CommandPool>,
}

impl VulkanDevice {
    pub fn new(
        instance: &Instance,
        surface_loader: &Surface,
        surface: vk::SurfaceKHR,
    ) -> Result<Self, RendererError> {
        let pdevices = unsafe { instance.enumerate_physical_devices()? };

        let (pdevice, graphics_family) = pdevices
            .iter()
            .map(|&p| (p, Self::score_device(instance, p)))
            .filter(|&(_, score)| score > 0)
            .max_by_key(|&(_, score)| score)
            .and_then(|(p, _)| {
                Self::find_queue_families(instance, surface_loader, surface, p)
                    .map(|family| (p, family))
            })
            .ok_or(RendererError::NoSuitableDevice)?;

        let queue_priorities = [1.0];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(graphics_family)
            .queue_priorities(&queue_priorities);

        let device_extension_names_raw = [ash::khr::swapchain::NAME.as_ptr()];

        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true);
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
            .descriptor_indexing(true)
            .shader_sampled_image_array_non_uniform_indexing(true)
            .descriptor_binding_partially_bound(true)
            .descriptor_binding_variable_descriptor_count(true)
            .runtime_descriptor_array(true)
            .draw_indirect_count(true)
            .buffer_device_address(true);

        let mut features11 = vk::PhysicalDeviceVulkan11Features::default()
            .shader_draw_parameters(true);

        let device_create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extension_names_raw)
            .push_next(&mut features11)
            .push_next(&mut features12)
            .push_next(&mut features13);

        let device = unsafe { instance.create_device(pdevice, &device_create_info, None)? };

        let graphics_queue = unsafe { device.get_device_queue(graphics_family, 0) };

        let msaa_samples = Self::get_max_usable_sample_count(instance, pdevice);
        let depth_format = Self::find_depth_format(instance, pdevice);
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(pdevice) };

        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(graphics_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?
        };

        let mut thread_command_pools = Vec::new();
        let thread_count = num_cpus::get();
        for _ in 0..thread_count {
            let pool = unsafe {
                device.create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(graphics_family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )?
            };
            thread_command_pools.push(pool);
        }

        Ok(Self {
            pdevice,
            device,
            graphics_queue,
            graphics_family,
            msaa_samples,
            depth_format,
            memory_properties,
            command_pool,
            thread_command_pools,
        })
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

        let handle = unsafe { self.device.create_buffer(&buffer_info, None).unwrap() };
        let bda_info = vk::BufferDeviceAddressInfo::default().buffer(handle);
        let address = if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
             unsafe { self.device.get_buffer_device_address(&bda_info) }
        } else {
             0
        };
        let mem_reqs = unsafe { self.device.get_buffer_memory_requirements(handle) };

        let mut type_idx = 0;
        for i in 0..self.memory_properties.memory_type_count {
            if (mem_reqs.memory_type_bits & (1 << i)) != 0
                && (self.memory_properties.memory_types[i as usize].property_flags & properties)
                    == properties
            {
                type_idx = i;
                break;
            }
        }

        let mut alloc_flags = vk::MemoryAllocateFlagsInfo::default();
        if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
            alloc_flags.flags = vk::MemoryAllocateFlags::DEVICE_ADDRESS;
        }

        let memory = unsafe {
            self.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(mem_reqs.size)
                    .memory_type_index(type_idx)
                    .push_next(&mut alloc_flags),
                None
            ).unwrap()
        };

        unsafe {
            self.device.bind_buffer_memory(handle, memory, 0).unwrap();
        }

        let ptr = if properties.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            unsafe {
                self.device.map_memory(memory, 0, mem_reqs.size, vk::MemoryMapFlags::empty()).unwrap()
            }
        } else {
            std::ptr::null_mut()
        };

        Buffer {
            handle,
            memory,
            size,
            ptr,
            address,
        }
    }

    pub fn create_image(
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
        let reqs = unsafe { self.device.get_image_memory_requirements(i) };

        let mut type_idx = 0;
        for j in 0..self.memory_properties.memory_type_count {
            if (reqs.memory_type_bits & (1 << j)) != 0
                && (self.memory_properties.memory_types[j as usize].property_flags & p) == p
            {
                type_idx = j;
                break;
            }
        }

        let m = unsafe {
            self.device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::default()
                        .allocation_size(reqs.size)
                        .memory_type_index(type_idx),
                    None,
                )
                .unwrap()
        };
        unsafe {
            self.device.bind_image_memory(i, m, 0).unwrap();
        }
        (i, m)
    }

    pub fn create_image_view(&self, image: vk::Image, format: vk::Format, mip_levels: u32) -> vk::ImageView {
        let aspect_mask = if format == vk::Format::D32_SFLOAT
            || format == vk::Format::D32_SFLOAT_S8_UINT
            || format == vk::Format::D24_UNORM_S8_UINT
            || format == vk::Format::D16_UNORM
        {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            });

        unsafe { self.device.create_image_view(&view_info, None).unwrap() }
    }

    pub fn transition_image_layout(
        &self,
        image: vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        mip_levels: u32,
    ) {
        let cb = self.begin_single_time_commands();

        let mut barrier = vk::ImageMemoryBarrier::default()
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
            _ => (
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
            ),
        };

        barrier.src_access_mask = match old_layout {
            vk::ImageLayout::UNDEFINED => vk::AccessFlags::empty(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => vk::AccessFlags::TRANSFER_WRITE,
            _ => vk::AccessFlags::empty(),
        };

        barrier.dst_access_mask = match new_layout {
            vk::ImageLayout::TRANSFER_DST_OPTIMAL => vk::AccessFlags::TRANSFER_WRITE,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => vk::AccessFlags::SHADER_READ,
            _ => vk::AccessFlags::empty(),
        };

        unsafe {
            self.device.cmd_pipeline_barrier(
                cb,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }

        self.end_single_time_commands(cb);
    }

    pub fn upload_to_buffer<T: Copy>(&self, b: &Buffer, data: &[T]) {
        if b.ptr.is_null() {
            panic!("Buffer is not host-visible for upload");
        }
        unsafe {
            let mut align = ash::util::Align::new(b.ptr, std::mem::align_of::<T>() as u64, b.size);
            align.copy_from_slice(data);
        }
    }

    pub fn create_command_buffer(&self, pool: vk::CommandPool, level: vk::CommandBufferLevel) -> vk::CommandBuffer {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .level(level)
            .command_pool(pool)
            .command_buffer_count(1);

        unsafe { self.device.allocate_command_buffers(&alloc_info).unwrap()[0] }
    }

    pub fn begin_single_time_commands(&self) -> vk::CommandBuffer {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(self.command_pool)
            .command_buffer_count(1);

        let cb = unsafe { self.device.allocate_command_buffers(&alloc_info).unwrap()[0] };

        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            self.device.begin_command_buffer(cb, &begin_info).unwrap();
        }

        cb
    }

    pub fn end_single_time_commands(&self, cb: vk::CommandBuffer) {
        unsafe {
            self.device.end_command_buffer(cb).unwrap();

            let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cb));

            self.device
                .queue_submit(self.graphics_queue, &[submit_info], vk::Fence::null())
                .expect("Failed to submit single-time commands");
            self.device.queue_wait_idle(self.graphics_queue).unwrap();

            self.device.free_command_buffers(self.command_pool, &[cb]);
        }
    }

    pub fn destroy_buffer(&self, buffer: Buffer) {
        unsafe {
            self.device.destroy_buffer(buffer.handle, None);
            self.device.free_memory(buffer.memory, None);
        }
    }

    fn get_max_usable_sample_count(
        instance: &Instance,
        pdevice: vk::PhysicalDevice,
    ) -> vk::SampleCountFlags {
        let props = unsafe { instance.get_physical_device_properties(pdevice) };
        let counts = props.limits.framebuffer_color_sample_counts
            & props.limits.framebuffer_depth_sample_counts;

        if counts.contains(vk::SampleCountFlags::TYPE_64) {
            return vk::SampleCountFlags::TYPE_64;
        }
        if counts.contains(vk::SampleCountFlags::TYPE_32) {
            return vk::SampleCountFlags::TYPE_32;
        }
        if counts.contains(vk::SampleCountFlags::TYPE_16) {
            return vk::SampleCountFlags::TYPE_16;
        }
        if counts.contains(vk::SampleCountFlags::TYPE_8) {
            return vk::SampleCountFlags::TYPE_8;
        }
        if counts.contains(vk::SampleCountFlags::TYPE_4) {
            return vk::SampleCountFlags::TYPE_4;
        }
        if counts.contains(vk::SampleCountFlags::TYPE_2) {
            return vk::SampleCountFlags::TYPE_2;
        }

        vk::SampleCountFlags::TYPE_1
    }

    fn find_depth_format(instance: &Instance, pdevice: vk::PhysicalDevice) -> vk::Format {
        let candidates = [
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ];

        for &format in &candidates {
            let props = unsafe { instance.get_physical_device_format_properties(pdevice, format) };
            if props
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
            {
                return format;
            }
        }

        vk::Format::D16_UNORM
    }

    fn score_device(instance: &Instance, pdevice: vk::PhysicalDevice) -> u32 {
        let props = unsafe { instance.get_physical_device_properties(pdevice) };
        let mut score = 0;

        match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => score += 1000,
            vk::PhysicalDeviceType::INTEGRATED_GPU => score += 500,
            _ => score += 100,
        }

        score += props.limits.max_image_dimension2_d;
        score
    }

    fn find_queue_families(
        instance: &Instance,
        surface_loader: &Surface,
        surface: vk::SurfaceKHR,
        pdevice: vk::PhysicalDevice,
    ) -> Option<u32> {
        let props = unsafe { instance.get_physical_device_queue_family_properties(pdevice) };
        for (index, prop) in props.iter().enumerate() {
            let index = index as u32;
            let graphics = prop.queue_flags.contains(vk::QueueFlags::GRAPHICS);
            let present = unsafe {
                surface_loader
                    .get_physical_device_surface_support(pdevice, index, surface)
                    .unwrap_or(false)
            };

            if graphics && present {
                return Some(index);
            }
        }
        None
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            for pool in self.thread_command_pools.drain(..) {
                self.device.destroy_command_pool(pool, None);
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
        }
    }
}
