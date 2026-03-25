use crate::error::RendererError;
use crate::resource::Buffer;
use ash::{khr::surface::Instance as Surface, vk, Device, Instance};
use gpu_allocator::vulkan::*;
use std::sync::{Arc, Mutex};

pub struct ImageCreateParams {
    pub width: u32,
    pub height: u32,
    pub mip_levels: u32,
    pub format: vk::Format,
    pub tiling: vk::ImageTiling,
    pub usage: vk::ImageUsageFlags,
    pub properties: vk::MemoryPropertyFlags,
    pub samples: vk::SampleCountFlags,
}

/// Represents the Vulkan logical device and its associated command pools and memory allocator.
pub struct VulkanDevice {
    pub pdevice: vk::PhysicalDevice,
    pub device: Device,
    pub graphics_queue: vk::Queue,
    pub graphics_family: u32,
    pub compute_queue: vk::Queue,
    pub compute_family: u32,
    pub rt_supported: bool,
    pub msaa_samples: vk::SampleCountFlags,
    pub depth_format: vk::Format,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub command_pool: vk::CommandPool,
    pub compute_command_pool: vk::CommandPool,
    pub thread_command_pools: Vec<[vk::CommandPool; crate::MAX_FRAMES_IN_FLIGHT]>,
    pub allocator: Arc<Mutex<Allocator>>,
    pub as_loader: Option<ash::khr::acceleration_structure::Device>,
    pub rt_loader: Option<ash::khr::ray_tracing_pipeline::Device>,
}

impl VulkanDevice {
    pub fn new(
        instance: &Instance,
        surface_loader: &Surface,
        surface: vk::SurfaceKHR,
    ) -> Result<Self, RendererError> {
        let pdevices = unsafe { instance.enumerate_physical_devices()? };

        let (pdevice, graphics_family, compute_family) = pdevices
            .iter()
            .map(|&p| (p, Self::score_device(instance, p)))
            .filter(|&(_, score)| score > 0)
            .max_by_key(|&(_, score)| score)
            .and_then(|(p, _)| {
                Self::find_queue_families(instance, surface_loader, surface, p)
                    .map(|(g, c)| (p, g, c))
            })
            .ok_or(RendererError::NoSuitableDevice)?;

        let queue_priorities = [1.0];
        let mut queue_infos = vec![
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(graphics_family)
                .queue_priorities(&queue_priorities)
        ];

        if graphics_family != compute_family {
            queue_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(compute_family)
                    .queue_priorities(&queue_priorities)
            );
        }

        let available_extensions = unsafe { instance.enumerate_device_extension_properties(pdevice)? };
        let mut rt_supported = true;
        for ext in &[
            ash::khr::acceleration_structure::NAME,
            ash::khr::ray_tracing_pipeline::NAME,
            ash::khr::deferred_host_operations::NAME,
        ] {
            if !available_extensions.iter().any(|e| {
                let name = unsafe { std::ffi::CStr::from_ptr(e.extension_name.as_ptr()) };
                name == *ext
            }) {
                rt_supported = false;
                break;
            }
        }

        let mut device_extension_names_raw = vec![ash::khr::swapchain::NAME.as_ptr()];
        if rt_supported {
            device_extension_names_raw.push(ash::khr::acceleration_structure::NAME.as_ptr());
            device_extension_names_raw.push(ash::khr::ray_tracing_pipeline::NAME.as_ptr());
            device_extension_names_raw.push(ash::khr::deferred_host_operations::NAME.as_ptr());
        }

        let mut features_as = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default()
            .acceleration_structure(rt_supported);
        let mut features_rt = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default()
            .ray_tracing_pipeline(rt_supported);

        let mut features13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
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
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&device_extension_names_raw);

        let mut device_create_info = device_create_info
            .push_next(&mut features11)
            .push_next(&mut features12)
            .push_next(&mut features13);

        if rt_supported {
            device_create_info = device_create_info
                .push_next(&mut features_as)
                .push_next(&mut features_rt);
        }

        let device = unsafe { instance.create_device(pdevice, &device_create_info, None)? };

        let graphics_queue = unsafe { device.get_device_queue(graphics_family, 0) };
        let compute_queue = unsafe { device.get_device_queue(compute_family, 0) };

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

        let compute_command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(compute_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?
        };

        let mut thread_command_pools = Vec::new();
        let thread_count = num_cpus::get();
        for _ in 0..thread_count {
            let mut pools = [vk::CommandPool::null(); crate::MAX_FRAMES_IN_FLIGHT];
            for p in pools.iter_mut() {
                *p = unsafe {
                    device.create_command_pool(
                        &vk::CommandPoolCreateInfo::default()
                            .queue_family_index(graphics_family)
                            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                        None,
                    )?
                };
            }
            thread_command_pools.push(pools);
        }

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device: pdevice,
            debug_settings: Default::default(),
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        }).map_err(|_| RendererError::NoSuitableDevice)?;

        let as_loader = if rt_supported {
            Some(ash::khr::acceleration_structure::Device::new(instance, &device))
        } else {
            None
        };

        let rt_loader = if rt_supported {
            Some(ash::khr::ray_tracing_pipeline::Device::new(instance, &device))
        } else {
            None
        };

        Ok(Self {
            pdevice,
            device,
            graphics_queue,
            graphics_family,
            compute_queue,
            compute_family,
            rt_supported,
            msaa_samples,
            depth_format,
            memory_properties,
            command_pool,
            compute_command_pool,
            thread_command_pools,
            allocator: Arc::new(Mutex::new(allocator)),
            as_loader,
            rt_loader,
        })
    }

    pub fn create_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> Result<Buffer, RendererError> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let handle = unsafe { self.device.create_buffer(&buffer_info, None)? };
        let mem_reqs = unsafe { self.device.get_buffer_memory_requirements(handle) };

        let location = if properties.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            gpu_allocator::MemoryLocation::CpuToGpu
        } else {
            gpu_allocator::MemoryLocation::GpuOnly
        };

        let allocation = self.allocator.lock().unwrap().allocate(&AllocationCreateDesc {
            name: "Buffer",
            requirements: mem_reqs,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }).map_err(|_| RendererError::NoSuitableDevice)?;

        unsafe {
            self.device.bind_buffer_memory(handle, allocation.memory(), allocation.offset())?;
        }

        let bda_info = vk::BufferDeviceAddressInfo::default().buffer(handle);
        let address = if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
             unsafe { self.device.get_buffer_device_address(&bda_info) }
        } else {
             0
        };

        let ptr = allocation.mapped_ptr().map(|p| p.as_ptr()).unwrap_or(std::ptr::null_mut());

        Ok(Buffer {
            handle,
            allocation: Arc::new(Mutex::new(Some(allocation))),
            size,
            ptr,
            address,
            version: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        })
    }

    pub fn create_image(
        &self,
        params: &ImageCreateParams,
    ) -> Result<(vk::Image, gpu_allocator::vulkan::Allocation), RendererError> {
        let i = unsafe {
            self.device
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .extent(vk::Extent3D {
                            width: params.width,
                            height: params.height,
                            depth: 1,
                        })
                        .mip_levels(params.mip_levels)
                        .array_layers(1)
                        .format(params.format)
                        .tiling(params.tiling)
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .usage(params.usage)
                        .samples(params.samples)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )?
        };
        let reqs = unsafe { self.device.get_image_memory_requirements(i) };

        let location = if params.properties.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            gpu_allocator::MemoryLocation::CpuToGpu
        } else {
            gpu_allocator::MemoryLocation::GpuOnly
        };

        let allocation = self.allocator.lock().unwrap().allocate(&AllocationCreateDesc {
            name: "Image",
            requirements: reqs,
            location,
            linear: false,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }).map_err(|_| RendererError::NoSuitableDevice)?;

        unsafe {
            self.device.bind_image_memory(i, allocation.memory(), allocation.offset())?;
        }
        Ok((i, allocation))
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
        let data_size = std::mem::size_of_val(data) as u64;
        if data_size > b.size {
            panic!("Data size exceeds buffer size");
        }
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), b.ptr as *mut T, data.len());
        }
        b.version.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn create_command_buffer(&self, pool: vk::CommandPool, level: vk::CommandBufferLevel) -> vk::CommandBuffer {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .level(level)
            .command_pool(pool)
            .command_buffer_count(1);

        unsafe { self.device.allocate_command_buffers(&alloc_info).unwrap()[0] }
    }

    pub fn submit_commands(&self, queue: vk::Queue, cb: vk::CommandBuffer, wait_semaphores: &[vk::Semaphore], signal_semaphores: &[vk::Semaphore], fence: vk::Fence) {
        let cbs = [cb];
        let submit_info = vk::SubmitInfo::default()
            .command_buffers(&cbs)
            .wait_semaphores(wait_semaphores)
            .signal_semaphores(signal_semaphores);

        unsafe {
            self.device.queue_submit(queue, &[submit_info], fence).expect("Failed to submit commands");
        }
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
            if let Some(alloc) = buffer.allocation.lock().unwrap().take() {
                self.allocator.lock().unwrap().free(alloc).unwrap();
            }
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
    ) -> Option<(u32, u32)> {
        let props = unsafe { instance.get_physical_device_queue_family_properties(pdevice) };
        let mut graphics = None;
        let mut compute = None;

        for (index, prop) in props.iter().enumerate() {
            let index = index as u32;
            let has_graphics = prop.queue_flags.contains(vk::QueueFlags::GRAPHICS);
            let has_compute = prop.queue_flags.contains(vk::QueueFlags::COMPUTE);
            let has_present = unsafe {
                surface_loader
                    .get_physical_device_surface_support(pdevice, index, surface)
                    .unwrap_or(false)
            };

            if has_graphics && has_present && graphics.is_none() {
                graphics = Some(index);
            }

            // Prefer a dedicated compute queue if available
            if has_compute && (!has_graphics) {
                compute = Some(index);
            }
        }

        // Fallback to shared graphics/compute queue
        if compute.is_none() {
            compute = graphics;
        }

        if let (Some(g), Some(c)) = (graphics, compute) {
            Some((g, c))
        } else {
            None
        }
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            for pools in self.thread_command_pools.drain(..) {
                for pool in pools {
                    self.device.destroy_command_pool(pool, None);
                }
            }
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_command_pool(self.compute_command_pool, None);
            // Allocator must be dropped before Device
            // We use Arc to share it, so we need to ensure it's the last reference
            // or we just let Arc handle it, but wait_idle is important.
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
        }
    }
}
