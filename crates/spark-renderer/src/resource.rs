use ash::vk;

pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

/// Represents a Vulkan buffer with its associated memory and size.
#[derive(Debug, Copy, Clone)]
pub struct Buffer {
    pub handle: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
}

/// Represents a framebuffer attachment (Image, Memory, View).
pub struct Attachment {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
}

/// Represents all resources and synchronization primitives for a single frame.
#[derive(Debug)]
pub struct RenderFrame {
    pub command_buffer: vk::CommandBuffer,
    pub image_available: vk::Semaphore,
    pub render_finished: vk::Semaphore,
    pub in_flight: vk::Fence,
    pub global_buffer: Option<Buffer>,
    pub light_buffer: Option<Buffer>,
    pub global_descriptor_set: vk::DescriptorSet,
    pub instance_pool: Vec<Buffer>,
    pub instance_index: usize,
}

impl Attachment {
    /// Destroys the attachment resources.
    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }

    pub fn create_image_resource(
        device: &ash::Device,
        mem_props: &vk::PhysicalDeviceMemoryProperties,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        samples: vk::SampleCountFlags,
    ) -> Self {
        let img_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let img = unsafe { device.create_image(&img_info, None).unwrap() };
        let reqs = unsafe { device.get_image_memory_requirements(img) };
        let mut type_idx = 0;
        for i in 0..mem_props.memory_type_count {
            if (reqs.memory_type_bits & (1 << i)) != 0
                && (mem_props.memory_types[i as usize].property_flags
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
        Self {
            image: img,
            memory: mem,
            view,
        }
    }
}

pub fn create_frame_attachments(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    extent: vk::Extent2D,
    format: vk::Format,
    msaa: vk::SampleCountFlags,
) -> Vec<Attachment> {
    (0..MAX_FRAMES_IN_FLIGHT)
        .map(|_| {
            Attachment::create_image_resource(
                device,
                mem_props,
                extent.width,
                extent.height,
                format,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                msaa,
            )
        })
        .collect()
}
