use ash::{
    khr::surface::Instance as Surface, khr::swapchain::Device as SwapchainLoader, vk, Device,
    Instance,
};

pub struct VulkanSwapchain {
    pub loader: SwapchainLoader,
    pub handle: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}

impl VulkanSwapchain {
    pub fn new(
        instance: &Instance,
        device: &Device,
        pdevice: vk::PhysicalDevice,
        surface_loader: &Surface,
        surface: vk::SurfaceKHR,
        width: u32,
        height: u32,
    ) -> Self {
        let loader = SwapchainLoader::new(instance, device);
        let (handle, images, views, format, extent) = Self::create_internal(
            pdevice,
            device,
            surface_loader,
            surface,
            &loader,
            width,
            height,
        );

        Self {
            loader,
            handle,
            images,
            views,
            format,
            extent,
        }
    }

    fn create_internal(
        pdevice: vk::PhysicalDevice,
        device: &Device,
        surface_loader: &Surface,
        surface: vk::SurfaceKHR,
        loader: &SwapchainLoader,
        width: u32,
        height: u32,
    ) -> (
        vk::SwapchainKHR,
        Vec<vk::Image>,
        Vec<vk::ImageView>,
        vk::Format,
        vk::Extent2D,
    ) {
        let formats = unsafe {
            surface_loader
                .get_physical_device_surface_formats(pdevice, surface)
                .unwrap()
        };

        // Prefer HDR/High-bit-depth formats
        let surface_format = formats
            .iter()
            .cloned()
            .find(|f| {
                f.format == vk::Format::A2B10G10R10_UNORM_PACK32
                    || f.format == vk::Format::R16G16B16A16_SFLOAT
            })
            .unwrap_or(formats[0]);

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

        let handle = unsafe {
            loader
                .create_swapchain(&swapchain_create_info, None)
                .expect("Failed to create swapchain")
        };

        let images = unsafe {
            loader
                .get_swapchain_images(handle)
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

        (handle, images, views, surface_format.format, extent)
    }
}

impl Drop for VulkanSwapchain {
    fn drop(&mut self) {
        // Views should be destroyed by Device, loader by us.
        // Actually, VulkanSwapchain doesn't have the Device handle.
        // We'll let Renderer handle destruction to maintain order.
    }
}
