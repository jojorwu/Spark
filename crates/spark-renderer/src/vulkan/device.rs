use ash::{vk, Instance, Device, khr::surface::Instance as Surface};

pub struct VulkanDevice {
    pub pdevice: vk::PhysicalDevice,
    pub device: Device,
    pub graphics_queue: vk::Queue,
    pub graphics_family: u32,
}

impl VulkanDevice {
    pub fn new(instance: &Instance, surface_loader: &Surface, surface: vk::SurfaceKHR) -> Self {
        let pdevices = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("Failed to enumerate physical devices")
        };

        let (pdevice, graphics_family) = pdevices
            .iter()
            .filter_map(|&pdevice| {
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
                        return Some((pdevice, index));
                    }
                }
                None
            })
            .next()
            .expect("Failed to find a suitable physical device");

        let queue_priorities = [1.0];
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(graphics_family)
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

        let graphics_queue = unsafe { device.get_device_queue(graphics_family, 0) };

        Self {
            pdevice,
            device,
            graphics_queue,
            graphics_family,
        }
    }
}

impl Drop for VulkanDevice {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}
