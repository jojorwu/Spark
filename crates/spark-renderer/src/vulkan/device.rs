use ash::{vk, Instance, Device, khr::surface::Instance as Surface};
use crate::error::RendererError;

pub struct VulkanDevice {
    pub pdevice: vk::PhysicalDevice,
    pub device: Device,
    pub graphics_queue: vk::Queue,
    pub graphics_family: u32,
}

impl VulkanDevice {
    pub fn new(instance: &Instance, surface_loader: &Surface, surface: vk::SurfaceKHR) -> Result<Self, RendererError> {
        let pdevices = unsafe {
            instance
                .enumerate_physical_devices()?
        };

        let (pdevice, graphics_family) = pdevices
            .iter()
            .map(|&p| (p, Self::score_device(instance, p)))
            .filter(|&(_, score)| score > 0)
            .max_by_key(|&(_, score)| score)
            .and_then(|(p, _)| {
                Self::find_queue_families(instance, surface_loader, surface, p).map(|family| (p, family))
            })
            .ok_or(RendererError::NoSuitableDevice)?;

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
                .create_device(pdevice, &device_create_info, None)?
        };

        let graphics_queue = unsafe { device.get_device_queue(graphics_family, 0) };

        Ok(Self {
            pdevice,
            device,
            graphics_queue,
            graphics_family,
        })
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
            self.device.destroy_device(None);
        }
    }
}
