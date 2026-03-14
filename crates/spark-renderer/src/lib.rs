use ash::{vk, Entry, Instance, Device};
use std::ffi::CString;
use winit::window::Window;

#[allow(dead_code)]
pub struct Renderer {
    entry: Entry,
    instance: Instance,
    device: Device,
}

impl Renderer {
    pub fn new(_window: &Window) -> Self {
        let entry = unsafe { Entry::load().expect("Failed to load Vulkan") };

        let app_name = CString::new("Spark Engine").unwrap();
        let engine_name = CString::new("Spark").unwrap();

        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        let instance_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info);

        let instance = unsafe {
            entry
                .create_instance(&instance_create_info, None)
                .expect("Failed to create Vulkan instance")
        };

        let pdevices = unsafe {
            instance
                .enumerate_physical_devices()
                .expect("Failed to enumerate physical devices")
        };

        if pdevices.is_empty() {
            panic!("No Vulkan devices found");
        }

        let pdevice = pdevices[0];

        let queue_priorities = [1.0];
        let queue_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(0)
            .queue_priorities(&queue_priorities);

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(std::slice::from_ref(&queue_info));

        let device = unsafe {
            instance
                .create_device(pdevice, &device_create_info, None)
                .expect("Failed to create logical device")
        };

        Self {
            entry,
            instance,
            device,
        }
    }
}
