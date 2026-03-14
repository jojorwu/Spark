use ash::{vk, Entry, Instance, khr::surface::Instance as Surface};
use std::ffi::CString;
use winit::window::Window;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

pub struct VulkanContext {
    pub entry: Entry,
    pub instance: Instance,
    pub surface_loader: Surface,
    pub surface: vk::SurfaceKHR,
}

impl VulkanContext {
    pub fn new(window: &Window) -> Self {
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

        Self {
            entry,
            instance,
            surface_loader,
            surface,
        }
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.surface_loader.destroy_surface(self.surface, None);
            self.instance.destroy_instance(None);
        }
    }
}
