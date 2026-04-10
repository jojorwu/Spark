use crate::error::RendererError;
use ash::{khr::surface::Instance as Surface, vk, Entry, Instance};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::ffi::CString;
use winit::window::Window;

/// Encapsulates the core Vulkan instance, debug messenger, and surface components.
pub struct VulkanContext {
    pub entry: Entry,
    pub instance: Instance,
    pub surface_loader: Surface,
    pub surface: vk::SurfaceKHR,
}

impl VulkanContext {
    pub fn new(window: &Window) -> Result<Self, RendererError> {
        let entry = unsafe { Entry::load()? };

        let app_name = CString::new("Spark Engine").expect("Failed to create app name CString");
        let engine_name = CString::new("Spark").expect("Failed to create engine name CString");

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        let display_handle = window
            .display_handle()
            .expect("Failed to get display handle")
            .as_raw();
        let window_handle = window
            .window_handle()
            .expect("Failed to get window handle")
            .as_raw();

        let extensions = ash_window::enumerate_required_extensions(display_handle)
            .map_err(|_| RendererError::SurfaceCreation)?;

        let instance_create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(extensions);

        let instance = unsafe { entry.create_instance(&instance_create_info, None)? };

        let surface = unsafe {
            ash_window::create_surface(&entry, &instance, display_handle, window_handle, None)
                .map_err(|_| RendererError::SurfaceCreation)?
        };
        let surface_loader = Surface::new(&entry, &instance);

        Ok(Self {
            entry,
            instance,
            surface_loader,
            surface,
        })
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
