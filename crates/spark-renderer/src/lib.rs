use ash::{vk, Entry, Instance, Device, extensions::khr::Surface, extensions::khr::Swapchain};
use std::ffi::CString;
use winit::window::Window;

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
}

impl Renderer {
    pub fn new(_window: &Window) -> Self {
        let entry = unsafe { Entry::load().expect("Failed to load Vulkan") };

        let extensions = {
            let mut ext = vec![Surface::name().as_ptr()];
            #[cfg(target_os = "windows")]
            {
                use ash::extensions::khr::Win32Surface;
                ext.push(Win32Surface::name().as_ptr());
            }
            #[cfg(target_os = "linux")]
            {
                use ash::extensions::khr::XlibSurface;
                use ash::extensions::khr::WaylandSurface;
                ext.push(XlibSurface::name().as_ptr());
                ext.push(WaylandSurface::name().as_ptr());
            }
            ext
        };

        let app_name = CString::new("Spark Engine").unwrap();
        let engine_name = CString::new("Spark").unwrap();

        let app_info = vk::ApplicationInfo::builder()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::API_VERSION_1_3);

        let instance_create_info = vk::InstanceCreateInfo::builder()
            .application_info(&app_info)
            .enabled_extension_names(&extensions);

        let instance = unsafe {
            entry
                .create_instance(&instance_create_info, None)
                .expect("Failed to create Vulkan instance")
        };

        let surface = vk::SurfaceKHR::null();
        let surface_loader = Surface::new(&entry, &instance);

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

        let device_extension_names_raw = [Swapchain::name().as_ptr()];

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(std::slice::from_ref(&queue_info))
            .enabled_extension_names(&device_extension_names_raw);

        let device = unsafe {
            instance
                .create_device(pdevice, &device_create_info, None)
                .expect("Failed to create logical device")
        };

        let swapchain_loader = Swapchain::new(&instance, &device);

        let swapchain = vk::SwapchainKHR::null();
        let swapchain_images = Vec::new();
        let swapchain_image_views = Vec::new();

        Self {
            entry,
            instance,
            pdevice,
            device,
            surface_loader,
            surface,
            swapchain_loader,
            swapchain,
            swapchain_images,
            swapchain_image_views,
        }
    }
}
