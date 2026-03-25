use ash::vk;
use crate::Renderer;

pub struct ReflectionProbe {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub irradiance_view: vk::ImageView,
    pub prefilter_view: vk::ImageView,
    pub position: spark_math::Vec3,
    pub range: f32,
    pub box_min: spark_math::Vec3,
    pub box_max: spark_math::Vec3,
}

pub struct IBLMaps {
    pub irradiance_image: vk::Image,
    pub irradiance_view: vk::ImageView,
    pub prefilter_image: vk::Image,
    pub prefilter_view: vk::ImageView,
    pub brdf_lut_image: vk::Image,
    pub brdf_lut_view: vk::ImageView,
    pub memory: vk::DeviceMemory,
    pub local_probes: Vec<ReflectionProbe>,
}

impl IBLMaps {
    pub fn new(
        _renderer: &Renderer,
        _env_view: vk::ImageView,
    ) -> Self {
        Self {
            irradiance_image: vk::Image::null(),
            irradiance_view: vk::ImageView::null(),
            prefilter_image: vk::Image::null(),
            prefilter_view: vk::ImageView::null(),
            brdf_lut_image: vk::Image::null(),
            brdf_lut_view: vk::ImageView::null(),
            memory: vk::DeviceMemory::null(),
            local_probes: Vec::new(),
        }
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            if self.irradiance_view != vk::ImageView::null() {
                device.destroy_image_view(self.irradiance_view, None);
                device.destroy_image(self.irradiance_image, None);
            }
            if self.prefilter_view != vk::ImageView::null() {
                device.destroy_image_view(self.prefilter_view, None);
                device.destroy_image(self.prefilter_image, None);
            }
            if self.brdf_lut_view != vk::ImageView::null() {
                device.destroy_image_view(self.brdf_lut_view, None);
                device.destroy_image(self.brdf_lut_image, None);
            }
            if self.memory != vk::DeviceMemory::null() {
                device.free_memory(self.memory, None);
            }
        }
    }
}
