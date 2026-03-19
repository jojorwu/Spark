use ash::vk;
use crate::resource::Attachment;
use crate::MAX_FRAMES_IN_FLIGHT;

pub struct GBuffer {
    pub hdr: Vec<Attachment>,
    pub albedo: Vec<Attachment>,
    pub normal: Vec<Attachment>,
    pub pbr: Vec<Attachment>,
    pub velocity: Vec<Attachment>,
    pub depth: Vec<Attachment>,
}

impl GBuffer {
    pub fn new(
        device: &ash::Device,
        pdevice: vk::PhysicalDevice,
        instance: &ash::Instance,
        extent: vk::Extent2D,
        msaa_samples: vk::SampleCountFlags,
        depth_format: vk::Format,
    ) -> Self {
        let props = unsafe { instance.get_physical_device_memory_properties(pdevice) };

        let hdr: Vec<Attachment> = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| {
                Attachment::create_image_resource(
                    device,
                    &props,
                    extent.width,
                    extent.height,
                    vk::Format::R16G16B16A16_SFLOAT,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT
                        | vk::ImageUsageFlags::INPUT_ATTACHMENT
                        | vk::ImageUsageFlags::SAMPLED,
                    vk::SampleCountFlags::TYPE_1,
                )
            })
            .collect();

        let albedo = crate::resource::create_frame_attachments(
            device,
            &props,
            extent,
            vk::Format::R8G8B8A8_UNORM,
            msaa_samples,
        );
        let velocity = crate::resource::create_frame_attachments(
            device,
            &props,
            extent,
            vk::Format::R16G16_SFLOAT,
            msaa_samples,
        );
        let normal = crate::resource::create_frame_attachments(
            device,
            &props,
            extent,
            vk::Format::A2B10G10R10_UNORM_PACK32,
            msaa_samples,
        );
        let pbr = crate::resource::create_frame_attachments(
            device,
            &props,
            extent,
            vk::Format::R8G8B8A8_UNORM,
            msaa_samples,
        );

        let depth: Vec<Attachment> = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| {
                Attachment::create_image_resource(
                    device,
                    &props,
                    extent.width,
                    extent.height,
                    depth_format,
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                        | vk::ImageUsageFlags::INPUT_ATTACHMENT
                        | vk::ImageUsageFlags::SAMPLED,
                    msaa_samples,
                )
            })
            .collect();

        Self {
            hdr,
            albedo,
            normal,
            pbr,
            velocity,
            depth,
        }
    }

    pub fn recreate(
        &mut self,
        device: &ash::Device,
        pdevice: vk::PhysicalDevice,
        instance: &ash::Instance,
        extent: vk::Extent2D,
        msaa_samples: vk::SampleCountFlags,
        depth_format: vk::Format,
    ) {
        self.destroy(device);
        let new_gb = Self::new(device, pdevice, instance, extent, msaa_samples, depth_format);
        *self = new_gb;
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        for a in self.hdr.drain(..) { a.destroy(device); }
        for a in self.albedo.drain(..) { a.destroy(device); }
        for a in self.normal.drain(..) { a.destroy(device); }
        for a in self.pbr.drain(..) { a.destroy(device); }
        for a in self.velocity.drain(..) { a.destroy(device); }
        for a in self.depth.drain(..) { a.destroy(device); }
    }
}
