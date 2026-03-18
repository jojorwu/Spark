use ash::vk;
use crate::resource::Attachment;
use crate::MAX_FRAMES_IN_FLIGHT;

pub struct GBuffer {
    pub hdr: Vec<Attachment>,
    pub albedo: Vec<Attachment>,
    pub normal: Vec<Attachment>,
    pub pbr: Vec<Attachment>,
    pub depth: Vec<Attachment>,
    pub render_pass: vk::RenderPass,
    pub framebuffers: Vec<vk::Framebuffer>,
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
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT,
                    msaa_samples,
                )
            })
            .collect();

        let render_pass = Self::create_render_pass(device, msaa_samples, depth_format);
        let framebuffers = Self::create_framebuffers(device, render_pass, &hdr, &albedo, &normal, &pbr, &depth, extent);

        Self {
            hdr,
            albedo,
            normal,
            pbr,
            depth,
            render_pass,
            framebuffers,
        }
    }

    fn create_render_pass(
        device: &ash::Device,
        msaa_samples: vk::SampleCountFlags,
        depth_format: vk::Format,
    ) -> vk::RenderPass {
        let albedo_att = vk::AttachmentDescription::default()
            .format(vk::Format::R8G8B8A8_UNORM)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let normal_att = vk::AttachmentDescription::default()
            .format(vk::Format::A2B10G10R10_UNORM_PACK32)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let pbr_att = vk::AttachmentDescription::default()
            .format(vk::Format::R8G8B8A8_UNORM)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let hdr_att = vk::AttachmentDescription::default()
            .format(vk::Format::R16G16B16A16_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let depth_att = vk::AttachmentDescription::default()
            .format(depth_format)
            .samples(msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let albedo_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let normal_ref = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let pbr_ref = vk::AttachmentReference::default()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let hdr_ref = vk::AttachmentReference::default()
            .attachment(3)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let depth_ref = vk::AttachmentReference::default()
            .attachment(4)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let subpass0_color_attachments = [albedo_ref, normal_ref, pbr_ref];
        let subpass0 = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&subpass0_color_attachments)
            .depth_stencil_attachment(&depth_ref);

        let subpass1_input_attachments = [
            vk::AttachmentReference::default()
                .attachment(0)
                .layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            vk::AttachmentReference::default()
                .attachment(1)
                .layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            vk::AttachmentReference::default()
                .attachment(2)
                .layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
            vk::AttachmentReference::default()
                .attachment(4)
                .layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL),
        ];
        let subpass1 = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(std::slice::from_ref(&hdr_ref))
            .input_attachments(&subpass1_input_attachments);

        let deps = [
            vk::SubpassDependency::default()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(vk::PipelineStageFlags::BOTTOM_OF_PIPE)
                .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags::MEMORY_READ)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
            vk::SubpassDependency::default()
                .src_subpass(0)
                .dst_subpass(1)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ),
            vk::SubpassDependency::default()
                .src_subpass(1)
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                .dst_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
                .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ),
        ];
        let attachments = [albedo_att, normal_att, pbr_att, hdr_att, depth_att];
        let subpasses = [subpass0, subpass1];
        unsafe {
            device
                .create_render_pass(
                    &vk::RenderPassCreateInfo::default()
                        .attachments(&attachments)
                        .subpasses(&subpasses)
                        .dependencies(&deps),
                    None,
                )
                .expect("Failed to create render pass")
        }
    }

    fn create_framebuffers(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        hdr: &[Attachment],
        albedo: &[Attachment],
        normal: &[Attachment],
        pbr: &[Attachment],
        depth: &[Attachment],
        extent: vk::Extent2D,
    ) -> Vec<vk::Framebuffer> {
        (0..MAX_FRAMES_IN_FLIGHT)
            .map(|i| {
                let attachments = [
                    albedo[i].view,
                    normal[i].view,
                    pbr[i].view,
                    hdr[i].view,
                    depth[i].view,
                ];
                unsafe {
                    device
                        .create_framebuffer(
                            &vk::FramebufferCreateInfo::default()
                                .render_pass(render_pass)
                                .attachments(&attachments)
                                .width(extent.width)
                                .height(extent.height)
                                .layers(1),
                            None,
                        )
                        .expect("Failed to create framebuffer")
                }
            })
            .collect()
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
        unsafe {
            for f in self.framebuffers.drain(..) {
                device.destroy_framebuffer(f, None);
            }
            device.destroy_render_pass(self.render_pass, None);
        }
        for a in self.hdr.drain(..) { a.destroy(device); }
        for a in self.albedo.drain(..) { a.destroy(device); }
        for a in self.normal.drain(..) { a.destroy(device); }
        for a in self.pbr.drain(..) { a.destroy(device); }
        for a in self.depth.drain(..) { a.destroy(device); }
    }
}
