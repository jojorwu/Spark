use super::{RenderContext, RenderPass};
use crate::resource::Attachment;
use crate::{Renderer, MAX_FRAMES_IN_FLIGHT};
use ash::vk;

pub struct SSGIPass {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub output_images: Vec<Attachment>,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
}

impl RenderPass for SSGIPass {
    fn name(&self) -> &str {
        "SSGIPass"
    }
    fn inputs(&self) -> Vec<&'static str> {
        vec!["HDRColor", "GBuffer"]
    }
    fn outputs(&self) -> Vec<&'static str> {
        vec!["SSGI"]
    }

    fn record_commands(&self, _ctx: &RenderContext) {
        if self.pipeline != vk::Pipeline::null() {
            // Implementation for drawing a fullscreen quad with SSGI shader
        }
    }

    fn destroy(&mut self, renderer: &mut Renderer) {
        unsafe {
            renderer.device.device.destroy_pipeline(self.pipeline, None);
            renderer
                .device
                .device
                .destroy_pipeline_layout(self.layout, None);
            renderer
                .device
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            for img in self.output_images.drain(..) {
                img.destroy(&renderer.device.device, &renderer.device.allocator);
            }
        }
    }
}

impl SSGIPass {
    pub fn new(
        renderer: &Renderer,
        _shader_spirv: &[u32],
    ) -> Result<Self, crate::error::RendererError> {
        let extent = renderer.get_extent();

        let output_images = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| {
                Attachment::create_image_resource(
                    &renderer.device,
                    extent.width,
                    extent.height,
                    vk::Format::R16G16B16A16_SFLOAT,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                    vk::SampleCountFlags::TYPE_1,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        // Descriptor set layout and pipeline creation omitted

        Ok(Self {
            pipeline: vk::Pipeline::null(),
            layout: vk::PipelineLayout::null(),
            output_images,
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            descriptor_sets: Vec::new(),
        })
    }
}
