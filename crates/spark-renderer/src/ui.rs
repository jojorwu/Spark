use ash::vk;
use crate::Renderer;

pub struct EguiRenderer {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub vertex_buffer: Option<crate::Buffer>,
    pub index_buffer: Option<crate::Buffer>,
    pub font_texture: Option<crate::vulkan::texture::Texture>,
}

impl EguiRenderer {
    pub fn new(device: &ash::Device, render_pass: vk::RenderPass) -> Self {
        let push_constant_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(8)]; // vec2 screen_size

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(&push_constant_ranges);

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .unwrap()
        };

        Self {
            pipeline: vk::Pipeline::null(),
            pipeline_layout,
            vertex_buffer: None,
            index_buffer: None,
            font_texture: None,
        }
    }

    pub fn draw(
        &mut self,
        _device: &ash::Device,
        _graphics_queue: vk::Queue,
        _command_buffer: vk::CommandBuffer,
        _full_output: egui::FullOutput,
    ) {
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            if self.pipeline != vk::Pipeline::null() {
                device.destroy_pipeline(self.pipeline, None);
            }
        }
    }
}
