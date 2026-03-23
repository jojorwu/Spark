use ash::vk;
use crate::Renderer;
use crate::pipeline::Pipeline;

use super::{RenderPass, RenderContext};

impl RenderPass for GridPass {
    fn name(&self) -> &str { "GridPass" }
    fn is_enabled(&self, renderer: &Renderer) -> bool { renderer.enable_grid }
    fn record_commands(&self, ctx: &RenderContext) {
        let renderer = ctx.renderer;
        let command_buffer = ctx.command_buffer;
        let current_frame = ctx.current_frame;

        let global_ds = renderer.frames[current_frame].global_descriptor_set;
        self.record_commands_impl(
            &renderer.device.device,
            command_buffer,
            renderer.swapchain.extent,
            global_ds,
            renderer.gbuffer.hdr[current_frame].view,
            renderer.gbuffer.depth[current_frame].view,
        );
    }

    fn destroy(&mut self, renderer: &mut Renderer) {
        unsafe {
            let device = &renderer.device.device;
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

pub struct GridPass {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
}

impl GridPass {
    pub fn new(
        device: &ash::Device,
        pipeline_cache: vk::PipelineCache,
        vert_spirv: &[u32],
        frag_spirv: &[u32],
        global_ds_layout: vk::DescriptorSetLayout,
        format: vk::Format,
    ) -> Result<Self, crate::error::RendererError> {
        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[global_ds_layout]),
                None,
            )?
        };

        let vert_module = Pipeline::create_shader_module(device, vert_spirv);
        let frag_module = Pipeline::create_shader_module(device, frag_spirv);
        let entry_point = std::ffi::CString::new("main").unwrap();

        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(&entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(&entry_point),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default().cull_mode(vk::CullModeFlags::NONE).line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_write_mask(vk::ColorComponentFlags::RGBA);
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(std::slice::from_ref(&color_blend_attachment));

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let color_formats = [format];
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(vk::Format::D32_SFLOAT);

        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .depth_stencil_state(&depth_stencil)
            .dynamic_state(&dynamic_state_info)
            .layout(layout)
            .push_next(&mut rendering_info);

        let pipeline = unsafe {
            device.create_graphics_pipelines(pipeline_cache, &[info], None).map_err(|e| e.1)?[0]
        };

        unsafe {
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
        }

        Ok(Self { pipeline, layout })
    }

    pub fn record_commands_impl(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        extent: vk::Extent2D,
        global_ds: vk::DescriptorSet,
        hdr_view: vk::ImageView,
        depth_view: vk::ImageView,
    ) {
        unsafe {
            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(hdr_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE);

            let depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(depth_view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE);

            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent })
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color_attachment))
                .depth_attachment(&depth_attachment);

            device.cmd_begin_rendering(command_buffer, &rendering_info);
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

            let viewport = vk::Viewport::default().width(extent.width as f32).height(extent.height as f32).max_depth(1.0);
            let scissor = vk::Rect2D::default().extent(extent);
            device.cmd_set_viewport(command_buffer, 0, &[viewport]);
            device.cmd_set_scissor(command_buffer, 0, &[scissor]);

            device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, &[global_ds], &[]);
            device.cmd_draw(command_buffer, 3, 1, 0, 0);
            device.cmd_end_rendering(command_buffer);
        }
    }

}
