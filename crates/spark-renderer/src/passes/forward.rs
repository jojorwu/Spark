use ash::vk;
use crate::Renderer;
use super::{RenderPass, RenderContext};

pub struct ForwardPass {
    pub pipeline: Option<vk::Pipeline>,
    pub layout: vk::PipelineLayout,
}

impl RenderPass for ForwardPass {
    fn name(&self) -> &str { "ForwardPass" }

    fn prepare(&self, _renderer: &Renderer, _current_frame: usize) {
        // Here we could implement back-to-front sorting of transparent objects
        // However, with MDI (Multi-Draw Indirect), we'd need to sort the ObjectDataSSBO
        // or use an index buffer to control the order.
    }

    fn record_commands(&self, ctx: &RenderContext) {
        let renderer = ctx.renderer;
        let cf = ctx.current_frame;

        if renderer.last_transparent_count == 0 {
             return;
        }

        let pipeline = match self.pipeline {
            Some(p) => p,
            None => return,
        };

        let extent = renderer.get_extent();

        unsafe {
            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(renderer.gbuffer.hdr[cf].view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE);

            let depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(renderer.gbuffer.depth[cf].view)
                .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE);

            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent })
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color_attachment))
                .depth_attachment(&depth_attachment);

            renderer.device.device.cmd_begin_rendering(ctx.command_buffer, &rendering_info);
            renderer.device.device.cmd_bind_pipeline(ctx.command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);

            let viewport = vk::Viewport::default().width(extent.width as f32).height(extent.height as f32).max_depth(1.0);
            let scissor = vk::Rect2D::default().extent(extent);
            renderer.device.device.cmd_set_viewport(ctx.command_buffer, 0, &[viewport]);
            renderer.device.device.cmd_set_scissor(ctx.command_buffer, 0, &[scissor]);

            #[repr(C)]
            struct PC {
                light_count: u32,
                metallic: f32,
                roughness: f32,
                width: f32,
                height: f32,
                padding: u32,
                object_buffer_address: u64,
                prev_view_proj: spark_math::Mat4,
                vertex_buffer_address: u64,
            }
            let pc = PC {
                light_count: renderer.light_count,
                metallic: 0.5,
                roughness: 0.5,
                width: extent.width as f32,
                height: extent.height as f32,
                padding: 0,
                object_buffer_address: renderer.frames[cf].transparent_object_buffer.as_ref().map_or(0, |b| b.address),
                prev_view_proj: renderer.prev_view_proj,
                vertex_buffer_address: renderer.global_vertex_buffer.as_ref().map_or(0, |b| b.address),
            };
            let pc_bytes = std::slice::from_raw_parts(&pc as *const _ as *const u8, std::mem::size_of::<PC>());

            renderer.device.device.cmd_push_constants(ctx.command_buffer, self.layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, pc_bytes);

            renderer.device.device.cmd_bind_descriptor_sets(
                ctx.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                &[renderer.frames[cf].global_descriptor_set, renderer.bindless_descriptor_set],
                &[],
            );

            if let Some(ref ib) = renderer.global_index_buffer {
                renderer.device.device.cmd_bind_index_buffer(ctx.command_buffer, ib.handle, 0, vk::IndexType::UINT32);
            }

            if let Some(ref indirect_buffer) = renderer.frames[cf].transparent_indirect_buffer {
                renderer.device.device.cmd_draw_indexed_indirect(
                    ctx.command_buffer,
                    indirect_buffer.handle,
                    0,
                    renderer.last_transparent_count,
                    std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                );
            }

            renderer.device.device.cmd_end_rendering(ctx.command_buffer);
        }
    }

    fn destroy(&mut self, renderer: &mut Renderer) {
        unsafe {
            renderer.device.device.destroy_pipeline(self.pipeline.unwrap(), None);
            renderer.device.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

impl ForwardPass {
    pub fn new(renderer: &Renderer, vert_spirv: &[u32], frag_spirv: &[u32]) -> Result<Self, crate::error::RendererError> {
        let device = &renderer.device.device;

        let push_constant_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(128)];

        let set_layouts = [renderer.global_descriptor_set_layout, renderer.bindless_descriptor_set_layout];

        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push_constant_ranges),
                None,
            )?
        };

        let vert_module = crate::pipeline::Pipeline::create_shader_module(device, vert_spirv);
        let frag_module = crate::pipeline::Pipeline::create_shader_module(device, frag_spirv);
        let entry_point = std::ffi::CString::new("main").unwrap();

        let stages = [
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vert_module).name(&entry_point),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(frag_module).name(&entry_point),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default().cull_mode(vk::CullModeFlags::NONE).front_face(vk::FrontFace::CLOCKWISE).line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(renderer.get_msaa_samples());

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);

        let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(std::slice::from_ref(&color_blend_attachment));

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::LESS);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let color_formats = [vk::Format::R16G16B16A16_SFLOAT];
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
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state_info)
            .layout(layout)
            .push_next(&mut rendering_info);

        let pipeline = unsafe { device.create_graphics_pipelines(renderer.pipeline_cache, &[info], None).unwrap()[0] };

        unsafe {
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
        }

        Ok(Self {
            pipeline: Some(pipeline),
            layout,
        })
    }
}
