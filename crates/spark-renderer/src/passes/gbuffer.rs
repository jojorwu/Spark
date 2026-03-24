use ash::vk;
use super::{RenderPass, RenderContext};

pub struct GBufferPass {
}

impl GBufferPass {
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for GBufferPass {
    fn default() -> Self {
        Self::new()
    }
}

use crate::Renderer;

impl GBufferPass {
    fn record_gbuffer_commands(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        renderer: &Renderer,
        current_frame: usize,
    ) {
        #[repr(C)]
        struct PC {
            count: u32,
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
            count: renderer.light_count,
            metallic: 0.5,
            roughness: 0.5,
            width: renderer.swapchain.extent.width as f32,
            height: renderer.swapchain.extent.height as f32,
            padding: 0,
            object_buffer_address: renderer.frames[current_frame].object_data_buffer.as_ref().map_or(0, |b| b.address),
            prev_view_proj: renderer.prev_view_proj,
            vertex_buffer_address: renderer.global_vertex_buffer.as_ref().map_or(0, |b| b.address),
        };
        let pc_bytes = unsafe { std::slice::from_raw_parts(&pc as *const _ as *const u8, std::mem::size_of::<PC>()) };

        let extent = renderer.get_extent();
        let global_ds = renderer.frames[current_frame].global_descriptor_set;

        unsafe {
            if let Some(pipeline) = &renderer.pipeline {
                device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline.graphics_pipeline);

                let viewport = vk::Viewport::default()
                    .x(0.0)
                    .y(extent.height as f32)
                    .width(extent.width as f32)
                    .height(-(extent.height as f32))
                    .min_depth(0.0)
                    .max_depth(1.0);
                let scissor = vk::Rect2D::default().extent(extent);
                device.cmd_set_viewport(command_buffer, 0, &[viewport]);
                device.cmd_set_scissor(command_buffer, 0, &[scissor]);

                device.cmd_push_constants(
                    command_buffer,
                    pipeline.layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    pc_bytes,
                );

                let color_attachments = [
                    vk::RenderingAttachmentInfo::default()
                        .image_view(renderer.get_pass_resource_view("", "GBufferAlbedo", current_frame).unwrap())
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } }),
                    vk::RenderingAttachmentInfo::default()
                        .image_view(renderer.get_pass_resource_view("", "GBufferNormal", current_frame).unwrap())
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } }),
                    vk::RenderingAttachmentInfo::default()
                        .image_view(renderer.get_pass_resource_view("", "GBufferPBR", current_frame).unwrap())
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } }),
                    vk::RenderingAttachmentInfo::default()
                        .image_view(renderer.get_pass_resource_view("", "GBufferVelocity", current_frame).unwrap())
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } }),
                ];
                let depth_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(renderer.get_pass_resource_view("", "GBufferDepth", current_frame).unwrap())
                    .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 } });

                let rendering_info = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent })
                    .layer_count(1)
                    .color_attachments(&color_attachments)
                    .depth_attachment(&depth_attachment);

                device.cmd_begin_rendering(command_buffer, &rendering_info);

                if let Some(ref indirect_buffer) = renderer.frames[current_frame].indirect_commands_buffer {
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.layout,
                        0,
                        &[global_ds, renderer.bindless_descriptor_set],
                        &[],
                    );

                    if let Some(ib) = renderer.global_index_buffer.as_ref() {
                        device.cmd_bind_index_buffer(command_buffer, ib.handle, 0, vk::IndexType::UINT32);
                    }

                    if let Some(ref count_buffer) = renderer.frames[current_frame].draw_count_buffer {
                         device.cmd_draw_indexed_indirect_count(
                            command_buffer,
                            indirect_buffer.handle,
                            0,
                            count_buffer.handle,
                            0,
                            renderer.last_object_count,
                            std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                        );
                    } else {
                        device.cmd_draw_indexed_indirect(
                            command_buffer,
                            indirect_buffer.handle,
                            0,
                            renderer.last_object_count,
                            std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                        );
                    }
                }

                device.cmd_end_rendering(command_buffer);
            }
        }
    }
}

impl RenderPass for GBufferPass {
    fn name(&self) -> &str { "GBufferPass" }
    fn dependencies(&self) -> Vec<&'static str> { vec!["CullingPass"] }

    fn record_secondary_commands(&self, ctx: &RenderContext) -> Vec<vk::CommandBuffer> {
        let renderer = ctx.renderer;
        let device = &renderer.device.device;
        let cf = ctx.current_frame;

        let cb = renderer.allocate_secondary_command_buffer();

        let color_formats = [vk::Format::R8G8B8A8_UNORM, vk::Format::A2B10G10R10_UNORM_PACK32, vk::Format::R8G8B8A8_UNORM, vk::Format::R16G16_SFLOAT];
        let mut rendering_info = vk::CommandBufferInheritanceRenderingInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(vk::Format::D32_SFLOAT);

        let inheritance = vk::CommandBufferInheritanceInfo::default()
            .push_next(&mut rendering_info);
        let begin = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE).inheritance_info(&inheritance);

        unsafe {
            device.begin_command_buffer(cb, &begin).unwrap();
            self.record_gbuffer_commands(device, cb, renderer, cf);
            device.end_command_buffer(cb).unwrap();
        }

        vec![cb]
    }


    fn record_commands(&self, ctx: &RenderContext) {
        let renderer = ctx.renderer;
        let command_buffer = ctx.command_buffer;
        let current_frame = ctx.current_frame;

        // Note: record_gbuffer_commands is called via record_secondary_commands.
        // We only add barriers here.

        // Barrier: G-Buffer to SHADER_READ_ONLY_OPTIMAL
        let gbuffer_barriers = [
            vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(renderer.render_graph.physical_attachments.get("GBufferAlbedo").unwrap()[current_frame].image)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 }),
            vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(renderer.render_graph.physical_attachments.get("GBufferNormal").unwrap()[current_frame].image)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 }),
            vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(renderer.render_graph.physical_attachments.get("GBufferPBR").unwrap()[current_frame].image)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 }),
            vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(renderer.render_graph.physical_attachments.get("GBufferDepth").unwrap()[current_frame].image)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::DEPTH, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 }),
        ];
        unsafe {
            renderer.device.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &gbuffer_barriers,
            );
        }
    }
}
