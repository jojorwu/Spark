use super::{RenderContext, RenderPass};
use ash::vk;

pub struct GBufferPass {}

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
        struct GBufferPushConstants {
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
        let pc = GBufferPushConstants {
            count: renderer.light_count,
            metallic: 0.5,
            roughness: 0.5,
            width: renderer.swapchain.extent.width as f32,
            height: renderer.swapchain.extent.height as f32,
            padding: 0,
            object_buffer_address: renderer.frame_manager.frames[current_frame]
                .object_data_buffer
                .as_ref()
                .map_or(0, |b| b.address),
            prev_view_proj: renderer.prev_view_proj,
            vertex_buffer_address: renderer
                .gpu_resource_manager
                .global_vertex_buffer
                .as_ref()
                .map_or(0, |b| b.address),
        };
        let pc_bytes = unsafe {
            std::slice::from_raw_parts(
                &pc as *const _ as *const u8,
                std::mem::size_of::<GBufferPushConstants>(),
            )
        };

        let extent = renderer.get_extent();
        let global_ds = renderer.frame_manager.frames[current_frame].global_descriptor_set;

        unsafe {
            if let Some(pipeline) = &renderer.pipeline {
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.graphics_pipeline,
                );

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
                    crate::vulkan::utils::RenderingAttachmentBuilder::new(
                        renderer
                            .get_pass_resource_view(self.name(), "GBufferAlbedo", current_frame)
                            .expect("GBufferAlbedo missing"),
                    )
                    .build(),
                    crate::vulkan::utils::RenderingAttachmentBuilder::new(
                        renderer
                            .get_pass_resource_view(self.name(), "GBufferNormal", current_frame)
                            .expect("GBufferNormal missing"),
                    )
                    .build(),
                    crate::vulkan::utils::RenderingAttachmentBuilder::new(
                        renderer
                            .get_pass_resource_view(self.name(), "GBufferPBR", current_frame)
                            .expect("GBufferPBR missing"),
                    )
                    .build(),
                    crate::vulkan::utils::RenderingAttachmentBuilder::new(
                        renderer
                            .get_pass_resource_view(self.name(), "GBufferVelocity", current_frame)
                            .expect("GBufferVelocity missing"),
                    )
                    .build(),
                ];

                let depth_attachment = crate::vulkan::utils::RenderingAttachmentBuilder::new(
                    renderer
                        .get_pass_resource_view(self.name(), "GBufferDepth", current_frame)
                        .expect("GBufferDepth missing"),
                )
                .with_clear_depth(1.0)
                .build();

                let rendering_info = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent,
                    })
                    .layer_count(1)
                    .color_attachments(&color_attachments)
                    .depth_attachment(&depth_attachment);

                device.cmd_begin_rendering(command_buffer, &rendering_info);

                if let Some(ref indirect_buffer) =
                    renderer.frame_manager.frames[current_frame].indirect_commands_buffer
                {
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.layout,
                        0,
                        &[global_ds, renderer.gpu_resource_manager.bindless.set],
                        &[],
                    );

                    if let Some(ib) = renderer.gpu_resource_manager.global_index_buffer.as_ref() {
                        device.cmd_bind_index_buffer(
                            command_buffer,
                            ib.handle,
                            0,
                            vk::IndexType::UINT32,
                        );
                    }

                    if let Some(ref count_buffer) =
                        renderer.frame_manager.frames[current_frame].draw_count_buffer
                    {
                        renderer.last_draw_calls.fetch_add(
                            renderer.last_object_count,
                            std::sync::atomic::Ordering::Relaxed,
                        );

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
                        renderer.last_draw_calls.fetch_add(
                            renderer.last_object_count,
                            std::sync::atomic::Ordering::Relaxed,
                        );
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
    fn name(&self) -> &str {
        "GBufferPass"
    }

    fn outputs(&self) -> Vec<&'static str> {
        vec![
            "GBufferAlbedo",
            "GBufferNormal",
            "GBufferPBR",
            "GBufferVelocity",
            "GBufferDepth",
        ]
    }

    fn gpu_resource_access(&self) -> Vec<super::GpuResourceAccess> {
        vec![
            super::GpuResourceAccess {
                resource_name: "GBufferAlbedo",
                access_flags: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                stage_flags: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            },
            super::GpuResourceAccess {
                resource_name: "GBufferNormal",
                access_flags: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                stage_flags: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            },
            super::GpuResourceAccess {
                resource_name: "GBufferPBR",
                access_flags: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                stage_flags: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            },
            super::GpuResourceAccess {
                resource_name: "GBufferVelocity",
                access_flags: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                stage_flags: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            },
            super::GpuResourceAccess {
                resource_name: "GBufferDepth",
                access_flags: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                stage_flags: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            },
        ]
    }

    fn destroy(&mut self, _renderer: &mut Renderer) {}
    fn dependencies(&self) -> Vec<&'static str> {
        vec!["CullingPass"]
    }

    fn record_commands(&self, ctx: &RenderContext) {
        let renderer = ctx.renderer;
        let command_buffer = ctx.command_buffer;
        let current_frame = ctx.current_frame;
        let device = &renderer.device.device;

        self.record_gbuffer_commands(device, command_buffer, renderer, current_frame);
    }
}
