use ash::vk;
use crate::resource::{Attachment, Buffer};
use crate::{Renderer, MAX_FRAMES_IN_FLIGHT};
use super::RenderPass;

impl RenderPass for DeferredPass {
    fn update_descriptor_sets(&self, renderer: &Renderer) {
        renderer.update_deferred_descriptor_sets();
    }

    fn record_commands(&self, renderer: &Renderer, command_buffer: vk::CommandBuffer, current_frame: usize) {
        // This pass is split into G-Buffer and Lighting, so it doesn't fit perfectly into a single record_commands
        // Unless we call both here.
    }
}

pub struct DeferredPass {
    pub pipeline: Option<vk::Pipeline>,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
}

impl DeferredPass {
    pub fn new(
        device: &ash::Device,
        descriptor_pool: vk::DescriptorPool,
        global_ds_layout: vk::DescriptorSetLayout,
    ) -> Result<Self, crate::error::RendererError> {
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(6)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(7)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(8)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(9)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(10)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];

        let ds_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?
        };

        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[global_ds_layout, ds_layout])
                    .push_constant_ranges(&[vk::PushConstantRange::default()
                        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                        .offset(0)
                        .size(128)]),
                None,
            )?
        };

        let layouts = vec![ds_layout; MAX_FRAMES_IN_FLIGHT];
        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&layouts),
            )?
        };

        Ok(Self {
            pipeline: None,
            layout,
            descriptor_set_layout: ds_layout,
            descriptor_sets,
        })
    }

    pub fn update_descriptor_sets(
        &self,
        device: &ash::Device,
        albedo: &[Attachment],
        normal: &[Attachment],
        pbr: &[Attachment],
        depth: &[Attachment],
        shadow_view: vk::ImageView,
        shadow_sampler: vk::Sampler,
        light_buffers: &[Buffer],
        object_data_buffers: &[Option<Buffer>],
        ssao_attachments: &[Attachment],
        irradiance_view: vk::ImageView,
        specular_view: vk::ImageView,
        brdf_lut_view: vk::ImageView,
    ) {
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let alb_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(albedo[i].view)];
            let norm_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(normal[i].view)];
            let pbr_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(pbr[i].view)];
            let depth_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(depth[i].view)];
            let shadow_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(shadow_view)
                .sampler(shadow_sampler)];

            let mut writes = vec![
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                    .image_info(&alb_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                    .image_info(&norm_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                    .image_info(&pbr_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(3)
                    .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                    .image_info(&depth_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(4)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&shadow_info),
            ];

            let mut buf_info = Vec::new();
            if let Some(lb) = light_buffers.get(i) {
                buf_info.push(
                    vk::DescriptorBufferInfo::default()
                        .buffer(lb.handle)
                        .offset(0)
                        .range(lb.size),
                );
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.descriptor_sets[i])
                        .dst_binding(5)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&buf_info),
                );
            }

            let mut obj_info = Vec::new();
            if let Some(Some(ob)) = object_data_buffers.get(i) {
                obj_info.push(
                    vk::DescriptorBufferInfo::default()
                        .buffer(ob.handle)
                        .offset(0)
                        .range(ob.size),
                );
                writes.push(
                    vk::WriteDescriptorSet::default()
                        .dst_set(self.descriptor_sets[i])
                        .dst_binding(6)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&obj_info),
                );
            }

            let ssao_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(ssao_attachments[i].view)
                .sampler(shadow_sampler)];
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(7)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&ssao_info),
            );

            let irr_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(irradiance_view)
                .sampler(shadow_sampler)];
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(8)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&irr_info),
            );

            let spec_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(specular_view)
                .sampler(shadow_sampler)];
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(9)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&spec_info),
            );

            let brdf_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(brdf_lut_view)
                .sampler(shadow_sampler)];
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(10)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&brdf_info),
            );

            unsafe {
                device.update_descriptor_sets(&writes, &[]);
            }
        }
    }

    pub fn record_gbuffer_commands(
        &self,
        renderer: &Renderer,
        command_buffer: vk::CommandBuffer,
        pc_bytes: &[u8],
        current_frame: usize,
        max_indirect_commands: u32,
    ) {
        let device = &renderer.device.device;
        let extent = renderer.get_extent();
        let global_ds = renderer.frames[current_frame].global_descriptor_set;

        unsafe {
            let color_formats = [vk::Format::R8G8B8A8_UNORM, vk::Format::A2B10G10R10_UNORM_PACK32, vk::Format::R8G8B8A8_UNORM, vk::Format::R16G16_SFLOAT];
            let mut inheritance_info = vk::CommandBufferInheritanceRenderingInfo::default()
                .color_attachment_formats(&color_formats)
                .depth_attachment_format(vk::Format::D32_SFLOAT);
            let inherit = vk::CommandBufferInheritanceInfo::default().push_next(&mut inheritance_info);
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE | vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                .inheritance_info(&inherit);

            device.begin_command_buffer(command_buffer, &begin_info).unwrap();

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
                        .image_view(renderer.gbuffer.albedo[current_frame].view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } }),
                    vk::RenderingAttachmentInfo::default()
                        .image_view(renderer.gbuffer.normal[current_frame].view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } }),
                    vk::RenderingAttachmentInfo::default()
                        .image_view(renderer.gbuffer.pbr[current_frame].view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } }),
                    vk::RenderingAttachmentInfo::default()
                        .image_view(renderer.gbuffer.velocity[current_frame].view)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } }),
                ];
                let depth_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(renderer.gbuffer.depth[current_frame].view)
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

                if let Some(indirect_buffer) = renderer.frames[current_frame].indirect_commands_buffer {
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline.layout,
                        0,
                        &[global_ds, renderer.bindless_descriptor_set],
                        &[],
                    );

                    if let (Some(vb), Some(ib)) = (renderer.global_vertex_buffer, renderer.global_index_buffer) {
                        device.cmd_bind_vertex_buffers(command_buffer, 0, &[vb.handle], &[0]);
                        device.cmd_bind_index_buffer(command_buffer, ib.handle, 0, vk::IndexType::UINT32);
                    }

                    if let Some(count_buffer) = renderer.frames[current_frame].draw_count_buffer {
                         device.cmd_draw_indexed_indirect_count(
                            command_buffer,
                            indirect_buffer.handle,
                            0,
                            count_buffer.handle,
                            0,
                            max_indirect_commands,
                            std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                        );
                    } else {
                        device.cmd_draw_indexed_indirect(
                            command_buffer,
                            indirect_buffer.handle,
                            0,
                            max_indirect_commands,
                            std::mem::size_of::<vk::DrawIndexedIndirectCommand>() as u32,
                        );
                    }
                }

                device.cmd_end_rendering(command_buffer);
            }
            device.end_command_buffer(command_buffer).unwrap();
        }
    }

    pub fn record_lighting_commands(
        &self,
        renderer: &Renderer,
        command_buffer: vk::CommandBuffer,
        pc_bytes: &[u8],
        current_frame: usize,
    ) {
        let device = &renderer.device.device;
        let extent = renderer.get_extent();
        let global_ds = renderer.frames[current_frame].global_descriptor_set;

        unsafe {
            let color_formats = [vk::Format::R16G16B16A16_SFLOAT];
            let mut inheritance_info = vk::CommandBufferInheritanceRenderingInfo::default()
                .color_attachment_formats(&color_formats);
            let inherit = vk::CommandBufferInheritanceInfo::default().push_next(&mut inheritance_info);
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE | vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
                .inheritance_info(&inherit);

            device.begin_command_buffer(command_buffer, &begin_info).unwrap();

            if let Some(pipeline) = self.pipeline {
                let color_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(renderer.gbuffer.hdr[current_frame].view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.1, 0.1, 0.1, 1.0] } });

                let rendering_info = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent })
                    .layer_count(1)
                    .color_attachments(std::slice::from_ref(&color_attachment));

                device.cmd_begin_rendering(command_buffer, &rendering_info);
                device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);

                device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.layout,
                    0,
                    &[global_ds],
                    &[],
                );

                let viewport = vk::Viewport::default()
                    .x(0.0)
                    .y(0.0)
                    .width(extent.width as f32)
                    .height(extent.height as f32)
                    .min_depth(0.0)
                    .max_depth(1.0);
                let scissor = vk::Rect2D::default().extent(extent);
                device.cmd_set_viewport(command_buffer, 0, &[viewport]);
                device.cmd_set_scissor(command_buffer, 0, &[scissor]);

                if !self.descriptor_sets.is_empty() {
                    device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        self.layout,
                        1,
                        &[self.descriptor_sets[current_frame]],
                        &[],
                    );
                }

                device.cmd_push_constants(
                    command_buffer,
                    self.layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    0,
                    pc_bytes,
                );

                device.cmd_draw(command_buffer, 3, 1, 0, 0);
                device.cmd_end_rendering(command_buffer);
            }
            device.end_command_buffer(command_buffer).unwrap();
        }
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            if let Some(p) = self.pipeline {
                device.destroy_pipeline(p, None);
            }
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}
