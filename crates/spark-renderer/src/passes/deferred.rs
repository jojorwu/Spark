use ash::vk;
use crate::resource::{Attachment, Buffer};
use crate::{Renderer, MAX_FRAMES_IN_FLIGHT};

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
                        .size(32)]),
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

        // Note: We need to store ds_layout to destroy it later,
        // but for now we'll assume it's managed or we'll refactor later.
        // Actually, let's include it in the struct.

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

            unsafe {
                device.update_descriptor_sets(&writes, &[]);
            }
        }
    }

    pub fn record_commands(
        &self,
        renderer: &Renderer,
        command_buffer: vk::CommandBuffer,
        renderables: &[(spark_math::Mat4, u32, Option<vk::ImageView>, Option<u32>)],
        instanced_renderables: &[(u32, u32, u32, Option<vk::ImageView>)],
        pc_bytes: &[u8],
        current_frame: usize,
    ) {
        let device = &renderer.device.device;
        let extent = renderer.get_extent();
        let global_ds = renderer.global_descriptor_sets[current_frame];

        unsafe {
            // Subpass 0: G-Buffer Generation
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

                for (vb_id, ib_id, count, tex_view) in instanced_renderables {
                    if let (Some(vb), Some(ib)) =
                        (renderer.get_buffer(*vb_id), renderer.get_instance_buffer(*ib_id))
                    {
                        let ds = tex_view
                            .and_then(|v| renderer.texture_descriptor_sets.get(&v))
                            .unwrap_or(&renderer.default_descriptor_set);

                        device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline.layout,
                            0,
                            &[global_ds, *ds],
                            &[],
                        );

                        device.cmd_bind_vertex_buffers(command_buffer, 0, &[vb.handle, ib.handle], &[0, 0]);
                        if let Some(idx_b) = renderer.index_buffer {
                            device.cmd_bind_index_buffer(command_buffer, idx_b.handle, 0, vk::IndexType::UINT32);
                            device.cmd_draw_indexed(command_buffer, (idx_b.size / 4) as u32, *count, 0, 0, 0);
                        } else {
                            device.cmd_draw(
                                command_buffer,
                                (vb.size / std::mem::size_of::<crate::vertex::Vertex>() as u64) as u32,
                                *count,
                                0,
                                0,
                            );
                        }
                    }
                }

                for (_model, _, tex_view, vb_id) in renderables {
                    if let Some(vb_id) = vb_id {
                        if let Some(vb) = renderer.get_buffer(*vb_id) {
                        let ds = tex_view
                            .and_then(|v| renderer.texture_descriptor_sets.get(&v))
                            .unwrap_or(&renderer.default_descriptor_set);

                        device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline.layout,
                            0,
                            &[global_ds, *ds],
                            &[],
                        );

                        device.cmd_bind_vertex_buffers(command_buffer, 0, &[vb.handle], &[0]);
                            if let Some(idx_b) = renderer.index_buffer {
                                device.cmd_bind_index_buffer(command_buffer, idx_b.handle, 0, vk::IndexType::UINT32);
                                device.cmd_draw_indexed(command_buffer, (idx_b.size / 4) as u32, 1, 0, 0, 0);
                            } else {
                                device.cmd_draw(
                                    command_buffer,
                                    (vb.size / std::mem::size_of::<crate::vertex::Vertex>() as u64) as u32,
                                    1,
                                    0,
                                    0,
                                );
                            }
                        }
                    }
                }
            }

            // Transition to Subpass 1: Lighting resolve
            device.cmd_next_subpass(command_buffer, vk::SubpassContents::INLINE);

            if let Some(pipeline) = self.pipeline {
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
            }
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
