use ash::vk;
use crate::resource::{Attachment, Buffer};
use crate::Renderer;
use crate::vertex::{Vertex, InstanceData};
use crate::pipeline::Pipeline;

pub struct ShadowPass {
    pub render_pass: vk::RenderPass,
    pub framebuffer: vk::Framebuffer,
    pub pipeline: Option<vk::Pipeline>,
    pub layout: vk::PipelineLayout,
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
}

impl ShadowPass {
    pub fn new(
        device: &ash::Device,
        pdevice: vk::PhysicalDevice,
        instance: &ash::Instance,
    ) -> Result<Self, crate::error::RendererError> {
        let props = unsafe { instance.get_physical_device_memory_properties(pdevice) };
        let attachment = Attachment::create_image_resource(
            device,
            &props,
            Renderer::SHADOW_MAP_CASCADE_SIZE,
            Renderer::SHADOW_MAP_CASCADE_SIZE,
            vk::Format::D32_SFLOAT,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            vk::SampleCountFlags::TYPE_1,
        );

        let (image, memory, view) = (attachment.image, attachment.memory, attachment.view);

        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_BORDER)
                    .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR),
                None,
            )?
        };

        let render_pass = Self::create_render_pass(device)?;

        let framebuffer = unsafe {
            device.create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&[view])
                    .width(Renderer::SHADOW_MAP_CASCADE_SIZE)
                    .height(Renderer::SHADOW_MAP_CASCADE_SIZE)
                    .layers(1),
                None,
            )?
        };

        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&[
                    vk::PushConstantRange::default()
                        .stage_flags(vk::ShaderStageFlags::VERTEX)
                        .offset(0)
                        .size(128),
                ]),
                None,
            )?
        };

        Ok(Self {
            render_pass,
            framebuffer,
            pipeline: None,
            layout,
            image,
            memory,
            view,
            sampler,
        })
    }

    fn create_render_pass(device: &ash::Device) -> Result<vk::RenderPass, crate::error::RendererError> {
        let att = vk::AttachmentDescription::default()
            .format(vk::Format::D32_SFLOAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let r = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        let sub = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .depth_stencil_attachment(&r);
        let dep = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::FRAGMENT_SHADER)
            .dst_stage_mask(vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
            .src_access_mask(vk::AccessFlags::SHADER_READ)
            .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE);

        Ok(unsafe {
            device.create_render_pass(
                &vk::RenderPassCreateInfo::default()
                    .attachments(std::slice::from_ref(&att))
                    .subpasses(std::slice::from_ref(&sub))
                    .dependencies(std::slice::from_ref(&dep)),
                None,
            )?
        })
    }

    pub fn create_pipeline(
        &mut self,
        device: &ash::Device,
        pipeline_cache: vk::PipelineCache,
        vert_spirv: &[u32],
        frag_spirv: &[u32],
    ) {
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

        let binding_descriptions = [
            Vertex::get_binding_description(),
            InstanceData::get_binding_description(),
        ];
        let mut attribute_descriptions = Vec::new();
        attribute_descriptions.extend_from_slice(&Vertex::get_attribute_descriptions());
        attribute_descriptions.extend_from_slice(&InstanceData::get_attribute_descriptions());

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport = vk::Viewport::default()
            .width(Renderer::SHADOW_MAP_CASCADE_SIZE as f32)
            .height(Renderer::SHADOW_MAP_CASCADE_SIZE as f32)
            .max_depth(1.0);
        let scissor = vk::Rect2D::default().extent(vk::Extent2D {
            width: Renderer::SHADOW_MAP_CASCADE_SIZE,
            height: Renderer::SHADOW_MAP_CASCADE_SIZE,
        });
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(std::slice::from_ref(&viewport))
            .scissors(std::slice::from_ref(&scissor));

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .line_width(1.0)
            .depth_bias_enable(true)
            .depth_bias_constant_factor(1.25)
            .depth_bias_slope_factor(1.75);

        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);

        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .layout(self.layout)
            .render_pass(self.render_pass)
            .subpass(0);

        self.pipeline = Some(unsafe {
            device
                .create_graphics_pipelines(pipeline_cache, &[info], None)
                .unwrap()[0]
        });

        unsafe {
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
        }
    }

    pub fn record_commands(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        light_view_proj: spark_math::Mat4,
        renderables: &[(spark_math::Mat4, u32, Option<vk::ImageView>, Option<u32>)],
        instanced_renderables: &[(u32, u32, u32, Option<vk::ImageView>)],
        renderer: &Renderer,
    ) {
        let pipeline = match self.pipeline {
            Some(p) => p,
            None => return,
        };

        let clear = [vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        }];

        unsafe {
            device.cmd_begin_render_pass(
                command_buffer,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.render_pass)
                    .framebuffer(self.framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: Renderer::SHADOW_MAP_CASCADE_SIZE,
                            height: Renderer::SHADOW_MAP_CASCADE_SIZE,
                        },
                    })
                    .clear_values(&clear),
                vk::SubpassContents::INLINE,
            );

            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);

            let shadow_viewport = vk::Viewport::default()
                .width(Renderer::SHADOW_MAP_CASCADE_SIZE as f32)
                .height(Renderer::SHADOW_MAP_CASCADE_SIZE as f32)
                .min_depth(0.0)
                .max_depth(1.0);
            let shadow_scissor = vk::Rect2D::default().extent(vk::Extent2D {
                width: Renderer::SHADOW_MAP_CASCADE_SIZE,
                height: Renderer::SHADOW_MAP_CASCADE_SIZE,
            });
            device.cmd_set_viewport(command_buffer, 0, &[shadow_viewport]);
            device.cmd_set_scissor(command_buffer, 0, &[shadow_scissor]);

            let lvp_bytes = std::slice::from_raw_parts(&light_view_proj as *const _ as *const u8, 64);

            for (vb_id, ib_id, count, _) in instanced_renderables {
                if let (Some(vb), Some(ib)) =
                    (renderer.get_buffer(*vb_id), renderer.get_instance_buffer(*ib_id))
                {
                    device.cmd_bind_vertex_buffers(command_buffer, 0, &[vb.handle, ib.handle], &[0, 0]);
                    device.cmd_push_constants(
                        command_buffer,
                        self.layout,
                        vk::ShaderStageFlags::VERTEX,
                        0,
                        lvp_bytes,
                    );
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

            for (model, _, _, vb_id) in renderables {
                if let Some(vb_id) = vb_id {
                    if let Some(vb) = renderer.get_buffer(*vb_id) {
                        device.cmd_bind_vertex_buffers(command_buffer, 0, &[vb.handle, vb.handle], &[0, 0]);
                        let mvp = light_view_proj * (*model);
                        let mvp_bytes = std::slice::from_raw_parts(&mvp as *const _ as *const u8, 64);
                        device.cmd_push_constants(
                            command_buffer,
                            self.layout,
                            vk::ShaderStageFlags::VERTEX,
                            0,
                            mvp_bytes,
                        );
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

            device.cmd_end_render_pass(command_buffer);
        }
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            if let Some(p) = self.pipeline {
                device.destroy_pipeline(p, None);
            }
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_framebuffer(self.framebuffer, None);
            device.destroy_render_pass(self.render_pass, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}
