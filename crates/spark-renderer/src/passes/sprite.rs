use super::{RenderContext, RenderPass};
use crate::Renderer;
use ash::vk;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SpriteData {
    pub model: spark_math::Mat4,
    pub color: [f32; 4],
    pub size: [f32; 2],
    pub texture_idx: i32,
    pub padding: i32,
}

pub struct SpritePass {
    pub pipeline: Option<vk::Pipeline>,
    pub layout: vk::PipelineLayout,
    pub sprite_data: Vec<SpriteData>,
    pub sprite_buffer: Vec<crate::resource::Buffer>,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
}

impl RenderPass for SpritePass {
    fn name(&self) -> &str {
        "SpritePass"
    }

    fn prepare(&self, renderer: &Renderer, current_frame: usize) {
        if self.sprite_data.is_empty() {
            return;
        }
        renderer.upload_to_buffer(&self.sprite_buffer[current_frame], &self.sprite_data);

        let buf_info = [vk::DescriptorBufferInfo::default()
            .buffer(self.sprite_buffer[current_frame].handle)
            .range(self.sprite_buffer[current_frame].size)];
        let writes = [vk::WriteDescriptorSet::default()
            .dst_set(self.descriptor_sets[current_frame])
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&buf_info)];
        unsafe {
            renderer.device.device.update_descriptor_sets(&writes, &[]);
        }
    }

    fn outputs(&self) -> Vec<&'static str> {
        vec!["SpriteColor"]
    }

    fn gpu_resource_access(&self) -> Vec<(String, vk::AccessFlags, vk::PipelineStageFlags)> {
        vec![(
            "SpriteColor".to_string(),
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        )]
    }

    fn record_commands(&self, ctx: &RenderContext) {
        let renderer = ctx.renderer;
        let cf = ctx.current_frame;
        let extent = renderer.get_extent();

        let sprite_view = renderer
            .get_pass_resource_view("SpritePass", "SpriteColor", cf)
            .unwrap();

        unsafe {
            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(sprite_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0],
                    },
                });

            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                })
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color_attachment));

            renderer
                .device
                .device
                .cmd_begin_rendering(ctx.command_buffer, &rendering_info);

            if self.sprite_data.is_empty() {
                renderer.device.device.cmd_end_rendering(ctx.command_buffer);
                return;
            }

            let pipeline = match self.pipeline {
                Some(p) => p,
                None => {
                    renderer.device.device.cmd_end_rendering(ctx.command_buffer);
                    return;
                }
            };

            renderer.device.device.cmd_bind_pipeline(
                ctx.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            );

            let viewport = vk::Viewport::default()
                .width(extent.width as f32)
                .height(extent.height as f32)
                .max_depth(1.0);
            let scissor = vk::Rect2D::default().extent(extent);
            renderer
                .device
                .device
                .cmd_set_viewport(ctx.command_buffer, 0, &[viewport]);
            renderer
                .device
                .device
                .cmd_set_scissor(ctx.command_buffer, 0, &[scissor]);

            let vp = renderer.current_view_proj;
            let pc_bytes = std::slice::from_raw_parts(
                &vp as *const _ as *const u8,
                std::mem::size_of::<spark_math::Mat4>(),
            );
            renderer.device.device.cmd_push_constants(
                ctx.command_buffer,
                self.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                pc_bytes,
            );

            renderer.device.device.cmd_bind_descriptor_sets(
                ctx.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                &[
                    self.descriptor_sets[cf],
                    renderer.gpu_resource_manager.bindless.set,
                ],
                &[],
            );

            renderer.device.device.cmd_draw(
                ctx.command_buffer,
                6,
                self.sprite_data.len() as u32,
                0,
                0,
            );

            renderer.device.device.cmd_end_rendering(ctx.command_buffer);
        }
    }

    fn destroy(&mut self, renderer: &mut Renderer) {
        unsafe {
            if let Some(p) = self.pipeline {
                renderer.device.device.destroy_pipeline(p, None);
            }
            renderer
                .device
                .device
                .destroy_pipeline_layout(self.layout, None);
            renderer
                .device
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            for b in self.sprite_buffer.drain(..) {
                renderer.destroy_buffer(b);
            }
        }
    }
}

impl SpritePass {
    pub fn clear_sprites(&mut self) {
        self.sprite_data.clear();
    }

    pub fn add_sprite(&mut self, data: SpriteData) {
        self.sprite_data.push(data);
    }

    pub fn new(
        renderer: &Renderer,
        vert_spirv: &[u32],
        frag_spirv: &[u32],
    ) -> Result<Self, crate::error::RendererError> {
        let device = &renderer.device.device;

        let sprite_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let ds_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&[sprite_binding]),
                None,
            )?
        };

        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[ds_layout, renderer.gpu_resource_manager.bindless.layout])
                    .push_constant_ranges(&[vk::PushConstantRange {
                        stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                        offset: 0,
                        size: 64, // Mat4
                    }]),
                None,
            )?
        };

        let mut sprite_buffer = Vec::new();
        for _ in 0..crate::MAX_FRAMES_IN_FLIGHT {
            sprite_buffer.push(renderer.create_buffer(
                1024 * 1024, // 1MB for sprites
                vk::BufferUsageFlags::STORAGE_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ));
        }

        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(renderer.gpu_resource_manager.descriptor_pool)
                    .set_layouts(&[ds_layout; crate::MAX_FRAMES_IN_FLIGHT]),
            )?
        };

        let vert_module = crate::pipeline::Pipeline::create_shader_module(device, vert_spirv);
        let frag_module = crate::pipeline::Pipeline::create_shader_module(device, frag_spirv);
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
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(renderer.get_msaa_samples());

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);

        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&color_blend_attachment));

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(false)
            .depth_write_enable(false);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let color_formats = [vk::Format::R16G16B16A16_SFLOAT];
        let mut rendering_info =
            vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);

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

        let pipeline = unsafe {
            device
                .create_graphics_pipelines(renderer.pipeline_cache, &[info], None)
                .unwrap()[0]
        };

        unsafe {
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
        }

        Ok(Self {
            pipeline: Some(pipeline),
            layout,
            sprite_data: Vec::new(),
            sprite_buffer,
            descriptor_set_layout: ds_layout,
            descriptor_sets,
        })
    }
}
