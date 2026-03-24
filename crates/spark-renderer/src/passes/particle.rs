use ash::vk;
use crate::resource::Buffer;
use crate::Renderer;
use super::{RenderPass, RenderContext};

#[repr(C)]
pub struct Particle {
    pub pos: [f32; 4],
    pub vel: [f32; 4],
    pub color: [f32; 4],
    pub life: f32,
    pub size: f32,
    pub padding: [f32; 2],
}

pub struct ParticlePass {
    pub compute_pipeline: vk::Pipeline,
    pub graphics_pipeline: vk::Pipeline,
    pub compute_layout: vk::PipelineLayout,
    pub graphics_layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub particle_buffer: Buffer,
    pub particle_count: u32,
}

impl RenderPass for ParticlePass {
    fn name(&self) -> &str { "ParticlePass" }
    fn outputs(&self) -> Vec<&'static str> { vec!["ParticleColor"] }

    fn record_commands(&self, ctx: &RenderContext) {
        let renderer = ctx.renderer;
        let cf = ctx.current_frame;
        let device = &renderer.device.device;

        unsafe {
            // 1. Simulation
            device.cmd_bind_pipeline(ctx.command_buffer, vk::PipelineBindPoint::COMPUTE, self.compute_pipeline);
            device.cmd_bind_descriptor_sets(ctx.command_buffer, vk::PipelineBindPoint::COMPUTE, self.compute_layout, 0, &[self.descriptor_sets[cf]], &[]);

            let pc = [ctx.delta, self.particle_count as f32];
            device.cmd_push_constants(ctx.command_buffer, self.compute_layout, vk::ShaderStageFlags::COMPUTE, 0, bytemuck::cast_slice(&pc));

            device.cmd_dispatch(ctx.command_buffer, self.particle_count.div_ceil(256), 1, 1);

            let barrier = vk::BufferMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .src_access_mask(vk::AccessFlags2::SHADER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                .buffer(self.particle_buffer.handle)
                .size(self.particle_buffer.size);
            let dep_info = vk::DependencyInfo::default().buffer_memory_barriers(std::slice::from_ref(&barrier));
            device.cmd_pipeline_barrier2(ctx.command_buffer, &dep_info);

            // 2. Rendering
            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(renderer.get_pass_resource_view("", "GBufferHDR", cf).unwrap_or(renderer.common_shadow_view))
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE);

            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: renderer.get_extent() })
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color_attachment));

            device.cmd_begin_rendering(ctx.command_buffer, &rendering_info);
            device.cmd_bind_pipeline(ctx.command_buffer, vk::PipelineBindPoint::GRAPHICS, self.graphics_pipeline);
            device.cmd_bind_descriptor_sets(ctx.command_buffer, vk::PipelineBindPoint::GRAPHICS, self.graphics_layout, 0, &[self.descriptor_sets[cf]], &[]);
            device.cmd_bind_descriptor_sets(ctx.command_buffer, vk::PipelineBindPoint::GRAPHICS, self.graphics_layout, 1, &[renderer.frames[cf].global_descriptor_set], &[]);

            device.cmd_draw(ctx.command_buffer, self.particle_count * 4, 1, 0, 0);
            device.cmd_end_rendering(ctx.command_buffer);
        }
    }

    fn destroy(&mut self, renderer: &mut Renderer) {
        let device = &renderer.device.device;
        unsafe {
            device.destroy_pipeline(self.compute_pipeline, None);
            device.destroy_pipeline(self.graphics_pipeline, None);
            device.destroy_pipeline_layout(self.compute_layout, None);
            device.destroy_pipeline_layout(self.graphics_layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            renderer.destroy_buffer(self.particle_buffer.clone());
        }
    }
}

impl ParticlePass {
    pub fn new(renderer: &Renderer, comp_spirv: &[u32], vert_spirv: &[u32], frag_spirv: &[u32]) -> Result<Self, crate::error::RendererError> {
        let device = &renderer.device.device;
        let count = 10000;

        let particle_buffer = renderer.create_buffer(
            (count * std::mem::size_of::<Particle>()) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE | vk::ShaderStageFlags::VERTEX);
        let ds_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&[binding]), None)? };

        let ds = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default().descriptor_pool(renderer.descriptor_pool).set_layouts(&[ds_layout; crate::MAX_FRAMES_IN_FLIGHT])
            )?
        };

        for descriptor_set in ds.iter().take(crate::MAX_FRAMES_IN_FLIGHT) {
            let info = [vk::DescriptorBufferInfo::default().buffer(particle_buffer.handle).range(particle_buffer.size)];
            let write = [vk::WriteDescriptorSet::default().dst_set(*descriptor_set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&info)];
            unsafe { device.update_descriptor_sets(&write, &[]); }
        }

        let comp_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[ds_layout]).push_constant_ranges(&[vk::PushConstantRange::default().stage_flags(vk::ShaderStageFlags::COMPUTE).offset(0).size(8)]),
                None,
            )?
        };

        let graph_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[ds_layout, renderer.global_descriptor_set_layout]),
                None,
            )?
        };

        let comp_module = crate::pipeline::Pipeline::create_shader_module(device, comp_spirv);
        let compute_pipeline = unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default().stage(vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::COMPUTE).module(comp_module).name(c"main")).layout(comp_layout)],
                None,
            ).map_err(|e| e.1)?[0]
        };

        let vert_module = crate::pipeline::Pipeline::create_shader_module(device, vert_spirv);
        let frag_module = crate::pipeline::Pipeline::create_shader_module(device, frag_spirv);
        let entry_point = std::ffi::CString::new("main").unwrap();

        let stages = [
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vert_module).name(&entry_point),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(frag_module).name(&entry_point),
        ];

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE)
            .color_blend_op(vk::BlendOp::ADD);

        let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(renderer.get_msaa_samples());
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default().depth_test_enable(true).depth_write_enable(false).depth_compare_op(vk::CompareOp::LESS);
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&[vk::Format::R16G16B16A16_SFLOAT]).depth_attachment_format(vk::Format::D32_SFLOAT);

        let vi_info = vk::PipelineVertexInputStateCreateInfo::default();
        let ia_info = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_STRIP);
        let vs_info = vk::PipelineViewportStateCreateInfo::default().viewport_count(1).scissor_count(1);
        let rs_info = vk::PipelineRasterizationStateCreateInfo::default().cull_mode(vk::CullModeFlags::NONE).line_width(1.0);
        let attachments = [color_blend_attachment];
        let cb_info = vk::PipelineColorBlendStateCreateInfo::default().attachments(&attachments);
        let dynamics = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dy_info = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamics);

        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vi_info)
            .input_assembly_state(&ia_info)
            .viewport_state(&vs_info)
            .rasterization_state(&rs_info)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&cb_info)
            .dynamic_state(&dy_info)
            .layout(graph_layout)
            .push_next(&mut rendering_info);

        let graphics_pipeline = unsafe { device.create_graphics_pipelines(renderer.pipeline_cache, &[info], None).unwrap()[0] };

        unsafe {
            device.destroy_shader_module(comp_module, None);
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
        }

        Ok(Self {
            compute_pipeline,
            graphics_pipeline,
            compute_layout: comp_layout,
            graphics_layout: graph_layout,
            descriptor_set_layout: ds_layout,
            descriptor_sets: ds,
            particle_buffer,
            particle_count: count as u32,
        })
    }
}
