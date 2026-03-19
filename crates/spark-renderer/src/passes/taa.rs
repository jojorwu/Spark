use ash::vk;
use crate::resource::Attachment;
use crate::Renderer;
use crate::MAX_FRAMES_IN_FLIGHT;
use super::RenderContext;

pub struct TAAPass {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub history_images: Vec<Attachment>,
}

use super::RenderPass;

impl RenderPass for TAAPass {
    fn name(&self) -> &str { "TAAPass" }
    fn is_enabled(&self, renderer: &Renderer) -> bool { renderer.enable_taa }
    fn prepare(&self, renderer: &Renderer, current_frame: usize) {
        let sampler = renderer.common_sampler;
        let prev_idx = (current_frame + MAX_FRAMES_IN_FLIGHT - 1) % MAX_FRAMES_IN_FLIGHT;
        let current_info = [vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).image_view(renderer.gbuffer.hdr[current_frame].view).sampler(sampler)];
        let history_info = [vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).image_view(self.history_images[prev_idx].view).sampler(sampler)];
        let velocity_info = [vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).image_view(renderer.gbuffer.velocity[current_frame].view).sampler(sampler)];
        let depth_info = [vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).image_view(renderer.gbuffer.depth[current_frame].view).sampler(sampler)];

        let writes = [
            vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[current_frame]).dst_binding(0).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&current_info),
            vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[current_frame]).dst_binding(1).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&history_info),
            vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[current_frame]).dst_binding(2).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&velocity_info),
            vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[current_frame]).dst_binding(3).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&depth_info),
        ];
        unsafe { renderer.device.device.update_descriptor_sets(&writes, &[]); }
    }

    fn update_descriptor_sets(&self, _renderer: &Renderer) {
        // Handled in prepare
    }

    fn record_commands(&self, ctx: &RenderContext) {
        self.record_commands_impl(
            &ctx.renderer.device.device,
            ctx.command_buffer,
            ctx.renderer.swapchain.extent,
            ctx.current_frame,
        );
    }

    fn get_resource_view(&self, name: &str, frame_index: usize) -> Option<vk::ImageView> {
        if name == "history" {
            Some(self.history_images[frame_index].view)
        } else {
            None
        }
    }

    fn destroy(&mut self, renderer: &Renderer) {
        let device = &renderer.device.device;
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            for img in self.history_images.drain(..) {
                img.destroy(device);
            }
        }
    }
}

impl TAAPass {
    pub fn new(
        renderer: &Renderer,
        shader: &[u32],
        vert_shader: &[u32],
    ) -> Result<Self, crate::error::RendererError> {
        let device = &renderer.device.device;
        let extent = renderer.get_extent();
        let props = unsafe { renderer.context.instance.get_physical_device_memory_properties(renderer.device.pdevice) };

        let history_images = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| {
                Attachment::create_image_resource(
                    device,
                    &props,
                    extent.width,
                    extent.height,
                    vk::Format::R16G16B16A16_SFLOAT,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST,
                    vk::SampleCountFlags::TYPE_1,
                )
            })
            .collect();

        let bindings = [
            vk::DescriptorSetLayoutBinding::default().binding(0).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default().binding(1).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default().binding(2).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default().binding(3).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::FRAGMENT),
        ];

        let ds_layout = unsafe { device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)? };
        let layout = unsafe { device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&[ds_layout]), None)? };

        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(renderer.descriptor_pool)
                    .set_layouts(&[ds_layout; MAX_FRAMES_IN_FLIGHT]),
            )?
        };

        let vert_module = crate::pipeline::Pipeline::create_shader_module(device, vert_shader);
        let frag_module = crate::pipeline::Pipeline::create_shader_module(device, shader);
        let entry_point = std::ffi::CString::new("main").unwrap();

        let stages = [
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vert_module).name(&entry_point),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(frag_module).name(&entry_point),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = vk::Viewport::default().width(extent.width as f32).height(extent.height as f32).max_depth(1.0);
        let scissor = vk::Rect2D::default().extent(extent);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default().viewports(std::slice::from_ref(&viewport)).scissors(std::slice::from_ref(&scissor));
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default().cull_mode(vk::CullModeFlags::NONE).line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA).blend_enable(false);
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(std::slice::from_ref(&color_blend_attachment));

        let color_formats = [vk::Format::R16G16B16A16_SFLOAT];
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default().color_attachment_formats(&color_formats);

        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .layout(layout)
            .push_next(&mut rendering_info);

        let pipeline = unsafe { device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None).unwrap()[0] };

        unsafe {
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
        }

        Ok(Self {
            pipeline,
            layout,
            descriptor_set_layout: ds_layout,
            descriptor_sets,
            history_images,
        })
    }

    pub fn update_descriptor_sets(
        &self,
        device: &ash::Device,
        current_frame_attachments: &[Attachment],
        velocity_attachments: &[Attachment],
        depth_attachments: &[Attachment],
        sampler: vk::Sampler,
    ) {
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let prev_idx = (i + MAX_FRAMES_IN_FLIGHT - 1) % MAX_FRAMES_IN_FLIGHT;
            let current_info = [vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).image_view(current_frame_attachments[i].view).sampler(sampler)];
            let history_info = [vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).image_view(self.history_images[prev_idx].view).sampler(sampler)];
            let velocity_info = [vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).image_view(velocity_attachments[i].view).sampler(sampler)];
            let depth_info = [vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).image_view(depth_attachments[i].view).sampler(sampler)];

            let writes = [
                vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[i]).dst_binding(0).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&current_info),
                vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[i]).dst_binding(1).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&history_info),
                vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[i]).dst_binding(2).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&velocity_info),
                vk::WriteDescriptorSet::default().dst_set(self.descriptor_sets[i]).dst_binding(3).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&depth_info),
            ];
            unsafe { device.update_descriptor_sets(&writes, &[]); }
        }
    }

    pub fn record_commands_impl(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        extent: vk::Extent2D,
        current_frame: usize,
    ) {
        unsafe {
            let history_barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(self.history_images[current_frame].image)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
            device.cmd_pipeline_barrier(command_buffer, vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::DependencyFlags::empty(), &[], &[], &[history_barrier]);

            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(self.history_images[current_frame].view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } });

            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent })
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color_attachment));

            device.cmd_begin_rendering(command_buffer, &rendering_info);
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_bind_descriptor_sets(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, &[self.descriptor_sets[current_frame]], &[]);
            device.cmd_draw(command_buffer, 3, 1, 0, 0);
            device.cmd_end_rendering(command_buffer);

            let to_shader_barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(self.history_images[current_frame].image)
                .subresource_range(vk::ImageSubresourceRange { aspect_mask: vk::ImageAspectFlags::COLOR, base_mip_level: 0, level_count: 1, base_array_layer: 0, layer_count: 1 });
            device.cmd_pipeline_barrier(command_buffer, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::PipelineStageFlags::FRAGMENT_SHADER, vk::DependencyFlags::empty(), &[], &[], &[to_shader_barrier]);
        }
    }

    pub fn destroy(&mut self, renderer: &Renderer) {
        let device = &renderer.device.device;
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            for img in self.history_images.drain(..) {
                img.destroy(device);
            }
        }
    }
}
