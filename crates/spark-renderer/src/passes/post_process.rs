use ash::vk;
use crate::resource::Attachment;
use crate::MAX_FRAMES_IN_FLIGHT;
use crate::pipeline::Pipeline;

pub struct PostProcessPass {
    pub pipeline: Option<vk::Pipeline>,
    pub bloom_pipeline: Option<vk::Pipeline>,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub bloom_images: Vec<vk::Image>,
    pub bloom_memories: Vec<vk::DeviceMemory>,
    pub bloom_views: Vec<vk::ImageView>,
    pub swapchain_format: vk::Format,
}

impl PostProcessPass {
    pub fn new(
        device: &ash::Device,
        pdevice: vk::PhysicalDevice,
        instance: &ash::Instance,
        format: vk::Format,
        extent: vk::Extent2D,
    ) -> Result<Self, crate::error::RendererError> {
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
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
                &vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&ds_layout)),
                None,
            )?
        };

        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(MAX_FRAMES_IN_FLIGHT as u32 * 2),
        ];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .pool_sizes(&pool_sizes)
                    .max_sets(MAX_FRAMES_IN_FLIGHT as u32),
                None,
            )?
        };

        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(descriptor_pool)
                    .set_layouts(&vec![ds_layout; MAX_FRAMES_IN_FLIGHT]),
            )?
        };

        let mut bloom_images = Vec::new();
        let mut bloom_memories = Vec::new();
        let mut bloom_views = Vec::new();
        let props = unsafe { instance.get_physical_device_memory_properties(pdevice) };
        for i in 1..6 {
            let att = Attachment::create_image_resource(
                device,
                &props,
                (extent.width >> i).max(1),
                (extent.height >> i).max(1),
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::INPUT_ATTACHMENT,
                vk::SampleCountFlags::TYPE_1,
            );
            bloom_images.push(att.image);
            bloom_memories.push(att.memory);
            bloom_views.push(att.view);
        }

        Ok(Self {
            pipeline: None,
            bloom_pipeline: None,
            layout,
            descriptor_set_layout: ds_layout,
            descriptor_pool,
            descriptor_sets,
            bloom_images,
            bloom_memories,
            bloom_views,
            swapchain_format: format,
        })
    }

    pub fn create_pipelines(
        &mut self,
        device: &ash::Device,
        pipeline_cache: vk::PipelineCache,
        extent: vk::Extent2D,
        vert_spirv: &[u32],
        frag_spirv: &[u32],
        bloom_frag_spirv: &[u32],
    ) {
        let vert_module = Pipeline::create_shader_module(device, vert_spirv);
        let frag_module = Pipeline::create_shader_module(device, frag_spirv);
        let bloom_frag_module = Pipeline::create_shader_module(device, bloom_frag_spirv);
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
        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default().cull_mode(vk::CullModeFlags::BACK).front_face(vk::FrontFace::CLOCKWISE).line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default().color_write_mask(vk::ColorComponentFlags::RGBA).blend_enable(false);
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(std::slice::from_ref(&color_blend_attachment));

        let color_formats = [self.swapchain_format];
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats);

        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .layout(self.layout)
            .push_next(&mut rendering_info);

        self.pipeline = Some(unsafe { device.create_graphics_pipelines(pipeline_cache, &[info], None).unwrap()[0] });

        let bloom_stages = [
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::VERTEX).module(vert_module).name(&entry_point),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::FRAGMENT).module(bloom_frag_module).name(&entry_point),
        ];

        let bloom_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&bloom_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .layout(self.layout)
            .push_next(&mut rendering_info);

        self.bloom_pipeline = Some(unsafe { device.create_graphics_pipelines(pipeline_cache, &[bloom_info], None).unwrap()[0] });

        unsafe {
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
            device.destroy_shader_module(bloom_frag_module, None);
        }
    }

    pub fn update_descriptor_sets(
        &self,
        device: &ash::Device,
        hdr_attachments: &[Attachment],
        sampler: vk::Sampler,
    ) {
        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let img_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(hdr_attachments[i].view)
                .sampler(sampler)];
            let blm_info = [vk::DescriptorImageInfo::default()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(self.bloom_views[0])
                .sampler(sampler)];
            let writes = [
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&img_info),
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&blm_info),
            ];
            unsafe {
                device.update_descriptor_sets(&writes, &[]);
            }
        }
    }

    pub fn record_commands(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        _image_index: u32,
        current_frame: usize,
        swapchain_image_view: vk::ImageView,
        swapchain_image: vk::Image,
        extent: vk::Extent2D,
    ) {
        let pipeline = match self.pipeline {
            Some(p) => p,
            None => return,
        };

        unsafe {
            let barrier = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(swapchain_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(swapchain_image_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] } });

            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent })
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color_attachment));

            device.cmd_begin_rendering(command_buffer, &rendering_info);

            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);

            let viewport = vk::Viewport::default()
                .width(extent.width as f32)
                .height(extent.height as f32)
                .max_depth(1.0);
            let scissor = vk::Rect2D::default().extent(extent);
            device.cmd_set_viewport(command_buffer, 0, &[viewport]);
            device.cmd_set_scissor(command_buffer, 0, &[scissor]);

            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.layout,
                0,
                &[self.descriptor_sets[current_frame]],
                &[],
            );

            device.cmd_draw(command_buffer, 3, 1, 0, 0);
            device.cmd_end_rendering(command_buffer);
        }
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            if let Some(p) = self.pipeline {
                device.destroy_pipeline(p, None);
            }
            if let Some(p) = self.bloom_pipeline {
                device.destroy_pipeline(p, None);
            }
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            for view in self.bloom_views.drain(..) {
                device.destroy_image_view(view, None);
            }
            for img in self.bloom_images.drain(..) {
                device.destroy_image(img, None);
            }
            for mem in self.bloom_memories.drain(..) {
                device.free_memory(mem, None);
            }
        }
    }
}
