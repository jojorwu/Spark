use ash::vk;

pub struct EguiRenderer {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
    pub vertex_buffer: Option<crate::Buffer>,
    pub index_buffer: Option<crate::Buffer>,
    pub font_texture: Option<crate::vulkan::texture::Texture>,
}

impl EguiRenderer {
    pub fn new(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        vert_shader_code: &[u32],
        frag_shader_code: &[u32],
        extent: vk::Extent2D,
    ) -> Self {
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(std::slice::from_ref(&binding));

        let descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout(&layout_info, None).unwrap()
        };

        let push_constant_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(8)]; // vec2 screen_size

        let layouts = [descriptor_set_layout];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(&layouts)
            .push_constant_ranges(&push_constant_ranges);

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .unwrap()
        };

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .pool_sizes(&pool_sizes)
            .max_sets(1);
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(&pool_info, None).unwrap()
        };

        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);
        let descriptor_set = unsafe {
            device.allocate_descriptor_sets(&alloc_info).unwrap()[0]
        };

        let pipeline = Self::create_pipeline(
            device,
            render_pass,
            pipeline_layout,
            vert_shader_code,
            frag_shader_code,
            extent,
        );

        Self {
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            vertex_buffer: None,
            index_buffer: None,
            font_texture: None,
        }
    }

    pub fn draw(
        &mut self,
        device: &ash::Device,
        _graphics_queue: vk::Queue,
        command_buffer: vk::CommandBuffer,
        _full_output: egui::FullOutput,
        screen_size: [f32; 2],
    ) {
        // In a real implementation, we would use the shapes from full_output
        // and tessellate them here or in the caller.
        // Since we are in a simplified environment, we will focus on the pipeline setup.

        if self.pipeline == vk::Pipeline::null() {
            return;
        }

        unsafe {
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );

            let bytes = std::slice::from_raw_parts(
                screen_size.as_ptr() as *const u8,
                8,
            );
            device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytes,
            );

            // Here we would iterate over tessellated shapes and issue draw calls.
            // For now, we'll just ensure the pipeline state is correctly set up.
            log::debug!("Drawing Egui with screen size: {:?}", screen_size);
        }
    }

    fn create_pipeline(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        layout: vk::PipelineLayout,
        vert_code: &[u32],
        frag_code: &[u32],
        extent: vk::Extent2D,
    ) -> vk::Pipeline {
        let vert_module = {
            let info = vk::ShaderModuleCreateInfo::default().code(vert_code);
            unsafe { device.create_shader_module(&info, None).unwrap() }
        };
        let frag_module = {
            let info = vk::ShaderModuleCreateInfo::default().code(frag_code);
            unsafe { device.create_shader_module(&info, None).unwrap() }
        };

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

        let vertex_binding = vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(20) // pos: 8, tc: 8, color: 4
            .input_rate(vk::VertexInputRate::VERTEX);

        let vertex_attrs = [
            vk::VertexInputAttributeDescription::default()
                .location(0)
                .binding(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescription::default()
                .location(1)
                .binding(0)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(8),
            vk::VertexInputAttributeDescription::default()
                .location(2)
                .binding(0)
                .format(vk::Format::R8G8B8A8_UNORM)
                .offset(16),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(std::slice::from_ref(&vertex_binding))
            .vertex_attribute_descriptions(&vertex_attrs);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport = vk::Viewport::default()
            .width(extent.width as f32)
            .height(extent.height as f32)
            .max_depth(1.0);
        let scissor = vk::Rect2D::default().extent(extent);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(std::slice::from_ref(&viewport))
            .scissors(std::slice::from_ref(&scissor));

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .cull_mode(vk::CullModeFlags::NONE)
            .line_width(1.0);

        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::ONE)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_write_mask(vk::ColorComponentFlags::RGBA);

        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&color_blend_attachment));

        let info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0);

        let pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)
                .unwrap()[0]
        };

        unsafe {
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
        }

        pipeline
    }

    pub fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            if self.pipeline != vk::Pipeline::null() {
                device.destroy_pipeline(self.pipeline, None);
            }
        }
    }
}
