use ash::{vk, Device};
use std::ffi::CString;

pub struct Pipeline {
    pub layout: vk::PipelineLayout,
    pub graphics_pipeline: vk::Pipeline,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
}

impl Pipeline {
    pub fn new(
        device: &Device,
        render_pass: vk::RenderPass,
        extent: vk::Extent2D,
        vert_shader_code: &[u32],
        frag_shader_code: &[u32],
    ) -> Self {
        let vert_shader_module = Self::create_shader_module(device, vert_shader_code);
        let frag_shader_module = Self::create_shader_module(device, frag_shader_code);

        let main_function_name = CString::new("main").unwrap();

        use crate::vertex::{Vertex, InstanceData};
        let binding_descriptions = [
            Vertex::get_binding_description(),
            InstanceData::get_binding_description(),
        ];
        let mut attribute_descriptions = Vec::new();
        attribute_descriptions.extend_from_slice(&Vertex::get_attribute_descriptions());
        attribute_descriptions.extend_from_slice(&InstanceData::get_attribute_descriptions());

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(vert_shader_module)
                .name(&main_function_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(frag_shader_module)
                .name(&main_function_name),
        ];

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewports = [vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(extent.width as f32)
            .height(extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0)];

        let scissors = [vk::Rect2D::default()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(extent)];

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewports(&viewports)
            .scissors(&scissors);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let color_blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(false);

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(std::slice::from_ref(&color_blend_attachment))
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        let push_constant_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size((std::mem::size_of::<spark_math::Mat4>() * 2) as u32)];

        let descriptor_set_layout_bindings = [
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

        let descriptor_set_layout_info = vk::DescriptorSetLayoutCreateInfo::default()
            .bindings(&descriptor_set_layout_bindings);

        let descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(&descriptor_set_layout_info, None)
                .expect("Failed to create descriptor set layout")
        };

        let set_layouts = [descriptor_set_layout];

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(&push_constant_ranges)
            .set_layouts(&set_layouts);
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .expect("Failed to create pipeline layout")
        };

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blending)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);

        let graphics_pipelines = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .expect("Failed to create graphics pipeline")
        };

        unsafe {
            device.destroy_shader_module(vert_shader_module, None);
            device.destroy_shader_module(frag_shader_module, None);
        }

        Self {
            layout: pipeline_layout,
            graphics_pipeline: graphics_pipelines[0],
            descriptor_set_layout,
        }
    }

    fn create_shader_module(device: &Device, code: &[u32]) -> vk::ShaderModule {
        let create_info = vk::ShaderModuleCreateInfo::default().code(code);
        unsafe {
            device
                .create_shader_module(&create_info, None)
                .expect("Failed to create shader module")
        }
    }
}
