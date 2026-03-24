use ash::{vk, Device};
use std::ffi::CString;

pub struct Pipeline {
    pub layout: vk::PipelineLayout,
    pub graphics_pipeline: vk::Pipeline,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
}

pub struct PostProcessPipeline {
    pub layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
}

pub struct PipelineCreateParams<'a> {
    pub extent: vk::Extent2D,
    pub vert_shader_code: &'a [u32],
    pub frag_shader_code: &'a [u32],
    pub msaa_samples: vk::SampleCountFlags,
    pub is_deferred_lighting: bool,
    pub input_attachments_count: u32,
    pub pipeline_cache: vk::PipelineCache,
    pub global_ds_layout: vk::DescriptorSetLayout,
    pub bindless_ds_layout: vk::DescriptorSetLayout,
}

impl Pipeline {
    pub fn new(
        device: &Device,
        params: &PipelineCreateParams,
    ) -> Self {
        let vert_shader_code = params.vert_shader_code;
        let frag_shader_code = params.frag_shader_code;
        let msaa_samples = params.msaa_samples;
        let is_deferred_lighting = params.is_deferred_lighting;
        let input_attachments_count = params.input_attachments_count;
        let pipeline_cache = params.pipeline_cache;
        let global_ds_layout = params.global_ds_layout;
        let bindless_ds_layout = params.bindless_ds_layout;
        let vert_shader_module = Self::create_shader_module(device, vert_shader_code);
        let frag_shader_module = Self::create_shader_module(device, frag_shader_code);

        let main_function_name = CString::new("main").unwrap();

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

        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::default();

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(if is_deferred_lighting {
                vk::CullModeFlags::NONE
            } else {
                vk::CullModeFlags::BACK
            })
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(false)
            .rasterization_samples(if is_deferred_lighting {
                vk::SampleCountFlags::TYPE_1
            } else {
                msaa_samples
            });

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(!is_deferred_lighting)
            .depth_write_enable(!is_deferred_lighting)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&dynamic_states);

        let mut color_blend_attachments = Vec::new();
        if !is_deferred_lighting {
            for _ in 0..5 {
                // Albedo, Normal, PBR, Velocity, HDR
                color_blend_attachments.push(
                    vk::PipelineColorBlendAttachmentState::default()
                        .color_write_mask(vk::ColorComponentFlags::RGBA)
                        .blend_enable(false),
                );
            }
        } else {
            // Lighting pass output (HDR)
            color_blend_attachments.push(
                vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(vk::ColorComponentFlags::RGBA)
                    .blend_enable(false),
            );
        }

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&color_blend_attachments)
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        let push_constant_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(128)];

        let mut descriptor_set_layout_bindings = Vec::new();
        if is_deferred_lighting {
            for i in 0..input_attachments_count {
                descriptor_set_layout_bindings.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                );
            }
            // Binding for Shadow Map
            descriptor_set_layout_bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(input_attachments_count)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            );
            // Binding for Light Buffer
            descriptor_set_layout_bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(input_attachments_count + 1)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            );
            // Binding for Object Data SSBO
            descriptor_set_layout_bindings.push(
                vk::DescriptorSetLayoutBinding::default()
                    .binding(input_attachments_count + 2)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            );
        }

        let descriptor_set_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(&descriptor_set_layout_bindings);

        let descriptor_set_layout = unsafe {
            device
                .create_descriptor_set_layout(&descriptor_set_layout_info, None)
                .expect("Failed to create descriptor set layout")
        };

        let set_layouts = [global_ds_layout, bindless_ds_layout, descriptor_set_layout];

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
            .push_constant_ranges(&push_constant_ranges)
            .set_layouts(&set_layouts);
        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .expect("Failed to create pipeline layout")
        };

        let mut rendering_info = vk::PipelineRenderingCreateInfo::default();
        let color_formats = if !is_deferred_lighting {
            vec![vk::Format::R8G8B8A8_UNORM, vk::Format::A2B10G10R10_UNORM_PACK32, vk::Format::R8G8B8A8_UNORM, vk::Format::R16G16_SFLOAT, vk::Format::R16G16B16A16_SFLOAT]
        } else {
            vec![vk::Format::R16G16B16A16_SFLOAT]
        };
        rendering_info = rendering_info.color_attachment_formats(&color_formats);
        if !is_deferred_lighting {
            rendering_info = rendering_info.depth_attachment_format(vk::Format::D32_SFLOAT);
        }

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_info)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisampling)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state_info)
            .layout(pipeline_layout)
            .push_next(&mut rendering_info);

        let graphics_pipelines = unsafe {
            device
                .create_graphics_pipelines(pipeline_cache, &[pipeline_info], None)
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

    pub fn create_shader_module(device: &Device, code: &[u32]) -> vk::ShaderModule {
        let create_info = vk::ShaderModuleCreateInfo::default().code(code);
        unsafe {
            device
                .create_shader_module(&create_info, None)
                .expect("Failed to create shader module")
        }
    }
}
