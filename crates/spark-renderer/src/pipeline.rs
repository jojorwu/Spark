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
    /// Creates a new graphics pipeline with the specified parameters.
    pub fn new(device: &Device, params: &PipelineCreateParams) -> Self {
        let (vert_module, frag_module) = Self::create_shader_modules(device, params);
        let entry_point = CString::new("main").expect("Failed to create entry point CString");

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

        let ds_layout = Self::create_pass_ds_layout(device, params);
        let layout = Self::create_pipeline_layout(device, params, ds_layout);

        let mut color_blend_attachments = Vec::new();
        let color_formats = if !params.is_deferred_lighting {
            vec![
                vk::Format::R8G8B8A8_UNORM,
                vk::Format::A2B10G10R10_UNORM_PACK32,
                vk::Format::R8G8B8A8_UNORM,
                vk::Format::R16G16_SFLOAT,
            ]
        } else {
            vec![vk::Format::R16G16B16A16_SFLOAT]
        };

        for _ in 0..color_formats.len() {
            color_blend_attachments.push(
                vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(vk::ColorComponentFlags::RGBA)
                    .blend_enable(false),
            );
        }

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::default()
            .cull_mode(if params.is_deferred_lighting { vk::CullModeFlags::NONE } else { vk::CullModeFlags::BACK })
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);

        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(if params.is_deferred_lighting { vk::SampleCountFlags::TYPE_1 } else { params.msaa_samples });

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(!params.is_deferred_lighting)
            .depth_write_enable(!params.is_deferred_lighting)
            .depth_compare_op(vk::CompareOp::LESS);

        let color_blending = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(&color_blend_attachments);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats);
        if !params.is_deferred_lighting {
            rendering_info = rendering_info.depth_attachment_format(vk::Format::D32_SFLOAT);
        }

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blending)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .push_next(&mut rendering_info);

        let pipeline = unsafe {
            device
                .create_graphics_pipelines(params.pipeline_cache, &[pipeline_info], None)
                .expect("Failed to create graphics pipeline")[0]
        };

        unsafe {
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
        }

        Self {
            layout,
            graphics_pipeline: pipeline,
            descriptor_set_layout: ds_layout,
        }
    }

    fn create_shader_modules(device: &Device, params: &PipelineCreateParams) -> (vk::ShaderModule, vk::ShaderModule) {
        (
            Self::create_shader_module(device, params.vert_shader_code),
            Self::create_shader_module(device, params.frag_shader_code),
        )
    }

    fn create_pass_ds_layout(device: &Device, params: &PipelineCreateParams) -> vk::DescriptorSetLayout {
        let mut bindings = Vec::new();
        if params.is_deferred_lighting {
            for i in 0..params.input_attachments_count {
                bindings.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i)
                        .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT),
                );
            }
            let next = params.input_attachments_count;
            bindings.push(vk::DescriptorSetLayoutBinding::default().binding(next).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::FRAGMENT));
            bindings.push(vk::DescriptorSetLayoutBinding::default().binding(next + 1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::FRAGMENT));
            bindings.push(vk::DescriptorSetLayoutBinding::default().binding(next + 2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT));
        }

        unsafe {
            device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)
                .expect("Failed to create pass descriptor set layout")
        }
    }

    fn create_pipeline_layout(device: &Device, params: &PipelineCreateParams, ds_layout: vk::DescriptorSetLayout) -> vk::PipelineLayout {
        let push_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .size(128)];
        let layouts = [params.global_ds_layout, params.bindless_ds_layout, ds_layout];

        unsafe {
            device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts).push_constant_ranges(&push_ranges), None)
                .expect("Failed to create pipeline layout")
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
