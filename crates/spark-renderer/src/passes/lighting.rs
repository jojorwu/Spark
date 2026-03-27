use super::{RenderContext, RenderPass};
use crate::resource::Buffer;
use crate::{Renderer, MAX_FRAMES_IN_FLIGHT};
use ash::vk;

pub struct LightingDescriptorParams<'a> {
    pub light_buffers: &'a [Buffer],
    pub object_data_buffers: &'a [Option<Buffer>],
    pub ssao_view: vk::ImageView,
    pub irradiance_view: vk::ImageView,
    pub specular_view: vk::ImageView,
    pub brdf_lut_view: vk::ImageView,
}

pub struct LightingPipelineParams<'a> {
    pub device: &'a ash::Device,
    pub pipeline_cache: vk::PipelineCache,
    pub extent: vk::Extent2D,
    pub vert_spirv: &'a [u32],
    pub frag_spirv: &'a [u32],
    pub msaa_samples: vk::SampleCountFlags,
    pub global_ds_layout: vk::DescriptorSetLayout,
    pub bindless_ds_layout: vk::DescriptorSetLayout,
}

pub struct LightingPass {
    pub pipeline: Option<vk::Pipeline>,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
}

impl LightingPass {
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
            vk::DescriptorSetLayoutBinding::default()
                .binding(6)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(7)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(8)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(9)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(10)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT),
            vk::DescriptorSetLayoutBinding::default()
                .binding(11)
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
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[global_ds_layout, ds_layout])
                    .push_constant_ranges(&[vk::PushConstantRange::default()
                        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
                        .offset(0)
                        .size(128)]),
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

        Ok(Self {
            pipeline: None,
            layout,
            descriptor_set_layout: ds_layout,
            descriptor_sets,
        })
    }

    pub fn create_pipeline(&mut self, params: LightingPipelineParams) {
        let deferred_pipeline = crate::pipeline::Pipeline::new(
            params.device,
            &crate::pipeline::PipelineCreateParams {
                extent: params.extent,
                vert_shader_code: params.vert_spirv,
                frag_shader_code: params.frag_spirv,
                msaa_samples: params.msaa_samples,
                is_deferred_lighting: true,
                input_attachments_count: 4,
                pipeline_cache: params.pipeline_cache,
                global_ds_layout: params.global_ds_layout,
                bindless_ds_layout: params.bindless_ds_layout,
            },
        );
        self.pipeline = Some(deferred_pipeline.graphics_pipeline);
    }

}

impl RenderPass for LightingPass {
    fn name(&self) -> &str {
        "LightingPass"
    }

    fn inputs(&self) -> Vec<&'static str> {
        vec![
            "GBufferAlbedo",
            "GBufferNormal",
            "GBufferPBR",
            "GBufferDepth",
            "ShadowMap",
            "SSAO",
        ]
    }

    fn outputs(&self) -> Vec<&'static str> {
        vec!["GBufferHDR"]
    }

    fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }

    fn set_descriptor_sets(&mut self, sets: Vec<vk::DescriptorSet>) {
        self.descriptor_sets = sets;
    }

    fn dependencies(&self) -> Vec<&'static str> {
        vec!["GBufferPass", "SSAOPass", "ShadowPass", "ClusteredPass"]
    }

    fn bindings(&self) -> Vec<super::ResourceBinding> {
        vec![
            super::ResourceBinding::InputAttachment(0, "GBufferAlbedo".to_string()),
            super::ResourceBinding::InputAttachment(1, "GBufferNormal".to_string()),
            super::ResourceBinding::InputAttachment(2, "GBufferPBR".to_string()),
            super::ResourceBinding::InputAttachment(3, "GBufferDepth".to_string()),
            super::ResourceBinding::SampledImage(4, "ShadowMap".to_string()),
            super::ResourceBinding::StorageBuffer(5, "light_buffer".to_string()),
            super::ResourceBinding::StorageBuffer(6, "object_data_buffer".to_string()),
            super::ResourceBinding::SampledImage(7, "SSAO".to_string()),
            super::ResourceBinding::SampledImage(8, "irradiance".to_string()),
            super::ResourceBinding::SampledImage(9, "specular".to_string()),
            super::ResourceBinding::SampledImage(10, "brdf_lut".to_string()),
            super::ResourceBinding::SampledImage(11, "SSGI".to_string()),
        ]
    }

    fn record_commands(&self, ctx: &RenderContext) {
        let renderer = ctx.renderer;
        let command_buffer = ctx.command_buffer;
        let current_frame = ctx.current_frame;
        let extent = renderer.get_extent();
        let global_ds = renderer.frame_manager.frames[current_frame].global_descriptor_set;

        #[repr(C)]
        struct PC {
            count: u32,
            metallic: f32,
            roughness: f32,
            width: f32,
            height: f32,
            ssgi_intensity: f32,
            shadow_pcf: u32,
            object_buffer_address: u64,
            prev_view_proj: spark_math::Mat4,
        }
        let pc = PC {
            count: renderer.light_count,
            metallic: 0.5,
            roughness: 0.5,
            width: extent.width as f32,
            height: extent.height as f32,
            ssgi_intensity: if renderer.settings.enable_ssgi {
                renderer.settings.ssgi_intensity
            } else {
                0.0
            },
            shadow_pcf: renderer.settings.shadow_pcf_samples,
            object_buffer_address: renderer.frame_manager.frames[current_frame]
                .object_data_buffer
                .as_ref()
                .map_or(0, |b| b.address),
            prev_view_proj: renderer.prev_view_proj,
        };
        let pc_bytes = unsafe {
            std::slice::from_raw_parts(&pc as *const _ as *const u8, std::mem::size_of::<PC>())
        };

        let device = &renderer.device.device;

        unsafe {
            if let Some(pipeline) = self.pipeline {
                let color_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(
                        renderer
                            .get_pass_resource_view("", "GBufferHDR", current_frame)
                            .unwrap_or(renderer.common_shadow_view),
                    )
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.1, 0.1, 0.1, 1.0],
                        },
                    });

                let rendering_info = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent,
                    })
                    .layer_count(1)
                    .color_attachments(std::slice::from_ref(&color_attachment));

                device.cmd_begin_rendering(command_buffer, &rendering_info);
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
                device.cmd_end_rendering(command_buffer);
            }
        }
    }

    fn destroy(&mut self, renderer: &mut Renderer) {
        unsafe {
            let device = &renderer.device.device;
            if let Some(p) = self.pipeline {
                device.destroy_pipeline(p, None);
            }
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}
