use ash::vk;
use crate::resource::Buffer;
use crate::{Renderer, MAX_FRAMES_IN_FLIGHT};
use super::{RenderPass, RenderContext};

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
            }
        );
        self.pipeline = Some(deferred_pipeline.graphics_pipeline);
    }

    pub fn update_descriptor_set_for_frame(
        &self,
        renderer: &Renderer,
        frame_idx: usize,
        params: &LightingDescriptorParams,
    ) {
        let i = frame_idx;
        let light_buffers = params.light_buffers;
        let object_data_buffers = params.object_data_buffers;
        let ssao_view = params.ssao_view;
        let irradiance_view = params.irradiance_view;
        let specular_view = params.specular_view;
        let brdf_lut_view = params.brdf_lut_view;
        let device = &renderer.device.device;
        let shadow_view = renderer.common_shadow_view;
        let shadow_sampler = renderer.common_sampler;

        let alb_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(renderer.gbuffer.albedo[i].view)];
        let norm_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(renderer.gbuffer.normal[i].view)];
        let pbr_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(renderer.gbuffer.pbr[i].view)];
        let depth_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(renderer.gbuffer.depth[i].view)];
        let shadow_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(shadow_view)
            .sampler(shadow_sampler)];

        let mut writes = vec![
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[i])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                .image_info(&alb_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[i])
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                .image_info(&norm_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[i])
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                .image_info(&pbr_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[i])
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::INPUT_ATTACHMENT)
                .image_info(&depth_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[i])
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&shadow_info),
        ];

        let mut buf_info = Vec::new();
        if let Some(lb) = light_buffers.get(i) {
            buf_info.push(
                vk::DescriptorBufferInfo::default()
                    .buffer(lb.handle)
                    .offset(0)
                    .range(lb.size),
            );
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(5)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&buf_info),
            );
        }

        let mut obj_info = Vec::new();
        if let Some(Some(ob)) = object_data_buffers.get(i) {
            obj_info.push(
                vk::DescriptorBufferInfo::default()
                    .buffer(ob.handle)
                    .offset(0)
                    .range(ob.size),
            );
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(6)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&obj_info),
            );
        }

        let ssao_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(ssao_view)
            .sampler(shadow_sampler)];
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[i])
                .dst_binding(7)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&ssao_info),
        );

        let irr_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(irradiance_view)
            .sampler(shadow_sampler)];
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[i])
                .dst_binding(8)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&irr_info),
        );

        let spec_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(specular_view)
            .sampler(shadow_sampler)];
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[i])
                .dst_binding(9)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&spec_info),
        );

        let brdf_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(brdf_lut_view)
            .sampler(shadow_sampler)];
        writes.push(
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[i])
                .dst_binding(10)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&brdf_info),
        );

        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
    }
}

impl RenderPass for LightingPass {
    fn name(&self) -> &str { "LightingPass" }
    fn dependencies(&self) -> Vec<&'static str> { vec!["GBufferPass", "SSAOPass", "ShadowPass", "ClusteredPass"] }

    fn update_descriptor_sets(&self, renderer: &Renderer) {
        let light_buffers: Vec<Buffer> = renderer.frames.iter().filter_map(|f| f.light_buffer.clone()).collect();
        let object_buffers: Vec<Option<Buffer>> = renderer.frames.iter().map(|f| f.object_data_buffer.clone()).collect();

        let irr_view = renderer.ibl_maps.as_ref().map(|m| m.irradiance_view).unwrap_or(renderer.common_shadow_view);
        let spec_view = renderer.ibl_maps.as_ref().map(|m| m.prefilter_view).unwrap_or(renderer.common_shadow_view);
        let brdf_view = renderer.ibl_maps.as_ref().map(|m| m.brdf_lut_view).unwrap_or(renderer.common_shadow_view);

        for i in 0..MAX_FRAMES_IN_FLIGHT {
            let mut ssao_view = renderer.common_shadow_view; // Fallback
            for pass in &renderer.render_passes {
                if pass.name() == "SSAOPass" {
                    if let Some(view) = pass.get_resource_view("ssao", i) {
                        ssao_view = view;
                    }
                }
            }

            let params = LightingDescriptorParams {
                light_buffers: &light_buffers,
                object_data_buffers: &object_buffers,
                ssao_view,
                irradiance_view: irr_view,
                specular_view: spec_view,
                brdf_lut_view: brdf_view,
            };
            self.update_descriptor_set_for_frame(
                renderer,
                i,
                &params,
            );
        }
    }

    fn record_commands(&self, ctx: &RenderContext) {
        let renderer = ctx.renderer;
        let command_buffer = ctx.command_buffer;
        let current_frame = ctx.current_frame;
        let extent = renderer.get_extent();
        let global_ds = renderer.frames[current_frame].global_descriptor_set;

        #[repr(C)]
        struct PC {
            count: u32,
            metallic: f32,
            roughness: f32,
            width: f32,
            height: f32,
            padding: u32,
            object_buffer_address: u64,
            prev_view_proj: spark_math::Mat4,
        }
        let pc = PC {
            count: renderer.light_count,
            metallic: 0.5,
            roughness: 0.5,
            width: extent.width as f32,
            height: extent.height as f32,
            padding: 0,
            object_buffer_address: renderer.frames[current_frame].object_data_buffer.as_ref().map_or(0, |b| b.address),
            prev_view_proj: renderer.prev_view_proj,
        };
        let pc_bytes = unsafe { std::slice::from_raw_parts(&pc as *const _ as *const u8, std::mem::size_of::<PC>()) };

        let device = &renderer.device.device;

        unsafe {
            if let Some(pipeline) = self.pipeline {
                let color_attachment = vk::RenderingAttachmentInfo::default()
                    .image_view(renderer.gbuffer.hdr[current_frame].view)
                    .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .clear_value(vk::ClearValue { color: vk::ClearColorValue { float32: [0.1, 0.1, 0.1, 1.0] } });

                let rendering_info = vk::RenderingInfo::default()
                    .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent })
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
