use super::{RenderContext, RenderPass};
use crate::resource::Attachment;
use crate::Renderer;
use crate::MAX_FRAMES_IN_FLIGHT;
use ash::vk;

pub struct SSRPass {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub output_images: Vec<Attachment>,
}

impl RenderPass for SSRPass {
    fn name(&self) -> &str {
        "SSRPass"
    }
    fn inputs(&self) -> Vec<&'static str> {
        vec![
            "GBufferAlbedo",
            "GBufferNormal",
            "GBufferPBR",
            "GBufferDepth",
            "GBufferHDR",
            "HiZ",
        ]
    }
    fn outputs(&self) -> Vec<&'static str> {
        vec!["SSR"]
    }

    fn declared_resources(&self) -> std::collections::HashMap<String, super::ResourceDesc> {
        let mut res = std::collections::HashMap::new();
        res.insert(
            "SSR".to_string(),
            super::ResourceDesc::Image(super::AttachmentDesc {
                format: vk::Format::R16G16B16A16_SFLOAT,
                usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                size: super::AttachmentSize::Relative(1.0, 1.0),
            }),
        );
        res
    }

    fn bindings(&self) -> Vec<super::ResourceBinding> {
        vec![
            super::ResourceBinding::SampledImage(0, "GBufferAlbedo".to_string()),
            super::ResourceBinding::SampledImage(1, "GBufferNormal".to_string()),
            super::ResourceBinding::SampledImage(2, "GBufferPBR".to_string()),
            super::ResourceBinding::SampledImage(3, "GBufferDepth".to_string()),
            super::ResourceBinding::SampledImage(4, "GBufferHDR".to_string()),
            super::ResourceBinding::SampledImage(5, "HiZ".to_string()),
            super::ResourceBinding::StorageImage(6, "SSR".to_string()),
        ]
    }

    fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }

    fn set_descriptor_sets(&mut self, sets: Vec<vk::DescriptorSet>) {
        self.descriptor_sets = sets;
    }

    fn record_commands(&self, ctx: &RenderContext) {
        if self.pipeline == vk::Pipeline::null() {
            return;
        }
        let renderer = ctx.renderer;
        let device = &renderer.device.device;
        let extent = renderer.get_extent();

        unsafe {
            device.cmd_bind_pipeline(
                ctx.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                ctx.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[
                    renderer.frame_manager.frames[ctx.current_frame].global_descriptor_set,
                    self.descriptor_sets[ctx.current_frame],
                ],
                &[],
            );

            #[repr(C)]
            struct Ssrpc {
                width: u32,
                height: u32,
                max_steps: u32,
                step_size: f32,
                thickness: f32,
                enabled: f32,
            }
            let pc = Ssrpc {
                width: extent.width,
                height: extent.height,
                max_steps: renderer.settings.ssr_max_steps,
                step_size: renderer.settings.ssr_step,
                thickness: renderer.settings.ssr_thickness,
                enabled: if renderer.settings.enable_ssr {
                    1.0
                } else {
                    0.0
                },
            };
            let pc_bytes = std::slice::from_raw_parts(
                &pc as *const _ as *const u8,
                std::mem::size_of::<Ssrpc>(),
            );
            device.cmd_push_constants(
                ctx.command_buffer,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );

            device.cmd_dispatch(
                ctx.command_buffer,
                extent.width.div_ceil(16),
                extent.height.div_ceil(16),
                1,
            );
        }
    }

    fn on_resize(&mut self, renderer: &Renderer, new_extent: vk::Extent2D) {
        for img in self.output_images.drain(..) {
            img.destroy(&renderer.device.device, &renderer.device.allocator);
        }
        self.output_images = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| {
                Attachment::create_image_resource(
                    &renderer.device,
                    new_extent.width,
                    new_extent.height,
                    vk::Format::R16G16B16A16_SFLOAT,
                    vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                    vk::SampleCountFlags::TYPE_1,
                )
                .unwrap()
            })
            .collect();
    }

    fn destroy(&mut self, renderer: &mut Renderer) {
        let device = &renderer.device.device;
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            for img in self.output_images.drain(..) {
                img.destroy(device, &renderer.device.allocator);
            }
        }
    }
}

impl SSRPass {
    pub fn new(
        renderer: &Renderer,
        shader_spirv: &[u32],
    ) -> Result<Self, crate::error::RendererError> {
        let device = &renderer.device.device;
        let extent = renderer.get_extent();

        let output_images = (0..MAX_FRAMES_IN_FLIGHT)
            .map(|_| {
                Attachment::create_image_resource(
                    &renderer.device,
                    extent.width,
                    extent.height,
                    vk::Format::R16G16B16A16_SFLOAT,
                    vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                    vk::SampleCountFlags::TYPE_1,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(5)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(6)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
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
                    .set_layouts(&[renderer.global_descriptor_set_layout, ds_layout])
                    .push_constant_ranges(&[vk::PushConstantRange {
                        stage_flags: vk::ShaderStageFlags::COMPUTE,
                        offset: 0,
                        size: 24,
                    }]),
                None,
            )?
        };

        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(renderer.gpu_resource_manager.descriptor_pool)
                    .set_layouts(&[ds_layout; MAX_FRAMES_IN_FLIGHT]),
            )?
        };

        let module = crate::pipeline::Pipeline::create_shader_module(device, shader_spirv);
        let entry_point = std::ffi::CString::new("main").unwrap();
        let stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(&entry_point);

        let pipeline = unsafe {
            device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage_info)
                        .layout(layout)],
                    None,
                )
                .unwrap()[0]
        };

        unsafe {
            device.destroy_shader_module(module, None);
        }

        Ok(Self {
            pipeline,
            layout,
            descriptor_set_layout: ds_layout,
            descriptor_sets,
            output_images,
        })
    }
}
