use super::{RenderContext, RenderPass};
use crate::resource::Attachment;
use crate::{Renderer, MAX_FRAMES_IN_FLIGHT};
use ash::vk;

pub struct DoFPass {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub output_images: Vec<Attachment>,
}

impl RenderPass for DoFPass {
    fn name(&self) -> &str {
        "DoFPass"
    }
    fn inputs(&self) -> Vec<&'static str> {
        vec!["HDRColor", "GBuffer"]
    }
    fn outputs(&self) -> Vec<&'static str> {
        vec!["DoF"]
    }

    fn prepare(&self, renderer: &Renderer, current_frame: usize) {
        let hdr_view = renderer
            .get_pass_resource_view("LightingPass", "HDRColor", current_frame)
            .unwrap_or(renderer.common_shadow_view);
        let depth_view = renderer
            .get_pass_resource_view("", "GBufferDepth", current_frame)
            .unwrap_or(renderer.common_shadow_view);

        let sampler = renderer.common_sampler;
        let hdr_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(hdr_view)
            .sampler(sampler)];
        let depth_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(depth_view)
            .sampler(sampler)];
        let out_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(self.output_images[current_frame].view)];

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[current_frame])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&hdr_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[current_frame])
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&depth_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[current_frame])
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&out_info),
        ];
        unsafe {
            renderer.device.device.update_descriptor_sets(&writes, &[]);
        }
    }

    fn record_commands(&self, ctx: &RenderContext) {
        let renderer = ctx.renderer;
        let extent = renderer.get_extent();
        unsafe {
            renderer.device.device.cmd_bind_pipeline(
                ctx.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );
            renderer.device.device.cmd_bind_descriptor_sets(
                ctx.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.descriptor_sets[ctx.current_frame]],
                &[],
            );

            #[repr(C)]
            struct DoFPC {
                focus_distance: f32,
                focus_range: f32,
                bokeh_size: f32,
                enabled: f32,
            }
            let pc = DoFPC {
                focus_distance: renderer.settings.dof_focus_distance,
                focus_range: renderer.settings.dof_focus_range,
                bokeh_size: renderer.settings.dof_bokeh_size,
                enabled: if renderer.settings.enable_dof {
                    1.0
                } else {
                    0.0
                },
            };
            let pc_bytes = std::slice::from_raw_parts(
                &pc as *const _ as *const u8,
                std::mem::size_of::<DoFPC>(),
            );
            renderer.device.device.cmd_push_constants(
                ctx.command_buffer,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );

            renderer.device.device.cmd_dispatch(
                ctx.command_buffer,
                extent.width.div_ceil(16),
                extent.height.div_ceil(16),
                1,
            );
        }
    }

    fn get_resource_view(&self, name: &str, frame_index: usize) -> Option<vk::ImageView> {
        if name == "output" {
            Some(self.output_images[frame_index].view)
        } else {
            None
        }
    }

    fn destroy(&mut self, renderer: &mut Renderer) {
        unsafe {
            renderer.device.device.destroy_pipeline(self.pipeline, None);
            renderer
                .device
                .device
                .destroy_pipeline_layout(self.layout, None);
            renderer
                .device
                .device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            for img in self.output_images.drain(..) {
                img.destroy(&renderer.device.device, &renderer.device.allocator);
            }
        }
    }
}

impl DoFPass {
    pub fn new(
        renderer: &Renderer,
        shader_spirv: &[u32],
    ) -> Result<Self, crate::error::RendererError> {
        let device = &renderer.device.device;
        let extent = renderer.get_extent();

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
                    .set_layouts(&[ds_layout])
                    .push_constant_ranges(&[vk::PushConstantRange {
                        stage_flags: vk::ShaderStageFlags::COMPUTE,
                        offset: 0,
                        size: 16,
                    }]),
                None,
            )?
        };

        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(renderer.descriptor_pool)
                    .set_layouts(&[ds_layout; MAX_FRAMES_IN_FLIGHT]),
            )?
        };

        let mut output_images = Vec::new();
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            output_images.push(Attachment::create_image_resource(
                &renderer.device,
                extent.width,
                extent.height,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                vk::SampleCountFlags::TYPE_1,
            )?);
        }

        let module = crate::pipeline::Pipeline::create_shader_module(device, shader_spirv);
        let entry = std::ffi::CString::new("main").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(&entry);
        let pipeline = unsafe {
            device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(layout)],
                    None,
                )
                .map_err(|e| e.1)?[0]
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
