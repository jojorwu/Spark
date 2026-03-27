use super::{RenderContext, RenderPass};
use crate::resource::Buffer;
use crate::{Renderer, MAX_FRAMES_IN_FLIGHT};
use ash::vk;

pub struct LuminancePass {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub luminance_buffer: Buffer,
}

impl RenderPass for LuminancePass {
    fn name(&self) -> &str {
        "LuminancePass"
    }
    fn inputs(&self) -> Vec<&'static str> {
        vec!["HDRColor"]
    }
    fn outputs(&self) -> Vec<&'static str> {
        vec!["Luminance"]
    }

    fn prepare(&self, renderer: &Renderer, current_frame: usize) {
        let hdr_view = renderer
            .get_pass_resource_view("LightingPass", "HDRColor", current_frame)
            .unwrap_or(renderer.common_shadow_view);

        let hdr_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(hdr_view)
            .sampler(renderer.common_sampler)];

        let buf_info = [vk::DescriptorBufferInfo::default()
            .buffer(self.luminance_buffer.handle)
            .range(self.luminance_buffer.size)];

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[current_frame])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&hdr_info),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_sets[current_frame])
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&buf_info),
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
            struct LuminancePC {
                width: u32,
                height: u32,
                min_log_lum: f32,
                max_log_lum: f32,
                speed: f32,
                delta_time: f32,
            }
            let pc = LuminancePC {
                width: extent.width,
                height: extent.height,
                min_log_lum: renderer.settings.auto_exposure_min.log2(),
                max_log_lum: renderer.settings.auto_exposure_max.log2(),
                speed: renderer.settings.auto_exposure_speed,
                delta_time: ctx.delta,
            };
            let pc_bytes = std::slice::from_raw_parts(
                &pc as *const _ as *const u8,
                std::mem::size_of::<LuminancePC>(),
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

    fn get_resource_buffer(&self, name: &str, _frame_index: usize) -> Option<Buffer> {
        if name == "Luminance" {
            Some(self.luminance_buffer.clone())
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
            renderer.destroy_buffer(self.luminance_buffer.clone());
        }
    }
}

impl LuminancePass {
    pub fn new(
        renderer: &Renderer,
        shader_spirv: &[u32],
    ) -> Result<Self, crate::error::RendererError> {
        let device = &renderer.device.device;

        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
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
                        size: 32,
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

        let luminance_buffer = renderer.create_buffer(
            16,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        let initial_data = [1.0f32, 1.0f32, 0.0f32, 0.0f32];
        renderer.upload_to_buffer(&luminance_buffer, &initial_data);

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
            luminance_buffer,
        })
    }
}
