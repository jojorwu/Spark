use crate::resource::Buffer;
use ash::vk;

use super::{RenderContext, RenderPass};
use crate::Renderer;

impl RenderPass for CullingPass {
    fn name(&self) -> &str {
        "CullingPass"
    }
    fn dependencies(&self) -> Vec<&'static str> {
        vec!["HiZPass"]
    }

    fn gpu_resource_buffer_access(&self) -> Vec<(String, vk::AccessFlags, vk::PipelineStageFlags)> {
        vec![
            (
                "object_data_buffer".to_string(),
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::COMPUTE_SHADER,
            ),
            (
                "indirect_commands_buffer".to_string(),
                vk::AccessFlags::SHADER_WRITE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
            ),
            (
                "draw_count_buffer".to_string(),
                vk::AccessFlags::SHADER_WRITE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
            ),
        ]
    }

    fn record_commands(&self, ctx: &RenderContext) {
        let renderer = ctx.renderer;
        let current_frame = ctx.current_frame;

        let global_ds = renderer.frame_manager.frames[current_frame].global_descriptor_set;
        if let (Some(_), Some(ind_buf), Some(cnt_buf)) = (
            renderer.frame_manager.frames[current_frame]
                .object_data_buffer
                .as_ref(),
            renderer.frame_manager.frames[current_frame]
                .indirect_commands_buffer
                .as_ref(),
            renderer.frame_manager.frames[current_frame]
                .draw_count_buffer
                .as_ref(),
        ) {
            // Reusing the main command buffer instead of allocating a separate compute one
            // This simplifies synchronization via the RenderGraph's barriers.
            let cb = ctx.command_buffer;

            let params = CullingRecordParams {
                device: &renderer.device.device,
                command_buffer: cb,
                object_count: renderer.last_object_count,
                global_ds,
                indirect_buffer: ind_buf,
                count_buffer: cnt_buf,
                renderer_ref_for_pc_extract: renderer,
            };
            self.record_commands_impl(&params);
        }
    }

    fn destroy(&mut self, renderer: &mut Renderer) {
        unsafe {
            let device = &renderer.device.device;
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

pub struct CullingRecordParams<'a> {
    pub device: &'a ash::Device,
    pub command_buffer: vk::CommandBuffer,
    pub object_count: u32,
    pub global_ds: vk::DescriptorSet,
    pub indirect_buffer: &'a Buffer,
    pub count_buffer: &'a Buffer,
    pub renderer_ref_for_pc_extract: &'a Renderer,
}

pub struct CullingPass {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
}

impl CullingPass {
    pub fn destroy_impl(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
    }
    pub fn new(
        device: &ash::Device,
        _descriptor_pool: vk::DescriptorPool,
        shader_code: &[u32],
        global_ub_layout: vk::DescriptorSetLayout,
    ) -> Result<Self, crate::error::RendererError> {
        let push_constant_ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(128)];

        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[global_ub_layout])
                    .push_constant_ranges(&push_constant_ranges),
                None,
            )?
        };

        let shader_module = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(shader_code),
                None,
            )?
        };

        let entry_point =
            std::ffi::CString::new("main").expect("Failed to create entry point name");
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry_point);

        let pipeline = unsafe {
            device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(layout)],
                    None,
                )
                .map_err(|e| e.1)
                .expect("Failed to create CullingPass compute pipeline")[0]
        };

        unsafe {
            device.destroy_shader_module(shader_module, None);
        }

        Ok(Self { pipeline, layout })
    }

    pub fn record_commands_impl(&self, params: &CullingRecordParams) {
        let device = params.device;
        let command_buffer = params.command_buffer;
        let object_count = params.object_count;
        let global_ds = params.global_ds;
        let indirect_buffer = params.indirect_buffer;
        let count_buffer = params.count_buffer;
        unsafe {
            // Reset count buffer
            device.cmd_fill_buffer(command_buffer, count_buffer.handle, 0, 4, 0);

            let barrier = vk::BufferMemoryBarrier::default()
                .buffer(count_buffer.handle)
                .size(4)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[barrier],
                &[],
            );

            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline,
            );

            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[global_ds],
                &[],
            );

            #[repr(C)]
            struct PC {
                light_count: u32,
                metallic: f32,
                roughness: f32,
                width: f32,
                height: f32,
                padding: u32,
                object_buffer_address: u64,
                prev_view_proj: spark_math::Mat4,
                vertex_buffer_address: u64,
            }
            let frame = &params.renderer_ref_for_pc_extract.frame_manager.frames[params
                .renderer_ref_for_pc_extract
                .frame_manager
                .current_frame];
            let pc = PC {
                light_count: params.renderer_ref_for_pc_extract.light_count,
                metallic: 0.0,
                roughness: 0.0,
                width: params.renderer_ref_for_pc_extract.get_extent().width as f32,
                height: params.renderer_ref_for_pc_extract.get_extent().height as f32,
                padding: 0,
                object_buffer_address: frame.object_data_buffer.as_ref().map_or(0, |b| b.address),
                prev_view_proj: params.renderer_ref_for_pc_extract.prev_view_proj,
                vertex_buffer_address: params
                    .renderer_ref_for_pc_extract
                    .gpu_resource_manager
                    .global_vertex_buffer
                    .as_ref()
                    .map_or(0, |b| b.address),
            };
            let pc_bytes =
                std::slice::from_raw_parts(&pc as *const _ as *const u8, std::mem::size_of::<PC>());

            device.cmd_push_constants(
                command_buffer,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );

            device.cmd_dispatch(command_buffer, object_count.div_ceil(256), 1, 1);

            // Barrier to ensure compute finish before indirect draw
            let indirect_barrier = vk::BufferMemoryBarrier::default()
                .buffer(indirect_buffer.handle)
                .size(indirect_buffer.size)
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ);
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::DRAW_INDIRECT,
                vk::DependencyFlags::empty(),
                &[],
                &[indirect_barrier],
                &[],
            );
        }
    }
}
