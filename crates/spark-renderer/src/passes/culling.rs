use ash::vk;
use crate::resource::Buffer;

pub struct CullingPass {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
}

impl CullingPass {
    pub fn new(
        device: &ash::Device,
        _descriptor_pool: vk::DescriptorPool,
        shader_code: &[u32],
        global_ub_layout: vk::DescriptorSetLayout,
    ) -> Self {
        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[global_ub_layout]),
                None,
            ).unwrap()
        };

        let shader_module = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(shader_code),
                None,
            ).unwrap()
        };

        let entry_point = std::ffi::CString::new("main").unwrap();
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(&entry_point);

        let pipeline = unsafe {
            device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(layout)],
                None,
            ).unwrap()[0]
        };

        unsafe { device.destroy_shader_module(shader_module, None); }

        Self {
            pipeline,
            layout,
        }
    }

    pub fn record_commands(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        object_count: u32,
        global_ds: vk::DescriptorSet,
        _object_buffer: &Buffer,
        indirect_buffer: &Buffer,
        count_buffer: &Buffer,
    ) {
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

            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, self.pipeline);

            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[global_ds],
                &[],
            );

            device.cmd_dispatch(command_buffer, (object_count + 255) / 256, 1, 1);

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

    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
    }
}
