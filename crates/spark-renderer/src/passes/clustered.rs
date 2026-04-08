use crate::resource::{Buffer, ClusterAABB, LightGrid};
use crate::Renderer;
use ash::vk;
use spark_math::Mat4;

use super::{RenderContext, RenderPass};

impl RenderPass for ClusteredPass {
    fn name(&self) -> &str {
        "ClusteredPass"
    }

    fn gpu_resource_buffer_access(&self) -> Vec<(String, vk::AccessFlags, vk::PipelineStageFlags)> {
        vec![
            (
                "light_grid".to_string(),
                vk::AccessFlags::SHADER_WRITE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
            ),
            (
                "index_list".to_string(),
                vk::AccessFlags::SHADER_WRITE,
                vk::PipelineStageFlags::COMPUTE_SHADER,
            ),
            (
                "Lights".to_string(),
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::COMPUTE_SHADER,
            ),
        ]
    }

    fn prepare(&self, renderer: &Renderer, _current_frame: usize) {
        if let Some(ref lb) =
            renderer.frame_manager.frames[renderer.frame_manager.current_frame].light_buffer
        {
            self.update_descriptor_sets(&renderer.device.device, lb);
        }
    }

    fn record_commands(&self, ctx: &RenderContext) {
        let renderer = ctx.renderer;
        let command_buffer = ctx.command_buffer;
        let view = renderer.scene_view_matrix_for_pos;
        let proj = spark_math::Mat4::perspective_rh(
            45.0f32.to_radians(),
            renderer.swapchain.extent.width as f32 / renderer.swapchain.extent.height as f32,
            0.1,
            100.0,
        );

        let screen_size = [
            renderer.swapchain.extent.width as f32,
            renderer.swapchain.extent.height as f32,
        ];

        let mut last_p = self
            .last_proj
            .lock()
            .expect("Failed to lock last projection in ClusteredPass");
        let mut last_s = self
            .last_screen_size
            .lock()
            .expect("Failed to lock last screen size in ClusteredPass");

        if *last_p != proj || *last_s != screen_size {
            self.record_build_commands(
                &renderer.device.device,
                command_buffer,
                proj.inverse(),
                screen_size,
                0.1,
                100.0,
            );
            *last_p = proj;
            *last_s = screen_size;
        }

        self.record_cull_commands(
            &renderer.device.device,
            command_buffer,
            view,
            renderer.light_count,
        );
    }

    fn get_resource_buffer(
        &self,
        name: &str,
        _frame_index: usize,
    ) -> Option<crate::resource::Buffer> {
        match name {
            "light_grid" => Some(self.light_grid_buffer.clone()),
            "index_list" => Some(self.global_index_list.clone()),
            _ => None,
        }
    }

    fn destroy(&mut self, renderer: &mut Renderer) {
        let device = &renderer.device.device;
        unsafe {
            device.destroy_pipeline(self.build_pipeline, None);
            device.destroy_pipeline(self.cull_pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            renderer.destroy_buffer(self.cluster_buffer.clone());
            renderer.destroy_buffer(self.light_grid_buffer.clone());
            renderer.destroy_buffer(self.global_index_list.clone());
            renderer.destroy_buffer(self.index_counter.clone());
        }
    }
}

pub struct ClusteredPass {
    pub build_pipeline: vk::Pipeline,
    pub cull_pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_set: vk::DescriptorSet,
    pub cluster_buffer: Buffer,
    pub light_grid_buffer: Buffer,
    pub global_index_list: Buffer,
    pub index_counter: Buffer,
    pub last_proj: std::sync::Mutex<Mat4>,
    pub last_screen_size: std::sync::Mutex<[f32; 2]>,
}

impl ClusteredPass {
    pub const GRID_SIZE: (u32, u32, u32) = (16, 9, 24); // Adjust as needed
    pub const TOTAL_CLUSTERS: u32 = Self::GRID_SIZE.0 * Self::GRID_SIZE.1 * Self::GRID_SIZE.2;

    pub fn new(
        renderer: &Renderer,
        build_shader: &[u32],
        cull_shader: &[u32],
    ) -> Result<Self, crate::error::RendererError> {
        let device = &renderer.device.device;

        // 1. Create Buffers
        let cluster_buffer = renderer.create_buffer(
            (Self::TOTAL_CLUSTERS as usize * std::mem::size_of::<ClusterAABB>()) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        let light_grid_buffer = renderer.create_buffer(
            (Self::TOTAL_CLUSTERS as usize * std::mem::size_of::<LightGrid>()) as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        let global_index_list = renderer.create_buffer(
            (Self::TOTAL_CLUSTERS * 128 * 4) as u64, // Max lights per cluster
            vk::BufferUsageFlags::STORAGE_BUFFER,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        let index_counter = renderer.create_buffer(
            4,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );

        // 2. Descriptor Sets
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE),
            vk::DescriptorSetLayoutBinding::default()
                .binding(4)
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

        let descriptor_set = unsafe {
            renderer.device.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(renderer.gpu_resource_manager.descriptor_pool)
                    .set_layouts(&[ds_layout]),
            )?[0]
        };

        // 3. Pipelines
        let layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[ds_layout])
                    .push_constant_ranges(&[vk::PushConstantRange {
                        stage_flags: vk::ShaderStageFlags::COMPUTE,
                        offset: 0,
                        size: 128,
                    }]),
                None,
            )?
        };

        let build_module = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(build_shader),
                None,
            )?
        };
        let cull_module = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(cull_shader),
                None,
            )?
        };

        let entry_point =
            std::ffi::CString::new("main").expect("Failed to create CString for entry point");

        let build_pipeline = unsafe {
            device
                .create_compute_pipelines(
                    renderer.pipeline_cache,
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::COMPUTE)
                                .module(build_module)
                                .name(&entry_point),
                        )
                        .layout(layout)],
                    None,
                )
                .expect("Failed to create ClusteredPass build pipeline")[0]
        };

        let cull_pipeline = unsafe {
            device
                .create_compute_pipelines(
                    renderer.pipeline_cache,
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::COMPUTE)
                                .module(cull_module)
                                .name(&entry_point),
                        )
                        .layout(layout)],
                    None,
                )
                .expect("Failed to create ClusteredPass cull pipeline")[0]
        };

        unsafe {
            device.destroy_shader_module(build_module, None);
            device.destroy_shader_module(cull_module, None);
        }

        Ok(Self {
            build_pipeline,
            cull_pipeline,
            layout,
            descriptor_set_layout: ds_layout,
            descriptor_set,
            cluster_buffer,
            light_grid_buffer,
            global_index_list,
            index_counter,
            last_proj: std::sync::Mutex::new(Mat4::ZERO),
            last_screen_size: std::sync::Mutex::new([0.0, 0.0]),
        })
    }

    pub fn update_descriptor_sets(&self, device: &ash::Device, light_buffer: &Buffer) {
        let b0 = [vk::DescriptorBufferInfo::default()
            .buffer(self.cluster_buffer.handle)
            .range(self.cluster_buffer.size)];
        let b1 = [vk::DescriptorBufferInfo::default()
            .buffer(light_buffer.handle)
            .range(light_buffer.size)];
        let b2 = [vk::DescriptorBufferInfo::default()
            .buffer(self.light_grid_buffer.handle)
            .range(self.light_grid_buffer.size)];
        let b3 = [vk::DescriptorBufferInfo::default()
            .buffer(self.global_index_list.handle)
            .range(self.global_index_list.size)];
        let b4 = [vk::DescriptorBufferInfo::default()
            .buffer(self.index_counter.handle)
            .range(self.index_counter.size)];

        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&b0),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&b1),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&b2),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&b3),
            vk::WriteDescriptorSet::default()
                .dst_set(self.descriptor_set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&b4),
        ];

        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
    }

    pub fn record_build_commands(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        inv_proj: Mat4,
        screen_size: [f32; 2],
        z_near: f32,
        z_far: f32,
    ) {
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.build_pipeline,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.descriptor_set],
                &[],
            );

            #[repr(C)]
            struct PC {
                inv_proj: Mat4,
                screen_size: [f32; 2],
                z_near: f32,
                z_far: f32,
            }
            let pc = PC {
                inv_proj,
                screen_size,
                z_near,
                z_far,
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

            device.cmd_dispatch(command_buffer, 1, 1, 6); // 16x9x24 total
        }
    }

    pub fn record_cull_commands(
        &self,
        device: &ash::Device,
        command_buffer: vk::CommandBuffer,
        view: Mat4,
        light_count: u32,
    ) {
        unsafe {
            // Reset counter
            device.cmd_fill_buffer(command_buffer, self.index_counter.handle, 0, 4, 0);

            let barrier = vk::BufferMemoryBarrier::default()
                .buffer(self.index_counter.handle)
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
                self.cull_pipeline,
            );
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[self.descriptor_set],
                &[],
            );

            #[repr(C)]
            struct PC {
                view: Mat4,
                light_count: u32,
            }
            let pc = PC { view, light_count };
            let pc_bytes =
                std::slice::from_raw_parts(&pc as *const _ as *const u8, std::mem::size_of::<PC>());
            device.cmd_push_constants(
                command_buffer,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                pc_bytes,
            );

            device.cmd_dispatch(command_buffer, 1, 1, 6);
        }
    }
}
