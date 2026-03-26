use ash::vk;
use crate::resource::{Attachment, MAX_FRAMES_IN_FLIGHT};
use crate::Renderer;
use super::{RenderPass, RenderContext};

const BINDING_TLAS: u32 = 0;
const BINDING_IMAGE: u32 = 1;
const BINDING_VERTICES: u32 = 2;
const BINDING_INDICES: u32 = 3;
const BINDING_MESHES: u32 = 4;
const BINDING_MATERIALS: u32 = 5;
const BINDING_LIGHTS: u32 = 6;

/// A rendering pass that performs hardware-accelerated ray tracing.
pub struct RayTracingPass {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub output_images: Vec<Attachment>,
    pub as_loader: ash::khr::acceleration_structure::Device,
    pub rt_loader: ash::khr::ray_tracing_pipeline::Device,
    pub sbt_buffer: Option<crate::resource::Buffer>,
    pub sbt_regions: [vk::StridedDeviceAddressRegionKHR; 4],
}

impl RenderPass for RayTracingPass {
    fn name(&self) -> &str { "RayTracingPass" }
    fn inputs(&self) -> Vec<&'static str> { vec!["GBufferDepth", "GBufferNormal", "GBufferPBR", "HiZ"] }
    fn outputs(&self) -> Vec<&'static str> { vec!["RTOutput"] }

    fn prepare(&self, renderer: &Renderer, current_frame: usize) {
        if self.pipeline == vk::Pipeline::null() { return; }
        let device = &renderer.device.device;
        let ds = self.descriptor_sets[current_frame];

        let out_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::GENERAL)
            .image_view(self.output_images[current_frame].view)];

        let mut writes = vec![
            vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(BINDING_IMAGE)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .image_info(&out_info),
        ];

        let vb_info;
        if let Some(ref vb) = renderer.global_vertex_buffer {
            vb_info = [vk::DescriptorBufferInfo::default().buffer(vb.handle).range(vb.size)];
            writes.push(vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(BINDING_VERTICES)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&vb_info));
        }

        let ib_info;
        if let Some(ref ib) = renderer.global_index_buffer {
            ib_info = [vk::DescriptorBufferInfo::default().buffer(ib.handle).range(ib.size)];
            writes.push(vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(BINDING_INDICES)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&ib_info));
        }

        let md_buf = renderer.frames[current_frame].object_data_buffer.as_ref().unwrap_or(&renderer.dummy_buffer);
        let md_info = [vk::DescriptorBufferInfo::default().buffer(md_buf.handle).range(md_buf.size)];
        writes.push(vk::WriteDescriptorSet::default()
            .dst_set(ds)
            .dst_binding(BINDING_MESHES)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&md_info));

        let mat_info;
        if let Some(ref mat_buf) = renderer.global_material_buffer {
            mat_info = [vk::DescriptorBufferInfo::default().buffer(mat_buf.handle).range(mat_buf.size)];
            writes.push(vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(BINDING_MATERIALS)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&mat_info));
        }

        let light_info;
        if let Some(ref light_buf) = renderer.frames[current_frame].light_buffer {
            light_info = [vk::DescriptorBufferInfo::default().buffer(light_buf.handle).range(light_buf.size)];
            writes.push(vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(BINDING_LIGHTS)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&light_info));
        }

        let mut as_info;
        if let Some(ref tlas) = renderer.as_manager.current_tlas[current_frame] {
            as_info = vk::WriteDescriptorSetAccelerationStructureKHR::default()
                .acceleration_structures(std::slice::from_ref(&tlas.handle));

            let w = vk::WriteDescriptorSet::default()
                .dst_set(ds)
                .dst_binding(BINDING_TLAS)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .push_next(&mut as_info);

            writes.push(w);
        }

        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
    }

    fn record_commands(&self, ctx: &RenderContext) {
        if self.pipeline == vk::Pipeline::null() { return; }
        let renderer = ctx.renderer;
        let device = &renderer.device.device;
        let extent = renderer.get_extent();

        unsafe {
            device.cmd_bind_pipeline(ctx.command_buffer, vk::PipelineBindPoint::RAY_TRACING_KHR, self.pipeline);
            device.cmd_bind_descriptor_sets(
                ctx.command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.layout,
                0,
                &[renderer.frames[ctx.current_frame].global_descriptor_set, self.descriptor_sets[ctx.current_frame]],
                &[],
            );

            #[repr(C)]
            struct RTPC {
                reflections: f32,
                shadows: f32,
                ao: f32,
                gi: f32,
            }
            let pc = RTPC {
                reflections: if renderer.settings.enable_rt_reflections { 1.0 } else { 0.0 },
                shadows: if renderer.settings.enable_rt_shadows { 1.0 } else { 0.0 },
                ao: if renderer.settings.enable_rt_ao { 1.0 } else { 0.0 },
                gi: if renderer.settings.enable_rt_gi { 1.0 } else { 0.0 },
            };
            let pc_bytes = std::slice::from_raw_parts(&pc as *const _ as *const u8, std::mem::size_of::<RTPC>());
            device.cmd_push_constants(ctx.command_buffer, self.layout, vk::ShaderStageFlags::RAYGEN_KHR, 0, pc_bytes);

            self.rt_loader.cmd_trace_rays(
                ctx.command_buffer,
                &self.sbt_regions[0],
                &self.sbt_regions[1],
                &self.sbt_regions[2],
                &self.sbt_regions[3],
                extent.width,
                extent.height,
                1,
            );
        }
    }

    fn on_resize(&mut self, renderer: &mut Renderer, new_extent: vk::Extent2D) {
        for img in self.output_images.drain(..) {
            img.destroy(&renderer.device.device, &renderer.device.allocator);
        }
        self.output_images = (0..MAX_FRAMES_IN_FLIGHT).map(|_| {
            Attachment::create_image_resource(
                &renderer.device,
                new_extent.width,
                new_extent.height,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                vk::SampleCountFlags::TYPE_1,
            ).unwrap()
        }).collect();
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
            if let Some(sbt) = self.sbt_buffer.take() {
                renderer.device.destroy_buffer(sbt);
            }
        }
    }
}

impl RayTracingPass {
    pub fn new(renderer: &Renderer, rgen_spirv: &[u32], rmiss_spirv: &[u32], rchit_spirv: &[u32]) -> Result<Self, crate::error::RendererError> {
        let device = &renderer.device.device;
        let as_loader = renderer.device.as_loader.as_ref().ok_or(crate::error::RendererError::NoSuitableDevice)?.clone();
        let rt_loader = renderer.device.rt_loader.as_ref().ok_or(crate::error::RendererError::NoSuitableDevice)?.clone();

        if !renderer.device.rt_supported {
            return Ok(Self::new_empty(as_loader, rt_loader));
        }

        let extent = renderer.get_extent();
        let output_images = (0..MAX_FRAMES_IN_FLIGHT).map(|_| {
            Attachment::create_image_resource(
                &renderer.device,
                extent.width,
                extent.height,
                vk::Format::R16G16B16A16_SFLOAT,
                vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                vk::SampleCountFlags::TYPE_1,
            ).unwrap()
        }).collect::<Vec<_>>();

        let ds_layout = Self::create_descriptor_set_layout(device)?;
        let layout = Self::create_pipeline_layout(renderer, device, ds_layout)?;
        let descriptor_sets = Self::allocate_descriptor_sets(renderer, device, ds_layout)?;

        let (pipeline, sbt_buffer, sbt_regions) = Self::create_pipeline_and_sbt(
            renderer, device, &rt_loader, layout, rgen_spirv, rmiss_spirv, rchit_spirv
        )?;

        Ok(Self {
            pipeline,
            layout,
            descriptor_set_layout: ds_layout,
            descriptor_sets,
            output_images,
            as_loader,
            rt_loader,
            sbt_buffer: Some(sbt_buffer),
            sbt_regions,
        })
    }

    fn new_empty(as_loader: ash::khr::acceleration_structure::Device, rt_loader: ash::khr::ray_tracing_pipeline::Device) -> Self {
        Self {
            pipeline: vk::Pipeline::null(),
            layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            descriptor_sets: Vec::new(),
            output_images: Vec::new(),
            as_loader,
            rt_loader,
            sbt_buffer: None,
            sbt_regions: [vk::StridedDeviceAddressRegionKHR::default(); 4],
        }
    }

    fn create_descriptor_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout, crate::error::RendererError> {
        let bindings = [
            vk::DescriptorSetLayoutBinding::default().binding(BINDING_TLAS).descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR).descriptor_count(1).stage_flags(vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR),
            vk::DescriptorSetLayoutBinding::default().binding(BINDING_IMAGE).descriptor_type(vk::DescriptorType::STORAGE_IMAGE).descriptor_count(1).stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            vk::DescriptorSetLayoutBinding::default().binding(BINDING_VERTICES).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR),
            vk::DescriptorSetLayoutBinding::default().binding(BINDING_INDICES).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR),
            vk::DescriptorSetLayoutBinding::default().binding(BINDING_MESHES).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR),
            vk::DescriptorSetLayoutBinding::default().binding(BINDING_MATERIALS).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR),
            vk::DescriptorSetLayoutBinding::default().binding(BINDING_LIGHTS).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).descriptor_count(1).stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
        ];

        unsafe {
            Ok(device.create_descriptor_set_layout(&vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings), None)?)
        }
    }

    fn create_pipeline_layout(renderer: &Renderer, device: &ash::Device, ds_layout: vk::DescriptorSetLayout) -> Result<vk::PipelineLayout, crate::error::RendererError> {
        let pc_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::RAYGEN_KHR,
            offset: 0,
            size: 16,
        };
        unsafe {
            Ok(device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[renderer.global_descriptor_set_layout, ds_layout])
                    .push_constant_ranges(&[pc_range]),
                None,
            )?)
        }
    }

    fn allocate_descriptor_sets(renderer: &Renderer, device: &ash::Device, ds_layout: vk::DescriptorSetLayout) -> Result<Vec<vk::DescriptorSet>, crate::error::RendererError> {
        let layouts = [ds_layout; MAX_FRAMES_IN_FLIGHT];
        unsafe {
            Ok(device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(renderer.descriptor_pool)
                    .set_layouts(&layouts),
            )?)
        }
    }

    fn create_pipeline_and_sbt(
        renderer: &Renderer,
        device: &ash::Device,
        rt_loader: &ash::khr::ray_tracing_pipeline::Device,
        layout: vk::PipelineLayout,
        rgen_spirv: &[u32],
        rmiss_spirv: &[u32],
        rchit_spirv: &[u32],
    ) -> Result<(vk::Pipeline, crate::resource::Buffer, [vk::StridedDeviceAddressRegionKHR; 4]), crate::error::RendererError> {
        let rgen_module = crate::pipeline::Pipeline::create_shader_module(device, rgen_spirv);
        let rmiss_module = crate::pipeline::Pipeline::create_shader_module(device, rmiss_spirv);
        let rchit_module = crate::pipeline::Pipeline::create_shader_module(device, rchit_spirv);

        let entry_point = std::ffi::CString::new("main").unwrap();
        let stages = [
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::RAYGEN_KHR).module(rgen_module).name(&entry_point),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::MISS_KHR).module(rmiss_module).name(&entry_point),
            vk::PipelineShaderStageCreateInfo::default().stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR).module(rchit_module).name(&entry_point),
        ];

        let groups = [
            vk::RayTracingShaderGroupCreateInfoKHR::default().ty(vk::RayTracingShaderGroupTypeKHR::GENERAL).general_shader(0).closest_hit_shader(vk::SHADER_UNUSED_KHR).any_hit_shader(vk::SHADER_UNUSED_KHR).intersection_shader(vk::SHADER_UNUSED_KHR),
            vk::RayTracingShaderGroupCreateInfoKHR::default().ty(vk::RayTracingShaderGroupTypeKHR::GENERAL).general_shader(1).closest_hit_shader(vk::SHADER_UNUSED_KHR).any_hit_shader(vk::SHADER_UNUSED_KHR).intersection_shader(vk::SHADER_UNUSED_KHR),
            vk::RayTracingShaderGroupCreateInfoKHR::default().ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP).general_shader(vk::SHADER_UNUSED_KHR).closest_hit_shader(2).any_hit_shader(vk::SHADER_UNUSED_KHR).intersection_shader(vk::SHADER_UNUSED_KHR),
        ];

        let pipeline = unsafe {
            rt_loader.create_ray_tracing_pipelines(vk::DeferredOperationKHR::null(), vk::PipelineCache::null(), &[
                vk::RayTracingPipelineCreateInfoKHR::default().stages(&stages).groups(&groups).max_pipeline_ray_recursion_depth(2).layout(layout)
            ], None).unwrap()[0]
        };

        unsafe {
            device.destroy_shader_module(rgen_module, None);
            device.destroy_shader_module(rmiss_module, None);
            device.destroy_shader_module(rchit_module, None);
        }

        let rt_props = unsafe {
            let mut props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut props);
            renderer.context.instance.get_physical_device_properties2(renderer.device.pdevice, &mut props2);
            props
        };

        let handle_size = rt_props.shader_group_handle_size;
        let handle_alignment = rt_props.shader_group_handle_alignment;
        let base_alignment = rt_props.shader_group_base_alignment as u64;

        let handle_size_aligned = (handle_size + handle_alignment - 1) & !(handle_alignment - 1);
        let region_size = (handle_size_aligned as u64 + base_alignment - 1) & !(base_alignment - 1);

        let sbt_buffer = renderer.device.create_buffer(
            region_size * 4,
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let handles = unsafe {
            rt_loader.get_ray_tracing_shader_group_handles(pipeline, 0, groups.len() as u32, (groups.len() as u32 * handle_size) as usize).unwrap()
        };

        let mut sbt_regions = [vk::StridedDeviceAddressRegionKHR::default(); 4];
        let sbt_address = sbt_buffer.address;

        unsafe {
            let ptr = sbt_buffer.ptr as *mut u8;
            for i in 0..groups.len() {
                std::ptr::copy_nonoverlapping(
                    handles.as_ptr().add(i * handle_size as usize),
                    ptr.add(i * region_size as usize),
                    handle_size as usize,
                );
            }
        }

        sbt_regions[0] = vk::StridedDeviceAddressRegionKHR { device_address: sbt_address, stride: handle_size_aligned as u64, size: region_size };
        sbt_regions[1] = vk::StridedDeviceAddressRegionKHR { device_address: sbt_address + region_size, stride: handle_size_aligned as u64, size: region_size };
        sbt_regions[2] = vk::StridedDeviceAddressRegionKHR { device_address: sbt_address + 2 * region_size, stride: handle_size_aligned as u64, size: region_size };

        Ok((pipeline, sbt_buffer, sbt_regions))
    }
}
