use super::{RenderContext, RenderPass};
use crate::resource::MAX_FRAMES_IN_FLIGHT;
use crate::Renderer;
use ash::vk;

const BINDING_TLAS: u32 = 0;
const BINDING_IMAGE: u32 = 1;
const BINDING_VERTICES: u32 = 2;
const BINDING_INDICES: u32 = 3;
const BINDING_MESHES: u32 = 4;
const BINDING_MATERIALS: u32 = 5;
const BINDING_LIGHTS: u32 = 6;
const BINDING_GBUFFER_DEPTH: u32 = 7;
const BINDING_GBUFFER_NORMAL: u32 = 8;
const BINDING_GBUFFER_ALBEDO: u32 = 9;
const BINDING_GBUFFER_PBR: u32 = 10;

use super::ResourceBinding;

/// A rendering pass that performs hardware-accelerated ray tracing.
pub struct RayTracingPass {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_sets: Vec<vk::DescriptorSet>,
    pub as_loader: ash::khr::acceleration_structure::Device,
    pub rt_loader: ash::khr::ray_tracing_pipeline::Device,
    pub sbt_buffer: Option<crate::resource::Buffer>,
    pub sbt_regions: [vk::StridedDeviceAddressRegionKHR; 4],
}

impl RenderPass for RayTracingPass {
    fn name(&self) -> &str {
        "RayTracingPass"
    }
    fn inputs(&self) -> Vec<&'static str> {
        vec![
            "GBufferDepth",
            "GBufferNormal",
            "GBufferAlbedo",
            "GBufferPBR",
            "HiZ",
        ]
    }
    fn on_resize(&mut self, _renderer: &Renderer, _new_extent: vk::Extent2D) {}

    fn bindings(&self) -> Vec<ResourceBinding> {
        vec![
            ResourceBinding::AccelerationStructure(BINDING_TLAS, "SceneTLAS".to_string()),
            ResourceBinding::StorageImage(BINDING_IMAGE, "RTOutput".to_string()),
            ResourceBinding::StorageBuffer(BINDING_VERTICES, "Vertices".to_string()),
            ResourceBinding::StorageBuffer(BINDING_INDICES, "Indices".to_string()),
            ResourceBinding::StorageBuffer(BINDING_MESHES, "MeshData".to_string()),
            ResourceBinding::StorageBuffer(BINDING_MATERIALS, "Materials".to_string()),
            ResourceBinding::StorageBuffer(BINDING_LIGHTS, "Lights".to_string()),
            ResourceBinding::SampledImage(BINDING_GBUFFER_DEPTH, "GBufferDepth".to_string()),
            ResourceBinding::SampledImage(BINDING_GBUFFER_NORMAL, "GBufferNormal".to_string()),
            ResourceBinding::SampledImage(BINDING_GBUFFER_ALBEDO, "GBufferAlbedo".to_string()),
            ResourceBinding::SampledImage(BINDING_GBUFFER_PBR, "GBufferPBR".to_string()),
        ]
    }

    fn destroy(&mut self, renderer: &mut Renderer) {
        let device = &renderer.device.device;
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            if let Some(sbt) = self.sbt_buffer.take() {
                renderer.device.destroy_buffer(sbt);
            }
        }
    }
    fn gpu_resource_access(&self) -> Vec<(String, vk::AccessFlags, vk::PipelineStageFlags)> {
        vec![
            (
                "SceneTLAS".to_string(),
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            ),
            (
                "GBufferDepth".to_string(),
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            ),
            (
                "GBufferNormal".to_string(),
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            ),
            (
                "GBufferAlbedo".to_string(),
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            ),
            (
                "GBufferPBR".to_string(),
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            ),
            (
                "RTOutput".to_string(),
                vk::AccessFlags::SHADER_WRITE,
                vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR,
            ),
        ]
    }
    fn outputs(&self) -> Vec<&'static str> {
        vec!["RTOutput"]
    }

    fn needs_descriptor_update(&self, _renderer: &Renderer, _frame_index: usize) -> bool {
        false // Now handled by RenderGraph
    }

    fn update_descriptor_sets(&self, _renderer: &Renderer) {
        // Now handled by RenderGraph
    }

    fn prepare(&self, _renderer: &Renderer, _current_frame: usize) {
        // Now handled by RenderGraph
    }

    fn record_commands(&self, ctx: &RenderContext) {
        self.record_commands_impl(ctx);
    }
}

impl RayTracingPass {
    fn record_commands_impl(&self, ctx: &RenderContext) {
        if self.pipeline == vk::Pipeline::null() {
            return;
        }
        let renderer = ctx.renderer;
        let device = &renderer.device.device;
        let extent = renderer.get_extent();

        unsafe {
            device.cmd_bind_pipeline(
                ctx.command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                ctx.command_buffer,
                vk::PipelineBindPoint::RAY_TRACING_KHR,
                self.layout,
                0,
                &[
                    renderer.frame_manager.frames[ctx.current_frame].global_descriptor_set,
                    self.descriptor_sets[ctx.current_frame],
                    renderer.gpu_resource_manager.bindless.set,
                ],
                &[],
            );

            #[repr(C)]
            struct Rtpc {
                reflections: f32,
                shadows: f32,
                ao: f32,
                gi: f32,
            }
            let pc = Rtpc {
                reflections: if renderer.settings.enable_rt_reflections {
                    1.0
                } else {
                    0.0
                },
                shadows: if renderer.settings.enable_rt_shadows {
                    1.0
                } else {
                    0.0
                },
                ao: if renderer.settings.enable_rt_ao {
                    1.0
                } else {
                    0.0
                },
                gi: if renderer.settings.enable_rt_gi {
                    1.0
                } else {
                    0.0
                },
            };
            let pc_bytes = std::slice::from_raw_parts(
                &pc as *const _ as *const u8,
                std::mem::size_of::<Rtpc>(),
            );
            device.cmd_push_constants(
                ctx.command_buffer,
                self.layout,
                vk::ShaderStageFlags::RAYGEN_KHR,
                0,
                pc_bytes,
            );

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
}

impl RayTracingPass {
    pub fn new(
        renderer: &Renderer,
        rgen_spirv: &[u32],
        rmiss_spirv: &[u32],
        rchit_spirv: &[u32],
    ) -> Result<Self, crate::error::RendererError> {
        let device = &renderer.device.device;
        let as_loader = renderer
            .device
            .as_loader
            .as_ref()
            .ok_or(crate::error::RendererError::NoSuitableDevice)?
            .clone();
        let rt_loader = renderer
            .device
            .rt_loader
            .as_ref()
            .ok_or(crate::error::RendererError::NoSuitableDevice)?
            .clone();

        if !renderer.device.rt_supported {
            return Ok(Self::new_empty(as_loader, rt_loader));
        }

        let ds_layout = Self::create_descriptor_set_layout(device)?;
        let layout = Self::create_pipeline_layout(renderer, device, ds_layout)?;
        let descriptor_sets = Self::allocate_descriptor_sets(renderer, device, ds_layout)?;

        let (pipeline, sbt_buffer, sbt_regions) = Self::create_pipeline_and_sbt(
            renderer,
            device,
            &rt_loader,
            layout,
            rgen_spirv,
            rmiss_spirv,
            rchit_spirv,
        )?;

        Ok(Self {
            pipeline,
            layout,
            descriptor_set_layout: ds_layout,
            descriptor_sets,
            as_loader,
            rt_loader,
            sbt_buffer: Some(sbt_buffer),
            sbt_regions,
        })
    }

    fn new_empty(
        as_loader: ash::khr::acceleration_structure::Device,
        rt_loader: ash::khr::ray_tracing_pipeline::Device,
    ) -> Self {
        Self {
            pipeline: vk::Pipeline::null(),
            layout: vk::PipelineLayout::null(),
            descriptor_set_layout: vk::DescriptorSetLayout::null(),
            descriptor_sets: Vec::new(),
            as_loader,
            rt_loader,
            sbt_buffer: None,
            sbt_regions: [vk::StridedDeviceAddressRegionKHR::default(); 4],
        }
    }

    fn create_descriptor_set_layout(
        device: &ash::Device,
    ) -> Result<vk::DescriptorSetLayout, crate::error::RendererError> {
        let bindings = [
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_TLAS)
                .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                .descriptor_count(1)
                .stage_flags(
                    vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                ),
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_IMAGE)
                .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_VERTICES)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR),
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_INDICES)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR),
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_MESHES)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR),
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_MATERIALS)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR),
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_LIGHTS)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_GBUFFER_DEPTH)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_GBUFFER_NORMAL)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_GBUFFER_ALBEDO)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
            vk::DescriptorSetLayoutBinding::default()
                .binding(BINDING_GBUFFER_PBR)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR),
        ];

        unsafe {
            Ok(device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?)
        }
    }

    fn create_pipeline_layout(
        renderer: &Renderer,
        device: &ash::Device,
        ds_layout: vk::DescriptorSetLayout,
    ) -> Result<vk::PipelineLayout, crate::error::RendererError> {
        let pc_range = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::RAYGEN_KHR,
            offset: 0,
            size: 16,
        };
        unsafe {
            Ok(device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&[
                        renderer.global_descriptor_set_layout,
                        ds_layout,
                        renderer.gpu_resource_manager.bindless.layout,
                    ])
                    .push_constant_ranges(&[pc_range]),
                None,
            )?)
        }
    }

    fn allocate_descriptor_sets(
        renderer: &Renderer,
        device: &ash::Device,
        ds_layout: vk::DescriptorSetLayout,
    ) -> Result<Vec<vk::DescriptorSet>, crate::error::RendererError> {
        let layouts = [ds_layout; MAX_FRAMES_IN_FLIGHT];
        unsafe {
            Ok(device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(renderer.gpu_resource_manager.descriptor_pool)
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
    ) -> Result<
        (
            vk::Pipeline,
            crate::resource::Buffer,
            [vk::StridedDeviceAddressRegionKHR; 4],
        ),
        crate::error::RendererError,
    > {
        let rgen_module = crate::pipeline::Pipeline::create_shader_module(device, rgen_spirv);
        let rmiss_module = crate::pipeline::Pipeline::create_shader_module(device, rmiss_spirv);
        let rchit_module = crate::pipeline::Pipeline::create_shader_module(device, rchit_spirv);

        let entry_point =
            std::ffi::CString::new("main").expect("Failed to create CString for entry point");
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::RAYGEN_KHR)
                .module(rgen_module)
                .name(&entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::MISS_KHR)
                .module(rmiss_module)
                .name(&entry_point),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                .module(rchit_module)
                .name(&entry_point),
        ];

        let groups = [
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(0)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::GENERAL)
                .general_shader(1)
                .closest_hit_shader(vk::SHADER_UNUSED_KHR)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
            vk::RayTracingShaderGroupCreateInfoKHR::default()
                .ty(vk::RayTracingShaderGroupTypeKHR::TRIANGLES_HIT_GROUP)
                .general_shader(vk::SHADER_UNUSED_KHR)
                .closest_hit_shader(2)
                .any_hit_shader(vk::SHADER_UNUSED_KHR)
                .intersection_shader(vk::SHADER_UNUSED_KHR),
        ];

        let pipeline = unsafe {
            rt_loader
                .create_ray_tracing_pipelines(
                    vk::DeferredOperationKHR::null(),
                    renderer.pipeline_cache,
                    &[vk::RayTracingPipelineCreateInfoKHR::default()
                        .stages(&stages)
                        .groups(&groups)
                        .max_pipeline_ray_recursion_depth(2)
                        .layout(layout)],
                    None,
                )
                .map_err(|e| e.1)
                .expect("Failed to create RayTracing pipeline")[0]
        };

        unsafe {
            device.destroy_shader_module(rgen_module, None);
            device.destroy_shader_module(rmiss_module, None);
            device.destroy_shader_module(rchit_module, None);
        }

        let rt_props = unsafe {
            let mut props = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
            let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut props);
            renderer
                .context
                .instance
                .get_physical_device_properties2(renderer.device.pdevice, &mut props2);
            props
        };

        let handle_size = rt_props.shader_group_handle_size;
        let handle_alignment = rt_props.shader_group_handle_alignment;
        let base_alignment = rt_props.shader_group_base_alignment as u64;

        let handle_size_aligned = (handle_size + handle_alignment - 1) & !(handle_alignment - 1);
        let region_size = (handle_size_aligned as u64 + base_alignment - 1) & !(base_alignment - 1);

        let sbt_buffer = renderer.device.create_buffer(
            region_size * 4,
            vk::BufferUsageFlags::SHADER_BINDING_TABLE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let handles = unsafe {
            rt_loader
                .get_ray_tracing_shader_group_handles(
                    pipeline,
                    0,
                    groups.len() as u32,
                    (groups.len() as u32 * handle_size) as usize,
                )
                .expect("Failed to get ray tracing shader group handles")
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

        sbt_regions[0] = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt_address,
            stride: handle_size_aligned as u64,
            size: region_size,
        };
        sbt_regions[1] = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt_address + region_size,
            stride: handle_size_aligned as u64,
            size: region_size,
        };
        sbt_regions[2] = vk::StridedDeviceAddressRegionKHR {
            device_address: sbt_address + 2 * region_size,
            stride: handle_size_aligned as u64,
            size: region_size,
        };

        Ok((pipeline, sbt_buffer, sbt_regions))
    }
}
