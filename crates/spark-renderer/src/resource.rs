use ash::vk;
use std::sync::{Arc, Mutex};
use serde::{Serialize, Deserialize};

pub const MAX_FRAMES_IN_FLIGHT: usize = 2;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RenderSettings {
    pub exposure: f32,
    pub gamma: f32,
    pub enable_ssao: bool,
    pub enable_taa: bool,
    pub enable_shadows: bool,
    pub enable_volumetric: bool,
    pub enable_grid: bool,
    pub enable_ibl: bool,
    pub enable_bloom: bool,
    pub ssao_radius: f32,
    pub ssao_strength: f32,
    pub bloom_threshold: f32,
    pub bloom_intensity: f32,
    pub vignette_intensity: f32,
    pub vignette_smoothness: f32,
    pub chromatic_aberration: f32,
    pub film_grain: f32,
    pub fog_color: [f32; 3],
    pub fog_density: f32,
    pub fog_height_falloff: f32,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            exposure: 1.0,
            gamma: 2.2,
            enable_ssao: true,
            enable_taa: true,
            enable_shadows: true,
            enable_volumetric: true,
            enable_grid: true,
            enable_ibl: true,
            enable_bloom: true,
            ssao_radius: 0.5,
            ssao_strength: 1.0,
            bloom_threshold: 1.0,
            bloom_intensity: 0.5,
            vignette_intensity: 0.15,
            vignette_smoothness: 0.5,
            chromatic_aberration: 0.002,
            film_grain: 0.02,
            fog_color: [0.5, 0.6, 0.7],
            fog_density: 0.01,
            fog_height_falloff: 0.1,
        }
    }
}

/// Represents a Vulkan buffer with its associated memory, size, and versioning for cache optimization.
pub struct Buffer {
    pub handle: vk::Buffer,
    pub allocation: Arc<Mutex<Option<gpu_allocator::vulkan::Allocation>>>,
    pub size: vk::DeviceSize,
    pub ptr: *mut std::ffi::c_void,
    pub address: u64,
    pub version: Arc<std::sync::atomic::AtomicU64>,
}

impl Clone for Buffer {
    fn clone(&self) -> Self {
        Self {
            handle: self.handle,
            allocation: self.allocation.clone(),
            size: self.size,
            ptr: self.ptr,
            address: self.address,
            version: self.version.clone(),
        }
    }
}


impl std::fmt::Debug for Buffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("handle", &self.handle)
            .field("size", &self.size)
            .field("address", &self.address)
            .field("version", &self.version)
            .finish()
    }
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

/// Represents a framebuffer attachment, including its Vulkan image, optional allocation, and view.
pub struct Attachment {
    pub image: vk::Image,
    pub allocation: Arc<Mutex<Option<gpu_allocator::vulkan::Allocation>>>,
    pub view: vk::ImageView,
    pub extent: vk::Extent2D,
    pub version: Arc<std::sync::atomic::AtomicU64>,
}

impl Clone for Attachment {
    fn clone(&self) -> Self {
        Self {
            image: self.image,
            allocation: self.allocation.clone(),
            view: self.view,
            extent: self.extent,
            version: self.version.clone(),
        }
    }
}

/// Represents the data required to draw a single mesh instance.
#[derive(Debug)]
pub struct MeshDraw {
    pub model: spark_math::Mat4,
    pub vertex_count: u32,
    pub index_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub material_index: u32,
    pub bounding_radius: f32,
}

pub struct LightDraw {
    pub position: spark_math::Vec3,
    pub direction: spark_math::Vec3,
    pub color: spark_math::Vec3,
    pub intensity: f32,
    pub range: f32,
    pub light_type: u32, // 0: Dir, 1: Point, 2: Spot
    pub spot_angles: [f32; 2], // inner, outer (cos)
}

pub struct FramePacket {
    pub view_matrix: spark_math::Mat4,
    pub projection_matrix: spark_math::Mat4,
    pub opaque_meshes: Vec<MeshDraw>,
    pub transparent_meshes: Vec<MeshDraw>,
    pub lights: Vec<LightDraw>,
}

#[derive(Debug)]
pub struct RenderFrame {
    pub command_buffer: vk::CommandBuffer,
    pub image_available: vk::Semaphore,
    pub render_finished: vk::Semaphore,
    pub in_flight: vk::Fence,
    pub global_buffer: Option<Buffer>,
    pub light_buffer: Option<Buffer>,
    pub global_descriptor_set: vk::DescriptorSet,
    pub instance_pool: Vec<Buffer>,
    pub instance_index: usize,
    pub indirect_commands_buffer: Option<Buffer>,
    pub object_data_buffer: Option<Buffer>,
    pub draw_count_buffer: Option<Buffer>,
    pub transparent_indirect_buffer: Option<Buffer>,
    pub transparent_object_buffer: Option<Buffer>,
    pub secondary_command_buffers: Vec<vk::CommandBuffer>,
    pub light_view_projs: [spark_math::Mat4; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ObjectDataSSBO {
    pub model_row0: spark_math::Vec4,
    pub model_row1: spark_math::Vec4,
    pub model_row2: spark_math::Vec4,
    pub sphere: spark_math::Vec4,
    pub index_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub material_index: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct MaterialDataSSBO {
    pub albedo_factor: spark_math::Vec4,
    pub emissive_factor: spark_math::Vec4,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub alpha_cutoff: f32,
    pub flags: u32,
    pub albedo_texture: i32,
    pub normal_texture: i32,
    pub metallic_roughness_texture: i32,
    pub emissive_texture: i32,
    pub occlusion_texture: i32,
    pub padding: [i32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct ClusterAABB {
    pub min: spark_math::Vec4,
    pub max: spark_math::Vec4,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct LightGrid {
    pub offset: u32,
    pub count: u32,
}

pub struct ResourceTracker {
    pub image_layouts: Arc<Mutex<std::collections::HashMap<vk::Image, vk::ImageLayout>>>,
}

impl Default for ResourceTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceTracker {
    pub fn new() -> Self {
        Self { image_layouts: Arc::new(Mutex::new(std::collections::HashMap::new())) }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn transition_image(
        &self,
        cb: vk::CommandBuffer,
        device: &ash::Device,
        image: vk::Image,
        new_layout: vk::ImageLayout,
        _src_access: vk::AccessFlags,
        dst_access: vk::AccessFlags,
        _src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
        aspect_mask: vk::ImageAspectFlags,
    ) {
        let mut layouts = self.image_layouts.lock().unwrap();
        let old_layout = *layouts.get(&image).unwrap_or(&vk::ImageLayout::UNDEFINED);
        if old_layout == new_layout { return; }

        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ)
            .dst_stage_mask(vk::PipelineStageFlags2::from_raw(dst_stage.as_raw() as u64))
            .dst_access_mask(vk::AccessFlags2::from_raw(dst_access.as_raw() as u64))
            .old_layout(old_layout)
            .new_layout(new_layout)
            .image(image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            });

        let dependency_info = vk::DependencyInfo::default()
            .image_memory_barriers(std::slice::from_ref(&barrier));

        unsafe {
            device.cmd_pipeline_barrier2(cb, &dependency_info);
        }
        layouts.insert(image, new_layout);
    }
}

impl Attachment {
    /// Destroys the attachment resources.
    pub fn destroy(&self, device: &ash::Device, allocator: &std::sync::Arc<std::sync::Mutex<gpu_allocator::vulkan::Allocator>>) {
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            if let Some(alloc) = self.allocation.lock().unwrap().take() {
                allocator.lock().unwrap().free(alloc).unwrap();
            }
        }
    }

    pub fn create_image_resource(
        device: &crate::vulkan::device::VulkanDevice,
        width: u32,
        height: u32,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        samples: vk::SampleCountFlags,
    ) -> Result<Self, crate::error::RendererError> {

        let (img, allocation) = device.create_image(&crate::vulkan::device::ImageCreateParams {
            width,
            height,
            mip_levels: 1,
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            usage,
            properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            samples,
        })?;
        let view = device.create_image_view(img, format, 1);
        Ok(Self {
            image: img,
            allocation: Arc::new(Mutex::new(Some(allocation))),
            view,
            extent: vk::Extent2D { width, height },
            version: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        })
    }

    pub fn recreate(
        &mut self,
        device: &crate::vulkan::device::VulkanDevice,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Result<(), crate::error::RendererError> {
        unsafe {
            device.device.destroy_image_view(self.view, None);
            device.device.destroy_image(self.image, None);
            if let Some(alloc) = self.allocation.lock().unwrap().take() {
                device.allocator.lock().unwrap().free(alloc).unwrap();
            }
        }

        let is_depth = format == vk::Format::D32_SFLOAT || format == vk::Format::D32_SFLOAT_S8_UINT || format == vk::Format::D24_UNORM_S8_UINT;
        let usage = if is_depth {
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC
        } else {
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::INPUT_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::STORAGE
        };

        let (img, allocation) = device.create_image(&crate::vulkan::device::ImageCreateParams {
            width,
            height,
            mip_levels: 1,
            format,
            tiling: vk::ImageTiling::OPTIMAL,
            usage,
            properties: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            samples: vk::SampleCountFlags::TYPE_1,
        })?;

        self.image = img;
        *self.allocation.lock().unwrap() = Some(allocation);
        self.view = device.create_image_view(img, format, 1);
        self.extent = vk::Extent2D { width, height };
        self.version.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(())
    }
}

pub fn create_frame_attachments(
    device: &crate::vulkan::device::VulkanDevice,
    extent: vk::Extent2D,
    format: vk::Format,
    msaa: vk::SampleCountFlags,
) -> Result<Vec<Attachment>, crate::error::RendererError> {
    (0..MAX_FRAMES_IN_FLIGHT)
        .map(|_| {
            Attachment::create_image_resource(
                device,
                extent.width,
                extent.height,
                format,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                msaa,
            )
        })
        .collect()
}
