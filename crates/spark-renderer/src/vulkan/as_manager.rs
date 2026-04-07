use crate::resource::Buffer;
use crate::vulkan::device::VulkanDevice;
use ash::vk;

/// Represents a Vulkan Acceleration Structure (BLAS or TLAS).
pub struct AccelerationStructure {
    pub handle: vk::AccelerationStructureKHR,
    pub buffer: Buffer,
    pub address: u64,
}

/// A unique key identifying a Bottom-Level Acceleration Structure (BLAS) in the cache.
#[derive(Hash, PartialEq, Eq, Clone, Copy, Debug)]
pub struct BlasKey {
    pub mesh_id: u32,
    pub vertex_offset: i32,
    pub first_index: u32,
}

/// Parameters for building a Top-Level Acceleration Structure (TLAS).
pub struct TlasBuildParams<'a> {
    pub device: &'a VulkanDevice,
    pub cb: vk::CommandBuffer,
    pub packet: &'a crate::resource::FramePacket,
    pub global_vb: &'a Buffer,
    pub global_ib: &'a Buffer,
    pub vertex_stride: u64,
    pub frame_index: usize,
    pub frame_id: u64,
}

/// Parameters for building a Bottom-Level Acceleration Structure (BLAS).
pub struct BlasBuildParams<'a> {
    pub device: &'a VulkanDevice,
    pub as_loader: &'a ash::khr::acceleration_structure::Device,
    pub cb: vk::CommandBuffer,
    pub vertex_buffer: &'a Buffer,
    pub index_buffer: &'a Buffer,
    pub vertex_count: u32,
    pub index_count: u32,
    pub vertex_stride: u64,
    pub vertex_offset: i32,
    pub first_index: u32,
}

/// Manages the creation, caching, and lifecycle of Vulkan Acceleration Structures.
pub struct AccelerationStructureManager {
    pub blas_cache: std::collections::HashMap<BlasKey, AccelerationStructure>,
    pub blas_usage: std::collections::HashMap<BlasKey, u64>,
    pub current_tlas: [Option<AccelerationStructure>; crate::MAX_FRAMES_IN_FLIGHT],
    pub frame_scratch: [Vec<Buffer>; crate::MAX_FRAMES_IN_FLIGHT],
}

impl AccelerationStructureManager {
    /// Creates a new AccelerationStructureManager.
    pub fn new() -> Self {
        Self {
            blas_cache: std::collections::HashMap::new(),
            blas_usage: std::collections::HashMap::new(),
            current_tlas: Default::default(),
            frame_scratch: Default::default(),
        }
    }
}

impl Default for AccelerationStructureManager {
    fn default() -> Self {
        Self::new()
    }
}

impl AccelerationStructureManager {
    /// Builds a Top-Level Acceleration Structure (TLAS) for the entire scene and builds/caches BLAS as needed.
    pub fn build_scene_tlas(
        &mut self,
        params: TlasBuildParams,
    ) -> Result<(), crate::error::RendererError> {
        self.cleanup_frame_resources(params.device, params.frame_index);

        let as_loader = params
            .device
            .as_loader
            .as_ref()
            .ok_or(crate::error::RendererError::NoSuitableDevice)?;
        let mut scratch_buffers = Vec::new();

        let instances = self.prepare_tlas_instances(&params, as_loader, &mut scratch_buffers)?;

        // Ensure all BLAS builds are complete before starting the TLAS build.
        let barrier_data = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
            .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
            .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
            .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR)];

        let as_barrier = vk::DependencyInfo::default().memory_barriers(&barrier_data);

        unsafe {
            params
                .device
                .device
                .cmd_pipeline_barrier2(params.cb, &as_barrier);
        }

        let (tlas, t_scratch, t_inst) =
            AccelerationStructure::new_tlas(params.device, as_loader, params.cb, &instances)?;
        scratch_buffers.push(t_scratch);
        scratch_buffers.push(t_inst);

        self.current_tlas[params.frame_index] = Some(tlas);
        self.frame_scratch[params.frame_index] = scratch_buffers;

        Ok(())
    }

    /// Evicts unused BLAS entries from the cache that haven't been used for over 100 frames.
    pub fn evict_unused_blas(&mut self, device: &VulkanDevice, current_frame_id: u64) {
        let keys_to_remove: Vec<BlasKey> = self
            .blas_usage
            .iter()
            .filter(|(_, &last_used)| current_frame_id > last_used + 100)
            .map(|(&key, _)| key)
            .collect();

        for key in keys_to_remove {
            if let Some(mut blas) = self.blas_cache.remove(&key) {
                blas.destroy(device);
            }
            self.blas_usage.remove(&key);
        }
    }

    pub fn cleanup(&mut self, device: &VulkanDevice) {
        for (_, mut blas) in self.blas_cache.drain() {
            blas.destroy(device);
        }
        for i in 0..crate::MAX_FRAMES_IN_FLIGHT {
            self.cleanup_frame_resources(device, i);
        }
    }

    fn cleanup_frame_resources(&mut self, device: &VulkanDevice, frame_index: usize) {
        if let Some(mut old_tlas) = self.current_tlas[frame_index].take() {
            old_tlas.destroy(device);
        }
        for b in self.frame_scratch[frame_index].drain(..) {
            device.destroy_buffer(b);
        }
    }

    fn prepare_tlas_instances(
        &mut self,
        params: &TlasBuildParams,
        as_loader: &ash::khr::acceleration_structure::Device,
        scratch_buffers: &mut Vec<Buffer>,
    ) -> Result<Vec<vk::AccelerationStructureInstanceKHR>, crate::error::RendererError> {
        let mut instances = Vec::with_capacity(params.packet.opaque_meshes.len());

        for (i, mesh) in params.packet.opaque_meshes.iter().enumerate() {
            let key = BlasKey {
                mesh_id: mesh.mesh_id,
                vertex_offset: mesh.vertex_offset,
                first_index: mesh.first_index,
            };

            let blas_address = self.get_or_build_blas(params, as_loader, key, mesh, scratch_buffers)?;
            self.blas_usage.insert(key, params.frame_id);

            instances.push(vk::AccelerationStructureInstanceKHR {
                transform: Self::convert_to_vk_transform(&mesh.model),
                instance_custom_index_and_mask: vk::Packed24_8::new(i as u32, 0xFF),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(
                    0,
                    vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
                ),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR {
                    device_handle: blas_address,
                },
            });
        }
        Ok(instances)
    }

    fn get_or_build_blas(
        &mut self,
        params: &TlasBuildParams,
        as_loader: &ash::khr::acceleration_structure::Device,
        key: BlasKey,
        mesh: &crate::resource::MeshDraw,
        scratch_buffers: &mut Vec<Buffer>,
    ) -> Result<u64, crate::error::RendererError> {
        match self.blas_cache.entry(key) {
            std::collections::hash_map::Entry::Vacant(entry) => {
                let blas_params = BlasBuildParams {
                    device: params.device,
                    as_loader,
                    cb: params.cb,
                    vertex_buffer: params.global_vb,
                    index_buffer: params.global_ib,
                    vertex_count: mesh.vertex_count,
                    index_count: mesh.index_count,
                    vertex_stride: params.vertex_stride,
                    vertex_offset: mesh.vertex_offset,
                    first_index: mesh.first_index,
                };
                let (b, scratch) = AccelerationStructure::new_blas(blas_params)?;
                scratch_buffers.push(scratch);
                let address = b.address;
                entry.insert(b);
                Ok(address)
            }
            std::collections::hash_map::Entry::Occupied(entry) => Ok(entry.get().address),
        }
    }

    fn convert_to_vk_transform(model: &spark_math::Mat4) -> vk::TransformMatrixKHR {
        let m = model.transpose();
        vk::TransformMatrixKHR {
            matrix: [
                m.row(0).x, m.row(0).y, m.row(0).z, m.row(0).w,
                m.row(1).x, m.row(1).y, m.row(1).z, m.row(1).w,
                m.row(2).x, m.row(2).y, m.row(2).z, m.row(2).w,
            ],
        }
    }
}

impl AccelerationStructure {
    /// Creates a new Bottom-Level Acceleration Structure (BLAS).
    pub fn new_blas(
        params: BlasBuildParams,
    ) -> Result<(Self, Buffer), crate::error::RendererError> {
        let vertex_address =
            params.vertex_buffer.address + (params.vertex_offset as u64 * params.vertex_stride);
        let index_address = params.index_buffer.address + (params.first_index as u64 * 4); // Assuming 32-bit indices

        let tri_data = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .vertex_data(vk::DeviceOrHostAddressConstKHR {
                device_address: vertex_address,
            })
            .vertex_stride(params.vertex_stride)
            .max_vertex(params.vertex_count)
            .index_type(vk::IndexType::UINT32)
            .index_data(vk::DeviceOrHostAddressConstKHR {
                device_address: index_address,
            });

        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                triangles: tri_data,
            })
            .flags(vk::GeometryFlagsKHR::OPAQUE);

        let build_range = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(params.index_count / 3)
            .primitive_offset(0)
            .first_vertex(0)
            .transform_offset(0);

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(std::slice::from_ref(&geometry))
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD);

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            params.as_loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[params.index_count / 3],
                &mut size_info,
            );
        }

        let as_buffer = params.device.create_buffer(
            size_info.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(as_buffer.handle)
            .size(size_info.acceleration_structure_size)
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);

        let handle = unsafe {
            params
                .as_loader
                .create_acceleration_structure(&create_info, None)
                .expect("Failed to create BLAS handle")
        };
        let address = unsafe {
            params.as_loader.get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(handle),
            )
        };

        let scratch_buffer = params.device.create_buffer(
            size_info.build_scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let mut build_info = build_info;
        build_info.dst_acceleration_structure = handle;
        build_info.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: scratch_buffer.address,
        };

        unsafe {
            params.as_loader.cmd_build_acceleration_structures(
                params.cb,
                &[build_info],
                &[&[build_range]],
            );
        }

        Ok((
            Self {
                handle,
                buffer: as_buffer,
                address,
            },
            scratch_buffer,
        ))
    }

    /// Creates a new Top-Level Acceleration Structure (TLAS).
    pub fn new_tlas(
        device: &VulkanDevice,
        as_loader: &ash::khr::acceleration_structure::Device,
        cb: vk::CommandBuffer,
        instances: &[vk::AccelerationStructureInstanceKHR],
    ) -> Result<(Self, Buffer, Buffer), crate::error::RendererError> {
        let instance_buffer = device.create_buffer(
            std::mem::size_of_val(instances) as u64,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        device.upload_to_buffer(&instance_buffer, instances);

        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default().data(
                    vk::DeviceOrHostAddressConstKHR {
                        device_address: instance_buffer.address,
                    },
                ),
            });

        let build_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL)
            .flags(vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
            .geometries(std::slice::from_ref(&geometry))
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD);

        let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
        unsafe {
            as_loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[instances.len() as u32],
                &mut size_info,
            );
        }

        let as_buffer = device.create_buffer(
            size_info.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(as_buffer.handle)
            .size(size_info.acceleration_structure_size)
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);

        let handle = unsafe { as_loader.create_acceleration_structure(&create_info, None).expect("Failed to create TLAS handle") };
        let address = unsafe {
            as_loader.get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default()
                    .acceleration_structure(handle),
            )
        };

        let scratch_buffer = device.create_buffer(
            size_info.build_scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let mut build_info = build_info;
        build_info.dst_acceleration_structure = handle;
        build_info.scratch_data = vk::DeviceOrHostAddressKHR {
            device_address: scratch_buffer.address,
        };

        let build_range = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(instances.len() as u32)
            .primitive_offset(0)
            .first_vertex(0)
            .transform_offset(0);

        unsafe {
            as_loader.cmd_build_acceleration_structures(cb, &[build_info], &[&[build_range]]);
        }

        Ok((
            Self {
                handle,
                buffer: as_buffer,
                address,
            },
            scratch_buffer,
            instance_buffer,
        ))
    }

    /// Destroys the acceleration structure and its buffer.
    pub fn destroy(&mut self, device: &VulkanDevice) {
        if let Some(ref as_loader) = device.as_loader {
            unsafe {
                as_loader.destroy_acceleration_structure(self.handle, None);
            }
        }
        device.destroy_buffer(std::mem::replace(
            &mut self.buffer,
            crate::resource::Buffer {
                handle: vk::Buffer::null(),
                allocation: std::sync::Arc::new(std::sync::Mutex::new(None)),
                size: 0,
                ptr: std::ptr::null_mut(),
                address: 0,
                version: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            },
        ));
    }
}
