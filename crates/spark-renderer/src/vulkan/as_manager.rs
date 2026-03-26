use ash::vk;
use crate::vulkan::device::VulkanDevice;
use crate::resource::Buffer;

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

    /// Builds a Top-Level Acceleration Structure (TLAS) for the entire scene and builds/caches BLAS as needed.
    ///
    /// # Arguments
    /// * `device` - The Vulkan device.
    /// * `cb` - The command buffer to record build commands into.
    /// * `packet` - The frame packet containing scene geometry.
    /// * `global_vb` - The global vertex buffer.
    /// * `global_ib` - The global index buffer.
    /// * `vertex_stride` - Stride of the vertex data in bytes.
    pub fn build_scene_tlas(
        &mut self,
        device: &VulkanDevice,
        cb: vk::CommandBuffer,
        packet: &crate::resource::FramePacket,
        global_vb: &Buffer,
        global_ib: &Buffer,
        vertex_stride: u64,
        frame_index: usize,
        frame_id: u64,
    ) -> Result<(), crate::error::RendererError> {
        // Cleanup old resources for this frame slot
        if let Some(mut old_tlas) = self.current_tlas[frame_index].take() {
            old_tlas.destroy(device);
        }
        for b in self.frame_scratch[frame_index].drain(..) {
            device.destroy_buffer(b);
        }

        let as_loader = device.as_loader.as_ref().ok_or(crate::error::RendererError::NoSuitableDevice)?;
        let mut scratch_buffers = Vec::new();
        let mut instances = Vec::new();

        for (i, mesh) in packet.opaque_meshes.iter().enumerate() {
            let key = BlasKey {
                mesh_id: mesh.mesh_id,
                vertex_offset: mesh.vertex_offset,
                first_index: mesh.first_index,
            };
            if !self.blas_cache.contains_key(&key) {
                let (b, scratch) = AccelerationStructure::new_blas(
                    device, as_loader, cb, global_vb, global_ib,
                    mesh.vertex_count, mesh.index_count, vertex_stride,
                    mesh.vertex_offset, mesh.first_index
                )?;
                scratch_buffers.push(scratch);
                self.blas_cache.insert(key, b);
            }
            self.blas_usage.insert(key, frame_id);
            let blas = self.blas_cache.get(&key).unwrap();

            let m = mesh.model.transpose();
            let transform = vk::TransformMatrixKHR {
                matrix: [
                    m.row(0).x, m.row(0).y, m.row(0).z, m.row(0).w,
                    m.row(1).x, m.row(1).y, m.row(1).z, m.row(1).w,
                    m.row(2).x, m.row(2).y, m.row(2).z, m.row(2).w,
                ],
            };

            instances.push(vk::AccelerationStructureInstanceKHR {
                transform,
                instance_custom_index_and_mask: vk::Packed24_8::new(i as u32, 0xFF),
                instance_shader_binding_table_record_offset_and_flags: vk::Packed24_8::new(0, vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8),
                acceleration_structure_reference: vk::AccelerationStructureReferenceKHR { device_handle: blas.address },
            });
        }

        // Barrier for BLAS builds to complete before TLAS build
        let barrier_data = [vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
            .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
            .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
            .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR)];

        let as_barrier = vk::DependencyInfo::default()
            .memory_barriers(&barrier_data);

        unsafe {
             device.device.cmd_pipeline_barrier2(cb, &as_barrier);
        }

        let (tlas, t_scratch, t_inst) = AccelerationStructure::new_tlas(device, as_loader, cb, &instances)?;
        scratch_buffers.push(t_scratch);
        scratch_buffers.push(t_inst);

        self.current_tlas[frame_index] = Some(tlas);
        self.frame_scratch[frame_index] = scratch_buffers;

        Ok(())
    }

    /// Cleans up all cached BLAS.
    /// Cleans up unused BLAS that haven't been seen in several frames.
    pub fn evict_unused_blas(&mut self, device: &VulkanDevice, current_frame_id: u64) {
        let keys_to_remove: Vec<BlasKey> = self.blas_usage.iter()
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
        for tlas in self.current_tlas.iter_mut() {
            if let Some(mut t) = tlas.take() {
                t.destroy(device);
            }
        }
        for scratch_list in self.frame_scratch.iter_mut() {
            for b in scratch_list.drain(..) {
                device.destroy_buffer(b);
            }
        }
    }
}

impl AccelerationStructure {
    /// Creates a new Bottom-Level Acceleration Structure (BLAS).
    pub fn new_blas(
        device: &VulkanDevice,
        as_loader: &ash::khr::acceleration_structure::Device,
        cb: vk::CommandBuffer,
        vertex_buffer: &Buffer,
        index_buffer: &Buffer,
        vertex_count: u32,
        index_count: u32,
        vertex_stride: u64,
        vertex_offset: i32,
        first_index: u32,
    ) -> Result<(Self, Buffer), crate::error::RendererError> {
        let vertex_address = vertex_buffer.address + (vertex_offset as u64 * vertex_stride);
        let index_address = index_buffer.address + (first_index as u64 * 4); // Assuming 32-bit indices

        let tri_data = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
            .vertex_format(vk::Format::R32G32B32_SFLOAT)
            .vertex_data(vk::DeviceOrHostAddressConstKHR { device_address: vertex_address })
            .vertex_stride(vertex_stride)
            .max_vertex(vertex_count)
            .index_type(vk::IndexType::UINT32)
            .index_data(vk::DeviceOrHostAddressConstKHR { device_address: index_address });

        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .geometry(vk::AccelerationStructureGeometryDataKHR { triangles: tri_data })
            .flags(vk::GeometryFlagsKHR::OPAQUE);

        let build_range = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(index_count / 3)
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
            as_loader.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build_info,
                &[index_count / 3],
                &mut size_info,
            );
        }

        let as_buffer = device.create_buffer(
            size_info.acceleration_structure_size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(as_buffer.handle)
            .size(size_info.acceleration_structure_size)
            .ty(vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);

        let handle = unsafe { as_loader.create_acceleration_structure(&create_info, None)? };
        let address = unsafe {
            as_loader.get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(handle)
            )
        };

        let scratch_buffer = device.create_buffer(
            size_info.build_scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let mut build_info = build_info;
        build_info.dst_acceleration_structure = handle;
        build_info.scratch_data = vk::DeviceOrHostAddressKHR { device_address: scratch_buffer.address };

        unsafe {
            as_loader.cmd_build_acceleration_structures(cb, &[build_info], &[&[build_range]]);
        }

        Ok((Self { handle, buffer: as_buffer, address }, scratch_buffer))
    }

    /// Creates a new Top-Level Acceleration Structure (TLAS).
    pub fn new_tlas(
        device: &VulkanDevice,
        as_loader: &ash::khr::acceleration_structure::Device,
        cb: vk::CommandBuffer,
        instances: &[vk::AccelerationStructureInstanceKHR],
    ) -> Result<(Self, Buffer, Buffer), crate::error::RendererError> {
        let instance_buffer = device.create_buffer(
            (instances.len() * std::mem::size_of::<vk::AccelerationStructureInstanceKHR>()) as u64,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;
        device.upload_to_buffer(&instance_buffer, instances);

        let geometry = vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                    .data(vk::DeviceOrHostAddressConstKHR { device_address: instance_buffer.address })
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
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(as_buffer.handle)
            .size(size_info.acceleration_structure_size)
            .ty(vk::AccelerationStructureTypeKHR::TOP_LEVEL);

        let handle = unsafe { as_loader.create_acceleration_structure(&create_info, None)? };
        let address = unsafe {
            as_loader.get_acceleration_structure_device_address(
                &vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(handle)
            )
        };

        let scratch_buffer = device.create_buffer(
            size_info.build_scratch_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let mut build_info = build_info;
        build_info.dst_acceleration_structure = handle;
        build_info.scratch_data = vk::DeviceOrHostAddressKHR { device_address: scratch_buffer.address };

        let build_range = vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(instances.len() as u32)
            .primitive_offset(0)
            .first_vertex(0)
            .transform_offset(0);

        unsafe {
            as_loader.cmd_build_acceleration_structures(cb, &[build_info], &[&[build_range]]);
        }

        Ok((Self { handle, buffer: as_buffer, address }, scratch_buffer, instance_buffer))
    }

    /// Destroys the acceleration structure and its buffer.
    pub fn destroy(&mut self, device: &VulkanDevice) {
        if let Some(ref as_loader) = device.as_loader {
            unsafe {
                as_loader.destroy_acceleration_structure(self.handle, None);
            }
        }
        device.destroy_buffer(std::mem::replace(&mut self.buffer, crate::resource::Buffer {
            handle: vk::Buffer::null(),
            allocation: std::sync::Arc::new(std::sync::Mutex::new(None)),
            size: 0,
            ptr: std::ptr::null_mut(),
            address: 0,
            version: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
        }));
    }
}
