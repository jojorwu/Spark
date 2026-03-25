use ash::vk;
use crate::vulkan::device::VulkanDevice;
use crate::resource::Buffer;

pub struct AccelerationStructure {
    pub handle: vk::AccelerationStructureKHR,
    pub buffer: Buffer,
}

impl AccelerationStructure {
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

        Ok((Self { handle, buffer: as_buffer }, scratch_buffer))
    }

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

        Ok((Self { handle, buffer: as_buffer }, scratch_buffer, instance_buffer))
    }

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
