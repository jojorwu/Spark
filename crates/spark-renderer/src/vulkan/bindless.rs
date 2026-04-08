use crate::vulkan::device::VulkanDevice;
use ash::vk;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

/// Manages a global bindless descriptor set for textures.
///
/// This manager provides a high-level API for allocating indices within a large
/// descriptor array (typically 10,000+ slots) and updating them with GPU texture resources.
pub struct BindlessManager {
    pub layout: vk::DescriptorSetLayout,
    pub set: vk::DescriptorSet,
    next_index: AtomicU32,
    free_indices: Mutex<Vec<u32>>,
    max_descriptors: u32,
}

impl BindlessManager {
    pub fn new(
        device: &VulkanDevice,
        pool: vk::DescriptorPool,
        max_descriptors: u32,
    ) -> Result<Self, crate::error::RendererError> {
        let layout = unsafe {
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(max_descriptors)
                .stage_flags(
                    vk::ShaderStageFlags::FRAGMENT
                        | vk::ShaderStageFlags::VERTEX
                        | vk::ShaderStageFlags::RAYGEN_KHR
                        | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                )];

            let flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND];

            let mut binding_flags =
                vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&flags);

            device
                .device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .bindings(&bindings)
                        .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                        .push_next(&mut binding_flags),
                    None,
                )
                .map_err(|_| crate::error::RendererError::NoSuitableDevice)?
        };

        let set = unsafe {
            device
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&[layout]),
                )
                .map_err(|_| crate::error::RendererError::NoSuitableDevice)?[0]
        };

        Ok(Self {
            layout,
            set,
            next_index: AtomicU32::new(0),
            free_indices: Mutex::new(Vec::new()),
            max_descriptors,
        })
    }

    /// Allocates a unique index for a bindless texture.
    ///
    /// This method first attempts to recycle an index from the free list.
    /// If the free list is empty, it increments the global index counter.
    pub fn allocate_index(&self) -> u32 {
        if let Some(idx) = self
            .free_indices
            .lock()
            .expect("Failed to lock free indices for allocation")
            .pop()
        {
            idx
        } else {
            let idx = self.next_index.fetch_add(1, Ordering::Relaxed);
            if idx >= self.max_descriptors {
                panic!(
                    "Exceeded maximum bindless descriptor count ({})",
                    self.max_descriptors
                );
            }
            idx
        }
    }

    /// Returns a bindless index to the free list for future reuse.
    pub fn deallocate_index(&self, index: u32) {
        self.free_indices
            .lock()
            .expect("Failed to lock free indices for deallocation")
            .push(index);
    }

    pub fn update_texture(
        &self,
        device: &ash::Device,
        index: u32,
        view: vk::ImageView,
        sampler: vk::Sampler,
    ) {
        let img_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(view)
            .sampler(sampler)];

        let writes = [vk::WriteDescriptorSet::default()
            .dst_set(self.set)
            .dst_binding(0)
            .dst_array_element(index)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&img_info)];

        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
    }

    /// Cleans up the bindless descriptor set layout.
    pub fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}
