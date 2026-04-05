use crate::resource::Buffer;
use crate::vulkan::device::VulkanDevice;
use crate::vulkan::texture::Texture;
use ash::vk;
use std::collections::HashMap;
use std::sync::atomic::AtomicU32;
use std::sync::Mutex;

pub struct GpuResourceManager {
    pub descriptor_pool: vk::DescriptorPool,
    pub texture_descriptor_sets: HashMap<vk::ImageView, vk::DescriptorSet>,
    pub global_vertex_buffer: Option<Buffer>,
    pub global_index_buffer: Option<Buffer>,
    pub global_material_buffer: Option<Buffer>,
    pub vertex_buffers: Vec<Buffer>,
    pub index_buffer: Option<Buffer>,
    pub bindless_descriptor_set_layout: vk::DescriptorSetLayout,
    pub bindless_descriptor_set: vk::DescriptorSet,
    pub next_bindless_index: AtomicU32,
    pub free_bindless_indices: Mutex<Vec<u32>>,
    pub default_texture: Option<Texture>,
}

impl GpuResourceManager {
    pub fn new(device: &VulkanDevice) -> Result<Self, crate::error::RendererError> {
        let descriptor_pool = Self::create_descriptor_pool(&device.device);
        let bindless_descriptor_set_layout = unsafe {
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(10000)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
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
        let bindless_descriptor_set = unsafe {
            device
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&[bindless_descriptor_set_layout]),
                )
                .map_err(|_| crate::error::RendererError::NoSuitableDevice)?[0]
        };
        Ok(Self {
            descriptor_pool,
            texture_descriptor_sets: HashMap::new(),
            global_vertex_buffer: None,
            global_index_buffer: None,
            global_material_buffer: None,
            vertex_buffers: Vec::new(),
            index_buffer: None,
            bindless_descriptor_set_layout,
            bindless_descriptor_set,
            next_bindless_index: AtomicU32::new(0),
            free_bindless_indices: Mutex::new(Vec::new()),
            default_texture: None,
        })
    }
    fn create_descriptor_pool(device: &ash::Device) -> vk::DescriptorPool {
        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(20000),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::INPUT_ATTACHMENT)
                .descriptor_count(100),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(100),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(100),
        ];
        unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .pool_sizes(&sizes)
                        .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
                        .max_sets(2000),
                    None,
                )
                .expect("Failed to create descriptor pool")
        }
    }
    pub fn destroy(&mut self, device: &VulkanDevice) {
        unsafe {
            for vb in self.vertex_buffers.drain(..) {
                device.destroy_buffer(vb);
            }
            if let Some(ib) = self.index_buffer.take() {
                device.destroy_buffer(ib);
            }
            if let Some(vb) = self.global_vertex_buffer.take() {
                device.destroy_buffer(vb);
            }
            if let Some(ib) = self.global_index_buffer.take() {
                device.destroy_buffer(ib);
            }
            if let Some(mb) = self.global_material_buffer.take() {
                device.destroy_buffer(mb);
            }
            device
                .device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            device
                .device
                .destroy_descriptor_set_layout(self.bindless_descriptor_set_layout, None);
        }
    }
}
