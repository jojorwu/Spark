use ash::vk;

pub struct Texture {
    pub image: vk::Image,
    pub allocation: gpu_allocator::vulkan::Allocation,
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
    pub mip_levels: u32,
    pub bindless_index: u32,
}
