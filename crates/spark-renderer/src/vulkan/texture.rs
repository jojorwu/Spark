use ash::vk;

pub struct Texture {
    pub image: vk::Image,
    pub allocation: Option<gpu_allocator::vulkan::Allocation>,
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
    pub mip_levels: u32,
    pub bindless_index: u32,
}

impl Clone for Texture {
    fn clone(&self) -> Self {
        Self {
            image: self.image,
            allocation: None, // Allocations cannot be trivially cloned
            view: self.view,
            sampler: self.sampler,
            mip_levels: self.mip_levels,
            bindless_index: self.bindless_index,
        }
    }
}
