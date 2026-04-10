use ash::vk;

/// Represents a GPU-resident texture.
pub struct Texture {
    pub image: vk::Image,
    pub allocation: Option<gpu_allocator::vulkan::Allocation>,
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
    pub mip_levels: u32,
    pub bindless_index: u32,
}

impl Texture {
    /// Creates a shallow copy of the texture handles.
    ///
    /// SAFETY: This does NOT clone the underlying allocation. The caller must ensure
    /// that the original texture remains valid for the lifetime of this copy, or that
    /// this copy is not used to free resources.
    pub unsafe fn shallow_copy(&self) -> Self {
        Self {
            image: self.image,
            allocation: None,
            view: self.view,
            sampler: self.sampler,
            mip_levels: self.mip_levels,
            bindless_index: self.bindless_index,
        }
    }
}
