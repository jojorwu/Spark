use thiserror::Error;
use ash::vk;

#[derive(Error, Debug)]
pub enum RendererError {
    #[error("Vulkan initialization failed: {0}")]
    VulkanInit(#[from] ash::LoadingError),

    #[error("Vulkan error: {0}")]
    Vulkan(#[from] vk::Result),

    #[error("No suitable physical device found")]
    NoSuitableDevice,

    #[error("Failed to create surface")]
    SurfaceCreation,
}
