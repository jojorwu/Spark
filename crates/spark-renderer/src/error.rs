use ash::vk;
use thiserror::Error;

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

    #[error("Shader compilation error: {0}")]
    ShaderCompilation(String),

    #[error("Resource loading failed: {0}")]
    ResourceLoading(String),
}
