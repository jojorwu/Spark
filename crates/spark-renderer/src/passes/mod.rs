pub mod shadow;
pub mod gbuffer;
pub mod lighting;
pub mod post_process;
pub mod culling;
pub mod hiz;
pub mod ssao;
pub mod clustered;
pub mod taa;
pub mod grid;
pub mod volumetric;
pub mod forward;
pub mod particle;

use ash::vk;
use crate::Renderer;

pub struct RenderContext<'a> {
    pub renderer: &'a Renderer,
    pub command_buffer: vk::CommandBuffer,
    pub current_frame: usize,
    pub image_index: u32,
}

/// A trait representing a modular rendering pass.
pub trait RenderPass: Send + Sync {
    /// Returns the unique name of the rendering pass.
    fn name(&self) -> &str;

    /// Returns true if the pass is currently enabled and should be executed.
    fn is_enabled(&self, _renderer: &Renderer) -> bool {
        true
    }

    /// Per-frame resource updates (e.g., uploading UBOs, updating dynamic descriptor sets).
    fn prepare(&self, _renderer: &Renderer, _current_frame: usize) {}

    /// Performs initial or global descriptor set updates for the pass.
    fn update_descriptor_sets(&self, _renderer: &Renderer) {}

    /// Checks if descriptors need updating based on resource versions.
    fn needs_descriptor_update(&self, _renderer: &Renderer, _frame_index: usize) -> bool { true }

    /// Records Vulkan commands for this pass into the provided command buffer.
    fn record_commands(&self, ctx: &RenderContext);

    /// Retrieves a specific image resource view from the pass for cross-pass communication.
    fn get_resource_view(&self, _name: &str, _frame_index: usize) -> Option<vk::ImageView> {
        None
    }

    /// Retrieves a specific buffer resource from the pass for cross-pass communication.
    fn get_resource_buffer(&self, _name: &str) -> Option<crate::resource::Buffer> {
        None
    }

    /// Cleans up resources managed by this pass.
    fn destroy(&mut self, _renderer: &Renderer) {}
}
