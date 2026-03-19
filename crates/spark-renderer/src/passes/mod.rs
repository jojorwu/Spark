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

use ash::vk;
use crate::Renderer;

pub struct RenderContext<'a> {
    pub renderer: &'a Renderer,
    pub command_buffer: vk::CommandBuffer,
    pub current_frame: usize,
    pub image_index: u32,
}

pub trait RenderPass: Send + Sync {
    fn name(&self) -> &str;

    fn is_enabled(&self, _renderer: &Renderer) -> bool {
        true
    }

    /// Per-frame resource updates (e.g., uploading UBOs, updating dynamic descriptor sets)
    fn prepare(&self, _renderer: &Renderer, _current_frame: usize) {}

    /// Initial or global descriptor set updates
    fn update_descriptor_sets(&self, _renderer: &Renderer) {}

    fn record_commands(&self, ctx: &RenderContext);

    /// Get a specific resource view from the pass (e.g., "output", "pyramid", "shadow_map")
    fn get_resource_view(&self, _name: &str, _frame_index: usize) -> Option<vk::ImageView> {
        None
    }

    /// Get a specific resource buffer from the pass (e.g., "light_grid", "index_list")
    fn get_resource_buffer(&self, _name: &str) -> Option<crate::resource::Buffer> {
        None
    }

    /// Optional cleanup for resources not managed by the pass itself
    fn destroy(&mut self, _renderer: &Renderer) {}
}
