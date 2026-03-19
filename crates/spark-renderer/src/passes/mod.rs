pub mod shadow;
pub mod deferred;
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

pub trait RenderPass: Send + Sync {
    fn update_descriptor_sets(&self, renderer: &Renderer);
    fn record_commands(
        &self,
        renderer: &Renderer,
        command_buffer: vk::CommandBuffer,
        current_frame: usize,
    );
}
