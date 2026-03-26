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
pub mod ssr;
pub mod sprite;
pub mod point_shadow;
pub mod ssgi;
pub mod luminance;
pub mod dof;
pub mod rt;
pub mod as_build;

use ash::vk;
use crate::Renderer;

pub struct RenderContext<'a> {
    pub renderer: &'a Renderer,
    pub command_buffer: vk::CommandBuffer,
    pub current_frame: usize,
    pub image_index: u32,
    pub delta: f32,
}

/// A trait representing a modular rendering pass.
pub enum ResourceBinding {
    StorageImage(String),
    SampledImage(String),
    StorageBuffer(String),
    UniformBuffer(String),
    AccelerationStructure(String),
}

pub trait RenderPass: Send + Sync {
    /// Returns the unique name of the rendering pass.
    fn name(&self) -> &str;

    /// Declarative GPU resource requirements for the pass.
    fn gpu_resource_access(&self) -> Vec<(String, vk::AccessFlags, vk::PipelineStageFlags)> { Vec::new() }

    /// Returns the list of resource bindings required by this pass.
    fn bindings(&self) -> Vec<ResourceBinding> { Vec::new() }

    /// Returns true if the pass is currently enabled and should be executed.
    fn is_enabled(&self, _renderer: &Renderer) -> bool {
        true
    }

    /// Per-frame resource updates (e.g., uploading UBOs, updating dynamic descriptor sets).
    fn prepare(&self, _renderer: &Renderer, _current_frame: usize) {}

    fn set_tlas(&self, _tlas: crate::vulkan::as_manager::AccelerationStructure, _frame_index: usize, _renderer: &Renderer) {}

    /// Performs initial or global descriptor set updates for the pass.
    fn update_descriptor_sets(&self, _renderer: &Renderer) {}

    /// Checks if descriptors need updating based on resource versions.
    fn needs_descriptor_update(&self, _renderer: &Renderer, _frame_index: usize) -> bool { true }

    fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout { vk::DescriptorSetLayout::null() }
    fn set_descriptor_sets(&mut self, _sets: Vec<vk::DescriptorSet>) {}

    /// Records Vulkan commands for this pass into the provided command buffer.
    fn record_commands(&self, ctx: &RenderContext);

    /// Records commands into secondary command buffers for parallel execution.
    fn record_secondary_commands(&self, _ctx: &RenderContext) -> Vec<vk::CommandBuffer> { Vec::new() }

    /// Retrieves a specific image resource view from the pass for cross-pass communication.
    fn get_resource_view(&self, _name: &str, _frame_index: usize) -> Option<vk::ImageView> {
        None
    }

    /// Retrieves a specific buffer resource from the pass for cross-pass communication.
    fn get_resource_buffer(&self, _name: &str) -> Option<crate::resource::Buffer> {
        None
    }

    /// Notifies the pass that the viewport or swapchain has been resized.
    fn on_resize(&mut self, _renderer: &mut Renderer, _new_extent: vk::Extent2D) {}

    /// Returns the dependencies of this pass.
    fn dependencies(&self) -> Vec<&'static str> { Vec::new() }

    /// Cleans up resources managed by this pass.
    fn destroy(&mut self, _renderer: &mut Renderer) {}

    /// Returns the input resource names for this pass.
    fn inputs(&self) -> Vec<&'static str> { Vec::new() }

    /// Returns the output resource names for this pass.
    fn outputs(&self) -> Vec<&'static str> { Vec::new() }
}
