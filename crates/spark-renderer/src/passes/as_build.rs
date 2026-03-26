use ash::vk;
use crate::Renderer;
use super::{RenderPass, RenderContext};

/// A rendering pass that builds and manages acceleration structures for ray tracing.
pub struct AccelerationStructurePass {
    pub vertex_stride: u64,
}

impl RenderPass for AccelerationStructurePass {
    fn name(&self) -> &str { "AccelerationStructurePass" }
    fn outputs(&self) -> Vec<&'static str> { vec!["SceneTLAS"] }

    fn gpu_resource_access(&self) -> Vec<(String, vk::AccessFlags, vk::PipelineStageFlags)> {
        vec![
            ("SceneTLAS".to_string(), vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR, vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR),
        ]
    }

    fn prepare(&self, _renderer: &Renderer, _current_frame: usize) {
        // Preparation is handled globally in Renderer::prepare_frame for now.
        // This pass exists primarily for the RenderGraph to know about SceneTLAS.
    }

    fn record_commands(&self, _ctx: &RenderContext) {}
}

impl AccelerationStructurePass {
    pub fn new() -> Self {
        Self { vertex_stride: std::mem::size_of::<crate::vertex::Vertex>() as u64 }
    }
}
