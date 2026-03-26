use ash::vk;
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

    fn record_commands(&self, ctx: &RenderContext) {
        let renderer = ctx.renderer;
        if !renderer.device.rt_supported { return; }

        if let Some(packet) = &renderer.current_packet {
            if !packet.opaque_meshes.is_empty() {
                if let (Some(vb), Some(ib)) = (renderer.global_vertex_buffer.as_ref(), renderer.global_index_buffer.as_ref()) {
                    let mut as_manager = renderer.as_manager.lock().unwrap();

                    let _ = as_manager.build_scene_tlas(
                        &renderer.device, ctx.command_buffer, packet, vb, ib, self.vertex_stride, ctx.current_frame, renderer.frame_index
                    );

                    if renderer.frame_index % 100 == 0 {
                        as_manager.evict_unused_blas(&renderer.device, renderer.frame_index);
                    }
                }
            }
        }
    }
}

impl AccelerationStructurePass {
    pub fn new() -> Self {
        Self { vertex_stride: std::mem::size_of::<crate::vertex::Vertex>() as u64 }
    }
}
