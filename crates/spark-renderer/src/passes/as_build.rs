use super::{RenderContext, RenderPass};
use ash::vk;

/// A rendering pass that builds and manages acceleration structures for ray tracing.
pub struct AccelerationStructurePass {
    pub vertex_stride: u64,
}

impl RenderPass for AccelerationStructurePass {
    fn name(&self) -> &str {
        "AccelerationStructurePass"
    }
    fn outputs(&self) -> Vec<&'static str> {
        vec!["SceneTLAS"]
    }

    fn gpu_resource_access(&self) -> Vec<(String, vk::AccessFlags, vk::PipelineStageFlags)> {
        vec![(
            "SceneTLAS".to_string(),
            vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR,
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR,
        )]
    }

    fn record_commands(&self, ctx: &RenderContext) {
        let renderer = ctx.renderer;
        if !renderer.device.rt_supported {
            return;
        }

        if let Some(packet) = &renderer.current_packet {
            if !packet.opaque_meshes.is_empty() {
                if let (Some(vb), Some(ib)) = (
                    renderer.gpu_resource_manager.global_vertex_buffer.as_ref(),
                    renderer.gpu_resource_manager.global_index_buffer.as_ref(),
                ) {
                    let mut as_manager = renderer.as_manager.lock().unwrap();

                    let params = crate::vulkan::as_manager::TlasBuildParams {
                        device: &renderer.device,
                        cb: ctx.command_buffer,
                        packet,
                        global_vb: vb,
                        global_ib: ib,
                        vertex_stride: self.vertex_stride,
                        frame_index: ctx.current_frame,
                        frame_id: renderer.frame_manager.frame_index,
                    };
                    let _ = as_manager.build_scene_tlas(params);

                    if renderer.frame_manager.frame_index % 100 == 0 {
                        as_manager.evict_unused_blas(
                            &renderer.device,
                            renderer.frame_manager.frame_index,
                        );
                    }
                }
            }
        }
    }
}

impl Default for AccelerationStructurePass {
    fn default() -> Self {
        Self::new()
    }
}

impl AccelerationStructurePass {
    pub fn new() -> Self {
        Self {
            vertex_stride: std::mem::size_of::<crate::vertex::Vertex>() as u64,
        }
    }
}
