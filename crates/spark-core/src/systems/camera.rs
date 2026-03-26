use crate::System;

pub struct CameraSystem;

impl System for CameraSystem {
    fn name(&self) -> &str {
        "CameraSystem"
    }

    fn resource_access(&self) -> crate::ResourceAccess {
        crate::ResourceAccess {
            scene: crate::Access::Read,
            renderer: crate::Access::Write,
            resource_manager: crate::Access::None,
        }
    }

    fn update(&mut self, ctx: &crate::FrameContext) {
        let _scene = ctx.scene();
        // The engine main loop already finds the active camera and updates the renderer.
        // This system could be used for more advanced camera logic, like following targets.
    }
}
