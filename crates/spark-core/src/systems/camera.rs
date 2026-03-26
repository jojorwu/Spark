use crate::System;

pub struct CameraSystem;

impl System for CameraSystem {
    fn name(&self) -> &str {
        "CameraSystem"
    }

    fn resource_access(&self) -> crate::ResourceAccess {
        crate::ResourceAccess::new()
            .with_scene(crate::Access::Read)
            .with_renderer(crate::Access::Write)
    }

    fn update(&mut self, ctx: &crate::FrameContext) {
        let _scene = ctx.scene();
        // The engine main loop already finds the active camera and updates the renderer.
        // This system could be used for more advanced camera logic, like following targets.
    }
}
