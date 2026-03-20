use crate::System;

pub mod component;

pub struct HierarchySystem;

impl System for HierarchySystem {
    fn update(&mut self, ctx: &mut crate::FrameContext) {
        ctx.scene.update_all_transforms();
    }
}

pub struct ResourceSystem;

impl System for ResourceSystem {
    fn update(&mut self, ctx: &mut crate::FrameContext) {
        ctx.resource_manager.upload_global_buffers(ctx.renderer);
    }
}
