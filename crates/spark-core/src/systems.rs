use crate::{Engine, System};

pub struct HierarchySystem;

impl System for HierarchySystem {
    fn update(&mut self, engine: &mut Engine, _delta: f32) {
        engine.scene.update_all_transforms();
    }
}

pub struct ResourceSystem;

impl System for ResourceSystem {
    fn update(&mut self, engine: &mut Engine, _delta: f32) {
        engine.resource_manager.upload_global_buffers(&mut engine.renderer);
    }
}
