use crate::System;

pub struct HierarchySystem;

impl System for HierarchySystem {
    fn update(
        &mut self,
        scene: &mut crate::scene::Scene,
        _renderer: &mut spark_renderer::Renderer,
        _resource_manager: &mut crate::resource::ResourceManager,
        _delta: f32,
    ) {
        scene.update_all_transforms();
    }
}

pub struct ResourceSystem;

impl System for ResourceSystem {
    fn update(
        &mut self,
        _scene: &mut crate::scene::Scene,
        renderer: &mut spark_renderer::Renderer,
        resource_manager: &mut crate::resource::ResourceManager,
        _delta: f32,
    ) {
        resource_manager.upload_global_buffers(renderer);
    }
}
