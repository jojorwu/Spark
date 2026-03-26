use crate::System;

pub struct ComponentSystem;

impl System for ComponentSystem {
    fn name(&self) -> &str {
        "ComponentSystem"
    }
    fn resource_access(&self) -> crate::ResourceAccess {
        crate::ResourceAccess::new().with_scene(crate::Access::Write)
    }
    fn update(&mut self, ctx: &crate::FrameContext) {
        use rayon::prelude::*;
        // Component updates in parallel:
        // We iterate over nodes and their components, allowing them to run in parallel.
        // Safety: We use the FrameContext to safely access shared resources or enqueue commands.
        // Direct mutable access to the scene is unsafe and should be avoided in on_update.

        unsafe {
            let scene = ctx.scene_mut();
            let nodes: Vec<_> = scene.nodes.iter_mut().collect();

            nodes.into_par_iter().for_each(|(key, node)| {
                for component in &mut node.components {
                    component.on_update(key, ctx);
                }
            });
        }
    }
}
