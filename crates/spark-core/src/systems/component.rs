use crate::System;
use crate::scene::NodeKey;

pub struct ComponentSystem;

impl System for ComponentSystem {
    fn name(&self) -> &str { "ComponentSystem" }
    fn resource_access(&self) -> crate::ResourceAccess {
        crate::ResourceAccess {
            scene: crate::Access::Write,
            renderer: crate::Access::None,
            resource_manager: crate::Access::None,
        }
    }
    fn update(&mut self, ctx: &mut crate::FrameContext) {
        // Component updates:
        // Due to the borrowing rules, we must take components out of the scene tree,
        // update them with a reference to the scene tree, and then put them back.

        let scene = ctx.scene();
        let node_keys: Vec<NodeKey> = scene.nodes.keys().collect();
        for key in node_keys {
            if let Some(node) = scene.nodes.get_mut(key) {
                let mut components = std::mem::take(&mut node.components);
                for component in &mut components {
                    component.on_update(key, scene, ctx.delta);
                }
                if let Some(node_after) = scene.nodes.get_mut(key) {
                    node_after.components = components;
                }
            }
        }
    }
}
