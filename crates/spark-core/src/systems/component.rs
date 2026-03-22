use crate::System;
use crate::scene::NodeKey;

pub struct ComponentSystem;

impl System for ComponentSystem {
    fn name(&self) -> &str { "ComponentSystem" }
    fn update(&mut self, ctx: &mut crate::FrameContext) {
        // Component updates:
        // Due to the borrowing rules, we must take components out of the scene tree,
        // update them with a reference to the scene tree, and then put them back.

        let node_keys: Vec<NodeKey> = ctx.scene.nodes.keys().collect();
        for key in node_keys {
            if let Some(node) = ctx.scene.nodes.get_mut(key) {
                let mut components = std::mem::take(&mut node.components);
                for component in &mut components {
                    component.on_update(key, ctx.scene, ctx.delta);
                }
                if let Some(node_after) = ctx.scene.nodes.get_mut(key) {
                    node_after.components = components;
                }
            }
        }
    }
}
