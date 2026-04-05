use crate::System;

pub struct ComponentSystem;

impl System for ComponentSystem {
    fn name(&self) -> &str {
        "ComponentSystem"
    }
    fn resource_access(&self) -> crate::ResourceAccess {
        crate::ResourceAccess::new().with_scene(crate::Access::Write)
    }
    /// Updates all active components in the scene in parallel.
    ///
    /// This system leverages Rayon to parallelize the execution of `on_update` across all nodes.
    ///
    /// # Safety and Optimization
    ///
    /// To maximize performance and minimize per-frame overhead, we use raw pointer access
    /// to the scene's node storage. This avoids expensive collections (like `Vec::collect`)
    /// within the hot loop. Each parallel task is guaranteed to operate on a unique node,
    /// ensuring thread safety for component mutation.
    fn update(&mut self, ctx: &crate::FrameContext) {
        use rayon::prelude::*;

        unsafe {
            let scene = ctx.scene_mut();

            // Optimization: Iterate directly over the slotmap's internal storage
            // if possible, or use a parallel iterator that avoids full re-collection.
            // Since SlotMap doesn't provide a direct par_iter_mut for (Key, &mut V),
            // we use the established raw pointer pattern for high performance.
            let nodes_ptr = &mut scene.nodes as *mut slotmap::SlotMap<crate::scene::NodeKey, crate::scene::Node> as usize;

            scene.nodes.keys().par_bridge().for_each(|key| {
                let nodes = &mut *(nodes_ptr as *mut slotmap::SlotMap<crate::scene::NodeKey, crate::scene::Node>);
                if let Some(node) = nodes.get_mut(key) {
                    for component in &mut node.components {
                        component.on_update(key, ctx);
                    }
                }
            });
        }
    }
}
