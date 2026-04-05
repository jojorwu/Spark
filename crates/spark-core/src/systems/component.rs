use crate::System;

pub struct ComponentSystem {
    cached_keys: Vec<crate::scene::NodeKey>,
    cached_version: u64,
}

impl Default for ComponentSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl ComponentSystem {
    pub fn new() -> Self {
        Self {
            cached_keys: Vec::new(),
            cached_version: 0,
        }
    }
}

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

            let current_version = scene
                .nodes_version
                .load(std::sync::atomic::Ordering::Relaxed);
            if current_version != self.cached_version {
                self.cached_keys = scene.nodes.keys().collect();
                self.cached_version = current_version;
            }

            // Optimization: Avoid par_bridge() which has high overhead due to internal channels.
            // Instead, we collect keys into a temporary buffer. For large scenes,
            // the overhead of this collection is significantly lower than par_bridge.
            // Even better, we use the raw pointer pattern to access nodes in parallel safely
            // because each key is unique.
            let nodes_ptr = &mut scene.nodes
                as *mut slotmap::SlotMap<crate::scene::NodeKey, crate::scene::Node>
                as usize;

            self.cached_keys.par_iter().for_each(|&key| {
                let nodes = &mut *(nodes_ptr
                    as *mut slotmap::SlotMap<crate::scene::NodeKey, crate::scene::Node>);
                if let Some(node) = nodes.get_mut(key) {
                    for component in &mut node.components {
                        component.on_update(key, ctx);
                    }
                }
            });
        }
    }
}
