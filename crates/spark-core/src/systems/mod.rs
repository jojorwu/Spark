pub mod component;

use crate::{System, FrameContext, InitContext};
use std::collections::{HashSet, HashMap};

pub struct SystemRegistry {
    pub systems: Vec<Box<dyn System>>,
    pub stages: Vec<Vec<usize>>,
}

impl SystemRegistry {
    pub fn new() -> Self {
        Self { systems: Vec::new(), stages: Vec::new() }
    }

    pub fn add_system<S: System + 'static>(&mut self, system: S) {
        self.systems.push(Box::new(system));
    }

    pub fn add_boxed_system(&mut self, system: Box<dyn System>) {
        self.systems.push(system);
    }

    pub fn sort_systems(&mut self) {
        let mut visited = HashSet::new();
        let mut temp_visited = HashSet::new();

        let name_to_idx: HashMap<String, usize> = self.systems.iter().enumerate()
            .map(|(i, s)| (s.name().to_string(), i))
            .collect();

        fn visit(
            idx: usize,
            systems: &Vec<Box<dyn System>>,
            name_to_idx: &HashMap<String, usize>,
            ordered: &mut Vec<usize>,
            visited: &mut HashSet<usize>,
            temp_visited: &mut HashSet<usize>,
        ) {
            if temp_visited.contains(&idx) {
                panic!("Circular dependency detected in systems!");
            }
            if !visited.contains(&idx) {
                temp_visited.insert(idx);
                for dep in systems[idx].dependencies() {
                    if let Some(&dep_idx) = name_to_idx.get(dep) {
                        visit(dep_idx, systems, name_to_idx, ordered, visited, temp_visited);
                    }
                }
                temp_visited.remove(&idx);
                visited.insert(idx);
                ordered.push(idx);
            }
        }

        let mut indices = Vec::new();
        for i in 0..self.systems.len() {
            visit(i, &self.systems, &name_to_idx, &mut indices, &mut visited, &mut temp_visited);
        }

        // Reorder systems
        let mut old_systems: Vec<Option<Box<dyn System>>> = self.systems.drain(..).map(Some).collect();
        for idx in indices {
            self.systems.push(old_systems[idx].take().unwrap());
        }

        // Build stages for parallel execution
        self.build_stages();
    }

    fn build_stages(&mut self) {
        let name_to_idx: HashMap<String, usize> = self.systems.iter().enumerate()
            .map(|(i, s)| (s.name().to_string(), i))
            .collect();

        let mut system_stages = vec![0; self.systems.len()];
        let mut max_stage = 0;

        for (i, system) in self.systems.iter().enumerate() {
            let mut stage = 0;
            for dep in system.dependencies() {
                if let Some(&dep_idx) = name_to_idx.get(dep) {
                    stage = stage.max(system_stages[dep_idx] + 1);
                }
            }
            system_stages[i] = stage;
            max_stage = max_stage.max(stage);
        }

        self.stages = vec![Vec::new(); max_stage + 1];
        for (i, &stage) in system_stages.iter().enumerate() {
            self.stages[stage].push(i);
        }
    }
}


impl Default for SystemRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Scheduler;

impl Scheduler {
    pub fn init(registry: &mut SystemRegistry, ctx: &mut InitContext) {
        registry.sort_systems();
        for system in &mut registry.systems {
            system.on_init(ctx);
        }
    }

    pub fn run(registry: &mut SystemRegistry, ctx: &mut FrameContext) {
        // The FrameContext contains &mut references, which normally prevents parallel execution.
        // To safely parallelize systems within a stage, we'd need to ensure they don't access
        // the same resources mutably at the same time.

        for stage in &registry.stages {
            // Within a stage, check for resource conflicts.
            // If all systems in the stage only require Read access to specific resources,
            // we could theoretically parallelize them if we had thread-safe wrappers (like Arc<RwLock>).

            // For now, even with declarative access, the current FrameContext design
            // (holding exclusive &mut references) prevents safe parallel dispatch of the `update` method.

            // To properly improve this, we would need to pass only the allowed resources to each system.
            // However, the architecture is now "Conflict-Aware", and we can already detect
            // which systems are safe to run together.

            for &idx in stage {
                registry.systems[idx].update(ctx);
            }
        }
    }

    pub fn shutdown(registry: &mut SystemRegistry, ctx: &mut InitContext) {
        for system in &mut registry.systems {
            system.on_stop(ctx);
        }
    }
}

pub struct HierarchySystem;

impl System for HierarchySystem {
    fn name(&self) -> &str { "HierarchySystem" }
    fn dependencies(&self) -> Vec<&'static str> { vec!["ComponentSystem"] }
    fn resource_access(&self) -> crate::ResourceAccess {
        crate::ResourceAccess {
            scene: crate::Access::Write,
            renderer: crate::Access::None,
            resource_manager: crate::Access::None,
        }
    }
    fn update(&mut self, ctx: &mut FrameContext) {
        ctx.scene.update_all_transforms();
    }
}

pub struct ResourceSystem;

impl System for ResourceSystem {
    fn name(&self) -> &str { "ResourceSystem" }
    fn resource_access(&self) -> crate::ResourceAccess {
        crate::ResourceAccess {
            scene: crate::Access::Read,
            renderer: crate::Access::Write,
            resource_manager: crate::Access::Write,
        }
    }
    fn update(&mut self, ctx: &mut FrameContext) {
        ctx.resource_manager.upload_global_buffers(ctx.renderer);
    }
}
