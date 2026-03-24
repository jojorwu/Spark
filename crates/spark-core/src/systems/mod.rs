pub mod component;
pub mod camera;
pub mod physics;
pub mod time;

use crate::{System, FrameContext, InitContext};
use std::collections::{HashSet, HashMap};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CoreStage {
    First,
    PreUpdate,
    Update,
    PostUpdate,
    Last,
}

pub struct SystemRegistry {
    pub systems: HashMap<CoreStage, Vec<Box<dyn System>>>,
    pub sorted_indices: HashMap<CoreStage, Vec<Vec<usize>>>,
}

impl SystemRegistry {
    pub fn new() -> Self {
        let mut systems = HashMap::new();
        systems.insert(CoreStage::First, Vec::new());
        systems.insert(CoreStage::PreUpdate, Vec::new());
        systems.insert(CoreStage::Update, Vec::new());
        systems.insert(CoreStage::PostUpdate, Vec::new());
        systems.insert(CoreStage::Last, Vec::new());

        Self { systems, sorted_indices: HashMap::new() }
    }

    pub fn add_system<S: System + 'static>(&mut self, system: S) {
        self.systems.get_mut(&CoreStage::Update).unwrap().push(Box::new(system));
    }

    pub fn add_system_to_stage<S: System + 'static>(&mut self, stage: CoreStage, system: S) {
        self.systems.get_mut(&stage).unwrap().push(Box::new(system));
    }

    pub fn add_boxed_system(&mut self, system: Box<dyn System>) {
        self.systems.get_mut(&CoreStage::Update).unwrap().push(system);
    }

    pub fn sort_systems(&mut self) {
        let stages = [CoreStage::First, CoreStage::PreUpdate, CoreStage::Update, CoreStage::PostUpdate, CoreStage::Last];

        for stage in stages {
            let mut visited = HashSet::new();
            let mut temp_visited = HashSet::new();
            let stage_systems = self.systems.get_mut(&stage).unwrap();

            let name_to_idx: HashMap<String, usize> = stage_systems.iter().enumerate()
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
            for i in 0..stage_systems.len() {
                visit(i, stage_systems, &name_to_idx, &mut indices, &mut visited, &mut temp_visited);
            }

            // Reorder systems within the stage
            let mut old_systems: Vec<Option<Box<dyn System>>> = stage_systems.drain(..).map(Some).collect();
            for idx in indices {
                stage_systems.push(old_systems[idx].take().unwrap());
            }

            // Build sub-stages for parallel execution within this stage
            self.build_stage_batches(stage);
        }
    }

    fn build_stage_batches(&mut self, stage: CoreStage) {
        let stage_systems = self.systems.get(&stage).unwrap();
        let name_to_idx: HashMap<String, usize> = stage_systems.iter().enumerate()
            .map(|(i, s)| (s.name().to_string(), i))
            .collect();

        let mut system_batches = vec![0; stage_systems.len()];
        let mut max_batch = 0;

        for (i, system) in stage_systems.iter().enumerate() {
            let mut batch = 0;

            // 1. Dependency constraints
            for dep in system.dependencies() {
                if let Some(&dep_idx) = name_to_idx.get(dep) {
                    batch = batch.max(system_batches[dep_idx] + 1);
                }
            }

            // 2. Resource conflict constraints
            let mut conflict = true;
            while conflict {
                conflict = false;
                for (j, &batch_j) in system_batches.iter().enumerate().take(i) {
                    if batch_j == batch
                        && system.resource_access().conflicts_with(&stage_systems[j].resource_access()) {
                        batch += 1;
                        conflict = true;
                        break;
                    }
                }
            }

            system_batches[i] = batch;
            max_batch = max_batch.max(batch);
        }

        let mut sorted_batches = vec![Vec::new(); max_batch + 1];
        for (i, &batch) in system_batches.iter().enumerate() {
            sorted_batches[batch].push(i);
        }
        self.sorted_indices.insert(stage, sorted_batches);
    }
}


impl Default for SystemRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// The core scheduler responsible for orchestrating system initialization, execution, and shutdown.
///
/// The scheduler ensures systems are executed in an order that respects their dependencies
/// and minimizes resource access conflicts through stage-based dispatch.
pub struct Scheduler;

impl Scheduler {
    pub fn init(registry: &mut SystemRegistry, ctx: &mut InitContext) {
        registry.sort_systems();
        for systems in registry.systems.values_mut() {
            for system in systems {
                system.on_init(ctx);
            }
        }
    }

    pub fn run(registry: &mut SystemRegistry, ctx: &mut FrameContext) {
        let stages = [CoreStage::First, CoreStage::PreUpdate, CoreStage::Update, CoreStage::PostUpdate, CoreStage::Last];
        for stage in stages {
            if let Some(batches) = registry.sorted_indices.get(&stage) {
                let systems = registry.systems.get_mut(&stage).unwrap();
                for batch in batches {
                    use rayon::prelude::*;
                    if batch.len() > 1 {
                        batch.par_iter().for_each(|&idx| {
                            unsafe {
                                let systems_ptr = systems.as_ptr() as *mut Box<dyn crate::System>;
                                let ctx_ptr = ctx as *const FrameContext as *mut FrameContext;
                                (*systems_ptr.add(idx)).update(&*ctx_ptr);
                            }
                        });
                    } else if let Some(&idx) = batch.first() {
                        systems[idx].update(ctx);
                    }
                }
            }
        }
    }

    pub fn shutdown(registry: &mut SystemRegistry, ctx: &mut InitContext) {
        for systems in registry.systems.values_mut() {
            for system in systems {
                system.on_stop(ctx);
            }
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
    fn update(&mut self, ctx: &FrameContext) {
        unsafe { ctx.scene_mut().update_all_transforms(); }
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
    fn update(&mut self, ctx: &FrameContext) {
        unsafe {
            let renderer = ctx.renderer_mut();
            ctx.resource_manager_mut().upload_global_buffers(renderer);
        }
    }
}
