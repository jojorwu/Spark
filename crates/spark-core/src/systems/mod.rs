pub mod camera;
pub mod component;
pub mod physics;
pub mod time;

use crate::{FrameContext, InitContext, System};
use std::collections::{HashMap, HashSet};

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
    pub active_state: Option<String>,
    pub pending_state_change: Option<String>,
}

impl SystemRegistry {
    pub fn new() -> Self {
        let mut systems = HashMap::new();
        systems.insert(CoreStage::First, Vec::new());
        systems.insert(CoreStage::PreUpdate, Vec::new());
        systems.insert(CoreStage::Update, Vec::new());
        systems.insert(CoreStage::PostUpdate, Vec::new());
        systems.insert(CoreStage::Last, Vec::new());

        Self {
            systems,
            sorted_indices: HashMap::new(),
            active_state: None,
            pending_state_change: None,
        }
    }

    pub fn add_system<S: System + 'static>(&mut self, system: S) {
        self.systems
            .get_mut(&CoreStage::Update)
            .unwrap()
            .push(Box::new(system));
    }

    pub fn add_system_to_stage<S: System + 'static>(&mut self, stage: CoreStage, system: S) {
        self.systems.get_mut(&stage).unwrap().push(Box::new(system));
    }

    pub fn add_boxed_system(&mut self, system: Box<dyn System>) {
        self.systems
            .get_mut(&CoreStage::Update)
            .unwrap()
            .push(system);
    }

    pub fn sort_systems(&mut self) {
        let stages = [
            CoreStage::First,
            CoreStage::PreUpdate,
            CoreStage::Update,
            CoreStage::PostUpdate,
            CoreStage::Last,
        ];

        for stage in stages {
            let mut visited = HashSet::new();
            let mut temp_visited = HashSet::new();
            let stage_systems = self.systems.get_mut(&stage).unwrap();

            let name_to_idx: HashMap<String, usize> = stage_systems
                .iter()
                .enumerate()
                .map(|(i, s)| (s.name().to_string(), i))
                .collect();

            let mut reverse_before: HashMap<String, Vec<usize>> = HashMap::new();
            for (i, system) in stage_systems.iter().enumerate() {
                for before in system.run_before() {
                    reverse_before
                        .entry(before.to_string())
                        .or_default()
                        .push(i);
                }
            }

            fn visit(
                idx: usize,
                systems: &Vec<Box<dyn System>>,
                name_to_idx: &HashMap<String, usize>,
                ordered: &mut Vec<usize>,
                visited: &mut HashSet<usize>,
                temp_visited: &mut HashSet<usize>,
                reverse_before: &HashMap<String, Vec<usize>>,
            ) {
                if temp_visited.contains(&idx) {
                    panic!("Circular dependency detected in systems!");
                }
                if !visited.contains(&idx) {
                    temp_visited.insert(idx);

                    // 1. Explicit dependencies
                    for dep in systems[idx].dependencies() {
                        if let Some(&dep_idx) = name_to_idx.get(dep) {
                            visit(
                                dep_idx,
                                systems,
                                name_to_idx,
                                ordered,
                                visited,
                                temp_visited,
                                reverse_before,
                            );
                        }
                    }

                    // 2. run_after labels
                    for after in systems[idx].run_after() {
                        if let Some(&after_idx) = name_to_idx.get(after) {
                            visit(
                                after_idx,
                                systems,
                                name_to_idx,
                                ordered,
                                visited,
                                temp_visited,
                                reverse_before,
                            );
                        }
                    }

                    // 3. run_before labels (others wanting to run after this one)
                    if let Some(others) = reverse_before.get(systems[idx].name()) {
                        // This logic is slightly different: if B runs before A, then A depends on B.
                        // So if we are visiting A, we need to visit B first.
                        // reverse_before maps A -> [B]
                        for &before_idx in others {
                            // This is actually wrong in my head. If B runs before A, A depends on B.
                            // Wait, no. If B says "run_before A", then A should be visited *after* B.
                            // So A depends on B.
                            // Correct.
                            visit(
                                before_idx,
                                systems,
                                name_to_idx,
                                ordered,
                                visited,
                                temp_visited,
                                reverse_before,
                            );
                        }
                    }

                    temp_visited.remove(&idx);
                    visited.insert(idx);
                    ordered.push(idx);
                }
            }

            let mut indices = Vec::new();
            for i in 0..stage_systems.len() {
                visit(
                    i,
                    stage_systems,
                    &name_to_idx,
                    &mut indices,
                    &mut visited,
                    &mut temp_visited,
                    &reverse_before,
                );
            }

            // Reorder systems within the stage
            let mut old_systems: Vec<Option<Box<dyn System>>> =
                stage_systems.drain(..).map(Some).collect();
            for idx in indices {
                stage_systems.push(old_systems[idx].take().unwrap());
            }

            // Build sub-stages for parallel execution within this stage
            self.build_stage_batches(stage);
        }
    }

    fn build_stage_batches(&mut self, stage: CoreStage) {
        let stage_systems = self.systems.get(&stage).unwrap();
        let name_to_idx: HashMap<String, usize> = stage_systems
            .iter()
            .enumerate()
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
                        && system
                            .resource_access()
                            .conflicts_with(&stage_systems[j].resource_access())
                    {
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

    pub fn set_state(registry: &mut SystemRegistry, state: &str) {
        registry.pending_state_change = Some(state.to_string());
    }

    pub fn apply_state_changes(registry: &mut SystemRegistry, ctx: &mut InitContext) {
        if let Some(state) = registry.pending_state_change.take() {
            let old_state = registry.active_state.clone();
            registry.active_state = Some(state.clone());

            for systems in registry.systems.values_mut() {
                for system in systems {
                    let allowed = system.run_in_states();
                    if allowed.is_empty() {
                        continue;
                    }

                    if let Some(ref old) = old_state {
                        if allowed.contains(old) && !allowed.contains(&state) {
                            system.on_exit(ctx);
                        }
                    }

                    if allowed.contains(&state)
                        && (old_state.is_none() || !allowed.contains(old_state.as_ref().unwrap()))
                    {
                        system.on_enter(ctx);
                    }
                }
            }
        }
    }

    /// Executes all registered systems according to their stages and dependencies.
    ///
    /// The scheduler processes systems in a predefined sequence of stages. Within each stage,
    /// systems are executed in parallel batches. A batch consists of systems that have
    /// no mutual dependencies and no conflicting resource requirements.
    ///
    /// # Safety and Multithreading
    ///
    /// Execution within a batch uses `rayon` for high-performance parallel dispatch.
    ///
    /// ## The Safety Contract (Raw Pointer Optimization)
    ///
    /// To maximize performance and bypass standard borrow checker restrictions during parallel
    /// execution, we employ a raw pointer optimization pattern:
    ///
    /// 1. **Context/System Capture**: The `FrameContext` and the system slice are converted
    ///    to raw pointers (represented as `usize`) to allow them to be captured by the
    ///    `Send + Sync` closures required by Rayon.
    /// 2. **Guaranteed Disjoint Access**: The engine's architecture ensures safety through
    ///    **System Batching**. During `sort_systems`, the engine analyzes the `ResourceAccess`
    ///    declarations of every system. Systems are only grouped into the same parallel batch
    ///    if their resource requirements are mutually compatible (e.g., multiple readers,
    ///    or a single writer with no other readers/writers of that specific resource).
    /// 3. **Controlled Mutation**: Since no two systems in a batch access the same mutable
    ///    data, they can safely execute in parallel despite the use of `unsafe` pointers.
    /// 4. **Sequential Consistency**: Stages themselves (First -> Last) are always executed
    ///    sequentially, providing synchronization points between groups of systems.
    pub fn run(registry: &mut SystemRegistry, ctx: &mut FrameContext) {
        let stages = [
            CoreStage::First,
            CoreStage::PreUpdate,
            CoreStage::Update,
            CoreStage::PostUpdate,
            CoreStage::Last,
        ];

        for stage in stages {
            if let Some(batches) = registry.sorted_indices.get(&stage) {
                let systems = registry.systems.get_mut(&stage).unwrap();
                let active_state = registry.active_state.as_deref();

                for batch in batches {
                    use rayon::prelude::*;

                    // Optimization: Use parallel iteration only for batches with multiple systems.
                    if batch.len() > 1 {
                        let systems_ptr = systems.as_ptr() as usize;
                        let ctx_ptr = ctx as *const FrameContext as usize;

                        batch.par_iter().for_each(|&idx| unsafe {
                            let systems_ptr = systems_ptr as *const Box<dyn crate::System>;
                            let system = &*systems_ptr.add(idx);

                            // Check state constraints.
                            if let Some(active) = active_state {
                                let allowed = system.run_in_states();
                                if !allowed.is_empty() && !allowed.contains(&active.to_string()) {
                                    return;
                                }
                            }

                            let ctx = &*(ctx_ptr as *const FrameContext);
                            let system_mut =
                                &mut *(systems_ptr.add(idx) as *mut Box<dyn crate::System>);
                            system_mut.update(ctx);
                        });
                    } else if let Some(&idx) = batch.first() {
                        // Sequential execution for single-system batches to avoid Rayon overhead.
                        let system = &mut systems[idx];
                        let mut should_run = true;

                        if let Some(active) = active_state {
                            let allowed = system.run_in_states();
                            if !allowed.is_empty() && !allowed.contains(&active.to_string()) {
                                should_run = false;
                            }
                        }

                        if should_run {
                            system.update(ctx);
                        }
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
    fn name(&self) -> &str {
        "HierarchySystem"
    }
    fn dependencies(&self) -> Vec<&'static str> {
        vec!["ComponentSystem"]
    }
    fn resource_access(&self) -> crate::ResourceAccess {
        crate::ResourceAccess::new().with_scene(crate::Access::Write)
    }
    fn update(&mut self, ctx: &FrameContext) {
        unsafe {
            ctx.scene_mut().update_all_transforms();
        }
    }
}

pub struct ResourceSystem;

impl System for ResourceSystem {
    fn name(&self) -> &str {
        "ResourceSystem"
    }
    fn resource_access(&self) -> crate::ResourceAccess {
        crate::ResourceAccess::new()
            .with_scene(crate::Access::Read)
            .with_renderer(crate::Access::Write)
            .with_resource_manager(crate::Access::Write)
    }
    fn update(&mut self, ctx: &FrameContext) {
        unsafe {
            let renderer = ctx.renderer_mut();
            ctx.resource_manager_mut().upload_global_buffers(renderer);
        }
    }
}
