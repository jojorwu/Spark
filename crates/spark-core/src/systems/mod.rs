pub mod component;

use crate::{System, FrameContext, InitContext};
use std::collections::{HashSet, HashMap};

pub struct SystemRegistry {
    pub systems: Vec<Box<dyn System>>,
}

impl SystemRegistry {
    pub fn new() -> Self {
        Self { systems: Vec::new() }
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

        let mut old_systems: Vec<Option<Box<dyn System>>> = self.systems.drain(..).map(Some).collect();
        for idx in indices {
            self.systems.push(old_systems[idx].take().unwrap());
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
        for system in &mut registry.systems {
            system.update(ctx);
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
    fn update(&mut self, ctx: &mut FrameContext) {
        ctx.scene.update_all_transforms();
    }
}

pub struct ResourceSystem;

impl System for ResourceSystem {
    fn name(&self) -> &str { "ResourceSystem" }
    fn update(&mut self, ctx: &mut FrameContext) {
        ctx.resource_manager.upload_global_buffers(ctx.renderer);
    }
}
