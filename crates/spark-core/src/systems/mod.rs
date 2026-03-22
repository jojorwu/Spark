pub mod component;

use crate::{System, FrameContext};

pub struct SystemRegistry {
    systems: Vec<Box<dyn System>>,
}

impl SystemRegistry {
    pub fn new() -> Self {
        Self { systems: Vec::new() }
    }

    pub fn add_system<S: System + 'static>(&mut self, system: S) {
        self.systems.push(Box::new(system));
    }

    pub fn take_systems(&mut self) -> Vec<Box<dyn System>> {
        std::mem::take(&mut self.systems)
    }

    pub fn restore_systems(&mut self, systems: Vec<Box<dyn System>>) {
        self.systems = systems;
    }
}

impl Default for SystemRegistry {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Scheduler;

impl Scheduler {
    pub fn run(registry: &mut SystemRegistry, ctx: &mut FrameContext) {
        let mut systems = registry.take_systems();
        for system in &mut systems {
            system.update(ctx);
        }
        registry.restore_systems(systems);
    }
}

pub struct HierarchySystem;

impl System for HierarchySystem {
    fn name(&self) -> &str { "HierarchySystem" }
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
