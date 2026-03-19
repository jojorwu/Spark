use crate::scene::Scene;

pub trait Plugin: Send + Sync {
    fn on_init(&mut self, scene: &mut Scene);
    fn on_update(&mut self, scene: &mut Scene, delta: f32);
}

pub struct PluginManager {
    plugins: Vec<Box<dyn Plugin>>,
}

impl PluginManager {
    pub fn new() -> Self {
        Self { plugins: Vec::new() }
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PluginManager {

    pub fn add_plugin(&mut self, plugin: Box<dyn Plugin>) {
        self.plugins.push(plugin);
    }

    pub fn init_plugins(&mut self, scene: &mut Scene) {
        for plugin in &mut self.plugins {
            plugin.on_init(scene);
        }
    }

    pub fn update_plugins(&mut self, scene: &mut Scene, delta: f32) {
        for plugin in &mut self.plugins {
            plugin.on_update(scene, delta);
        }
    }
}
