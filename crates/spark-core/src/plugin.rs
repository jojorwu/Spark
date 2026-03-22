use crate::scene::Scene;
use crate::resource::ResourceManager;
use spark_renderer::Renderer;
use crate::FrameContext;

/// Context passed to plugins during initialization and cleanup.
pub struct InitContext<'a> {
    pub scene: &'a mut Scene,
    pub renderer: &'a mut Renderer,
    pub resource_manager: &'a mut ResourceManager,
}

pub trait Plugin: Send + Sync {
    /// Returns the name of the plugin.
    fn name(&self) -> &str;

    /// Returns the version of the plugin.
    fn version(&self) -> &str { "0.1.0" }

    /// Called when the plugin is loaded and initialized.
    fn on_init(&mut self, ctx: &mut InitContext);

    /// Called once per frame during the update phase.
    fn on_update(&mut self, ctx: &mut FrameContext);

    /// Called when the plugin is about to be unloaded or the engine is shutting down.
    fn on_stop(&mut self, ctx: &mut InitContext);
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

    pub fn init_plugins(&mut self, scene: &mut Scene, renderer: &mut Renderer, resource_manager: &mut ResourceManager) {
        let mut ctx = InitContext {
            scene,
            renderer,
            resource_manager,
        };
        for plugin in &mut self.plugins {
            plugin.on_init(&mut ctx);
        }
    }

    pub fn update_plugins(&mut self, ctx: &mut FrameContext) {
        for plugin in &mut self.plugins {
            plugin.on_update(ctx);
        }
    }

    pub fn stop_plugins(&mut self, scene: &mut Scene, renderer: &mut Renderer, resource_manager: &mut ResourceManager) {
        let mut ctx = InitContext {
            scene,
            renderer,
            resource_manager,
        };
        for plugin in &mut self.plugins {
            plugin.on_stop(&mut ctx);
        }
    }
}
