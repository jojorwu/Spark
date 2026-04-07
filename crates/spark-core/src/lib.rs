pub mod asset;
pub mod command;
pub mod event;
pub mod event_bus;
pub mod event_mapper;
pub mod gltf_loader;
pub mod input;
pub mod logger;
pub mod prefab;
pub mod resource;
pub mod resource_container;
pub mod scene;
pub mod systems;
pub mod systems_events;
pub mod task;

use std::any::TypeId;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Access {
    None,
    Read,
    Write,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResourceAccess {
    pub accesses: HashMap<TypeId, Access>,
}

impl ResourceAccess {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with<T: 'static>(mut self, access: Access) -> Self {
        self.accesses.insert(TypeId::of::<T>(), access);
        self
    }

    pub fn with_scene(self, access: Access) -> Self {
        self.with::<crate::scene::Scene>(access)
    }

    pub fn with_renderer(self, access: Access) -> Self {
        self.with::<spark_renderer::Renderer>(access)
    }

    pub fn with_resource_manager(self, access: Access) -> Self {
        self.with::<crate::resource::ResourceManager>(access)
    }

    pub fn get<T: 'static>(&self) -> Access {
        self.accesses
            .get(&TypeId::of::<T>())
            .copied()
            .unwrap_or(Access::None)
    }

    pub fn conflicts_with(&self, other: &Self) -> bool {
        for (type_id, access) in &self.accesses {
            if *access == Access::None {
                continue;
            }
            let other_access = other.accesses.get(type_id).copied().unwrap_or(Access::None);
            if other_access == Access::None {
                continue;
            }

            if *access == Access::Write || other_access == Access::Write {
                return true;
            }
        }
        false
    }
}

use crate::resource::ResourceManager;
use crate::scene::Scene;
use crate::task::TaskSystem;
use serde::{Deserialize, Serialize};
use spark_renderer::resource::RenderSettings;
use spark_renderer::Renderer;
use std::path::PathBuf;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

#[derive(Serialize, Deserialize, Clone)]
pub struct PhysicsSettings {
    pub gravity: spark_math::Vec3,
    pub simulation_frequency: f32,
}

impl Default for PhysicsSettings {
    fn default() -> Self {
        Self {
            gravity: spark_math::Vec3::new(0.0, -9.81, 0.0),
            simulation_frequency: 60.0,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct Project {
    pub name: String,
    pub asset_root: PathBuf,
    pub startup_scene: PathBuf,
    pub render_settings: RenderSettings,
    pub physics_settings: PhysicsSettings,
}

/// Context passed to systems during initialization and cleanup.
pub struct InitContext<'a> {
    pub scene: &'a mut Scene,
    pub renderer: &'a mut Renderer,
    pub resource_manager: &'a mut ResourceManager,
    pub asset_manager: &'a mut crate::asset::AssetManager,
    pub resources: &'a mut crate::resource_container::Resources,
    pub task_system: &'a TaskSystem,
}

/// Context passed to systems during the update phase.
///
/// `FrameContext` provides high-performance, shared access to core engine subsystems.
/// It encapsulates raw pointers to bypass standard borrow checker restrictions during
/// parallel system execution.
///
/// ### Thread Safety and the Multi-threading Contract
///
/// While `FrameContext` uses `unsafe` pointers internally, its usage is safe when
/// orchestrated by the engine's `Scheduler`. The safety is guaranteed through:
///
/// 1.  **System Batching**: The `Scheduler` analyzes the `ResourceAccess` of every system
///     before execution. Systems are grouped into parallel batches only if their
///     resource requirements are mutually compatible (e.g., multiple readers, or
///     one writer with no other readers/writers).
/// 2.  **Disjoint Access**: Within a parallel batch, no two systems will attempt to
///     mutably access the same resource simultaneously.
/// 3.  **Deferred Mutation**: Structural changes to shared resources (like adding/removing
///     nodes in the `Scene`) must be deferred using the provided `command_queue`
///     during the parallel update phase to avoid data races.
pub struct FrameContext<'a> {
    scene: *mut Scene,
    renderer: *mut Renderer,
    resource_manager: *mut ResourceManager,
    asset_manager: *mut crate::asset::AssetManager,
    pub project: &'a Project,
    pub resources: &'a crate::resource_container::Resources,
    pub task_system: &'a TaskSystem,
    pub delta: f32,
    pub input: &'a crate::input::InputManager,
    pub command_queue: &'a crate::command::CommandQueue,
    pub event_bus: &'a crate::event_bus::EventBus,
}

unsafe impl<'a> Send for FrameContext<'a> {}
unsafe impl<'a> Sync for FrameContext<'a> {}

impl<'a> FrameContext<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        scene: &mut Scene,
        renderer: &mut Renderer,
        resource_manager: &mut ResourceManager,
        asset_manager: &mut crate::asset::AssetManager,
        project: &'a Project,
        resources: &'a crate::resource_container::Resources,
        task_system: &'a TaskSystem,
        delta: f32,
        input: &'a crate::input::InputManager,
        command_queue: &'a crate::command::CommandQueue,
        event_bus: &'a crate::event_bus::EventBus,
    ) -> Self {
        Self {
            scene: scene as *mut Scene,
            renderer: renderer as *mut Renderer,
            resource_manager: resource_manager as *mut ResourceManager,
            asset_manager: asset_manager as *mut crate::asset::AssetManager,
            project,
            resources,
            task_system,
            delta,
            input,
            command_queue,
            event_bus,
        }
    }

    /// Provides read-only access to the scene.
    pub fn scene(&self) -> &Scene {
        unsafe { &*self.scene }
    }

    /// Provides read-only access to the renderer.
    pub fn renderer(&self) -> &Renderer {
        unsafe { &*self.renderer }
    }

    /// Provides read-only access to the resource manager.
    pub fn resource_manager(&self) -> &ResourceManager {
        unsafe { &*self.resource_manager }
    }

    /// Provides read-only access to the asset manager.
    pub fn asset_manager(&self) -> &crate::asset::AssetManager {
        unsafe { &*self.asset_manager }
    }

    /// Returns a mutable reference to the scene.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the current system has declared `Access::Write`
    /// for the `Scene` resource. This ensures the `Scheduler` has placed this
    /// system in a batch where it has exclusive mutable access.
    ///
    /// ### Structural Changes
    ///
    /// Even with mutable access, structural changes to the scene (adding or removing
    /// nodes) MUST be performed via the `command_queue` to ensure consistency
    /// across parallel systems.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn scene_mut(&self) -> &mut Scene {
        &mut *self.scene
    }

    /// Returns a mutable reference to the renderer.
    ///
    /// # Safety
    ///
    /// The caller must ensure that no other systems or threads are accessing the renderer
    /// concurrently. This is typically guaranteed by the `Scheduler`.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn renderer_mut(&self) -> &mut Renderer {
        &mut *self.renderer
    }

    /// Returns a mutable reference to the resource manager.
    ///
    /// # Safety
    ///
    /// The caller must ensure that no other systems or threads are accessing the
    /// resource manager concurrently.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn resource_manager_mut(&self) -> &mut ResourceManager {
        &mut *self.resource_manager
    }

    /// Returns a mutable reference to the asset manager.
    ///
    /// # Safety
    ///
    /// The caller must ensure that no other systems or threads are accessing the
    /// asset manager concurrently.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn asset_manager_mut(&self) -> &mut crate::asset::AssetManager {
        &mut *self.asset_manager
    }

    pub fn get_component<T: 'static>(&self, node: crate::scene::NodeKey) -> Option<&T> {
        if let Some(node) = self.scene().nodes.get(node) {
            for comp in &node.components {
                if let Some(c) = comp.as_any().downcast_ref::<T>() {
                    return Some(c);
                }
            }
        }
        None
    }

    pub fn query_nodes<T: 'static>(&self) -> Vec<crate::scene::NodeKey> {
        self.scene().query_components::<T>()
    }

    /// Creates a new query to filter nodes in the scene.
    pub fn query(&self) -> crate::scene::Query<'_> {
        self.scene().query()
    }

    pub fn get_resource<T: 'static>(
        &self,
    ) -> Option<std::sync::Arc<std::sync::RwLock<Box<dyn std::any::Any + Send + Sync>>>> {
        self.resources.get::<T>()
    }

    pub fn get_resource_by_id(
        &self,
        id: TypeId,
    ) -> Option<std::sync::Arc<std::sync::RwLock<Box<dyn std::any::Any + Send + Sync>>>> {
        self.resources.get_by_id(id)
    }
}

/// A trait representing a system that processes engine state.
pub trait System: Send + Sync {
    fn name(&self) -> &str;
    fn version(&self) -> &str {
        "0.1.0"
    }
    fn on_init(&mut self, _ctx: &mut InitContext) {}
    fn update(&mut self, ctx: &FrameContext);
    fn on_stop(&mut self, _ctx: &mut InitContext) {}
    fn on_enter(&mut self, _ctx: &mut InitContext) {}
    fn on_exit(&mut self, _ctx: &mut InitContext) {}
    fn run_in_states(&self) -> Vec<String> {
        Vec::new()
    }
    fn dependencies(&self) -> Vec<&'static str> {
        Vec::new()
    }
    fn run_before(&self) -> Vec<&'static str> {
        Vec::new()
    }
    fn run_after(&self) -> Vec<&'static str> {
        Vec::new()
    }
    fn resource_access(&self) -> ResourceAccess {
        ResourceAccess::new()
            .with_scene(Access::Write)
            .with_renderer(Access::Write)
            .with_resource_manager(Access::Write)
            .with::<crate::asset::AssetManager>(Access::Write)
    }
}

/// A trait for engine plugins.
pub trait Plugin {
    fn build(&self, app: &mut App);
}

pub struct App {
    pub engine: Engine,
    pub startup_systems: Vec<Box<dyn System>>,
}

impl App {
    pub fn new(title: &str) -> Self {
        let engine = Engine::new(title, None).expect("Failed to initialize engine");
        Self {
            engine,
            startup_systems: Vec::new(),
        }
    }

    pub fn with_ui_shaders(title: &str, vert: &[u32], frag: &[u32]) -> Self {
        let engine = Engine::new(title, Some((vert, frag))).expect("Failed to initialize engine");
        Self {
            engine,
            startup_systems: Vec::new(),
        }
    }

    pub fn add_plugin<P: Plugin + 'static>(mut self, plugin: P) -> Self {
        plugin.build(&mut self);
        self
    }

    pub fn add_system<S: System + 'static>(mut self, system: S) -> Self {
        self.engine.add_system(system);
        self
    }

    pub fn add_system_to_stage<S: System + 'static>(
        mut self,
        stage: crate::systems::CoreStage,
        system: S,
    ) -> Self {
        self.engine.add_system_to_stage(stage, system);
        self
    }

    pub fn add_plugin_to_app<P: Plugin + 'static>(&mut self, plugin: P) {
        plugin.build(self);
    }

    pub fn add_startup_system<S: System + 'static>(mut self, system: S) -> Self {
        self.startup_systems.push(Box::new(system));
        self
    }

    pub fn insert_resource<T: Send + Sync + 'static>(mut self, resource: T) -> Self {
        self.engine.resources.insert(resource);
        self
    }

    pub fn set_state(&mut self, state: &str) {
        crate::systems::Scheduler::set_state(&mut self.engine.system_registry, state);
    }

    pub fn run(mut self) {
        self.run_startup();
        self.engine.run(|_, _, _, _, _, _, _, _, _| (false, None));
    }

    pub fn run_with_ui<F>(mut self, ui_callback: F)
    where
        F: FnMut(
                &winit::window::Window,
                &winit::event::Event<()>,
                &mut Scene,
                &mut ResourceManager,
                &mut crate::asset::AssetManager,
                &mut Renderer,
                &mut Project,
                &mut crate::resource_container::Resources,
                f32,
            ) -> (bool, Option<(egui::FullOutput, egui::Context)>)
            + 'static,
    {
        self.run_startup();
        self.engine.run(ui_callback);
    }

    fn run_startup(&mut self) {
        let mut init_ctx = InitContext {
            scene: &mut self.engine.scene,
            renderer: &mut self.engine.renderer,
            resource_manager: &mut self.engine.resource_manager,
            asset_manager: &mut self.engine.asset_manager,
            resources: &mut self.engine.resources,
            task_system: &self.engine.task_system,
        };
        for system in &mut self.startup_systems {
            system.on_init(&mut init_ctx);
        }
    }
}

pub struct Engine {
    pub window: winit::window::Window,
    pub event_loop: Option<EventLoop<()>>,
    pub scene: Scene,
    pub renderer: Renderer,
    pub task_system: TaskSystem,
    pub resource_manager: ResourceManager,
    pub asset_manager: crate::asset::AssetManager,
    pub resources: crate::resource_container::Resources,
    pub input_manager: crate::input::InputManager,
    pub command_queue: crate::command::CommandQueue,
    pub event_bus: crate::event_bus::EventBus,
    pub system_registry: crate::systems::SystemRegistry,
    pub last_frame_time: instant::Instant,
    pub current_fps: f32,
    pub system_events: std::sync::Mutex<Vec<crate::systems_events::events::SystemEvent>>,
    pub project: Project,
}

impl Engine {
    pub fn new(
        title: &str,
        ui_shaders: Option<(&[u32], &[u32])>,
    ) -> Result<Self, spark_renderer::error::RendererError> {
        let event_loop = EventLoop::new().expect("Failed to create event loop");
        let window = WindowBuilder::new()
            .with_title(title)
            .build(&event_loop)
            .expect("Failed to build window");

        let renderer = Renderer::new(&window, ui_shaders)?;
        let scene = Scene::new();
        let task_system = TaskSystem::new();
        let resource_manager = ResourceManager::new();
        let asset_manager = crate::asset::AssetManager::new();
        let resources = crate::resource_container::Resources::new();
        let input_manager = crate::input::InputManager::new();
        let command_queue = crate::command::CommandQueue::new();
        let event_bus = crate::event_bus::EventBus::new();
        let system_registry = crate::systems::SystemRegistry::new();

        Ok(Self {
            window,
            event_loop: Some(event_loop),
            scene,
            renderer,
            task_system,
            resource_manager,
            asset_manager,
            resources,
            input_manager,
            command_queue,
            event_bus,
            system_registry,
            last_frame_time: instant::Instant::now(),
            current_fps: 0.0,
            system_events: std::sync::Mutex::new(Vec::new()),
            project: Project::default(),
        })
    }

    pub fn add_system<S: System + 'static>(&mut self, system: S) {
        self.system_registry.add_system(system);
    }

    pub fn add_system_to_stage<S: System + 'static>(
        &mut self,
        stage: crate::systems::CoreStage,
        system: S,
    ) {
        self.system_registry.add_system_to_stage(stage, system);
    }

    pub fn add_boxed_system(&mut self, system: Box<dyn System>) {
        self.system_registry.add_boxed_system(system);
    }

    fn handle_window_event(
        &mut self,
        event: &WindowEvent,
        elwt: &winit::event_loop::EventLoopWindowTarget<()>,
    ) {
        if let WindowEvent::CloseRequested = event {
            elwt.exit();
            return;
        }

        if let Some(engine_event) = crate::event_mapper::EventMapper::map_window_event(event) {
            self.event_bus.publish(engine_event);
        }
    }

    pub fn run<F>(mut self, mut ui_callback: F)
    where
        F: FnMut(
                &winit::window::Window,
                &winit::event::Event<()>,
                &mut Scene,
                &mut ResourceManager,
                &mut crate::asset::AssetManager,
                &mut Renderer,
                &mut Project,
                &mut crate::resource_container::Resources,
                f32,
            ) -> (bool, Option<(egui::FullOutput, egui::Context)>)
            + 'static,
    {
        let event_loop = self.event_loop.take().expect("Engine event loop already taken or not initialized");

        {
            let mut init_ctx = InitContext {
                scene: &mut self.scene,
                renderer: &mut self.renderer,
                resource_manager: &mut self.resource_manager,
                asset_manager: &mut self.asset_manager,
                resources: &mut self.resources,
                task_system: &self.task_system,
            };
            crate::systems::Scheduler::init(&mut self.system_registry, &mut init_ctx);
        }

        event_loop
            .run(move |event, elwt| {
                let (ui_consumed, egui_output) = ui_callback(
                    &self.window,
                    &event,
                    &mut self.scene,
                    &mut self.resource_manager,
                    &mut self.asset_manager,
                    &mut self.renderer,
                    &mut self.project,
                    &mut self.resources,
                    self.current_fps,
                );
                if ui_consumed {
                    // UI consumed the event
                }

                match &event {
                    Event::WindowEvent { event, .. } => {
                        self.handle_window_event(event, elwt);
                    }
                    Event::AboutToWait => {
                        let now = instant::Instant::now();
                        let delta = now.duration_since(self.last_frame_time).as_secs_f32();
                        self.last_frame_time = now;
                        self.current_fps = 0.9 * self.current_fps + 0.1 * (1.0 / delta.max(0.001));

                        self.on_frame_start(delta, egui_output);
                    }
                    _ => (),
                }

                if elwt.exiting() {
                    self.on_shutdown();
                }
            })
            .expect("Event loop failed");
    }

    /// Handles the transition at the start of a frame, including event buffer swapping,
    /// state changes, and update/render execution.
    fn on_frame_start(
        &mut self,
        delta: f32,
        egui_output: Option<(egui::FullOutput, egui::Context)>,
    ) {
        self.event_bus.swap_buffers();

        {
            let mut init_ctx = InitContext {
                scene: &mut self.scene,
                renderer: &mut self.renderer,
                resource_manager: &mut self.resource_manager,
                asset_manager: &mut self.asset_manager,
                resources: &mut self.resources,
                task_system: &self.task_system,
            };
            crate::systems::Scheduler::apply_state_changes(
                &mut self.system_registry,
                &mut init_ctx,
            );
        }

        self.update_phase(delta);
        self.render_phase(egui_output, delta);
    }

    /// Handles engine shutdown logic and system cleanup.
    fn on_shutdown(&mut self) {
        let mut init_ctx = InitContext {
            scene: &mut self.scene,
            renderer: &mut self.renderer,
            resource_manager: &mut self.resource_manager,
            asset_manager: &mut self.asset_manager,
            resources: &mut self.resources,
            task_system: &self.task_system,
        };
        crate::systems::Scheduler::shutdown(&mut self.system_registry, &mut init_ctx);
    }

    /// Processes a single frame's update logic.
    ///
    /// This includes updating input state from the event bus and executing
    /// all registered systems across multiple parallel stages.
    pub fn update_phase(&mut self, delta: f32) {
        let events = self.event_bus.read_events::<crate::event::EngineEvent>();
        self.input_manager.update(&events);

        {
            let mut ctx = FrameContext::new(
                &mut self.scene,
                &mut self.renderer,
                &mut self.resource_manager,
                &mut self.asset_manager,
                &self.project,
                &self.resources,
                &self.task_system,
                delta,
                &self.input_manager,
                &self.command_queue,
                &self.event_bus,
            );

            crate::systems::Scheduler::run(&mut self.system_registry, &mut ctx);
        }

        // Execute all deferred commands after system updates
        self.command_queue
            .execute_all(&mut self.scene, &mut self.resource_manager);
    }

    pub fn render_phase(
        &mut self,
        egui_output: Option<(egui::FullOutput, egui::Context)>,
        delta: f32,
    ) {
        let total_objects =
            self.renderer
                .prepare_frame(&self.scene, &self.resource_manager, &self.asset_manager);

        // Draw the frame
        self.renderer
            .draw_frame(&self.window, egui_output, total_objects, delta);

        self.renderer.clear_instance_buffers();
    }
}
