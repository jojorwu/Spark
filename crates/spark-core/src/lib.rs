pub mod command;
pub mod event;
pub mod event_bus;
pub mod event_mapper;
pub mod input;
pub mod logger;
pub mod prefab;
pub mod resource;
pub mod resource_container;
pub mod scene;
pub mod systems;
pub mod systems_events;
pub mod task;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Access {
    None,
    Read,
    Write,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceAccess {
    pub scene: Access,
    pub renderer: Access,
    pub resource_manager: Access,
}

impl ResourceAccess {
    pub const NONE: Self = Self {
        scene: Access::None,
        renderer: Access::None,
        resource_manager: Access::None,
    };

    pub fn conflicts_with(&self, other: &Self) -> bool {
        let scene_conflict = (self.scene == Access::Write && other.scene != Access::None)
            || (other.scene == Access::Write && self.scene != Access::None);
        let renderer_conflict = (self.renderer == Access::Write && other.renderer != Access::None)
            || (other.renderer == Access::Write && self.renderer != Access::None);
        let resource_manager_conflict = (self.resource_manager == Access::Write
            && other.resource_manager != Access::None)
            || (other.resource_manager == Access::Write && self.resource_manager != Access::None);

        scene_conflict || renderer_conflict || resource_manager_conflict
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
    pub resources: &'a mut crate::resource_container::Resources,
    pub task_system: &'a TaskSystem,
}

/// Context passed to systems during the update phase.
pub struct FrameContext<'a> {
    pub scene: *mut Scene,
    pub renderer: *mut Renderer,
    pub resource_manager: *mut ResourceManager,
    pub project: &'a Project,
    pub resources: &'a crate::resource_container::Resources,
    pub task_system: &'a TaskSystem,
    pub delta: f32,
    pub event_proxy: crate::systems_events::events::EventProxy<'a>,
    pub input: &'a crate::input::InputManager,
    pub command_queue: &'a crate::command::CommandQueue,
    pub event_bus: &'a crate::event_bus::EventBus,
}

unsafe impl<'a> Send for FrameContext<'a> {}
unsafe impl<'a> Sync for FrameContext<'a> {}

impl<'a> FrameContext<'a> {
    pub fn scene(&self) -> &Scene {
        unsafe { &*self.scene }
    }
    pub fn renderer(&self) -> &Renderer {
        unsafe { &*self.renderer }
    }
    pub fn resource_manager(&self) -> &ResourceManager {
        unsafe { &*self.resource_manager }
    }

    /// Returns a mutable reference to the scene.
    ///
    /// # Safety
    ///
    /// Caller must ensure no other threads are accessing the scene concurrently.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn scene_mut(&self) -> &mut Scene {
        &mut *self.scene
    }

    /// Returns a mutable reference to the renderer.
    ///
    /// # Safety
    ///
    /// Caller must ensure no other threads are accessing the renderer concurrently.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn renderer_mut(&self) -> &mut Renderer {
        &mut *self.renderer
    }

    /// Returns a mutable reference to the resource manager.
    ///
    /// # Safety
    ///
    /// Caller must ensure no other threads are accessing the resource manager concurrently.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn resource_manager_mut(&self) -> &mut ResourceManager {
        &mut *self.resource_manager
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

    pub fn query(&self) -> crate::scene::Query<'_> {
        self.scene().query()
    }

    pub fn get_resource<T: 'static>(
        &self,
    ) -> Option<std::sync::Arc<std::sync::RwLock<Box<dyn std::any::Any + Send + Sync>>>> {
        self.resources.get::<T>()
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
    fn resource_access(&self) -> ResourceAccess {
        ResourceAccess {
            scene: Access::Write,
            renderer: Access::Write,
            resource_manager: Access::Write,
        }
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
        let mut init_ctx = InitContext {
            scene: &mut self.engine.scene,
            renderer: &mut self.engine.renderer,
            resource_manager: &mut self.engine.resource_manager,
            resources: &mut self.engine.resources,
            task_system: &self.engine.task_system,
        };
        crate::systems::Scheduler::set_state(
            &mut self.engine.system_registry,
            state,
            &mut init_ctx,
        );
    }

    pub fn run(mut self) {
        self.run_startup();
        self.engine.run(|_, _, _, _, _, _, _, _| (false, None));
    }

    pub fn run_with_ui<F>(mut self, ui_callback: F)
    where
        F: FnMut(
                &winit::window::Window,
                &winit::event::Event<()>,
                &mut Scene,
                &mut ResourceManager,
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
                &mut Renderer,
                &mut Project,
                &mut crate::resource_container::Resources,
                f32,
            ) -> (bool, Option<(egui::FullOutput, egui::Context)>)
            + 'static,
    {
        let event_loop = self.event_loop.take().unwrap();

        {
            let mut init_ctx = InitContext {
                scene: &mut self.scene,
                renderer: &mut self.renderer,
                resource_manager: &mut self.resource_manager,
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

                        self.update_phase(delta);
                        self.render_phase(egui_output, delta);

                        self.event_bus.clear_events();
                    }
                    _ => (),
                }

                if elwt.exiting() {
                    let mut init_ctx = InitContext {
                        scene: &mut self.scene,
                        renderer: &mut self.renderer,
                        resource_manager: &mut self.resource_manager,
                        resources: &mut self.resources,
                        task_system: &self.task_system,
                    };
                    crate::systems::Scheduler::shutdown(&mut self.system_registry, &mut init_ctx);
                }
            })
            .expect("Event loop failed");
    }

    fn update_phase(&mut self, delta: f32) {
        let events = self.event_bus.read_events::<crate::event::EngineEvent>();
        self.input_manager.update(&events);

        self.system_events.lock().unwrap().clear(); // Reset for this frame

        {
            let mut ctx = FrameContext {
                scene: &mut self.scene as *mut Scene,
                renderer: &mut self.renderer as *mut Renderer,
                resource_manager: &mut self.resource_manager as *mut ResourceManager,
                project: &self.project,
                resources: &self.resources,
                task_system: &self.task_system,
                delta,
                event_proxy: crate::systems_events::events::EventProxy {
                    events: &events,
                    outgoing: &self.system_events,
                },
                input: &self.input_manager,
                command_queue: &self.command_queue,
                event_bus: &self.event_bus,
            };

            crate::systems::Scheduler::run(&mut self.system_registry, &mut ctx);
        }

        // Execute all deferred commands after system updates
        self.command_queue
            .execute_all(&mut self.scene, &mut self.resource_manager);
    }

    fn render_phase(&mut self, egui_output: Option<(egui::FullOutput, egui::Context)>, delta: f32) {
        // Find active camera to calculate frustum
        let mut camera_matrix = spark_math::Mat4::IDENTITY;
        let mut projection_matrix = spark_math::Mat4::IDENTITY;

        for node in self.scene.nodes.values() {
            for component in &node.components {
                if let Some(camera) = component
                    .as_any()
                    .downcast_ref::<crate::scene::CameraComponent>()
                {
                    let view = node.global_transform.inverse();
                    if camera.orthographic {
                        let aspect = self.renderer.get_extent().width as f32
                            / self.renderer.get_extent().height as f32;
                        let size = camera.ortho_size;
                        projection_matrix = spark_math::Mat4::orthographic_rh(
                            -size * aspect,
                            size * aspect,
                            -size,
                            size,
                            camera.near,
                            camera.far,
                        );
                    } else {
                        projection_matrix = spark_math::Mat4::perspective_rh(
                            camera.fov.to_radians(),
                            self.renderer.get_extent().width as f32
                                / self.renderer.get_extent().height as f32,
                            camera.near,
                            camera.far,
                        );
                    }
                    camera_matrix = projection_matrix * view;
                    break;
                }
            }
        }

        let frustum_obj = spark_math::Frustum::from_matrix(camera_matrix);
        let frustum_ref = if camera_matrix != spark_math::Mat4::IDENTITY {
            Some(&frustum_obj)
        } else {
            None
        };

        // Collect visibility and light data
        let mut packet = self
            .scene
            .collect_frame_packet(frustum_ref, &self.resource_manager);
        packet.projection_matrix = projection_matrix;
        let total_objects = self.renderer.prepare_frame(packet);

        // Draw the frame
        self.renderer
            .draw_frame(&self.window, egui_output, total_objects, delta);

        self.renderer.clear_instance_buffers();
    }
}
