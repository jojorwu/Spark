pub mod scene;
pub mod task;
pub mod resource;
pub mod event;
pub mod logger;
pub mod systems;
pub mod systems_events;
pub mod event_mapper;
pub mod input;
pub mod command;
pub mod event_bus;

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
        let scene_conflict = (self.scene == Access::Write && other.scene != Access::None) ||
                             (other.scene == Access::Write && self.scene != Access::None);
        let renderer_conflict = (self.renderer == Access::Write && other.renderer != Access::None) ||
                                (other.renderer == Access::Write && self.renderer != Access::None);
        let resource_manager_conflict = (self.resource_manager == Access::Write && other.resource_manager != Access::None) ||
                                        (other.resource_manager == Access::Write && self.resource_manager != Access::None);

        scene_conflict || renderer_conflict || resource_manager_conflict
    }
}

use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};
use crate::scene::Scene;
use crate::task::TaskSystem;
use crate::resource::ResourceManager;
use crate::event::EventQueue;
use spark_renderer::Renderer;
use spark_renderer::resource::RenderSettings;
use serde::{Serialize, Deserialize};
use std::path::PathBuf;

#[derive(Serialize, Deserialize, Clone, Default)]
pub struct Project {
    pub name: String,
    pub asset_root: PathBuf,
    pub startup_scene: PathBuf,
    pub render_settings: RenderSettings,
}

/// Context passed to systems during initialization and cleanup.
pub struct InitContext<'a> {
    pub scene: &'a mut Scene,
    pub renderer: &'a mut Renderer,
    pub resource_manager: &'a mut ResourceManager,
    pub task_system: &'a TaskSystem,
}

/// Context passed to systems during the update phase.
pub struct FrameContext<'a> {
    pub scene: *mut Scene,
    pub renderer: *mut Renderer,
    pub resource_manager: *mut ResourceManager,
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
    pub fn scene(&self) -> &Scene { unsafe { &*self.scene } }
    pub fn renderer(&self) -> &Renderer { unsafe { &*self.renderer } }
    pub fn resource_manager(&self) -> &ResourceManager { unsafe { &*self.resource_manager } }

    /// Returns a mutable reference to the scene.
    /// Safety: Caller must ensure no other threads are accessing the scene concurrently.
    pub unsafe fn scene_mut(&self) -> &mut Scene { &mut *self.scene }
    /// Returns a mutable reference to the renderer.
    /// Safety: Caller must ensure no other threads are accessing the renderer concurrently.
    pub unsafe fn renderer_mut(&self) -> &mut Renderer { &mut *self.renderer }
    /// Returns a mutable reference to the resource manager.
    /// Safety: Caller must ensure no other threads are accessing the resource manager concurrently.
    pub unsafe fn resource_manager_mut(&self) -> &mut ResourceManager { &mut *self.resource_manager }
}

/// A trait representing a system that processes engine state.
pub trait System: Send + Sync {
    fn name(&self) -> &str;
    fn version(&self) -> &str { "0.1.0" }
    fn on_init(&mut self, _ctx: &mut InitContext) {}
    fn update(&mut self, ctx: &FrameContext);
    fn on_stop(&mut self, _ctx: &mut InitContext) {}
    fn dependencies(&self) -> Vec<&'static str> { Vec::new() }
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
}

impl App {
    pub fn new(title: &str) -> Self {
        let engine = Engine::new(title, None).expect("Failed to initialize engine");
        Self {
            engine,
        }
    }

    pub fn with_ui_shaders(title: &str, vert: &[u32], frag: &[u32]) -> Self {
        let engine = Engine::new(title, Some((vert, frag))).expect("Failed to initialize engine");
        Self {
            engine,
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

    pub fn run(self) {
        self.engine.run(|_, _, _, _, _, _| (false, None));
    }

    pub fn run_with_ui<F>(self, ui_callback: F)
    where
        F: FnMut(&winit::window::Window, &winit::event::Event<()>, &mut Scene, &mut ResourceManager, &mut Renderer, f32) -> (bool, Option<(egui::FullOutput, egui::Context)>) + 'static,
    {
        self.engine.run(ui_callback);
    }
}

pub struct Engine {
    pub window: winit::window::Window,
    pub event_loop: Option<EventLoop<()>>,
    pub scene: Scene,
    pub renderer: Renderer,
    pub task_system: TaskSystem,
    pub resource_manager: ResourceManager,
    pub event_queue: EventQueue,
    pub input_manager: crate::input::InputManager,
    pub command_queue: crate::command::CommandQueue,
    pub event_bus: crate::event_bus::EventBus,
    pub system_registry: crate::systems::SystemRegistry,
    pub last_frame_time: instant::Instant,
    pub current_fps: f32,
    pub system_events: std::sync::Mutex<Vec<crate::systems_events::events::SystemEvent>>,
}

impl Engine {
    pub fn new(
        title: &str,
        ui_shaders: Option<(&[u32], &[u32])>
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
        let event_queue = EventQueue::new();
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
            event_queue,
            input_manager,
            command_queue,
            event_bus,
            system_registry,
            last_frame_time: instant::Instant::now(),
            current_fps: 0.0,
            system_events: std::sync::Mutex::new(Vec::new()),
        })
    }

    pub fn add_system<S: System + 'static>(&mut self, system: S) {
        self.system_registry.add_system(system);
    }

    pub fn add_boxed_system(&mut self, system: Box<dyn System>) {
        self.system_registry.add_boxed_system(system);
    }

    fn handle_window_event(&mut self, event: &WindowEvent, elwt: &winit::event_loop::EventLoopWindowTarget<()>) {
        if let WindowEvent::CloseRequested = event {
            elwt.exit();
            return;
        }

        crate::event_mapper::EventMapper::map_window_event(event, &mut self.event_queue);
    }

    pub fn run<F>(mut self, mut ui_callback: F)
    where
        F: FnMut(&winit::window::Window, &winit::event::Event<()>, &mut Scene, &mut ResourceManager, &mut Renderer, f32) -> (bool, Option<(egui::FullOutput, egui::Context)>) + 'static,
    {
        let event_loop = self.event_loop.take().unwrap();

        {
            let mut init_ctx = InitContext {
                scene: &mut self.scene,
                renderer: &mut self.renderer,
                resource_manager: &mut self.resource_manager,
                task_system: &self.task_system,
            };
            crate::systems::Scheduler::init(&mut self.system_registry, &mut init_ctx);
        }

        event_loop.run(move |event, elwt| {
            let (ui_consumed, egui_output) = ui_callback(&self.window, &event, &mut self.scene, &mut self.resource_manager, &mut self.renderer, self.current_fps);
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

                    self.event_queue.clear();
                }
                _ => (),
            }

            if elwt.exiting() {
                 let mut init_ctx = InitContext {
                    scene: &mut self.scene,
                    renderer: &mut self.renderer,
                    resource_manager: &mut self.resource_manager,
                    task_system: &self.task_system,
                };
                crate::systems::Scheduler::shutdown(&mut self.system_registry, &mut init_ctx);
            }
        }).expect("Event loop failed");
    }

    fn update_phase(&mut self, delta: f32) {
        self.input_manager.update(&self.event_queue.events);

        self.system_events.lock().unwrap().clear(); // Reset for this frame

        {
            let mut ctx = FrameContext {
                scene: &mut self.scene as *mut Scene,
                renderer: &mut self.renderer as *mut Renderer,
                resource_manager: &mut self.resource_manager as *mut ResourceManager,
                task_system: &self.task_system,
                delta,
                event_proxy: crate::systems_events::events::EventProxy {
                    events: &self.event_queue.events,
                    outgoing: &self.system_events,
                },
                input: &self.input_manager,
                command_queue: &self.command_queue,
                event_bus: &self.event_bus,
            };

            crate::systems::Scheduler::run(&mut self.system_registry, &mut ctx);
        }

        // Execute all deferred commands after system updates
        self.command_queue.execute_all(&mut self.scene, &mut self.resource_manager);
    }

    fn render_phase(&mut self, egui_output: Option<(egui::FullOutput, egui::Context)>, delta: f32) {
        // Find active camera to calculate frustum
        let mut camera_matrix = spark_math::Mat4::IDENTITY;
        let mut projection_matrix = spark_math::Mat4::IDENTITY;

        for node in self.scene.nodes.values() {
            for component in &node.components {
                if let Some(camera) = component.as_any().downcast_ref::<crate::scene::CameraComponent>() {
                    let view = node.global_transform.inverse();
                    if camera.orthographic {
                        let aspect = self.renderer.get_extent().width as f32 / self.renderer.get_extent().height as f32;
                        let size = camera.ortho_size;
                        projection_matrix = spark_math::Mat4::orthographic_rh(
                            -size * aspect, size * aspect, -size, size, camera.near, camera.far
                        );
                    } else {
                        projection_matrix = spark_math::Mat4::perspective_rh(
                            camera.fov.to_radians(),
                            self.renderer.get_extent().width as f32 / self.renderer.get_extent().height as f32,
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
        let frustum_ref = if camera_matrix != spark_math::Mat4::IDENTITY { Some(&frustum_obj) } else { None };

        // Collect visibility and light data
        let mut packet = self.scene.collect_frame_packet(frustum_ref, &self.resource_manager);
        packet.projection_matrix = projection_matrix;
        let total_objects = self.renderer.prepare_frame(packet);

        // Draw the frame
        self.renderer.draw_frame(
            &self.window,
            egui_output,
            total_objects,
            delta
        );

        self.renderer.clear_instance_buffers();
    }
}
