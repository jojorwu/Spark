pub mod scene;
pub mod task;
pub mod resource;
pub mod event;
pub mod logger;
pub mod systems;
pub mod systems_events;
pub mod event_mapper;
pub mod input;

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

/// Context passed to systems during initialization and cleanup.
pub struct InitContext<'a> {
    pub scene: &'a mut Scene,
    pub renderer: &'a mut Renderer,
    pub resource_manager: &'a mut ResourceManager,
    pub task_system: &'a TaskSystem,
}

/// Context passed to systems during the update phase.
pub struct FrameContext<'a> {
    pub scene: &'a mut Scene,
    pub renderer: &'a mut Renderer,
    pub resource_manager: &'a mut ResourceManager,
    pub task_system: &'a TaskSystem,
    pub delta: f32,
    pub event_proxy: crate::systems_events::events::EventProxy<'a>,
    pub input: &'a crate::input::InputManager,
}

/// A trait representing a system that processes engine state.
pub trait System: Send + Sync {
    fn name(&self) -> &str;
    fn version(&self) -> &str { "0.1.0" }
    fn on_init(&mut self, _ctx: &mut InitContext) {}
    fn update(&mut self, ctx: &mut FrameContext);
    fn on_stop(&mut self, _ctx: &mut InitContext) {}
    fn dependencies(&self) -> Vec<&'static str> { Vec::new() }
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
    pub system_registry: crate::systems::SystemRegistry,
    pub last_frame_time: instant::Instant,
    pub current_fps: f32,
    pub system_events: Vec<crate::systems_events::events::SystemEvent>,
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
            system_registry,
            last_frame_time: instant::Instant::now(),
            current_fps: 0.0,
            system_events: Vec::new(),
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

        let mut system_events = std::mem::take(&mut self.system_events);
        system_events.clear(); // Reset for this frame

        {
            let mut ctx = FrameContext {
                scene: &mut self.scene,
                renderer: &mut self.renderer,
                resource_manager: &mut self.resource_manager,
                task_system: &self.task_system,
                delta,
                event_proxy: crate::systems_events::events::EventProxy {
                    events: &self.event_queue.events,
                    outgoing: &mut system_events,
                },
                input: &self.input_manager,
            };

            crate::systems::Scheduler::run(&mut self.system_registry, &mut ctx);
        }

        self.system_events = system_events;
    }

    fn render_phase(&mut self, egui_output: Option<(egui::FullOutput, egui::Context)>, delta: f32) {
        // Collect visibility and light data
        let packet = self.scene.collect_frame_packet(None, &self.resource_manager);
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
