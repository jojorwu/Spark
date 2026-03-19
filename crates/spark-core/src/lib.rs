pub mod scene;
pub mod task;
pub mod plugin;
pub mod resource;
pub mod event;
pub mod logger;
pub mod systems;

use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};
use crate::scene::Scene;
use crate::task::TaskSystem;
use crate::plugin::PluginManager;
use crate::resource::ResourceManager;
use crate::event::EventQueue;
use spark_renderer::Renderer;
use spark_math::Vec4Swizzles;

pub trait System {
    fn update(
        &mut self,
        scene: &mut Scene,
        renderer: &mut Renderer,
        resource_manager: &mut ResourceManager,
        delta: f32,
    );
}

pub struct Engine {
    pub window: winit::window::Window,
    pub event_loop: Option<EventLoop<()>>,
    pub scene: Scene,
    pub renderer: Renderer,
    pub task_system: TaskSystem,
    pub plugin_manager: PluginManager,
    pub resource_manager: ResourceManager,
    pub event_queue: EventQueue,
    pub last_frame_time: instant::Instant,
    pub current_fps: f32,
    pub systems: Vec<Box<dyn System>>,
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
        let plugin_manager = PluginManager::new();
        let resource_manager = ResourceManager::new();
        let event_queue = EventQueue::new();

        Ok(Self {
            window,
            event_loop: Some(event_loop),
            scene,
            renderer,
            task_system,
            plugin_manager,
            resource_manager,
            event_queue,
            last_frame_time: instant::Instant::now(),
            current_fps: 0.0,
            systems: Vec::new(),
        })
    }

    pub fn add_system<S: System + 'static>(&mut self, system: S) {
        self.systems.push(Box::new(system));
    }

    fn handle_window_event(&mut self, event: &WindowEvent, elwt: &winit::event_loop::EventLoopWindowTarget<()>) {
        match event {
            WindowEvent::CloseRequested => {
                elwt.exit();
            }
            WindowEvent::Resized(size) => {
                self.event_queue.push(crate::event::EngineEvent::WindowResized { width: size.width, height: size.height });
            }
            WindowEvent::KeyboardInput { event: input_event, .. } => {
                if let winit::keyboard::PhysicalKey::Code(code) = input_event.physical_key {
                    if input_event.state == winit::event::ElementState::Pressed {
                        self.event_queue.push(crate::event::EngineEvent::KeyDown { key_code: code as u32 });
                    } else {
                        self.event_queue.push(crate::event::EngineEvent::KeyUp { key_code: code as u32 });
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.event_queue.push(crate::event::EngineEvent::MouseMoved { x: position.x, y: position.y });
            }
            _ => {}
        }
    }

    pub fn run<F>(mut self, mut ui_callback: F)
    where
        F: FnMut(&winit::window::Window, &winit::event::Event<()>, &mut Scene, &mut ResourceManager, &mut Renderer, f32) -> (bool, Option<(egui::FullOutput, egui::Context)>) + 'static,
    {
        let event_loop = self.event_loop.take().unwrap();
        self.plugin_manager.init_plugins(&mut self.scene);

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

                    self.plugin_manager.update_plugins(&mut self.scene, delta);

                    let mut systems = std::mem::take(&mut self.systems);
                    for system in &mut systems {
                        system.update(&mut self.scene, &mut self.renderer, &mut self.resource_manager, delta);
                    }
                    self.systems = systems;

                    // Rendering orchestration
                    let view_matrix = self.scene.last_view_matrix;
                    let (renderables_raw, instanced_raw, _, lights) = self.scene.collect_render_data(None);
                    let renderer_lights: Vec<(spark_math::Vec3, spark_math::Vec3, f32)> = lights.iter().map(|(trans, _type, col, intensity, _range)| {
                        (trans.w_axis.xyz(), *col, *intensity)
                    }).collect();

                    let total_objects = self.renderer.prepare_frame(view_matrix, &renderables_raw, &instanced_raw, &renderer_lights);

                    self.renderer.draw_frame(
                        &self.window,
                        egui_output,
                        total_objects
                    );

                    self.renderer.clear_instance_buffers();
                    self.event_queue.clear();
                }
                _ => (),
            }
        }).expect("Event loop failed");
    }
}
