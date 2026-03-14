pub mod scene;
pub mod task;
pub mod plugin;
pub mod resource;
pub mod event;

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

pub struct Engine {
    pub window: winit::window::Window,
    pub event_loop: Option<EventLoop<()>>,
    pub scene: Scene,
    pub renderer: Renderer,
    pub task_system: TaskSystem,
    pub plugin_manager: PluginManager,
    pub resource_manager: ResourceManager,
    pub event_queue: EventQueue,
    last_frame_time: instant::Instant,
}

impl Engine {
    pub fn new(title: &str) -> Result<Self, spark_renderer::error::RendererError> {
        let event_loop = EventLoop::new().expect("Failed to create event loop");
        let window = WindowBuilder::new()
            .with_title(title)
            .build(&event_loop)
            .expect("Failed to build window");

        let renderer = Renderer::new(&window)?;
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
        })
    }

    pub fn run<F>(mut self, mut ui_callback: F)
    where
        F: FnMut(&winit::window::Window, &winit::event::Event<()>, &mut Scene) -> bool + 'static,
    {
        let event_loop = self.event_loop.take().unwrap();
        self.plugin_manager.init_plugins(&mut self.scene);

        event_loop.run(move |event, elwt| {
            if ui_callback(&self.window, &event, &mut self.scene) {
                // UI consumed the event or handled frame begin/end
            }

            use crate::event::EngineEvent;
            match &event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    elwt.exit();
                }
                Event::WindowEvent { event, .. } => {
                    match event {
                        WindowEvent::Resized(size) => {
                            self.event_queue.push(EngineEvent::WindowResized { width: size.width, height: size.height });
                        }
                        WindowEvent::KeyboardInput {
                            event: input_event,
                            ..
                        } => {
                            // Map to EngineEvent
                            log::info!("Keyboard input: {:?}", input_event);
                        }
                        WindowEvent::CursorMoved {
                            position,
                            ..
                        } => {
                            self.event_queue.push(EngineEvent::MouseMoved { x: position.x, y: position.y });
                        }
                        _ => {}
                    }
                }
                Event::AboutToWait => {
                    // Logic that uses event_queue would go here
                    self.event_queue.clear();
                    let now = instant::Instant::now();
                    let delta = now.duration_since(self.last_frame_time).as_secs_f32();
                    self.last_frame_time = now;

                    self.plugin_manager.update_plugins(&mut self.scene, delta);
                    self.scene.update_all_transforms();
                    let (renderables, view_matrix) = self.scene.collect_render_data();

                    let extent = self.renderer.get_extent();
                    let projection = spark_math::Mat4::perspective_rh(
                        45.0f32.to_radians(),
                        extent.width as f32 / extent.height as f32,
                        0.1,
                        100.0,
                    );
                    let view_proj = projection * view_matrix;

                    self.renderer.draw_frame(&renderables, view_proj, &self.window);
                }
                _ => (),
            }
        }).expect("Event loop failed");
    }
}
