pub mod scene;

use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};
use crate::scene::Scene;
use spark_renderer::Renderer;

pub struct Engine {
    pub window: winit::window::Window,
    pub event_loop: Option<EventLoop<()>>,
    pub scene: Scene,
    pub renderer: Renderer,
}

impl Engine {
    pub fn new(title: &str) -> Self {
        let event_loop = EventLoop::new().expect("Failed to create event loop");
        let window = WindowBuilder::new()
            .with_title(title)
            .build(&event_loop)
            .expect("Failed to build window");

        let renderer = Renderer::new(&window);
        let scene = Scene::new();

        Self {
            window,
            event_loop: Some(event_loop),
            scene,
            renderer,
        }
    }

    pub fn run(mut self) {
        let event_loop = self.event_loop.take().unwrap();

        event_loop.run(move |event, elwt| {
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    elwt.exit();
                }
                Event::WindowEvent {
                    event: WindowEvent::KeyboardInput {
                        event: input_event,
                        ..
                    },
                    ..
                } => {
                    log::info!("Keyboard input: {:?}", input_event);
                }
                Event::WindowEvent {
                    event: WindowEvent::CursorMoved {
                        position,
                        ..
                    },
                    ..
                } => {
                    log::debug!("Mouse position: {:?}", position);
                }
                Event::AboutToWait => {
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

                    self.renderer.draw_frame(&renderables, view_proj);
                }
                _ => (),
            }
        }).expect("Event loop failed");
    }
}
