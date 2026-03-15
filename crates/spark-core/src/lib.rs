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
use spark_math::Vec4Swizzles;

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
        })
    }

    pub fn run<F>(mut self, mut ui_callback: F)
    where
        F: FnMut(&winit::window::Window, &winit::event::Event<()>, &mut Scene) -> (bool, Option<(egui::FullOutput, egui::Context)>) + 'static,
    {
        let event_loop = self.event_loop.take().unwrap();
        self.plugin_manager.init_plugins(&mut self.scene);

        event_loop.run(move |event, elwt| {
            let (ui_consumed, egui_output) = ui_callback(&self.window, &event, &mut self.scene);
            if ui_consumed {
                // UI consumed the event
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

                    let extent = self.renderer.get_extent();
                    let projection = spark_math::Mat4::perspective_rh(
                        45.0f32.to_radians(),
                        extent.width as f32 / extent.height as f32,
                        0.1,
                        100.0,
                    );

                    // First pass to find camera
                    let (_, _, view_matrix, _) = self.scene.collect_render_data(None);
                    let frustum = spark_math::Frustum::from_matrix(projection * view_matrix);

                    // Second pass for culled rendering
                    let (renderables, instanced_raw, _, lights) = self.scene.collect_render_data(Some(&frustum));

                    // Prepare instance buffers
                    let mut instanced_renderables = Vec::new();
                    for (vb_id, transforms) in instanced_raw {
                        let instance_buffer = self.renderer.create_buffer(
                            (std::mem::size_of::<spark_math::Mat4>() * transforms.len()) as u64,
                            spark_renderer::ash::vk::BufferUsageFlags::VERTEX_BUFFER,
                            spark_renderer::ash::vk::MemoryPropertyFlags::HOST_VISIBLE | spark_renderer::ash::vk::MemoryPropertyFlags::HOST_COHERENT,
                        );
                        self.renderer.upload_to_buffer(&instance_buffer, &transforms);
                        let ib_id = self.renderer.add_instance_buffer(instance_buffer);
                        instanced_renderables.push((vb_id, ib_id, transforms.len() as u32));
                    }

                    let view_proj = projection * view_matrix;

                    // Calculate Light View-Projection for Shadows
                    let light_pos = spark_math::Vec3::new(10.0, 10.0, 10.0);
                    let light_view = spark_math::Mat4::look_at_rh(
                        light_pos,
                        spark_math::Vec3::ZERO,
                        spark_math::Vec3::Y,
                    );
                    let light_proj = spark_math::Mat4::orthographic_rh(-20.0, 20.0, -20.0, 20.0, 0.1, 100.0);
                    let light_view_proj = light_proj * light_view;

                    let _main_light = lights.first().cloned().unwrap_or((
                        spark_math::Mat4::IDENTITY,
                        crate::scene::LightType::Directional,
                        spark_math::Vec3::ONE,
                        1.0,
                        10.0
                    ));

                    // Convert lights for renderer
                    let renderer_lights: Vec<(spark_math::Vec3, spark_math::Vec3, f32)> = lights.iter().map(|(trans, _type, col, intensity, _range)| {
                        (trans.w_axis.xyz(), *col, *intensity)
                    }).collect();
                    self.renderer.update_lights(&renderer_lights);

                    self.renderer.draw_frame(
                        &renderables,
                        &instanced_renderables,
                        view_proj,
                        light_view_proj,
                        &self.window,
                        egui_output
                    );

                    // Clear temporary instance buffers for next frame
                    // In a real engine, we'd reuse them or use a ring buffer.
                    self.renderer.clear_instance_buffers();
                }
                _ => (),
            }
        }).expect("Event loop failed");
    }
}
