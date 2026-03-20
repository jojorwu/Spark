use winit::event::{WindowEvent, ElementState};
use winit::keyboard::PhysicalKey;
use crate::event::{EventQueue, EngineEvent};

/// Helper to map winit events to Spark engine events.
pub struct EventMapper;

impl EventMapper {
    pub fn map_window_event(event: &WindowEvent, queue: &mut EventQueue) -> bool {
        match event {
            WindowEvent::Resized(size) => {
                queue.push(EngineEvent::WindowResized { width: size.width, height: size.height });
                true
            }
            WindowEvent::KeyboardInput { event: input_event, .. } => {
                if let PhysicalKey::Code(code) = input_event.physical_key {
                    if input_event.state == ElementState::Pressed {
                        queue.push(EngineEvent::KeyDown { key: code });
                    } else {
                        queue.push(EngineEvent::KeyUp { key: code });
                    }
                    return true;
                }
                false
            }
            WindowEvent::CursorMoved { position, .. } => {
                queue.push(EngineEvent::MouseMoved { x: position.x, y: position.y });
                true
            }
            _ => false,
        }
    }
}
