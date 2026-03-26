use crate::event::EngineEvent;
use winit::event::{ElementState, WindowEvent};
use winit::keyboard::PhysicalKey;

/// Helper to map winit events to Spark engine events.
pub struct EventMapper;

impl EventMapper {
    pub fn map_window_event(event: &WindowEvent) -> Option<EngineEvent> {
        match event {
            WindowEvent::Resized(size) => Some(EngineEvent::WindowResized {
                width: size.width,
                height: size.height,
            }),
            WindowEvent::KeyboardInput {
                event: input_event, ..
            } => {
                if let PhysicalKey::Code(code) = input_event.physical_key {
                    if input_event.state == ElementState::Pressed {
                        Some(EngineEvent::KeyDown { key: code })
                    } else {
                        Some(EngineEvent::KeyUp { key: code })
                    }
                } else {
                    None
                }
            }
            WindowEvent::CursorMoved { position, .. } => Some(EngineEvent::MouseMoved {
                x: position.x,
                y: position.y,
            }),
            _ => None,
        }
    }
}
