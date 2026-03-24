#[derive(Clone, Debug)]
pub enum EngineEvent {
    WindowResized { width: u32, height: u32 },
    KeyDown { key: winit::keyboard::KeyCode },
    KeyUp { key: winit::keyboard::KeyCode },
    MouseMoved { x: f64, y: f64 },
}
