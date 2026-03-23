pub enum EngineEvent {
    WindowResized { width: u32, height: u32 },
    KeyDown { key: winit::keyboard::KeyCode },
    KeyUp { key: winit::keyboard::KeyCode },
    MouseMoved { x: f64, y: f64 },
}

pub struct EventQueue {
    pub(crate) events: Vec<EngineEvent>,
}

impl EventQueue {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }
}

impl Default for EventQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl EventQueue {

    pub fn push(&mut self, event: EngineEvent) {
        self.events.push(event);
    }

    pub fn clear(&mut self) {
        self.events.clear();
    }

    pub fn iter(&self) -> std::slice::Iter<'_, EngineEvent> {
        self.events.iter()
    }
}
