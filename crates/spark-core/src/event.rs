pub enum EngineEvent {
    WindowResized { width: u32, height: u32 },
    KeyDown { key_code: u32 },
    KeyUp { key_code: u32 },
    MouseMoved { x: f64, y: f64 },
}

pub struct EventQueue {
    events: Vec<EngineEvent>,
}

impl EventQueue {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

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
