use std::collections::HashSet;
use winit::keyboard::KeyCode;
use crate::event::EngineEvent;

pub struct InputManager {
    keys_pressed: HashSet<KeyCode>,
    mouse_position: (f64, f64),
}

impl InputManager {
    pub fn new() -> Self {
        Self {
            keys_pressed: HashSet::new(),
            mouse_position: (0.0, 0.0),
        }
    }

    pub fn update(&mut self, events: &[EngineEvent]) {
        for event in events {
            match event {
                EngineEvent::KeyDown { key } => {
                    self.keys_pressed.insert(*key);
                }
                EngineEvent::KeyUp { key } => {
                    self.keys_pressed.remove(key);
                }
                EngineEvent::MouseMoved { x, y } => {
                    self.mouse_position = (*x, *y);
                }
                _ => {}
            }
        }
    }

    pub fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.keys_pressed.contains(&key)
    }

    pub fn mouse_position(&self) -> (f64, f64) {
        self.mouse_position
    }
}

impl Default for InputManager {
    fn default() -> Self {
        Self::new()
    }
}
