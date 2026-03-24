use std::collections::{HashSet, HashMap};
use winit::keyboard::KeyCode;
use crate::event::EngineEvent;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InputAction {
    MoveForward,
    MoveBackward,
    MoveLeft,
    MoveRight,
    Jump,
    Interact,
}

pub struct InputManager {
    keys_pressed: HashSet<KeyCode>,
    mouse_position: (f64, f64),
    action_map: HashMap<KeyCode, InputAction>,
}

impl InputManager {
    pub fn new() -> Self {
        let mut action_map = HashMap::new();
        action_map.insert(KeyCode::KeyW, InputAction::MoveForward);
        action_map.insert(KeyCode::KeyS, InputAction::MoveBackward);
        action_map.insert(KeyCode::KeyA, InputAction::MoveLeft);
        action_map.insert(KeyCode::KeyD, InputAction::MoveRight);
        action_map.insert(KeyCode::Space, InputAction::Jump);
        action_map.insert(KeyCode::KeyE, InputAction::Interact);

        Self {
            keys_pressed: HashSet::new(),
            mouse_position: (0.0, 0.0),
            action_map,
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

    pub fn is_action_pressed(&self, action: InputAction) -> bool {
        for (key, mapped_action) in &self.action_map {
            if *mapped_action == action && self.keys_pressed.contains(key) {
                return true;
            }
        }
        false
    }

    pub fn mouse_position(&self) -> (f64, f64) {
        self.mouse_position
    }

    pub fn set_action_mapping(&mut self, key: KeyCode, action: InputAction) {
        self.action_map.insert(key, action);
    }
}

impl Default for InputManager {
    fn default() -> Self {
        Self::new()
    }
}
