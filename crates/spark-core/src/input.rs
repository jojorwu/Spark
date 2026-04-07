use crate::event::EngineEvent;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use winit::keyboard::KeyCode;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InputAction {
    MoveForward,
    MoveBackward,
    MoveLeft,
    MoveRight,
    Jump,
    Interact,
}

/// Manages user input by tracking key states and mapping them to logical actions.
pub struct InputManager {
    keys_pressed: HashSet<KeyCode>,
    mouse_position: (f64, f64),
    action_map: HashMap<KeyCode, InputAction>,
    /// Cache for actions that are currently pressed to avoid per-query resolution.
    active_actions: HashSet<InputAction>,
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
            active_actions: HashSet::new(),
        }
    }

    /// Processes input events and updates the internal key states and active actions.
    pub fn update(&mut self, events: &[crate::event::EngineEvent]) {
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

        // Rebuild active actions cache
        self.active_actions.clear();
        for (key, action) in &self.action_map {
            if self.keys_pressed.contains(key) {
                self.active_actions.insert(*action);
            }
        }
    }

    /// Returns true if the specified key is currently held down.
    pub fn is_key_pressed(&self, key: KeyCode) -> bool {
        self.keys_pressed.contains(&key)
    }

    /// Returns true if any key mapped to the logical action is currently held down.
    pub fn is_action_pressed(&self, action: InputAction) -> bool {
        self.active_actions.contains(&action)
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
