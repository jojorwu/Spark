use crate::event::EngineEvent;

pub enum SystemEvent {
    Engine(EngineEvent),
    Custom(String, serde_json::Value),
}

use std::sync::Mutex;

pub struct EventProxy<'a> {
    pub events: &'a [EngineEvent],
    pub(crate) outgoing: &'a Mutex<Vec<SystemEvent>>,
}

impl<'a> EventProxy<'a> {
    pub fn iter_engine_events(&self) -> std::slice::Iter<'_, EngineEvent> {
        self.events.iter()
    }

    pub fn publish_custom(&self, name: &str, data: serde_json::Value) {
        self.outgoing.lock().unwrap().push(SystemEvent::Custom(name.to_string(), data));
    }
}
