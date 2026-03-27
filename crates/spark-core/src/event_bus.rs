use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub trait Event: Any + Send + Sync {}
impl Event for crate::event::EngineEvent {}

use std::sync::RwLock;

type EventHandler = Box<dyn Fn(&dyn Any) + Send + Sync>;
type EventMap = HashMap<TypeId, Vec<Arc<dyn Any + Send + Sync>>>;

pub struct EventBus {
    handlers: RwLock<HashMap<TypeId, Vec<EventHandler>>>,
    incoming_events: Mutex<EventMap>,
    active_events: RwLock<EventMap>,
}

impl EventBus {
    pub fn new() -> Self {
        Self {
            handlers: RwLock::new(HashMap::new()),
            incoming_events: Mutex::new(HashMap::new()),
            active_events: RwLock::new(HashMap::new()),
        }
    }

    pub fn subscribe<E: Event, F>(&self, handler: F)
    where
        F: Fn(&E) + Send + Sync + 'static,
    {
        let mut handlers = self.handlers.write().unwrap();
        let entry = handlers.entry(TypeId::of::<E>()).or_default();
        entry.push(Box::new(move |any_event| {
            if let Some(event) = any_event.downcast_ref::<E>() {
                handler(event);
            }
        }));
    }

    pub fn publish<E: Event + Send + Sync + 'static>(&self, event: E) {
        // Instant handlers
        {
            let handlers = self.handlers.read().unwrap();
            if let Some(event_handlers) = handlers.get(&TypeId::of::<E>()) {
                for handler in event_handlers {
                    handler(&event);
                }
            }
        }

        // Buffered events
        let mut events = self.incoming_events.lock().unwrap();
        events
            .entry(TypeId::of::<E>())
            .or_default()
            .push(Arc::new(event));
    }

    /// Reads events from the ACTIVE buffer (published in previous frames and swapped).
    pub fn read_events<E: Clone + Send + Sync + 'static>(&self) -> Vec<E> {
        let events = self.active_events.read().unwrap();
        if let Some(event_list) = events.get(&TypeId::of::<E>()) {
            event_list
                .iter()
                .filter_map(|e| e.downcast_ref::<E>().cloned())
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Swaps the incoming buffer into the active buffer and clears incoming.
    pub fn swap_buffers(&self) {
        let mut incoming = self.incoming_events.lock().unwrap();
        let mut active = self.active_events.write().unwrap();
        *active = std::mem::take(&mut *incoming);
    }

    pub fn clear_events(&self) {
        self.active_events.write().unwrap().clear();
        self.incoming_events.lock().unwrap().clear();
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}
