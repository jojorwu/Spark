use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub trait Event: Any + Send + Sync {}
impl Event for crate::event::EngineEvent {}

type EventHandler = Box<dyn Fn(&dyn Any) + Send + Sync>;

pub struct EventBus {
    handlers: Arc<Mutex<HashMap<TypeId, Vec<EventHandler>>>>,
    events: Arc<Mutex<HashMap<TypeId, Vec<Box<dyn Any + Send + Sync>>>>>,
}

impl EventBus {
    pub fn new() -> Self {
        Self {
            handlers: Arc::new(Mutex::new(HashMap::new())),
            events: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn subscribe<E: Event, F>(&self, handler: F)
    where
        F: Fn(&E) + Send + Sync + 'static,
    {
        let mut handlers = self.handlers.lock().unwrap();
        let entry = handlers.entry(TypeId::of::<E>()).or_default();
        entry.push(Box::new(move |any_event| {
            if let Some(event) = any_event.downcast_ref::<E>() {
                handler(event);
            }
        }));
    }

    pub fn publish<E: Event + Send + Sync + 'static>(&self, event: E) {
        // Instant handlers
        let handlers = self.handlers.lock().unwrap();
        if let Some(event_handlers) = handlers.get(&TypeId::of::<E>()) {
            for handler in event_handlers {
                handler(&event);
            }
        }

        // Buffered events
        let mut events = self.events.lock().unwrap();
        events.entry(TypeId::of::<E>()).or_default().push(Box::new(event));
    }

    pub fn read_events<E: 'static>(&self) -> Vec<E>
    where E: Clone + Send + Sync {
        let events = self.events.lock().unwrap();
        if let Some(event_list) = events.get(&TypeId::of::<E>()) {
            event_list.iter().filter_map(|e| e.downcast_ref::<E>().cloned()).collect()
        } else {
            Vec::new()
        }
    }

    pub fn clear_events(&self) {
        self.events.lock().unwrap().clear();
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}
