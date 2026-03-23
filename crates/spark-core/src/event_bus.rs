use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub trait Event: Any + Send + Sync {}

type EventHandler = Box<dyn Fn(&dyn Any) + Send + Sync>;

pub struct EventBus {
    handlers: Arc<Mutex<HashMap<TypeId, Vec<EventHandler>>>>,
}

impl EventBus {
    pub fn new() -> Self {
        Self {
            handlers: Arc::new(Mutex::new(HashMap::new())),
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

    pub fn publish<E: Event>(&self, event: E) {
        let handlers = self.handlers.lock().unwrap();
        if let Some(event_handlers) = handlers.get(&TypeId::of::<E>()) {
            for handler in event_handlers {
                handler(&event);
            }
        }
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}
