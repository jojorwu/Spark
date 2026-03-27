use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub struct Resources {
    storage: HashMap<TypeId, Arc<RwLock<Box<dyn Any + Send + Sync>>>>,
}

impl Resources {
    pub fn new() -> Self {
        Self {
            storage: HashMap::new(),
        }
    }

    pub fn insert<T: Send + Sync + 'static>(&mut self, resource: T) {
        self.storage
            .insert(TypeId::of::<T>(), Arc::new(RwLock::new(Box::new(resource))));
    }

    pub fn get<T: 'static>(&self) -> Option<Arc<RwLock<Box<dyn Any + Send + Sync>>>> {
        self.storage.get(&TypeId::of::<T>()).cloned()
    }

    pub fn remove<T: 'static>(&mut self) {
        self.storage.remove(&TypeId::of::<T>());
    }
}

impl Default for Resources {
    fn default() -> Self {
        Self::new()
    }
}
