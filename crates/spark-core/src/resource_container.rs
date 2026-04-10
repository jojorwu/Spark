use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// A type-safe container for global engine resources.
///
/// Resources are stored as boxed `Any` types wrapped in an `Arc<RwLock<...>>`
/// to allow for shared, thread-safe access across parallel systems.
pub struct Resources {
    storage: HashMap<TypeId, Arc<RwLock<Box<dyn Any + Send + Sync>>>>,
}

impl Resources {
    pub fn new() -> Self {
        Self {
            storage: HashMap::new(),
        }
    }

    /// Inserts a new resource into the container.
    pub fn insert<T: Send + Sync + 'static>(&mut self, resource: T) {
        self.storage
            .insert(TypeId::of::<T>(), Arc::new(RwLock::new(Box::new(resource))));
    }

    /// Retrieves a shared reference to a resource by its type.
    pub fn get<T: 'static>(&self) -> Option<Arc<RwLock<Box<dyn Any + Send + Sync>>>> {
        self.storage.get(&TypeId::of::<T>()).cloned()
    }

    /// Retrieves a shared reference to a resource by its `TypeId`.
    pub fn get_by_id(&self, id: TypeId) -> Option<Arc<RwLock<Box<dyn Any + Send + Sync>>>> {
        self.storage.get(&id).cloned()
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
