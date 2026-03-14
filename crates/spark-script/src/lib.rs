use spark_core::plugin::Plugin;
use libloading::{Library, Symbol};
use std::rc::Rc;

pub struct ScriptHost {
    libraries: Vec<Rc<Library>>,
}

impl ScriptHost {
    pub fn new() -> Self {
        Self { libraries: Vec::new() }
    }

    pub fn load_rust_plugin(&mut self, path: &str) -> Box<dyn Plugin> {
        log::info!("Loading Rust plugin from: {}", path);

        let lib = unsafe { Library::new(path).expect("Failed to load library") };
        let lib = Rc::new(lib);
        self.libraries.push(lib.clone());

        unsafe {
            let constructor: Symbol<fn() -> Box<dyn Plugin>> = lib.get(b"create_plugin").expect("Failed to find constructor");
            constructor()
        }
    }
}
