pub mod ffi;
use spark_core::plugin::Plugin;
use libloading::{Library, Symbol};
use std::rc::Rc;
use netcorehost::{nethost, hostfxr::Hostfxr, pdcstring::PdCString};

pub struct ScriptHost {
    libraries: Vec<Rc<Library>>,
    pub hostfxr: Option<Hostfxr>,
}

impl ScriptHost {
    pub fn new() -> Self {
        let hostfxr = nethost::load_hostfxr().ok();
        Self {
            libraries: Vec::new(),
            hostfxr,
        }
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

    pub fn init_dotnet(&mut self, config_path: &str) {
        log::info!("Initializing .NET Runtime with config: {}", config_path);

        if let Some(hostfxr) = &self.hostfxr {
            let config_path_pdc = PdCString::from_os_str(config_path).unwrap();
            let context = hostfxr.initialize_for_runtime_config(config_path_pdc).expect("Failed to initialize .NET core");
            let _loader = context.get_delegate_loader().expect("Failed to get delegate loader");
            log::info!(".NET Runtime initialized successfully");
        } else {
            log::warn!(".NET Hostfxr not found. C# scripting will be unavailable.");
        }
    }
}
