pub trait Script {
    fn on_init(&mut self);
    fn on_update(&mut self, delta: f32);
}

pub struct ScriptHost {}

impl ScriptHost {
    pub fn new() -> Self {
        Self {}
    }

    pub fn load_rust_plugin(&mut self, path: &str) {
        log::info!("Loading Rust plugin from: {}", path);
    }

    pub fn init_dotnet(&mut self) {
        log::info!("Initializing .NET Runtime...");
    }
}
