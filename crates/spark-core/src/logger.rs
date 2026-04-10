use log::{Log, Metadata, Record};
use std::sync::{Arc, Mutex};

pub struct EditorLogger {
    pub logs: Arc<Mutex<Vec<String>>>,
}

impl EditorLogger {
    pub fn new() -> (Self, Arc<Mutex<Vec<String>>>) {
        let logs = Arc::new(Mutex::new(Vec::new()));
        (Self { logs: logs.clone() }, logs)
    }

    pub fn init(self) {
        log::set_boxed_logger(Box::new(self)).expect("Failed to initialize global logger");
        log::set_max_level(log::LevelFilter::Info);
        log_panics::init();
    }
}

impl Log for EditorLogger {
    fn enabled(&self, _metadata: &Metadata) -> bool {
        true
    }

    fn log(&self, record: &Record) {
        if self.enabled(record.metadata()) {
            let mut logs = self.logs.lock().expect("Failed to lock logs for writing");
            let msg = format!("[{}] {}", record.level(), record.args());
            logs.push(msg);
            if logs.len() > 1000 {
                logs.remove(0);
            }
        }
    }

    fn flush(&self) {}
}
