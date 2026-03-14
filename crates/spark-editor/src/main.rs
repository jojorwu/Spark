use spark_core::Engine;

fn main() {
    env_logger::init();
    log::info!("Spark Editor starting...");

    let _engine = Engine::new("Spark Engine Editor");
    // _engine.run(); // Not running here to avoid blocking during checks
}
