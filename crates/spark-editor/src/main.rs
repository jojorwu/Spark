use spark_core::Engine;
use spark_renderer::pipeline::Pipeline;
use std::fs;

fn main() {
    env_logger::init();
    log::info!("Spark Editor starting...");

    let mut engine = Engine::new("Spark Engine Editor");

    let vert_code = fs::read_to_string("assets/shaders/triangle.vert").expect("Failed to read vertex shader");
    let frag_code = fs::read_to_string("assets/shaders/triangle.frag").expect("Failed to read fragment shader");

    let compiler = shaderc::Compiler::new().unwrap();

    let vert_spirv = compiler.compile_into_spirv(&vert_code, shaderc::ShaderKind::Vertex, "triangle.vert", "main", None).unwrap();
    let frag_spirv = compiler.compile_into_spirv(&frag_code, shaderc::ShaderKind::Fragment, "triangle.frag", "main", None).unwrap();

    let pipeline = Pipeline::new(
        engine.renderer.get_device(),
        engine.renderer.render_pass,
        engine.renderer.get_extent(),
        vert_spirv.as_binary(),
        frag_spirv.as_binary(),
    );

    engine.renderer.set_pipeline(pipeline);

    engine.run();
}
