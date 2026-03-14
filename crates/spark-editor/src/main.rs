use spark_core::Engine;
pub mod ui;

use spark_renderer::pipeline::Pipeline;
use std::fs;

use spark_script::ScriptHost;
use crate::ui::EditorUI;

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

    use spark_renderer::vertex::Vertex;
    use spark_math::{Vec2, Vec3, Mat4};

    let vertices = [
        Vertex { pos: Vec3::new(0.0, -0.5, 0.0), color: Vec3::new(1.0, 0.0, 0.0), tex_coord: Vec2::ZERO },
        Vertex { pos: Vec3::new(0.5, 0.5, 0.0), color: Vec3::new(0.0, 1.0, 0.0), tex_coord: Vec2::ZERO },
        Vertex { pos: Vec3::new(-0.5, 0.5, 0.0), color: Vec3::new(0.0, 0.0, 1.0), tex_coord: Vec2::ZERO },
    ];

    let vb = engine.renderer.create_buffer(
        (std::mem::size_of::<Vertex>() * vertices.len()) as u64,
        ash::vk::BufferUsageFlags::VERTEX_BUFFER,
        ash::vk::MemoryPropertyFlags::HOST_VISIBLE | ash::vk::MemoryPropertyFlags::HOST_COHERENT,
    );
    engine.renderer.upload_to_buffer(&vb, &vertices);
    engine.renderer.set_vertex_buffer(vb);

    use spark_core::scene::{Node, NodeData};

    let triangle_node = Node {
        name: "MyTriangle".to_string(),
        local_transform: Mat4::from_translation(Vec3::new(0.0, 0.0, -5.0)),
        global_transform: Mat4::IDENTITY,
        parent: None,
        children: Vec::new(),
        data: NodeData::Mesh { vertex_count: 3, texture_id: None, vertex_buffer_id: Some(0) },
    };

    engine.scene.add_node(engine.scene.root, triangle_node);

    let _camera_node = Node {
        name: "MainCamera".to_string(),
        local_transform: Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0)),
        global_transform: Mat4::IDENTITY,
        parent: None,
        children: Vec::new(),
        data: NodeData::Camera { fov: 45.0, near: 0.1, far: 100.0 },
    };

    engine.scene.add_node(engine.scene.root, _camera_node);

    let _script_host = ScriptHost::new();
    // Initialize .NET logic would go here if we had a valid runtimeconfig.json
    // script_host.init_dotnet("path/to/runtimeconfig.json");

    let mut ui = EditorUI::new(&engine.window);

    engine.run(move |window, event, scene| {
        match event {
            winit::event::Event::WindowEvent { event, .. } => {
                ui.handle_event(window, event)
            }
            winit::event::Event::AboutToWait => {
                ui.begin_frame(window);
                ui.draw_ui(scene);
                let _full_output = ui.end_frame(window);
                // We need to render egui output here eventually
                false
            }
            _ => false,
        }
    });
}
