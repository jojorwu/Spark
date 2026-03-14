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

    use spark_renderer::vertex::Vertex;
    use spark_math::Vec2;
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
    use spark_math::{Mat4, Vec3};

    let triangle_node = Node {
        name: "MyTriangle".to_string(),
        local_transform: Mat4::from_translation(Vec3::new(0.0, 0.0, -5.0)),
        global_transform: Mat4::IDENTITY,
        parent: None,
        children: Vec::new(),
        data: NodeData::Mesh { vertex_count: 3 },
    };

    engine.scene.add_node(engine.scene.root, triangle_node);

    let camera_node = Node {
        name: "MainCamera".to_string(),
        local_transform: Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0)),
        global_transform: Mat4::IDENTITY,
        parent: None,
        children: Vec::new(),
        data: NodeData::Camera { fov: 45.0, near: 0.1, far: 100.0 },
    };

    engine.scene.add_node(engine.scene.root, camera_node);

    engine.run();
}
