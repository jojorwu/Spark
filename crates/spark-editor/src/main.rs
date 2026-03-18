use spark_core::Engine;
use spark_renderer::pipeline::Pipeline;
use std::fs;

mod ui;

struct ShaderCompiler {
    compiler: shaderc::Compiler,
}

impl ShaderCompiler {
    fn new() -> Self {
        Self { compiler: shaderc::Compiler::new().unwrap() }
    }

    fn compile(&self, path: &str, kind: shaderc::ShaderKind) -> Vec<u32> {
        let code = fs::read_to_string(path).expect(&format!("Failed to read shader: {}", path));
        let name = std::path::Path::new(path).file_name().unwrap().to_str().unwrap();
        self.compiler.compile_into_spirv(&code, kind, name, "main", None).unwrap().as_binary().to_vec()
    }

    fn compile_with_options(&self, path: &str, kind: shaderc::ShaderKind, options: &shaderc::CompileOptions) -> Vec<u32> {
        let code = fs::read_to_string(path).expect(&format!("Failed to read shader: {}", path));
        let name = std::path::Path::new(path).file_name().unwrap().to_str().unwrap();
        self.compiler.compile_into_spirv(&code, kind, name, "main", Some(options)).unwrap().as_binary().to_vec()
    }
}
use spark_script::ScriptHost;
use crate::ui::EditorUI;

fn main() {
    env_logger::init();
    log::info!("Spark Editor starting...");

    let compiler = ShaderCompiler::new();

    let culling_spirv = compiler.compile("assets/shaders/culling.comp", shaderc::ShaderKind::Compute);
    let hiz_spirv = compiler.compile("assets/shaders/hiz.comp", shaderc::ShaderKind::Compute);

    let ui_vert_spirv = compiler.compile("assets/shaders/ui.vert", shaderc::ShaderKind::Vertex);
    let ui_frag_spirv = compiler.compile("assets/shaders/ui.frag", shaderc::ShaderKind::Fragment);

    let mut engine = Engine::new(
        "Spark Engine Editor",
        Some((&ui_vert_spirv, &ui_frag_spirv))
    ).expect("Failed to initialize engine");

    let vert_spirv = compiler.compile("assets/shaders/gbuffer.vert", shaderc::ShaderKind::Vertex);
    let frag_spirv = compiler.compile("assets/shaders/gbuffer.frag", shaderc::ShaderKind::Fragment);

    let shadow_vert_spirv = compiler.compile("assets/shaders/shadow.vert", shaderc::ShaderKind::Vertex);
    let shadow_frag_spirv = compiler.compile("assets/shaders/shadow.frag", shaderc::ShaderKind::Fragment);

    engine.renderer.create_shadow_pipeline(&shadow_vert_spirv, &shadow_frag_spirv);
    engine.renderer.create_culling_pipeline(&culling_spirv);
    engine.renderer.create_hiz_pipeline(&hiz_spirv);

    let post_vert_spirv = compiler.compile("assets/shaders/fullscreen.vert", shaderc::ShaderKind::Vertex);
    let post_frag_spirv = compiler.compile("assets/shaders/tonemap_bloom.frag", shaderc::ShaderKind::Fragment);
    let bloom_frag_spirv = compiler.compile("assets/shaders/bloom_filter.frag", shaderc::ShaderKind::Fragment);

    engine.renderer.create_post_process_pipeline(&post_vert_spirv, &post_frag_spirv, &bloom_frag_spirv);

    let pipeline = Pipeline::new(
        engine.renderer.get_device(),
        engine.renderer.get_extent(),
        &vert_spirv,
        &frag_spirv,
        engine.renderer.get_msaa_samples(),
        false, // Not deferred lighting
        0,
        engine.renderer.pipeline_cache,
        engine.renderer.global_descriptor_set_layout,
        engine.renderer.bindless_descriptor_set_layout,
    );

    engine.renderer.set_pipeline(pipeline);

    // Initialize Deferred Pipeline
    let def_vert_spirv = compiler.compile("assets/shaders/deferred.vert", shaderc::ShaderKind::Vertex);

    let mut def_options = shaderc::CompileOptions::new().unwrap();
    let msaa_count = match engine.renderer.get_msaa_samples() {
        ash::vk::SampleCountFlags::TYPE_1 => 1,
        ash::vk::SampleCountFlags::TYPE_2 => 2,
        ash::vk::SampleCountFlags::TYPE_4 => 4,
        ash::vk::SampleCountFlags::TYPE_8 => 8,
        ash::vk::SampleCountFlags::TYPE_16 => 16,
        ash::vk::SampleCountFlags::TYPE_32 => 32,
        ash::vk::SampleCountFlags::TYPE_64 => 64,
        _ => 1,
    };
    def_options.add_macro_definition("MSAA_SAMPLES", Some(&msaa_count.to_string()));
    let def_frag_spirv = compiler.compile_with_options("assets/shaders/deferred.frag", shaderc::ShaderKind::Fragment, &def_options);

    let deferred_pipeline = Pipeline::new(
        engine.renderer.get_device(),
        engine.renderer.get_extent(),
        &def_vert_spirv,
        &def_frag_spirv,
        engine.renderer.get_msaa_samples(),
        true, // Deferred lighting
        4,    // 4 input attachments (Albedo, Normal, PBR, Depth)
        engine.renderer.pipeline_cache,
        engine.renderer.global_descriptor_set_layout,
        engine.renderer.bindless_descriptor_set_layout,
    );

    engine.renderer.set_deferred_pipeline(deferred_pipeline.graphics_pipeline);

    use spark_renderer::vertex::Vertex;
    use spark_math::{Vec2, Vec3, Mat4};

    let vertices = [
        Vertex { pos: Vec3::new(0.0, -0.5, 0.0), normal: Vec3::Z, color: Vec3::new(1.0, 0.0, 0.0), tex_coord: Vec2::ZERO },
        Vertex { pos: Vec3::new(0.5, 0.5, 0.0), normal: Vec3::Z, color: Vec3::new(0.0, 1.0, 0.0), tex_coord: Vec2::ZERO },
        Vertex { pos: Vec3::new(-0.5, 0.5, 0.0), normal: Vec3::Z, color: Vec3::new(0.0, 0.0, 1.0), tex_coord: Vec2::ZERO },
    ];

    let vb = engine.renderer.create_buffer(
        (std::mem::size_of::<Vertex>() * vertices.len()) as u64,
        ash::vk::BufferUsageFlags::VERTEX_BUFFER,
        ash::vk::MemoryPropertyFlags::HOST_VISIBLE | ash::vk::MemoryPropertyFlags::HOST_COHERENT,
    );
    engine.renderer.upload_to_buffer(&vb, &vertices);
    engine.renderer.add_vertex_buffer(vb);

    use spark_core::scene::{Node, NodeData};

    let triangle_node = Node {
        name: "MyTriangle".to_string(),
        local_transform: Mat4::from_translation(Vec3::new(0.0, 0.0, -5.0)),
        global_transform: Mat4::IDENTITY,
        parent: None,
        children: Vec::new(),
        data: NodeData::Mesh {
            vertex_count: 3,
            index_count: 3,
            first_index: 0,
            vertex_offset: 0,
            texture_id: None,
            vertex_buffer_id: Some(0),
            bounding_radius: 1.0,
        },
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

    let mut ui = EditorUI::new(&engine.window);

    engine.run(move |window, event, scene| {
        match event {
            winit::event::Event::WindowEvent { event, .. } => {
                (ui.handle_event(window, event), None)
            }
            winit::event::Event::AboutToWait => {
                ui.begin_frame(window);
                ui.draw_ui(scene);
                let full_output = ui.end_frame(window);
                (false, Some((full_output, ui.egui_ctx.clone())))
            }
            _ => (false, None),
        }
    });
}
