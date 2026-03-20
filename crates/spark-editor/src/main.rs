use spark_core::Engine;
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
        let code = fs::read_to_string(path).unwrap_or_else(|_| panic!("Failed to read shader: {}", path));
        let name = std::path::Path::new(path).file_name().unwrap().to_str().unwrap();
        self.compiler.compile_into_spirv(&code, kind, name, "main", None).unwrap().as_binary().to_vec()
    }

    fn compile_with_options(&self, path: &str, kind: shaderc::ShaderKind, options: &shaderc::CompileOptions) -> Vec<u32> {
        let code = fs::read_to_string(path).unwrap_or_else(|_| panic!("Failed to read shader: {}", path));
        let name = std::path::Path::new(path).file_name().unwrap().to_str().unwrap();
        self.compiler.compile_into_spirv(&code, kind, name, "main", Some(options)).unwrap().as_binary().to_vec()
    }
}
use spark_script::ScriptHost;
use crate::ui::EditorUI;

fn main() {
    let (logger, logs) = spark_core::logger::EditorLogger::new();
    logger.init();
    log::info!("Spark Editor starting...");

    let compiler = ShaderCompiler::new();

    let ui_vert_spirv = compiler.compile("assets/shaders/ui.vert", shaderc::ShaderKind::Vertex);
    let ui_frag_spirv = compiler.compile("assets/shaders/ui.frag", shaderc::ShaderKind::Fragment);

    let mut engine = Engine::new(
        "Spark Engine Editor",
        Some((&ui_vert_spirv, &ui_frag_spirv))
    ).expect("Failed to initialize engine");

    engine.add_system(spark_core::systems::component::ComponentSystem);
    engine.add_system(spark_core::systems::HierarchySystem);
    engine.add_system(spark_core::systems::ResourceSystem);

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

    let shaders = spark_renderer::factory::PassShaders {
        culling: compiler.compile("assets/shaders/culling.comp", shaderc::ShaderKind::Compute),
        hiz: compiler.compile("assets/shaders/hiz.comp", shaderc::ShaderKind::Compute),
        cluster_build: compiler.compile("assets/shaders/cluster_build.comp", shaderc::ShaderKind::Compute),
        cluster_cull: compiler.compile("assets/shaders/cluster_cull.comp", shaderc::ShaderKind::Compute),
        shadow_vert: compiler.compile("assets/shaders/shadow.vert", shaderc::ShaderKind::Vertex),
        shadow_frag: compiler.compile("assets/shaders/shadow.frag", shaderc::ShaderKind::Fragment),
        gbuffer_vert: compiler.compile("assets/shaders/gbuffer.vert", shaderc::ShaderKind::Vertex),
        gbuffer_frag: compiler.compile("assets/shaders/gbuffer.frag", shaderc::ShaderKind::Fragment),
        ssao_vert: compiler.compile("assets/shaders/ssao.vert", shaderc::ShaderKind::Vertex),
        ssao_frag: compiler.compile("assets/shaders/ssao.frag", shaderc::ShaderKind::Fragment),
        ssao_blur_frag: compiler.compile("assets/shaders/ssao_blur.frag", shaderc::ShaderKind::Fragment),
        deferred_vert: compiler.compile("assets/shaders/deferred.vert", shaderc::ShaderKind::Vertex),
        deferred_frag: compiler.compile_with_options("assets/shaders/deferred.frag", shaderc::ShaderKind::Fragment, &def_options),
        grid_vert: compiler.compile("assets/shaders/grid.vert", shaderc::ShaderKind::Vertex),
        grid_frag: compiler.compile("assets/shaders/grid.frag", shaderc::ShaderKind::Fragment),
        volumetric: compiler.compile("assets/shaders/volumetric.comp", shaderc::ShaderKind::Compute),
        taa_vert: compiler.compile("assets/shaders/taa.vert", shaderc::ShaderKind::Vertex),
        taa_frag: compiler.compile("assets/shaders/taa.frag", shaderc::ShaderKind::Fragment),
        fullscreen_vert: compiler.compile("assets/shaders/fullscreen.vert", shaderc::ShaderKind::Vertex),
        tonemap_frag: compiler.compile("assets/shaders/tonemap_bloom.frag", shaderc::ShaderKind::Fragment),
        bloom_downsample: compiler.compile("assets/shaders/bloom_downsample.frag", shaderc::ShaderKind::Fragment),
        bloom_upsample: compiler.compile("assets/shaders/bloom_upsample.frag", shaderc::ShaderKind::Fragment),
    };

    engine.renderer.setup_default_passes(shaders).expect("Failed to setup render passes");


    use spark_renderer::vertex::Vertex;
    use spark_math::{Vec2, Vec3, Mat4};

    let vertices = [
        Vertex::pack(Vec3::new(0.0, -0.5, 0.0), Vec3::Z, Vec2::ZERO, Vec3::new(1.0, 0.0, 0.0)),
        Vertex::pack(Vec3::new(0.5, 0.5, 0.0), Vec3::Z, Vec2::ZERO, Vec3::new(0.0, 1.0, 0.0)),
        Vertex::pack(Vec3::new(-0.5, 0.5, 0.0), Vec3::Z, Vec2::ZERO, Vec3::new(0.0, 0.0, 1.0)),
    ];

    let vb = engine.renderer.create_buffer(
        (std::mem::size_of::<Vertex>() * vertices.len()) as u64,
        ash::vk::BufferUsageFlags::VERTEX_BUFFER,
        ash::vk::MemoryPropertyFlags::HOST_VISIBLE | ash::vk::MemoryPropertyFlags::HOST_COHERENT,
    );
    engine.renderer.upload_to_buffer(&vb, &vertices);
    engine.renderer.add_vertex_buffer(vb);

    use spark_core::scene::{Node, MeshComponent, CameraComponent};

    let triangle_node = Node {
        name: "MyTriangle".to_string(),
        local_transform: Mat4::from_translation(Vec3::new(0.0, 0.0, -5.0)),
        global_transform: Mat4::IDENTITY,
        parent: None,
        children: Vec::new(),
        components: vec![Box::new(MeshComponent {
            vertex_count: 3,
            index_count: 3,
            first_index: 0,
            vertex_offset: 0,
            texture_handle: None,
            material_index: Some(0),
            bounding_radius: 1.0,
        })],
    };

    engine.scene.add_node(engine.scene.root, triangle_node);

    let _camera_node = Node {
        name: "MainCamera".to_string(),
        local_transform: Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0)),
        global_transform: Mat4::IDENTITY,
        parent: None,
        children: Vec::new(),
        components: vec![Box::new(CameraComponent { fov: 45.0, near: 0.1, far: 100.0 })],
    };

    engine.scene.add_node(engine.scene.root, _camera_node);

    let _script_host = ScriptHost::new();

    let mut ui = EditorUI::new(&engine.window, logs);
    engine.renderer.create_viewport_attachment(1280, 720);
    let viewport_view = engine.renderer.viewport_attachment.as_ref().unwrap().view;
    let viewport_sampler = engine.renderer.common_sampler;
    ui.viewport_texture_id = Some(engine.renderer.register_egui_texture(viewport_view, viewport_sampler));

    engine.run(move |window, event, scene, rm, renderer, fps| {
        match event {
            winit::event::Event::WindowEvent { event, .. } => {
                (ui.handle_event(window, event), None)
            }
            winit::event::Event::AboutToWait => {
                ui.begin_frame(window);
                ui.draw_ui(scene, rm, renderer, fps);
                ui.draw_viewport(scene, fps);
                let full_output = ui.end_frame(window);
                (false, Some((full_output, ui.egui_ctx.clone())))
            }
            _ => (false, None),
        }
    });
}
