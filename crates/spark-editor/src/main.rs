use spark_core::{App, Plugin};
use std::fs;

mod ui;

struct ShaderCompiler {
    compiler: shaderc::Compiler,
}

impl ShaderCompiler {
    fn new() -> Self {
        Self { compiler: shaderc::Compiler::new().unwrap() }
    }

    fn compile(&self, path: &str, kind: shaderc::ShaderKind) -> Result<Vec<u32>, String> {
        let mut options = shaderc::CompileOptions::new().unwrap();
        options.set_include_callback(|name, _, _, _| {
            let p = format!("assets/shaders/{}", name);
            fs::read_to_string(&p)
                .map(|content| shaderc::ResolvedInclude {
                    resolved_name: name.to_string(),
                    content,
                })
                .map_err(|e| e.to_string())
        });

        let code = fs::read_to_string(path).map_err(|e| format!("Failed to read shader {}: {}", path, e))?;
        let name = std::path::Path::new(path).file_name().unwrap().to_str().unwrap();
        self.compiler.compile_into_spirv(&code, kind, name, "main", Some(&options))
            .map(|artifact| artifact.as_binary().to_vec())
            .map_err(|e| format!("Shader compilation error in {}: {}", path, e))
    }

    fn compile_with_options(&self, path: &str, kind: shaderc::ShaderKind, options: &shaderc::CompileOptions) -> Result<Vec<u32>, String> {
        let code = fs::read_to_string(path).map_err(|e| format!("Failed to read shader {}: {}", path, e))?;
        let name = std::path::Path::new(path).file_name().unwrap().to_str().unwrap();
        self.compiler.compile_into_spirv(&code, kind, name, "main", Some(options))
            .map(|artifact| artifact.as_binary().to_vec())
            .map_err(|e| format!("Shader compilation error in {}: {}", path, e))
    }
}
use spark_script::ScriptHost;
use crate::ui::EditorUI;

struct EditorPlugin {
    _logs: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}

impl Plugin for EditorPlugin {
    fn build(&self, app: &mut App) {
        app.engine.add_system_to_stage(spark_core::systems::CoreStage::First, spark_core::systems::time::TimeSystem);
        app.engine.add_system(spark_core::systems::component::ComponentSystem);
        app.engine.add_system(spark_core::systems::HierarchySystem);
        app.engine.add_system_to_stage(spark_core::systems::CoreStage::Last, spark_core::systems::ResourceSystem);
    }
}

fn main() {
    let (logger, logs) = spark_core::logger::EditorLogger::new();
    logger.init();
    log::info!("Spark Editor starting...");

    let task_system = spark_core::task::TaskSystem::new();
    let compiler = ShaderCompiler::new();

    let ui_vert_spirv = compiler.compile("assets/shaders/ui.vert", shaderc::ShaderKind::Vertex).expect("Failed to compile UI vertex shader");
    let ui_frag_spirv = compiler.compile("assets/shaders/ui.frag", shaderc::ShaderKind::Fragment).expect("Failed to compile UI fragment shader");

    let mut app = App::with_ui_shaders("Spark Engine Editor", &ui_vert_spirv, &ui_frag_spirv);

    app = app.add_plugin(EditorPlugin { _logs: logs.clone() });

    let mut def_options = shaderc::CompileOptions::new().unwrap();
    let msaa_count = match app.engine.renderer.get_msaa_samples() {
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
        culling: compiler.compile("assets/shaders/culling.comp", shaderc::ShaderKind::Compute).expect("Failed to compile culling.comp"),
        hiz: compiler.compile("assets/shaders/hiz.comp", shaderc::ShaderKind::Compute).expect("Failed to compile hiz.comp"),
        cluster_build: compiler.compile("assets/shaders/cluster_build.comp", shaderc::ShaderKind::Compute).expect("Failed to compile cluster_build.comp"),
        cluster_cull: compiler.compile("assets/shaders/cluster_cull.comp", shaderc::ShaderKind::Compute).expect("Failed to compile cluster_cull.comp"),
        shadow_vert: compiler.compile("assets/shaders/shadow.vert", shaderc::ShaderKind::Vertex).expect("Failed to compile shadow.vert"),
        shadow_frag: compiler.compile("assets/shaders/shadow.frag", shaderc::ShaderKind::Fragment).expect("Failed to compile shadow.frag"),
        gbuffer_vert: compiler.compile("assets/shaders/gbuffer.vert", shaderc::ShaderKind::Vertex).expect("Failed to compile gbuffer.vert"),
        gbuffer_frag: compiler.compile("assets/shaders/gbuffer.frag", shaderc::ShaderKind::Fragment).expect("Failed to compile gbuffer.frag"),
        ssao_vert: compiler.compile("assets/shaders/ssao.vert", shaderc::ShaderKind::Vertex).expect("Failed to compile ssao.vert"),
        ssao_frag: compiler.compile("assets/shaders/ssao.frag", shaderc::ShaderKind::Fragment).expect("Failed to compile ssao.frag"),
        ssao_blur_frag: compiler.compile("assets/shaders/ssao_blur.frag", shaderc::ShaderKind::Fragment).expect("Failed to compile ssao_blur.frag"),
        deferred_vert: compiler.compile("assets/shaders/deferred.vert", shaderc::ShaderKind::Vertex).expect("Failed to compile deferred.vert"),
        deferred_frag: compiler.compile_with_options("assets/shaders/deferred.frag", shaderc::ShaderKind::Fragment, &def_options).expect("Failed to compile deferred.frag"),
        grid_vert: compiler.compile("assets/shaders/grid.vert", shaderc::ShaderKind::Vertex).expect("Failed to compile grid.vert"),
        grid_frag: compiler.compile("assets/shaders/grid.frag", shaderc::ShaderKind::Fragment).expect("Failed to compile grid.frag"),
        volumetric: compiler.compile("assets/shaders/volumetric.comp", shaderc::ShaderKind::Compute).expect("Failed to compile volumetric.comp"),
        taa_vert: compiler.compile("assets/shaders/taa.vert", shaderc::ShaderKind::Vertex).expect("Failed to compile taa.vert"),
        taa_frag: compiler.compile("assets/shaders/taa.frag", shaderc::ShaderKind::Fragment).expect("Failed to compile taa.frag"),
        fullscreen_vert: compiler.compile("assets/shaders/fullscreen.vert", shaderc::ShaderKind::Vertex).expect("Failed to compile fullscreen.vert"),
        tonemap_frag: compiler.compile("assets/shaders/tonemap.frag", shaderc::ShaderKind::Fragment).expect("Failed to compile tonemap.frag"),
        bloom_downsample: compiler.compile("assets/shaders/bloom_downsample.frag", shaderc::ShaderKind::Fragment).expect("Failed to compile bloom_downsample.frag"),
        bloom_upsample: compiler.compile("assets/shaders/bloom_upsample.frag", shaderc::ShaderKind::Fragment).expect("Failed to compile bloom_upsample.frag"),
        forward_vert: compiler.compile("assets/shaders/forward.vert", shaderc::ShaderKind::Vertex).expect("Failed to compile forward.vert"),
        forward_frag: compiler.compile("assets/shaders/forward.frag", shaderc::ShaderKind::Fragment).expect("Failed to compile forward.frag"),
        particle_comp: compiler.compile("assets/shaders/particle.comp", shaderc::ShaderKind::Compute).expect("Failed to compile particle.comp"),
        particle_vert: compiler.compile("assets/shaders/particle.vert", shaderc::ShaderKind::Vertex).expect("Failed to compile particle.vert"),
        particle_frag: compiler.compile("assets/shaders/particle.frag", shaderc::ShaderKind::Fragment).expect("Failed to compile particle.frag"),
        ssr_comp: compiler.compile("assets/shaders/ssr.comp", shaderc::ShaderKind::Compute).expect("Failed to compile ssr.comp"),
        sprite_vert: compiler.compile("assets/shaders/sprite.vert", shaderc::ShaderKind::Vertex).expect("Failed to compile sprite.vert"),
        sprite_frag: compiler.compile("assets/shaders/sprite.frag", shaderc::ShaderKind::Fragment).expect("Failed to compile sprite.frag"),
        luminance: compiler.compile("assets/shaders/luminance.comp", shaderc::ShaderKind::Compute).expect("Failed to compile luminance.comp"),
        dof: compiler.compile("assets/shaders/dof.comp", shaderc::ShaderKind::Compute).expect("Failed to compile dof.comp"),
        skinning: compiler.compile("assets/shaders/skinning.comp", shaderc::ShaderKind::Compute).expect("Failed to compile skinning.comp"),
        point_shadow_vert: compiler.compile("assets/shaders/point_shadow.vert", shaderc::ShaderKind::Vertex).expect("Failed to compile point_shadow.vert"),
        point_shadow_frag: compiler.compile("assets/shaders/point_shadow.frag", shaderc::ShaderKind::Fragment).expect("Failed to compile point_shadow.frag"),
        ssgi: compiler.compile("assets/shaders/ssgi.frag", shaderc::ShaderKind::Fragment).expect("Failed to compile ssgi.frag"),
        rgen: compiler.compile("assets/shaders/raytrace.rgen", shaderc::ShaderKind::RayGeneration).expect("Failed to compile raytrace.rgen"),
        rmiss: compiler.compile("assets/shaders/raytrace.rmiss", shaderc::ShaderKind::Miss).expect("Failed to compile raytrace.rmiss"),
        rchit: compiler.compile("assets/shaders/raytrace.rchit", shaderc::ShaderKind::ClosestHit).expect("Failed to compile raytrace.rchit"),
    };

    app.engine.renderer.setup_default_passes(shaders).expect("Failed to setup render passes");


    use spark_renderer::vertex::Vertex;
    use spark_math::{Vec2, Vec3, Mat4};

    let vertices = [
        Vertex::pack(Vec3::new(0.0, -0.5, 0.0), Vec3::Z, Vec2::ZERO, Vec3::new(1.0, 0.0, 0.0), Vec3::X),
        Vertex::pack(Vec3::new(0.5, 0.5, 0.0), Vec3::Z, Vec2::ZERO, Vec3::new(0.0, 1.0, 0.0), Vec3::X),
        Vertex::pack(Vec3::new(-0.5, 0.5, 0.0), Vec3::Z, Vec2::ZERO, Vec3::new(0.0, 0.0, 1.0), Vec3::X),
    ];

    let vb = app.engine.renderer.create_buffer(
        (std::mem::size_of::<Vertex>() * vertices.len()) as u64,
        ash::vk::BufferUsageFlags::VERTEX_BUFFER,
        ash::vk::MemoryPropertyFlags::HOST_VISIBLE | ash::vk::MemoryPropertyFlags::HOST_COHERENT,
    );
    app.engine.renderer.upload_to_buffer(&vb, &vertices);
    app.engine.renderer.add_vertex_buffer(vb);

    use spark_core::scene::{Node, MeshComponent, CameraComponent};

    let triangle_node = Node {
        name: "MyTriangle".to_string(),
        visible: true,
        locked: false,
        is_dirty: true,
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
            skin_index: None,
        })],
    };

    app.engine.scene.add_node(app.engine.scene.root, triangle_node);

    let _camera_node = Node {
        name: "MainCamera".to_string(),
        visible: true,
        locked: false,
        is_dirty: true,
        local_transform: Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0)),
        global_transform: Mat4::IDENTITY,
        parent: None,
        children: Vec::new(),
        components: vec![Box::new(CameraComponent { fov: 45.0, near: 0.1, far: 100.0, orthographic: false, ortho_size: 5.0 })],
    };

    app.engine.scene.add_node(app.engine.scene.root, _camera_node);

    let _script_host = ScriptHost::new();

    let mut ui = EditorUI::new(&app.engine.window, logs);
    app.engine.renderer.create_viewport_attachment(1280, 720);
    let viewport_view = app.engine.renderer.viewport_attachment.as_ref().unwrap().view;
    let viewport_sampler = app.engine.renderer.common_sampler;
    ui.viewport_texture_id = Some(app.engine.renderer.register_egui_texture(viewport_view, viewport_sampler));

    app.run_with_ui(move |window, event, scene, rm, renderer, project, resources, fps| {
        match event {
            winit::event::Event::WindowEvent { event, .. } => {
                (ui.handle_event(window, event), None)
            }
            winit::event::Event::AboutToWait => {
                ui.begin_frame(window);
                ui.draw_ui(scene, rm, renderer, project, fps);
                ui.draw_viewport(scene, renderer, fps);

                // Update simulation state
                let delta = 1.0 / fps.max(0.001);
                if ui.sim_state == crate::ui::SimulationState::Playing {
                    scene.update_components(delta, renderer, rm, project, &task_system, resources);
                }

                let full_output = ui.end_frame(window);
                (false, Some((full_output, ui.egui_ctx.clone())))
            }
            _ => (false, None),
        }
    });
}
