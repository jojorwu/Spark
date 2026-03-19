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

    let culling_spirv = compiler.compile("assets/shaders/culling.comp", shaderc::ShaderKind::Compute);
    let hiz_spirv = compiler.compile("assets/shaders/hiz.comp", shaderc::ShaderKind::Compute);

    let ui_vert_spirv = compiler.compile("assets/shaders/ui.vert", shaderc::ShaderKind::Vertex);
    let ui_frag_spirv = compiler.compile("assets/shaders/ui.frag", shaderc::ShaderKind::Fragment);

    let mut engine = Engine::new(
        "Spark Engine Editor",
        Some((&ui_vert_spirv, &ui_frag_spirv))
    ).expect("Failed to initialize engine");

    engine.add_system(spark_core::systems::HierarchySystem);
    engine.add_system(spark_core::systems::ResourceSystem);

    let vert_spirv = compiler.compile("assets/shaders/gbuffer.vert", shaderc::ShaderKind::Vertex);
    let frag_spirv = compiler.compile("assets/shaders/gbuffer.frag", shaderc::ShaderKind::Fragment);

    let shadow_vert_spirv = compiler.compile("assets/shaders/shadow.vert", shaderc::ShaderKind::Vertex);
    let shadow_frag_spirv = compiler.compile("assets/shaders/shadow.frag", shaderc::ShaderKind::Fragment);

    let hiz_pass = spark_renderer::passes::hiz::HiZPass::new(
        &engine.renderer.device,
        engine.renderer.descriptor_pool,
        &hiz_spirv,
        engine.renderer.get_extent().width,
        engine.renderer.get_extent().height,
    );
    engine.renderer.set_hiz_view(hiz_pass.pyramid_view);

    let clustered_pass = spark_renderer::passes::clustered::ClusteredPass::new(
        &engine.renderer,
        &compiler.compile("assets/shaders/cluster_build.comp", shaderc::ShaderKind::Compute),
        &compiler.compile("assets/shaders/cluster_cull.comp", shaderc::ShaderKind::Compute),
    ).unwrap();

    let culling_pass = spark_renderer::passes::culling::CullingPass::new(
        &engine.renderer.device.device,
        engine.renderer.descriptor_pool,
        &culling_spirv,
        engine.renderer.global_descriptor_set_layout,
    );

    let mut shadow_pass = spark_renderer::passes::shadow::ShadowPass::new(
        &engine.renderer.device.device,
        engine.renderer.device.pdevice,
        &engine.renderer.context.instance
    ).unwrap();
    shadow_pass.create_pipeline(&engine.renderer.device.device, engine.renderer.pipeline_cache, &shadow_vert_spirv, &shadow_frag_spirv);
    engine.renderer.set_common_shadow_view(shadow_pass.view);

    let gbuffer_pass = spark_renderer::passes::gbuffer::GBufferPass::new();

    let mut ssao_pass = spark_renderer::passes::ssao::SSAOPass::new(
        &engine.renderer,
        engine.renderer.descriptor_pool,
    ).unwrap();
    ssao_pass.create_pipelines(
        &engine.renderer.device.device,
        engine.renderer.pipeline_cache,
        engine.renderer.get_extent(),
        &compiler.compile("assets/shaders/ssao.vert", shaderc::ShaderKind::Vertex),
        &compiler.compile("assets/shaders/ssao.frag", shaderc::ShaderKind::Fragment),
        &compiler.compile("assets/shaders/ssao_blur.frag", shaderc::ShaderKind::Fragment),
    );

    let mut lighting_pass = spark_renderer::passes::lighting::LightingPass::new(
        &engine.renderer.device.device,
        engine.renderer.descriptor_pool,
        engine.renderer.global_descriptor_set_layout,
    ).unwrap();

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
        &spark_renderer::pipeline::PipelineCreateParams {
            extent: engine.renderer.get_extent(),
            vert_shader_code: &compiler.compile("assets/shaders/deferred.vert", shaderc::ShaderKind::Vertex),
            frag_shader_code: &def_frag_spirv,
            msaa_samples: engine.renderer.get_msaa_samples(),
            is_deferred_lighting: true,
            input_attachments_count: 4,
            pipeline_cache: engine.renderer.pipeline_cache,
            global_ds_layout: engine.renderer.global_descriptor_set_layout,
            bindless_ds_layout: engine.renderer.bindless_descriptor_set_layout,
        }
    );
    lighting_pass.pipeline = Some(deferred_pipeline.graphics_pipeline);

    let grid_pass = spark_renderer::passes::grid::GridPass::new(
        &engine.renderer.device.device,
        engine.renderer.pipeline_cache,
        &compiler.compile("assets/shaders/grid.vert", shaderc::ShaderKind::Vertex),
        &compiler.compile("assets/shaders/grid.frag", shaderc::ShaderKind::Fragment),
        engine.renderer.global_descriptor_set_layout,
        ash::vk::Format::R16G16B16A16_SFLOAT,
    );

    let taa_pass = spark_renderer::passes::taa::TAAPass::new(
        &engine.renderer,
        &compiler.compile("assets/shaders/taa.frag", shaderc::ShaderKind::Fragment),
        &compiler.compile("assets/shaders/taa.vert", shaderc::ShaderKind::Vertex),
    ).unwrap();

    let volumetric_pass = spark_renderer::passes::volumetric::VolumetricPass::new(
        &engine.renderer,
        &compiler.compile("assets/shaders/volumetric.comp", shaderc::ShaderKind::Compute),
    );

    let mut post_process_pass = spark_renderer::passes::post_process::PostProcessPass::new(
        &engine.renderer.device.device,
        engine.renderer.device.pdevice,
        &engine.renderer.context.instance,
        ash::vk::Format::B8G8R8A8_UNORM, // TODO: Dynamic
        engine.renderer.get_extent(),
    ).unwrap();
    post_process_pass.create_pipelines(
        spark_renderer::passes::post_process::PostProcessPipelineParams {
            device: &engine.renderer.device.device,
            pipeline_cache: engine.renderer.pipeline_cache,
            extent: engine.renderer.get_extent(),
            vert_spirv: &compiler.compile("assets/shaders/fullscreen.vert", shaderc::ShaderKind::Vertex),
            frag_spirv: &compiler.compile("assets/shaders/tonemap_bloom.frag", shaderc::ShaderKind::Fragment),
            downsample_spirv: &compiler.compile("assets/shaders/bloom_downsample.frag", shaderc::ShaderKind::Fragment),
            upsample_spirv: &compiler.compile("assets/shaders/bloom_upsample.frag", shaderc::ShaderKind::Fragment),
        }
    );

    engine.renderer.add_render_pass(hiz_pass);
    engine.renderer.add_render_pass(clustered_pass);
    engine.renderer.add_render_pass(culling_pass);
    engine.renderer.add_render_pass(shadow_pass);
    engine.renderer.add_render_pass(gbuffer_pass);
    engine.renderer.add_render_pass(ssao_pass);
    engine.renderer.add_render_pass(lighting_pass);
    engine.renderer.add_render_pass(grid_pass);
    engine.renderer.add_render_pass(volumetric_pass);
    engine.renderer.add_render_pass(taa_pass);
    engine.renderer.add_render_pass(post_process_pass);

    engine.renderer.update_all_descriptor_sets();

    let pipeline = Pipeline::new(
        engine.renderer.get_device(),
        &spark_renderer::pipeline::PipelineCreateParams {
            extent: engine.renderer.get_extent(),
            vert_shader_code: &vert_spirv,
            frag_shader_code: &frag_spirv,
            msaa_samples: engine.renderer.get_msaa_samples(),
            is_deferred_lighting: false,
            input_attachments_count: 0,
            pipeline_cache: engine.renderer.pipeline_cache,
            global_ds_layout: engine.renderer.global_descriptor_set_layout,
            bindless_ds_layout: engine.renderer.bindless_descriptor_set_layout,
        }
    );

    engine.renderer.set_pipeline(pipeline);


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
                ui.draw_ui(scene, rm, renderer);
                ui.draw_viewport(scene, fps);
                let full_output = ui.end_frame(window);
                (false, Some((full_output, ui.egui_ctx.clone())))
            }
            _ => (false, None),
        }
    });
}
