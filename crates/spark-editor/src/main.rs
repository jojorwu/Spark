use spark_core::Engine;
use spark_renderer::pipeline::Pipeline;
use std::fs;

mod ui;
use spark_script::ScriptHost;
use crate::ui::EditorUI;

fn main() {
    env_logger::init();
    log::info!("Spark Editor starting...");

    let compiler = shaderc::Compiler::new().unwrap();

    let ui_vert_code = fs::read_to_string("assets/shaders/ui.vert").unwrap();
    let ui_frag_code = fs::read_to_string("assets/shaders/ui.frag").unwrap();
    let ui_vert_spirv = compiler.compile_into_spirv(&ui_vert_code, shaderc::ShaderKind::Vertex, "ui.vert", "main", None).unwrap();
    let ui_frag_spirv = compiler.compile_into_spirv(&ui_frag_code, shaderc::ShaderKind::Fragment, "ui.frag", "main", None).unwrap();

    let mut engine = Engine::new(
        "Spark Engine Editor",
        Some((ui_vert_spirv.as_binary(), ui_frag_spirv.as_binary()))
    ).expect("Failed to initialize engine");

    let vert_code = fs::read_to_string("assets/shaders/gbuffer.vert").expect("Failed to read vertex shader");
    let frag_code = fs::read_to_string("assets/shaders/gbuffer.frag").expect("Failed to read fragment shader");

    let vert_spirv = compiler.compile_into_spirv(&vert_code, shaderc::ShaderKind::Vertex, "gbuffer.vert", "main", None).unwrap();
    let frag_spirv = compiler.compile_into_spirv(&frag_code, shaderc::ShaderKind::Fragment, "gbuffer.frag", "main", None).unwrap();

    let shadow_vert_code = fs::read_to_string("assets/shaders/shadow.vert").expect("Failed to read shadow vertex shader");
    let shadow_frag_code = fs::read_to_string("assets/shaders/shadow.frag").expect("Failed to read shadow fragment shader");
    let shadow_vert_spirv = compiler.compile_into_spirv(&shadow_vert_code, shaderc::ShaderKind::Vertex, "shadow.vert", "main", None).unwrap();
    let shadow_frag_spirv = compiler.compile_into_spirv(&shadow_frag_code, shaderc::ShaderKind::Fragment, "shadow.frag", "main", None).unwrap();

    let shadow_pipeline = {
        use spark_renderer::vertex::{Vertex, InstanceData};
        let device = engine.renderer.get_device();
        let render_pass = engine.renderer.shadow_render_pass;
        let layout = engine.renderer.shadow_pipeline_layout;

        let vert_module = {
            let info = ash::vk::ShaderModuleCreateInfo::default().code(shadow_vert_spirv.as_binary());
            unsafe { device.create_shader_module(&info, None).unwrap() }
        };
        let frag_module = {
            let info = ash::vk::ShaderModuleCreateInfo::default().code(shadow_frag_spirv.as_binary());
            unsafe { device.create_shader_module(&info, None).unwrap() }
        };
        let entry_point = std::ffi::CString::new("main").unwrap();
        let stages = [
            ash::vk::PipelineShaderStageCreateInfo::default()
                .stage(ash::vk::ShaderStageFlags::VERTEX)
                .module(vert_module)
                .name(&entry_point),
            ash::vk::PipelineShaderStageCreateInfo::default()
                .stage(ash::vk::ShaderStageFlags::FRAGMENT)
                .module(frag_module)
                .name(&entry_point),
        ];

        let binding_descriptions = [
            Vertex::get_binding_description(),
            InstanceData::get_binding_description(),
        ];
        let mut attribute_descriptions = Vec::new();
        attribute_descriptions.extend_from_slice(&Vertex::get_attribute_descriptions());
        attribute_descriptions.extend_from_slice(&InstanceData::get_attribute_descriptions());

        let vertex_input = ash::vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        let input_assembly = ash::vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(ash::vk::PrimitiveTopology::TRIANGLE_LIST);

        let viewport = ash::vk::Viewport::default()
            .width(2048.0)
            .height(2048.0)
            .max_depth(1.0);
        let scissor = ash::vk::Rect2D::default().extent(ash::vk::Extent2D { width: 2048, height: 2048 });
        let viewport_state = ash::vk::PipelineViewportStateCreateInfo::default()
            .viewports(std::slice::from_ref(&viewport))
            .scissors(std::slice::from_ref(&scissor));

        let rasterizer = ash::vk::PipelineRasterizationStateCreateInfo::default()
            .cull_mode(ash::vk::CullModeFlags::BACK)
            .front_face(ash::vk::FrontFace::CLOCKWISE)
            .line_width(1.0)
            .depth_bias_enable(true)
            .depth_bias_constant_factor(1.25)
            .depth_bias_slope_factor(1.75);

        let multisample = ash::vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(ash::vk::SampleCountFlags::TYPE_1);

        let depth_stencil = ash::vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(ash::vk::CompareOp::LESS_OR_EQUAL);

        let info = ash::vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0);

        let p = unsafe {
            device
                .create_graphics_pipelines(engine.renderer.pipeline_cache, &[info], None)
                .unwrap()[0]
        };
        unsafe {
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
        }
        p
    };

    engine.renderer.set_shadow_pipeline(shadow_pipeline);

    let post_vert_code = fs::read_to_string("assets/shaders/fullscreen.vert").unwrap();
    let post_frag_code = fs::read_to_string("assets/shaders/tonemap_bloom.frag").unwrap();
    let post_vert_spirv = compiler.compile_into_spirv(&post_vert_code, shaderc::ShaderKind::Vertex, "fullscreen.vert", "main", None).unwrap();
    let post_frag_spirv = compiler.compile_into_spirv(&post_frag_code, shaderc::ShaderKind::Fragment, "tonemap_bloom.frag", "main", None).unwrap();

    let bloom_frag_code = fs::read_to_string("assets/shaders/bloom_filter.frag").unwrap();
    let bloom_frag_spirv = compiler.compile_into_spirv(&bloom_frag_code, shaderc::ShaderKind::Fragment, "bloom_filter.frag", "main", None).unwrap();

    let (post_pipeline, post_layout, post_ds_layout, bloom_pipeline) = {
        let device = engine.renderer.get_device();
        let render_pass = engine.renderer.post_process_render_pass;

        let bindings = [
            ash::vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(ash::vk::ShaderStageFlags::FRAGMENT),
            ash::vk::DescriptorSetLayoutBinding::default()
                .binding(1)
                .descriptor_type(ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(ash::vk::ShaderStageFlags::FRAGMENT),
        ];
        let ds_layout_info = ash::vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let ds_layout = unsafe { device.create_descriptor_set_layout(&ds_layout_info, None).unwrap() };

        let layout_info = ash::vk::PipelineLayoutCreateInfo::default().set_layouts(std::slice::from_ref(&ds_layout));
        let layout = unsafe { device.create_pipeline_layout(&layout_info, None).unwrap() };

        let vert_module = {
            let info = ash::vk::ShaderModuleCreateInfo::default().code(post_vert_spirv.as_binary());
            unsafe { device.create_shader_module(&info, None).unwrap() }
        };
        let frag_module = {
            let info = ash::vk::ShaderModuleCreateInfo::default().code(post_frag_spirv.as_binary());
            unsafe { device.create_shader_module(&info, None).unwrap() }
        };
        let entry_point = std::ffi::CString::new("main").unwrap();
        let stages = [
            ash::vk::PipelineShaderStageCreateInfo::default().stage(ash::vk::ShaderStageFlags::VERTEX).module(vert_module).name(&entry_point),
            ash::vk::PipelineShaderStageCreateInfo::default().stage(ash::vk::ShaderStageFlags::FRAGMENT).module(frag_module).name(&entry_point),
        ];

        let vertex_input = ash::vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = ash::vk::PipelineInputAssemblyStateCreateInfo::default().topology(ash::vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport = ash::vk::Viewport::default().width(engine.renderer.get_extent().width as f32).height(engine.renderer.get_extent().height as f32).max_depth(1.0);
        let scissor = ash::vk::Rect2D::default().extent(engine.renderer.get_extent());
        let viewport_state = ash::vk::PipelineViewportStateCreateInfo::default().viewports(std::slice::from_ref(&viewport)).scissors(std::slice::from_ref(&scissor));
        let rasterizer = ash::vk::PipelineRasterizationStateCreateInfo::default().cull_mode(ash::vk::CullModeFlags::BACK).front_face(ash::vk::FrontFace::CLOCKWISE).line_width(1.0);
        let multisample = ash::vk::PipelineMultisampleStateCreateInfo::default().rasterization_samples(ash::vk::SampleCountFlags::TYPE_1);
        let color_blend_attachment = ash::vk::PipelineColorBlendAttachmentState::default().color_write_mask(ash::vk::ColorComponentFlags::RGBA).blend_enable(false);
        let color_blend = ash::vk::PipelineColorBlendStateCreateInfo::default().attachments(std::slice::from_ref(&color_blend_attachment));

        let info = ash::vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0);

        let p = unsafe { device.create_graphics_pipelines(engine.renderer.pipeline_cache, &[info], None).unwrap()[0] };

        let bloom_frag_module = {
            let info = ash::vk::ShaderModuleCreateInfo::default().code(bloom_frag_spirv.as_binary());
            unsafe { device.create_shader_module(&info, None).unwrap() }
        };
        let mut bloom_stages = stages.clone();
        bloom_stages[1] = ash::vk::PipelineShaderStageCreateInfo::default().stage(ash::vk::ShaderStageFlags::FRAGMENT).module(bloom_frag_module).name(&entry_point);

        let bloom_info = ash::vk::GraphicsPipelineCreateInfo::default()
            .stages(&bloom_stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterizer)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .layout(layout)
            .render_pass(render_pass)
            .subpass(0);

        let bp = unsafe { device.create_graphics_pipelines(engine.renderer.pipeline_cache, &[bloom_info], None).unwrap()[0] };

        unsafe {
            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);
            device.destroy_shader_module(bloom_frag_module, None);
        }
        (p, layout, ds_layout, bp)
    };

    engine.renderer.set_post_process_pipeline(post_pipeline, post_layout, post_ds_layout, bloom_pipeline);

    let pipeline = Pipeline::new(
        engine.renderer.get_device(),
        engine.renderer.render_pass,
        0, // Subpass 0: Geometry
        engine.renderer.get_extent(),
        vert_spirv.as_binary(),
        frag_spirv.as_binary(),
        engine.renderer.get_msaa_samples(),
        false, // Not deferred lighting
        0,
        engine.renderer.pipeline_cache,
    );

    engine.renderer.set_pipeline(pipeline);

    // Initialize Deferred Pipeline
    let def_vert_code = fs::read_to_string("assets/shaders/deferred.vert").unwrap();
    let def_frag_code = fs::read_to_string("assets/shaders/deferred.frag").unwrap();
    let def_vert_spirv = compiler.compile_into_spirv(&def_vert_code, shaderc::ShaderKind::Vertex, "deferred.vert", "main", None).unwrap();

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
    let def_frag_spirv = compiler.compile_into_spirv(&def_frag_code, shaderc::ShaderKind::Fragment, "deferred.frag", "main", Some(&def_options)).unwrap();

    let deferred_pipeline = Pipeline::new(
        engine.renderer.get_device(),
        engine.renderer.render_pass,
        1, // Subpass 1: Lighting
        engine.renderer.get_extent(),
        def_vert_spirv.as_binary(),
        def_frag_spirv.as_binary(),
        engine.renderer.get_msaa_samples(),
        true, // Deferred lighting
        4,    // 4 input attachments (Albedo, Normal, PBR, Depth)
        engine.renderer.pipeline_cache,
    );

    engine.renderer.set_deferred_pipeline(deferred_pipeline.graphics_pipeline, deferred_pipeline.layout, deferred_pipeline.descriptor_set_layout);

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
