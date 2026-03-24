use egui::{Context, Visuals, Ui};
use egui_winit::State;
use winit::window::Window;
use winit::event::WindowEvent;
use egui_gizmo::{Gizmo, GizmoMode};
use spark_math::Vec4Swizzles;

use spark_core::scene::{Scene, NodeKey, Node};

pub trait Command {
    fn execute(&mut self, scene: &mut Scene);
    fn undo(&mut self, scene: &mut Scene);
}

pub struct TransformCommand {
    pub node_key: NodeKey,
    pub old_transform: spark_math::Mat4,
    pub new_transform: spark_math::Mat4,
}

impl Command for TransformCommand {
    fn execute(&mut self, scene: &mut Scene) {
        if let Some(node) = scene.nodes.get_mut(self.node_key) {
            node.local_transform = self.new_transform;
        }
    }
    fn undo(&mut self, scene: &mut Scene) {
        if let Some(node) = scene.nodes.get_mut(self.node_key) {
            node.local_transform = self.old_transform;
        }
    }
}

pub struct EditorUI {
    pub egui_ctx: Context,
    pub egui_state: State,
    pub selected_nodes: std::collections::HashSet<NodeKey>,
    pub viewport_texture_id: Option<egui::TextureId>,
    pub undo_stack: Vec<Box<dyn Command>>,
    pub redo_stack: Vec<Box<dyn Command>>,
    pub gizmo_mode: GizmoMode,
    pub camera_pos: spark_math::Vec3,
    pub camera_rot: spark_math::Vec2, // Yaw, Pitch
    pub logs: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    pub active_bottom_tab: BottomTab,
    pub camera_speed: f32,
    pub node_to_delete: Option<NodeKey>,
    pub node_to_add_child: Option<(NodeKey, NodeType)>,
    pub initial_gizmo_transform: Option<spark_math::Mat4>,
    pub component_to_remove: Option<(NodeKey, usize)>,
}

pub enum NodeType {
    Mesh,
    Light,
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum BottomTab {
    Console,
    Assets,
    Scripts,
    Project,
    Settings,
    Statistics,
}

impl EditorUI {
    pub fn new(window: &Window, logs: std::sync::Arc<std::sync::Mutex<Vec<String>>>) -> Self {
        let egui_ctx = Context::default();
        egui_ctx.set_visuals(Visuals::dark());

        let egui_state = State::new(
            egui_ctx.clone(),
            egui_ctx.viewport_id(),
            &window,
            None,
            None,
        );

        Self {
            egui_ctx,
            egui_state,
            selected_nodes: std::collections::HashSet::new(),
            viewport_texture_id: None,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            gizmo_mode: GizmoMode::Translate,
            camera_pos: spark_math::Vec3::new(0.0, 2.0, 10.0),
            camera_rot: spark_math::Vec2::new(-90.0f32.to_radians(), 0.0),
            camera_speed: 0.1,
            logs,
            active_bottom_tab: BottomTab::Console,
            node_to_delete: None,
            node_to_add_child: None,
            initial_gizmo_transform: None,
            component_to_remove: None,
        }
    }

    pub fn handle_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        let response = self.egui_state.on_window_event(window, event);
        response.consumed
    }

    pub fn begin_frame(&mut self, window: &Window) {
        let raw_input = self.egui_state.take_egui_input(window);
        self.egui_ctx.begin_frame(raw_input);
    }

    pub fn end_frame(&mut self, window: &Window) -> egui::FullOutput {
        let full_output = self.egui_ctx.end_frame();
        self.egui_state.handle_platform_output(window, full_output.platform_output.clone());
        full_output
    }

    fn execute_command(&mut self, mut command: Box<dyn Command>, scene: &mut Scene) {
        command.execute(scene);
        self.undo_stack.push(command);
        self.redo_stack.clear();
    }

    fn undo(&mut self, scene: &mut Scene) {
        if let Some(mut command) = self.undo_stack.pop() {
            command.undo(scene);
            self.redo_stack.push(command);
        }
    }

    fn redo(&mut self, scene: &mut Scene) {
        if let Some(mut command) = self.redo_stack.pop() {
            command.execute(scene);
            self.undo_stack.push(command);
        }
    }

    pub fn draw_ui(&mut self, scene: &mut Scene, resource_manager: &mut spark_core::resource::ResourceManager, renderer: &mut spark_renderer::Renderer, fps: f32) {
        self.draw_menu_bar(scene, resource_manager, renderer);
        self.draw_bottom_panel(scene, resource_manager, renderer, fps);
        self.draw_hierarchy_panel(scene);
        self.draw_inspector_panel(scene, renderer);

        if let Some((parent, node_type)) = self.node_to_add_child.take() {
            match node_type {
                NodeType::Mesh => {
                    let mut node = Node {
                        name: "New Mesh".to_string(),
                        local_transform: spark_math::Mat4::IDENTITY,
                        global_transform: spark_math::Mat4::IDENTITY,
                        parent: None,
                        children: Vec::new(),
                        components: Vec::new(),
                    };
                    node.components.push(Box::new(spark_core::scene::MeshComponent {
                        vertex_count: 0, index_count: 0, first_index: 0, vertex_offset: 0,
                        texture_handle: None, material_index: None, bounding_radius: 1.0,
                    }));
                    scene.add_node(parent, node);
                }
                NodeType::Light => {
                    let mut node = Node {
                        name: "New Light".to_string(),
                        local_transform: spark_math::Mat4::IDENTITY,
                        global_transform: spark_math::Mat4::IDENTITY,
                        parent: None,
                        children: Vec::new(),
                        components: Vec::new(),
                    };
                    node.components.push(Box::new(spark_core::scene::LightComponent {
                        light_type: spark_core::scene::LightType::Point,
                        color: spark_math::Vec3::ONE,
                        intensity: 1.0,
                        range: 10.0,
                    }));
                    scene.add_node(parent, node);
                }
            }
        }

        if let Some(key) = self.node_to_delete.take() {
            self.delete_node(scene, key);
            self.selected_nodes.remove(&key);
        }

        if let Some((node_key, comp_idx)) = self.component_to_remove.take() {
            if let Some(node) = scene.nodes.get_mut(node_key) {
                if comp_idx < node.components.len() {
                    node.components.remove(comp_idx);
                }
            }
        }
    }

    fn draw_menu_bar(&mut self, scene: &mut Scene, resource_manager: &mut spark_core::resource::ResourceManager, renderer: &mut spark_renderer::Renderer) {
        let ctx = self.egui_ctx.clone();
        egui::TopBottomPanel::top("menu").show(&ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("New").clicked() {
                        *scene = Scene::new();
                        ui.close_menu();
                    }
                    if ui.button("Open").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Spark Scene", &["json"])
                            .pick_file() {
                            if let Ok(new_scene) = Scene::load_from_file(path.to_str().unwrap()) {
                                *scene = new_scene;
                            }
                        }
                        ui.close_menu();
                    }
                    if ui.button("Import glTF").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("glTF", &["gltf", "glb"])
                            .pick_file() {
                            resource_manager.load_scene(path, scene, renderer);
                        }
                        ui.close_menu();
                    }
                    if ui.button("Save").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Spark Scene", &["json"])
                            .save_file() {
                            let _ = scene.save_to_file(path.to_str().unwrap());
                        }
                        ui.close_menu();
                    }
                });
                ui.menu_button("Edit", |ui| {
                    if ui.button("Undo").clicked() {
                        self.undo(scene);
                        ui.close_menu();
                    }
                    if ui.button("Redo").clicked() {
                        self.redo(scene);
                        ui.close_menu();
                    }
                });
            });

            ui.separator();

            ui.horizontal(|ui| {
                ui.label("Tools:");
                ui.selectable_value(&mut self.gizmo_mode, egui_gizmo::GizmoMode::Translate, "⬈ Translate");
                ui.selectable_value(&mut self.gizmo_mode, egui_gizmo::GizmoMode::Rotate, "⟲ Rotate");
                ui.selectable_value(&mut self.gizmo_mode, egui_gizmo::GizmoMode::Scale, "⤢ Scale");

                ui.separator();

                if ui.button("⟲ Undo").clicked() {
                    self.undo(scene);
                }
                if ui.button("⟳ Redo").clicked() {
                    self.redo(scene);
                }
            });
        });
    }

    fn draw_bottom_panel(&mut self, scene: &mut Scene, resource_manager: &mut spark_core::resource::ResourceManager, renderer: &mut spark_renderer::Renderer, fps: f32) {
        let ctx = self.egui_ctx.clone();
        egui::TopBottomPanel::bottom("bottom_panel").show(&ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.active_bottom_tab, BottomTab::Console, "Console");
                ui.selectable_value(&mut self.active_bottom_tab, BottomTab::Assets, "Assets");
                ui.selectable_value(&mut self.active_bottom_tab, BottomTab::Settings, "Settings");
                ui.selectable_value(&mut self.active_bottom_tab, BottomTab::Statistics, "Statistics");
            });
            ui.separator();

            match self.active_bottom_tab {
                BottomTab::Console => {
                    egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                        let logs = self.logs.lock().unwrap();
                        for log in logs.iter() {
                            ui.label(log);
                        }
                    });
                }
                BottomTab::Assets => {
                    let mut asset_to_load = None;
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        for entry in walkdir::WalkDir::new("assets")
                            .into_iter()
                            .filter_map(|e| e.ok()) {
                            let path = entry.path();
                            if path.is_file() {
                                let label = path.file_name().unwrap().to_string_lossy();
                                let is_gltf = path.extension().is_some_and(|ext| ext == "gltf" || ext == "glb");

                                ui.horizontal(|ui| {
                                    let icon = if is_gltf { "📦" } else { "📄" };
                                    if ui.selectable_label(false, format!("{} {}", icon, label)).clicked() {
                                        log::info!("Selected asset: {:?}", path);
                                        if is_gltf {
                                            asset_to_load = Some(path.to_path_buf());
                                        }
                                    }
                                });
                            } else if path.is_dir() && path != std::path::Path::new("assets") {
                                let label = path.file_name().unwrap().to_string_lossy();
                                ui.label(format!("📁 {}", label));
                            }
                        }
                    });
                    if let Some(path) = asset_to_load {
                        resource_manager.load_scene(path, scene, renderer);
                    }
                }
                BottomTab::Scripts => {
                    ui.heading("Plugin Management");
                    ui.label("Add Rust plugins to enhance engine functionality.");
                    if ui.button("Load Plugin...").clicked() {
                         if let Some(path) = rfd::FileDialog::new()
                            .add_filter("Rust Plugin", &["so", "dll", "dylib"])
                            .pick_file() {
                                log::info!("Loading plugin: {:?}", path);
                                // Logic to load and register system via ScriptHost would go here
                            }
                    }
                    ui.separator();
                    ui.label("Active Systems:");
                    // List systems from scheduler
                }
                BottomTab::Project => {
                    ui.heading("Project Settings");
                    ui.label("Manage project paths and metadata.");
                    // Project settings UI
                }
                BottomTab::Settings => {
                    ui.heading("Renderer Settings");
                    ui.horizontal(|ui| {
                        ui.label("Exposure:");
                        ui.add(egui::Slider::new(&mut renderer.exposure, 0.1..=10.0));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Gamma:");
                        ui.add(egui::Slider::new(&mut renderer.gamma, 1.0..=3.0));
                    });
                    ui.separator();
                    ui.heading("Visual Features");
                    ui.checkbox(&mut renderer.enable_shadows, "Shadows");
                    ui.checkbox(&mut renderer.enable_ssao, "SSAO");
                    ui.checkbox(&mut renderer.enable_taa, "TAA");
                    ui.checkbox(&mut renderer.enable_volumetric, "Volumetric Fog");
                    ui.checkbox(&mut renderer.enable_grid, "Ground Grid");
                    ui.checkbox(&mut renderer.enable_ibl, "IBL");
                    ui.checkbox(&mut renderer.enable_bloom, "Bloom");
                }
                BottomTab::Statistics => {
                    ui.horizontal(|ui| {
                        ui.label("Camera Speed:");
                        ui.add(egui::Slider::new(&mut self.camera_speed, 0.01..=1.0));
                    });
                    ui.separator();
                    ui.label(format!("FPS: {:.1}", fps));
                    ui.label(format!("Active Objects: {}", renderer.last_object_count));
                    ui.label("Draw Calls: TODO");
                    ui.label("GPU Memory: TODO");
                }
            }
        });
    }

    fn draw_hierarchy_panel(&mut self, scene: &mut Scene) {
        let ctx = self.egui_ctx.clone();
        egui::SidePanel::left("hierarchy").show(&ctx, |ui| {
            ui.heading("Scene Hierarchy");

            ui.horizontal(|ui| {
                if ui.button("Add Mesh").clicked() {
                    self.add_default_mesh(scene);
                }
                if ui.button("Add Light").clicked() {
                    self.add_default_light(scene);
                }
            });

            ui.separator();

            Self::draw_node_tree(ui, scene, scene.root, &mut self.selected_nodes, &mut self.node_to_delete, &mut self.node_to_add_child);

            if !self.selected_nodes.is_empty() {
                if ui.button("Delete Selected").clicked() {
                    let keys: Vec<_> = self.selected_nodes.drain().collect();
                    for key in keys {
                        self.delete_node(scene, key);
                    }
                }
            }
        });
    }

    fn add_default_mesh(&mut self, scene: &mut Scene) {
        let mut new_node = Node {
            name: "New Mesh".to_string(),
            local_transform: spark_math::Mat4::IDENTITY,
            global_transform: spark_math::Mat4::IDENTITY,
            parent: None,
            children: Vec::new(),
            components: Vec::new(),
        };
        new_node.components.push(Box::new(spark_core::scene::MeshComponent {
            vertex_count: 0,
            index_count: 0,
            first_index: 0,
            vertex_offset: 0,
            texture_handle: None,
            material_index: None,
            bounding_radius: 1.0,
        }));
        scene.add_node(scene.root, new_node);
    }

    fn add_default_light(&mut self, scene: &mut Scene) {
        let mut new_node = Node {
            name: "New Light".to_string(),
            local_transform: spark_math::Mat4::IDENTITY,
            global_transform: spark_math::Mat4::IDENTITY,
            parent: None,
            children: Vec::new(),
            components: Vec::new(),
        };
        new_node.components.push(Box::new(spark_core::scene::LightComponent {
            light_type: spark_core::scene::LightType::Point,
            color: spark_math::Vec3::ONE,
            intensity: 1.0,
            range: 10.0,
        }));
        scene.add_node(scene.root, new_node);
    }

    fn delete_node(&mut self, scene: &mut Scene, key: NodeKey) {
        scene.remove_node(key);
    }

    fn draw_inspector_panel(&mut self, scene: &mut Scene, _renderer: &mut spark_renderer::Renderer) {
        let ctx = self.egui_ctx.clone();
        egui::SidePanel::right("inspector").show(&ctx, |ui| {
            ui.heading("Inspector");
            if self.selected_nodes.len() > 1 {
                ui.label(format!("{} nodes selected", self.selected_nodes.len()));
            } else if let Some(&selected_key) = self.selected_nodes.iter().next() {
                let mut changed_transform = None;
                if let Some(node) = scene.nodes.get_mut(selected_key) {
                    ui.horizontal(|ui| {
                        ui.label("Name:");
                        ui.text_edit_singleline(&mut node.name);
                    });
                    ui.separator();

                    changed_transform = self.draw_transform_editor(ui, node);

                    ui.separator();
                    ui.horizontal(|ui| {
                        ui.label("Components");
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            ui.menu_button("✚ Add", |ui| {
                                if ui.button("Mesh").clicked() {
                                    node.components.push(Box::new(spark_core::scene::MeshComponent {
                                        vertex_count: 0, index_count: 0, first_index: 0, vertex_offset: 0,
                                        texture_handle: None, material_index: None, bounding_radius: 1.0,
                                    }));
                                    ui.close_menu();
                                }
                                if ui.button("Light").clicked() {
                                    node.components.push(Box::new(spark_core::scene::LightComponent {
                                        light_type: spark_core::scene::LightType::Point,
                                        color: spark_math::Vec3::ONE,
                                        intensity: 1.0,
                                        range: 10.0,
                                    }));
                                    ui.close_menu();
                                }
                                if ui.button("Camera").clicked() {
                                    node.components.push(Box::new(spark_core::scene::CameraComponent {
                                        fov: 45.0, near: 0.1, far: 100.0,
                                    }));
                                    ui.close_menu();
                                }
                            });
                        });
                    });

                    let mut to_remove = None;
                    for (idx, component) in node.components.iter_mut().enumerate() {
                        let header = match component.as_any().type_id() {
                            t if t == std::any::TypeId::of::<spark_core::scene::MeshComponent>() => "Mesh",
                            t if t == std::any::TypeId::of::<spark_core::scene::LightComponent>() => "Light",
                            t if t == std::any::TypeId::of::<spark_core::scene::CameraComponent>() => "Camera",
                            _ => "Unknown",
                        };

                        let response = ui.collapsing(header, |ui| {
                            self.draw_component_editor(ui, component);
                        });

                        response.header_response.context_menu(|ui| {
                            if ui.button("Remove").clicked() {
                                to_remove = Some(idx);
                                ui.close_menu();
                            }
                        });
                    }

                    if let Some(idx) = to_remove {
                        self.component_to_remove = Some((selected_key, idx));
                    }

                    ui.separator();
                    ui.label("Gizmo Mode");
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.gizmo_mode, egui_gizmo::GizmoMode::Translate, "T");
                        ui.selectable_value(&mut self.gizmo_mode, egui_gizmo::GizmoMode::Rotate, "R");
                        ui.selectable_value(&mut self.gizmo_mode, egui_gizmo::GizmoMode::Scale, "S");
                    });
                }

                if let Some((old, new)) = changed_transform {
                    self.execute_command(Box::new(TransformCommand {
                        node_key: selected_key,
                        old_transform: old,
                        new_transform: new,
                    }), scene);
                }
            } else {
                ui.label("Select a node to inspect");
            }
        });
    }

    fn draw_transform_editor(&mut self, ui: &mut Ui, node: &mut Node) -> Option<(spark_math::Mat4, spark_math::Mat4)> {
        ui.label("Transform");
        let (mut scale, mut rotation, mut translation) = node.local_transform.to_scale_rotation_translation();

        let mut changed = false;
        let initial_transform = node.local_transform;
        ui.horizontal(|ui| {
            ui.label("Pos:");
            changed |= ui.add(egui::DragValue::new(&mut translation.x).speed(0.1)).changed();
            changed |= ui.add(egui::DragValue::new(&mut translation.y).speed(0.1)).changed();
            changed |= ui.add(egui::DragValue::new(&mut translation.z).speed(0.1)).changed();
        });

        let mut euler = rotation.to_euler(spark_math::EulerRot::XYZ);
        ui.horizontal(|ui| {
            ui.label("Rot:");
            let mut deg_x = euler.0.to_degrees();
            let mut deg_y = euler.1.to_degrees();
            let mut deg_z = euler.2.to_degrees();
            if ui.add(egui::DragValue::new(&mut deg_x).speed(1.0).suffix("°")).changed() {
                euler.0 = deg_x.to_radians();
                changed = true;
            }
            if ui.add(egui::DragValue::new(&mut deg_y).speed(1.0).suffix("°")).changed() {
                euler.1 = deg_y.to_radians();
                changed = true;
            }
            if ui.add(egui::DragValue::new(&mut deg_z).speed(1.0).suffix("°")).changed() {
                euler.2 = deg_z.to_radians();
                changed = true;
            }
        });
        if changed {
            rotation = spark_math::Quat::from_euler(spark_math::EulerRot::XYZ, euler.0, euler.1, euler.2);
        }

        ui.horizontal(|ui| {
            ui.label("Scale:");
            changed |= ui.add(egui::DragValue::new(&mut scale.x).speed(0.1)).changed();
            changed |= ui.add(egui::DragValue::new(&mut scale.y).speed(0.1)).changed();
            changed |= ui.add(egui::DragValue::new(&mut scale.z).speed(0.1)).changed();
        });

        if changed {
            node.local_transform = spark_math::Mat4::from_scale_rotation_translation(scale, rotation, translation);
        }

        if ui.input(|i| i.pointer.any_released()) && changed {
             return Some((initial_transform, node.local_transform));
        }
        None
    }

    fn draw_component_editor(&mut self, ui: &mut Ui, component: &mut Box<dyn spark_core::scene::Component>) {
        let any = component.as_any_mut();
        if let Some(light) = any.downcast_mut::<spark_core::scene::LightComponent>() {
            ui.collapsing("Light Component", |ui| {
                ui.horizontal(|ui| {
                    ui.label("Color:");
                    ui.color_edit_button_rgb(light.color.as_mut());
                });
                ui.horizontal(|ui| {
                    ui.label("Intensity:");
                    ui.add(egui::DragValue::new(&mut light.intensity).speed(0.1));
                });
                ui.horizontal(|ui| {
                    ui.label("Range:");
                    ui.add(egui::DragValue::new(&mut light.range).speed(0.1));
                });
            });
        } else if let Some(mesh) = any.downcast_mut::<spark_core::scene::MeshComponent>() {
            ui.collapsing("Mesh Component", |ui| {
                ui.horizontal(|ui| {
                    ui.label("Radius:");
                    ui.add(egui::DragValue::new(&mut mesh.bounding_radius).speed(0.1));
                });
            });
        } else if let Some(camera) = any.downcast_mut::<spark_core::scene::CameraComponent>() {
            ui.collapsing("Camera Component", |ui| {
                ui.horizontal(|ui| {
                    ui.label("FOV:");
                    ui.add(egui::DragValue::new(&mut camera.fov).speed(1.0).clamp_range(1.0..=179.0));
                });
                ui.horizontal(|ui| {
                    ui.label("Near:");
                    ui.add(egui::DragValue::new(&mut camera.near).speed(0.01));
                });
                ui.horizontal(|ui| {
                    ui.label("Far:");
                    ui.add(egui::DragValue::new(&mut camera.far).speed(1.0));
                });
            });
        }
    }

    fn draw_node_tree(ui: &mut egui::Ui, scene: &mut Scene, node_key: NodeKey, selected_nodes: &mut std::collections::HashSet<NodeKey>, node_to_delete: &mut Option<NodeKey>, node_to_add_child: &mut Option<(NodeKey, NodeType)>) {
        let (label, children) = if let Some(node) = scene.nodes.get(node_key) {
            (node.name.clone(), node.children.clone())
        } else {
            return;
        };

        let is_selected = selected_nodes.contains(&node_key);

        let response = ui.selectable_label(is_selected, &label);

        let mut delete_requested = false;
        response.context_menu(|ui| {
                if ui.button("Delete").clicked() {
                    delete_requested = true;
                    ui.close_menu();
                }
                ui.separator();
                if ui.button("Add Child Mesh").clicked() {
                    *node_to_add_child = Some((node_key, NodeType::Mesh));
                    ui.close_menu();
                }
                if ui.button("Add Child Light").clicked() {
                    *node_to_add_child = Some((node_key, NodeType::Light));
                    ui.close_menu();
                }
            });

        if delete_requested {
            *node_to_delete = Some(node_key);
        }

        if response.clicked() {
            if ui.input(|i| i.modifiers.shift || i.modifiers.command || i.modifiers.ctrl) {
                if is_selected {
                    selected_nodes.remove(&node_key);
                } else {
                    selected_nodes.insert(node_key);
                }
            } else {
                selected_nodes.clear();
                selected_nodes.insert(node_key);
            }
        }

        response.dnd_set_drag_payload(node_key);
        if let Some(payload) = response.dnd_hover_payload::<NodeKey>() {
            let dragged_key = *payload;
            if ui.input(|i| i.pointer.any_released()) && dragged_key != node_key {
                // Reparenting logic
                let mut can_reparent = true;
                // Check for cycles
                let mut current = Some(node_key);
                while let Some(k) = current {
                    if k == dragged_key { can_reparent = false; break; }
                    current = scene.nodes.get(k).and_then(|n| n.parent);
                }

                if can_reparent {
                    // Remove from old parent
                    let old_parent = scene.nodes.get(dragged_key).and_then(|n| n.parent);
                    if let Some(opk) = old_parent {
                        if let Some(op) = scene.nodes.get_mut(opk) {
                            op.children.retain(|&k| k != dragged_key);
                        }
                    }

                    // Set new parent
                    if let Some(n) = scene.nodes.get_mut(dragged_key) {
                        n.parent = Some(node_key);
                    }
                    if let Some(p) = scene.nodes.get_mut(node_key) {
                        p.children.push(dragged_key);
                    }
                    scene.update_all_transforms();
                }
            }
        }

        for &child_key in &children {
            ui.indent(&label, |ui| {
                Self::draw_node_tree(ui, scene, child_key, selected_nodes, node_to_delete, node_to_add_child);
            });
        }
    }

    pub fn draw_viewport(&mut self, scene: &mut Scene, fps: f32) {
        if let Some(texture_id) = self.viewport_texture_id {
            let ctx = self.egui_ctx.clone();
            egui::Window::new("Viewport").show(&ctx, |ui| {
                ui.label(format!("FPS: {:.1}", fps));
                // Keyboard shortcuts
                if ui.input(|i| i.key_pressed(egui::Key::T)) { self.gizmo_mode = egui_gizmo::GizmoMode::Translate; }
                if ui.input(|i| i.key_pressed(egui::Key::R)) { self.gizmo_mode = egui_gizmo::GizmoMode::Rotate; }
                if ui.input(|i| i.key_pressed(egui::Key::S)) { self.gizmo_mode = egui_gizmo::GizmoMode::Scale; }

                let size = ui.available_size();
                let response = ui.image(egui::load::SizedTexture::new(texture_id, size));
                let rect = response.rect;

                // Camera Controls
                if response.hovered() {
                    let speed = self.camera_speed;
                    let rot_speed = 0.005;

                    if ui.input(|i| i.pointer.button_down(egui::PointerButton::Secondary)) {
                        let delta = ui.input(|i| i.pointer.delta());
                        self.camera_rot.x += delta.x * rot_speed;
                        self.camera_rot.y -= delta.y * rot_speed;
                        self.camera_rot.y = self.camera_rot.y.clamp(-1.5, 1.5);
                    }

                    let forward = spark_math::Vec3::new(
                        self.camera_rot.x.cos() * self.camera_rot.y.cos(),
                        self.camera_rot.y.sin(),
                        self.camera_rot.x.sin() * self.camera_rot.y.cos(),
                    ).normalize();
                    let right = forward.cross(spark_math::Vec3::Y).normalize();

                    if ui.input(|i| i.key_down(egui::Key::W)) { self.camera_pos += forward * speed; }
                    if ui.input(|i| i.key_down(egui::Key::S)) { self.camera_pos -= forward * speed; }
                    if ui.input(|i| i.key_down(egui::Key::A)) { self.camera_pos -= right * speed; }
                    if ui.input(|i| i.key_down(egui::Key::D)) { self.camera_pos += right * speed; }

                    scene.last_view_matrix = spark_math::Mat4::look_at_rh(
                        self.camera_pos,
                        self.camera_pos + forward,
                        spark_math::Vec3::Y
                    );
                }

                // Picking
                if response.clicked_by(egui::PointerButton::Primary) {
                    if let Some(pointer_pos) = ui.input(|i| i.pointer.interact_pos()) {
                        if rect.contains(pointer_pos) {
                            let normalized_x = (pointer_pos.x - rect.min.x) / rect.width();
                            let normalized_y = (pointer_pos.y - rect.min.y) / rect.height();

                            let view = scene.last_view_matrix;
                            let projection = spark_math::Mat4::perspective_rh(
                                45.0f32.to_radians(),
                                size.x / size.y,
                                0.1,
                                100.0,
                            );
                            let inv_vp = (projection * view).inverse();

                            let ndc_x = normalized_x * 2.0 - 1.0;
                            let ndc_y = (1.0 - normalized_y) * 2.0 - 1.0;

                            let near_ndc = spark_math::Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
                            let far_ndc = spark_math::Vec4::new(ndc_x, ndc_y, 1.0, 1.0);

                            let near_world = inv_vp * near_ndc;
                            let far_world = inv_vp * far_ndc;

                            let origin = near_world.xyz() / near_world.w;
                            let target = far_world.xyz() / far_world.w;
                            let direction = (target - origin).normalize();

                            let ray = spark_math::Ray::new(origin, direction);
                            if let Some((key, _)) = scene.pick_node(&ray) {
                                if ui.input(|i| i.modifiers.shift || i.modifiers.command || i.modifiers.ctrl) {
                                    if self.selected_nodes.contains(&key) {
                                        self.selected_nodes.remove(&key);
                                    } else {
                                        self.selected_nodes.insert(key);
                                    }
                                } else {
                                    self.selected_nodes.clear();
                                    self.selected_nodes.insert(key);
                                }
                            } else if !ui.input(|i| i.modifiers.shift || i.modifiers.command || i.modifiers.ctrl) {
                                self.selected_nodes.clear();
                            }
                        }
                    }
                }

                if !self.selected_nodes.is_empty() {
                    // Calculate selection centroid
                    let mut centroid = spark_math::Vec3::ZERO;
                    let mut count = 0;
                    for &key in &self.selected_nodes {
                        if let Some(node) = scene.nodes.get(key) {
                            centroid += node.global_transform.w_axis.xyz();
                            count += 1;
                        }
                    }
                    if count > 0 { centroid /= count as f32; }

                    let (view, projection, model, parent_key) = {
                        let selected_key = *self.selected_nodes.iter().next().unwrap();
                        let node = scene.nodes.get(selected_key).unwrap();
                        let view = scene.last_view_matrix;
                        let projection = spark_math::Mat4::perspective_rh(
                            45.0f32.to_radians(),
                            size.x / size.y,
                            0.1,
                            100.0,
                        );
                        (view, projection, node.global_transform, node.parent)
                    };

                    let gizmo_model = spark_math::Mat4::from_translation(centroid);

                    let gizmo = Gizmo::new("scene_gizmo")
                        .view_matrix(view.to_cols_array_2d().into())
                        .projection_matrix(projection.to_cols_array_2d().into())
                        .model_matrix(gizmo_model.to_cols_array_2d().into())
                        .mode(self.gizmo_mode)
                        .viewport(rect);

                    if let Some(response) = gizmo.interact(ui) {
                        let m = response.transform();
                        let new_gizmo_model = spark_math::Mat4::from_cols_array_2d(&[
                            m.x.into(), m.y.into(), m.z.into(), m.w.into()
                        ]);

                        let delta = new_gizmo_model * gizmo_model.inverse();

                        for &key in &self.selected_nodes {
                            let parent_global_inv = if let Some(node) = scene.nodes.get(key) {
                                if let Some(pk) = node.parent {
                                    scene.nodes.get(pk).map(|p| p.global_transform.inverse()).unwrap_or(spark_math::Mat4::IDENTITY)
                                } else {
                                    spark_math::Mat4::IDENTITY
                                }
                            } else {
                                spark_math::Mat4::IDENTITY
                            };

                            if let Some(node) = scene.nodes.get_mut(key) {
                                let new_global = delta * node.global_transform;
                                node.local_transform = parent_global_inv * new_global;
                            }
                        }
                        scene.update_all_transforms();
                    }
                }
            });
        }
    }
}
