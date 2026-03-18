use egui::{Context, Visuals};
use egui_winit::State;
use winit::window::Window;
use winit::event::WindowEvent;
use egui_gizmo::{Gizmo, GizmoMode};

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
    pub selected_node: Option<NodeKey>,
    pub viewport_texture_id: Option<egui::TextureId>,
    pub undo_stack: Vec<Box<dyn Command>>,
    pub redo_stack: Vec<Box<dyn Command>>,
    pub gizmo_mode: GizmoMode,
}

impl EditorUI {
    pub fn new(window: &Window) -> Self {
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
            selected_node: None,
            viewport_texture_id: None,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            gizmo_mode: GizmoMode::Translate,
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

    pub fn draw_ui(&mut self, scene: &mut Scene) {
        egui::TopBottomPanel::top("menu").show(&self.egui_ctx, |ui| {
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
                        if let Some(mut cmd) = self.undo_stack.pop() {
                            cmd.undo(scene);
                            self.redo_stack.push(cmd);
                        }
                        ui.close_menu();
                    }
                    if ui.button("Redo").clicked() {
                        if let Some(mut cmd) = self.redo_stack.pop() {
                            cmd.execute(scene);
                            self.undo_stack.push(cmd);
                        }
                        ui.close_menu();
                    }
                });
            });
        });

        egui::TopBottomPanel::bottom("assets").show(&self.egui_ctx, |ui| {
            ui.heading("Asset Browser");
            ui.label("assets/shaders/triangle.vert");
            ui.label("assets/shaders/triangle.frag");
        });

        egui::SidePanel::left("hierarchy").show(&self.egui_ctx, |ui| {
            ui.heading("Scene Hierarchy");

            ui.horizontal(|ui| {
                if ui.button("Add Mesh").clicked() {
                    let new_node = Node {
                        name: "New Mesh".to_string(),
                        local_transform: spark_math::Mat4::IDENTITY,
                        global_transform: spark_math::Mat4::IDENTITY,
                        parent: None,
                        children: Vec::new(),
                        data: spark_core::scene::NodeData::Mesh {
                            vertex_count: 0,
                            index_count: 0,
                            first_index: 0,
                            vertex_offset: 0,
                            texture_id: None,
                            vertex_buffer_id: None,
                            bounding_radius: 1.0,
                        },
                    };
                    scene.add_node(scene.root, new_node);
                }
                if ui.button("Add Light").clicked() {
                    let new_node = Node {
                        name: "New Light".to_string(),
                        local_transform: spark_math::Mat4::IDENTITY,
                        global_transform: spark_math::Mat4::IDENTITY,
                        parent: None,
                        children: Vec::new(),
                        data: spark_core::scene::NodeData::Light {
                            light_type: spark_core::scene::LightType::Point,
                            color: spark_math::Vec3::ONE,
                            intensity: 1.0,
                            range: 10.0,
                        },
                    };
                    scene.add_node(scene.root, new_node);
                }
            });

            ui.separator();

            Self::draw_node_tree(ui, scene, scene.root, &mut self.selected_node);

            if let Some(selected_key) = self.selected_node {
                if ui.button("Delete Selected").clicked() {
                    // Simplified deletion: just remove from parent and slotmap
                    if let Some(node) = scene.nodes.get(selected_key) {
                        if let Some(parent_key) = node.parent {
                            if let Some(parent) = scene.nodes.get_mut(parent_key) {
                                parent.children.retain(|&k| k != selected_key);
                            }
                        }
                    }
                    scene.nodes.remove(selected_key);
                    self.selected_node = None;
                }
            }
        });

        egui::SidePanel::right("inspector").show(&self.egui_ctx, |ui| {
            ui.heading("Inspector");
            if let Some(selected_key) = self.selected_node {
                if let Some(node) = scene.nodes.get_mut(selected_key) {
                    ui.horizontal(|ui| {
                        ui.label("Name:");
                        ui.text_edit_singleline(&mut node.name);
                    });
                    ui.separator();

                    ui.label("Transform");
                    let (mut scale, mut rotation, mut translation) = node.local_transform.to_scale_rotation_translation();

                    let mut changed = false;
                    ui.horizontal(|ui| {
                        ui.label("Pos:");
                        changed |= ui.drag_angle(&mut translation.x).changed();
                        changed |= ui.drag_angle(&mut translation.y).changed();
                        changed |= ui.drag_angle(&mut translation.z).changed();
                    });
                    // Actually use DragValue for non-angles
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

                    ui.separator();
                    ui.label("Node Data");
                    match &mut node.data {
                        spark_core::scene::NodeData::Light { light_type, color, intensity, range } => {
                            ui.label("Type: Light");
                            ui.horizontal(|ui| {
                                ui.label("Color:");
                                ui.color_edit_button_rgb(color.as_mut());
                            });
                            ui.horizontal(|ui| {
                                ui.label("Intensity:");
                                ui.add(egui::DragValue::new(intensity).speed(0.1));
                            });
                            ui.horizontal(|ui| {
                                ui.label("Range:");
                                ui.add(egui::DragValue::new(range).speed(0.1));
                            });
                        }
                        spark_core::scene::NodeData::Mesh { bounding_radius, .. } => {
                            ui.label("Type: Mesh");
                            ui.horizontal(|ui| {
                                ui.label("Radius:");
                                ui.add(egui::DragValue::new(bounding_radius).speed(0.1));
                            });
                        }
                        spark_core::scene::NodeData::Camera { fov, near, far } => {
                            ui.label("Type: Camera");
                            ui.horizontal(|ui| {
                                ui.label("FOV:");
                            ui.add(egui::DragValue::new(fov).speed(1.0).clamp_range(1.0..=179.0));
                            });
                            ui.horizontal(|ui| {
                                ui.label("Near:");
                                ui.add(egui::DragValue::new(near).speed(0.01));
                            });
                            ui.horizontal(|ui| {
                                ui.label("Far:");
                                ui.add(egui::DragValue::new(far).speed(1.0));
                            });
                        }
                        _ => { ui.label("No data components"); }
                    }

                    ui.separator();
                    ui.label("Gizmo Mode");
                    ui.horizontal(|ui| {
                        ui.selectable_value(&mut self.gizmo_mode, egui_gizmo::GizmoMode::Translate, "T");
                        ui.selectable_value(&mut self.gizmo_mode, egui_gizmo::GizmoMode::Rotate, "R");
                        ui.selectable_value(&mut self.gizmo_mode, egui_gizmo::GizmoMode::Scale, "S");
                    });
                }
            } else {
                ui.label("Select a node to inspect");
            }
        });
    }

    fn draw_node_tree(ui: &mut egui::Ui, scene: &Scene, node_key: NodeKey, selected_node: &mut Option<NodeKey>) {
        if let Some(node) = scene.nodes.get(node_key) {
            let label = &node.name;
            let is_selected = Some(node_key) == *selected_node;

            let response = ui.selectable_label(is_selected, label);
            if response.clicked() {
                *selected_node = Some(node_key);
            }

            for &child_key in &node.children {
                ui.indent(label, |ui| {
                    Self::draw_node_tree(ui, scene, child_key, selected_node);
                });
            }
        }
    }

    pub fn draw_viewport(&mut self, scene: &mut Scene) {
        if let Some(texture_id) = self.viewport_texture_id {
            egui::Window::new("Viewport").show(&self.egui_ctx, |ui| {
                let size = ui.available_size();
                let rect = ui.image(egui::load::SizedTexture::new(texture_id, size)).rect;

                if let Some(selected_key) = self.selected_node {
                    let (view, projection, model, parent_key) = {
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

                    let gizmo = Gizmo::new("scene_gizmo")
                        .view_matrix(view.to_cols_array_2d().into())
                        .projection_matrix(projection.to_cols_array_2d().into())
                        .model_matrix(model.to_cols_array_2d().into())
                        .mode(self.gizmo_mode)
                        .viewport(rect);

                    if let Some(response) = gizmo.interact(ui) {
                        let m = response.transform();
                        let new_model = spark_math::Mat4::from_cols_array_2d(&[
                            m.x.into(), m.y.into(), m.z.into(), m.w.into()
                        ]);

                        let parent_global_inv = if let Some(pk) = parent_key {
                            scene.nodes.get(pk).map(|p| p.global_transform.inverse()).unwrap_or(spark_math::Mat4::IDENTITY)
                        } else {
                            spark_math::Mat4::IDENTITY
                        };

                        let new_local = parent_global_inv * new_model;

                        if let Some(node) = scene.nodes.get_mut(selected_key) {
                            node.local_transform = new_local;
                        }
                    }
                }
            });
        }
    }
}
