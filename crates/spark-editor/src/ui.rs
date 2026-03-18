use egui::{Context, Visuals};
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
    pub selected_node: Option<NodeKey>,
    pub viewport_texture_id: Option<egui::TextureId>,
    pub undo_stack: Vec<Box<dyn Command>>,
    pub redo_stack: Vec<Box<dyn Command>>,
    pub gizmo_mode: GizmoMode,
    pub camera_pos: spark_math::Vec3,
    pub camera_rot: spark_math::Vec2, // Yaw, Pitch
    pub logs: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
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
            selected_node: None,
            viewport_texture_id: None,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            gizmo_mode: GizmoMode::Translate,
            camera_pos: spark_math::Vec3::new(0.0, 2.0, 10.0),
            camera_rot: spark_math::Vec2::new(-90.0f32.to_radians(), 0.0),
            logs,
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

    pub fn draw_ui(&mut self, scene: &mut Scene, resource_manager: &mut spark_core::resource::ResourceManager, renderer: &mut spark_renderer::Renderer) {
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

        egui::TopBottomPanel::bottom("bottom_panel").show(&self.egui_ctx, |ui| {
            ui.horizontal(|ui| {
                let _ = ui.selectable_label(true, "Console");
                let _ = ui.selectable_label(false, "Assets");
            });
            ui.separator();

            egui::ScrollArea::vertical().stick_to_bottom(true).show(ui, |ui| {
                let logs = self.logs.lock().unwrap();
                for log in logs.iter() {
                    ui.label(log);
                }
            });
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
                         self.undo_stack.push(Box::new(TransformCommand {
                            node_key: selected_key,
                            old_transform: initial_transform,
                            new_transform: node.local_transform,
                        }));
                        self.redo_stack.clear();
                    }

                    ui.separator();
                    ui.label("Node Data");
                    match &mut node.data {
                        spark_core::scene::NodeData::Light { light_type: _, color, intensity, range } => {
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

    fn draw_node_tree(ui: &mut egui::Ui, scene: &mut Scene, node_key: NodeKey, selected_node: &mut Option<NodeKey>) {
        let (label, children) = if let Some(node) = scene.nodes.get(node_key) {
            (node.name.clone(), node.children.clone())
        } else {
            return;
        };

        let is_selected = Some(node_key) == *selected_node;

        let response = ui.selectable_label(is_selected, &label);
        response.context_menu(|ui| {
                if ui.button("Rename").clicked() { ui.close_menu(); }
                if ui.button("Delete").clicked() {
                    ui.close_menu();
                }
            });

        if response.clicked() {
            *selected_node = Some(node_key);
        }

        // Drag and drop
        if ui.memory(|m| m.is_being_dragged(ui.id())) {
             // simplified: only visualize for now
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
                    scene.update_transforms(dragged_key);
                }
            }
        }

        for &child_key in &children {
            ui.indent(&label, |ui| {
                Self::draw_node_tree(ui, scene, child_key, selected_node);
            });
        }
    }

    pub fn draw_viewport(&mut self, scene: &mut Scene, fps: f32) {
        if let Some(texture_id) = self.viewport_texture_id {
            egui::Window::new("Viewport").show(&self.egui_ctx, |ui| {
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
                    let speed = 0.1;
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
                                self.selected_node = Some(key);
                            } else {
                                self.selected_node = None;
                            }
                        }
                    }
                }

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
