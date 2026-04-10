use egui::{Context, Visuals};
use egui_gizmo::GizmoMode;
use egui_winit::State;
use winit::event::WindowEvent;
use winit::window::Window;

use spark_core::scene::{Node, NodeKey, Scene};

pub mod editor_ui;

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
            node.is_dirty = true;
        }
    }
    fn undo(&mut self, scene: &mut Scene) {
        if let Some(node) = scene.nodes.get_mut(self.node_key) {
            node.local_transform = self.old_transform;
            node.is_dirty = true;
        }
    }
}

pub struct AddComponentCommand {
    pub node_key: NodeKey,
    pub component: Option<Box<dyn spark_core::scene::Component>>,
    pub component_index: Option<usize>,
}

impl Command for AddComponentCommand {
    fn execute(&mut self, scene: &mut Scene) {
        if let Some(node) = scene.nodes.get_mut(self.node_key) {
            if let Some(comp) = self.component.take() {
                node.components.push(comp);
                self.component_index = Some(node.components.len() - 1);
                scene.rebuild_component_registry();
            }
        }
    }
    fn undo(&mut self, scene: &mut Scene) {
        if let (Some(node), Some(idx)) = (scene.nodes.get_mut(self.node_key), self.component_index)
        {
            if idx < node.components.len() {
                self.component = Some(node.components.remove(idx));
                scene.rebuild_component_registry();
            }
        }
    }
}

pub struct RemoveComponentCommand {
    pub node_key: NodeKey,
    pub component_index: usize,
    pub removed_component: Option<Box<dyn spark_core::scene::Component>>,
}

impl Command for RemoveComponentCommand {
    fn execute(&mut self, scene: &mut Scene) {
        if let Some(node) = scene.nodes.get_mut(self.node_key) {
            if self.component_index < node.components.len() {
                self.removed_component = Some(node.components.remove(self.component_index));
                scene.rebuild_component_registry();
            }
        }
    }
    fn undo(&mut self, scene: &mut Scene) {
        if let Some(node) = scene.nodes.get_mut(self.node_key) {
            if let Some(comp) = self.removed_component.take() {
                if self.component_index <= node.components.len() {
                    node.components.insert(self.component_index, comp);
                } else {
                    node.components.push(comp);
                }
                scene.rebuild_component_registry();
            }
        }
    }
}

pub struct AddNodeCommand {
    pub parent_key: NodeKey,
    pub node: Option<Node>,
    pub added_key: Option<NodeKey>,
}

impl Command for AddNodeCommand {
    fn execute(&mut self, scene: &mut Scene) {
        if let Some(node) = self.node.take() {
            let key = scene.add_node(self.parent_key, node);
            self.added_key = Some(key);
        }
    }
    fn undo(&mut self, scene: &mut Scene) {
        if let Some(key) = self.added_key.take() {
            if let Some(node) = scene.nodes.remove(key) {
                if let Some(parent_key) = node.parent {
                    if let Some(parent) = scene.nodes.get_mut(parent_key) {
                        parent.children.retain(|&k| k != key);
                    }
                }
                self.node = Some(node);
            }
        }
    }
}

pub struct DeleteNodeCommand {
    pub node_key: NodeKey,
    pub node: Option<Node>,
    pub parent_key: Option<NodeKey>,
    pub index_in_parent: usize,
}

impl Command for DeleteNodeCommand {
    fn execute(&mut self, scene: &mut Scene) {
        if let Some(node) = scene.nodes.get(self.node_key) {
            self.parent_key = node.parent;
            if let Some(pk) = self.parent_key {
                if let Some(parent) = scene.nodes.get_mut(pk) {
                    self.index_in_parent = parent
                        .children
                        .iter()
                        .position(|&k| k == self.node_key)
                        .unwrap_or(0);
                    parent.children.retain(|&k| k != self.node_key);
                }
            }
            self.node = scene.nodes.remove(self.node_key);
        }
    }
    fn undo(&mut self, scene: &mut Scene) {
        if let Some(node) = self.node.take() {
            let key = scene.nodes.insert(node);
            self.node_key = key;

            if let Some(pk) = self.parent_key {
                if let Some(parent) = scene.nodes.get_mut(pk) {
                    if self.index_in_parent < parent.children.len() {
                        parent.children.insert(self.index_in_parent, key);
                    } else {
                        parent.children.push(key);
                    }
                }
            }
        }
    }
}

pub struct DuplicateNodeCommand {
    pub original_key: NodeKey,
    pub new_key: Option<NodeKey>,
}

impl Command for DuplicateNodeCommand {
    fn execute(&mut self, scene: &mut Scene) {
        if let Some(key) = scene.duplicate_node(self.original_key) {
            self.new_key = Some(key);
        }
    }
    fn undo(&mut self, scene: &mut Scene) {
        if let Some(key) = self.new_key.take() {
            scene.remove_node(key);
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum SimulationState {
    Stopped,
    Playing,
    Paused,
}

pub struct EditorUI {
    pub asset_current_dir: std::path::PathBuf,
    pub hierarchy_search: String,
    pub asset_search: String,
    pub material_search: String,
    pub log_search: String,
    pub log_filter_info: bool,
    pub log_filter_warn: bool,
    pub log_filter_error: bool,
    pub egui_ctx: Context,
    pub egui_state: State,
    pub selected_node: Option<NodeKey>,
    pub viewport_texture_id: Option<egui::TextureId>,
    pub undo_stack: Vec<Box<dyn Command>>,
    pub redo_stack: Vec<Box<dyn Command>>,
    pub gizmo_mode: GizmoMode,
    pub gizmo_local: bool,
    pub snap_enabled: bool,
    pub snap_distance: f32,
    pub camera_pos: spark_math::Vec3,
    pub camera_rot: spark_math::Vec2, // Yaw, Pitch
    pub logs: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    pub active_bottom_tab: BottomTab,
    pub camera_speed: f32,
    pub node_to_delete: Option<NodeKey>,
    pub node_to_add: Option<(NodeKey, Node)>,
    pub node_to_duplicate: Option<NodeKey>,
    pub node_to_add_child: Option<(NodeKey, NodeType)>,
    pub initial_gizmo_transform: Option<spark_math::Mat4>,
    pub component_to_remove: Option<(NodeKey, usize)>,
    pub component_to_add: Option<(NodeKey, Box<dyn spark_core::scene::Component>)>,
    pub sim_state: SimulationState,
    pub scene_snapshot: Option<Scene>,
    pub show_hierarchy: bool,
    pub show_inspector: bool,
    pub show_bottom_panel: bool,
    pub status_message: String,
    pub hierarchy_force_state: Option<bool>,
    pub asset_rename_state: Option<(std::path::PathBuf, String)>,
}

pub enum NodeType {
    Mesh,
    Light,
    Sprite,
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum BottomTab {
    Console,
    Assets,
    Materials,
    Settings,
    Statistics,
}

impl EditorUI {
    pub fn new(window: &Window, logs: std::sync::Arc<std::sync::Mutex<Vec<String>>>) -> Self {
        let egui_ctx = Context::default();

        let mut visuals = Visuals::dark();
        visuals.widgets.noninteractive.bg_fill = egui::Color32::from_gray(20);
        visuals.widgets.noninteractive.fg_stroke =
            egui::Stroke::new(1.0, egui::Color32::from_gray(180));
        visuals.widgets.active.bg_fill = egui::Color32::from_rgb(60, 100, 150);
        visuals.widgets.hovered.bg_fill = egui::Color32::from_gray(45);
        visuals.window_rounding = 0.0.into();
        egui_ctx.set_visuals(visuals);

        let egui_state = State::new(
            egui_ctx.clone(),
            egui_ctx.viewport_id(),
            &window,
            None,
            None,
        );

        Self {
            asset_current_dir: std::path::PathBuf::from("assets"),
            hierarchy_search: String::new(),
            node_to_add: None,
            asset_search: String::new(),
            material_search: String::new(),
            log_search: String::new(),
            log_filter_info: true,
            log_filter_warn: true,
            log_filter_error: true,
            egui_ctx,
            egui_state,
            selected_node: None,
            viewport_texture_id: None,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            gizmo_mode: GizmoMode::Translate,
            gizmo_local: false,
            snap_enabled: false,
            snap_distance: 1.0,
            camera_pos: spark_math::Vec3::new(0.0, 2.0, 10.0),
            camera_rot: spark_math::Vec2::new(-90.0f32.to_radians(), 0.0),
            camera_speed: 0.1,
            logs,
            active_bottom_tab: BottomTab::Console,
            node_to_delete: None,
            node_to_duplicate: None,
            node_to_add_child: None,
            initial_gizmo_transform: None,
            component_to_remove: None,
            component_to_add: None,
            sim_state: SimulationState::Stopped,
            scene_snapshot: None,
            show_hierarchy: true,
            show_inspector: true,
            show_bottom_panel: true,
            status_message: "Ready".to_string(),
            hierarchy_force_state: None,
            asset_rename_state: None,
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
        self.egui_state
            .handle_platform_output(window, full_output.platform_output.clone());
        full_output
    }

    pub fn execute_command(&mut self, mut command: Box<dyn Command>, scene: &mut Scene) {
        command.execute(scene);
        self.undo_stack.push(command);
        self.redo_stack.clear();
    }

    pub fn undo(&mut self, scene: &mut Scene) {
        if let Some(mut command) = self.undo_stack.pop() {
            command.undo(scene);
            self.redo_stack.push(command);
        }
    }

    pub fn redo(&mut self, scene: &mut Scene) {
        if let Some(mut command) = self.redo_stack.pop() {
            command.execute(scene);
            self.undo_stack.push(command);
        }
    }

    pub fn draw_ui(
        &mut self,
        scene: &mut Scene,
        resource_manager: &mut spark_core::resource::ResourceManager,
        asset_manager: &mut spark_core::asset::AssetManager,
        renderer: &mut spark_renderer::Renderer,
        project: &mut spark_core::Project,
        fps: f32,
    ) {
        let ctx = self.egui_ctx.clone();

        ctx.input(|i| {
            if i.modifiers.command && i.key_pressed(egui::Key::Z) {
                if i.modifiers.shift {
                    self.redo(scene);
                } else {
                    self.undo(scene);
                }
            }
            if i.modifiers.command && i.key_pressed(egui::Key::Y) {
                self.redo(scene);
            }
        });

        self.draw_menu_bar(scene, resource_manager, asset_manager, renderer);
        self.draw_toolbar(scene);
        self.draw_status_bar(fps);

        if self.show_bottom_panel {
            self.draw_bottom_panel(
                scene,
                resource_manager,
                asset_manager,
                renderer,
                project,
                fps,
            );
        }
        if self.show_hierarchy {
            self.draw_hierarchy_panel(scene);
        }
        if self.show_inspector {
            self.draw_inspector_panel(scene, renderer, asset_manager);
        }

        if let Some(key) = self.node_to_duplicate.take() {
            self.execute_command(
                Box::new(DuplicateNodeCommand {
                    original_key: key,
                    new_key: None,
                }),
                scene,
            );
        }

        if let Some((parent, node_type)) = self.node_to_add_child.take() {
            match node_type {
                NodeType::Mesh => {
                    let mut node = Node {
                        name: "New Mesh".to_string(),
                        visible: true,
                        locked: false,
                        is_dirty: true,
                        local_transform: spark_math::Mat4::IDENTITY,
                        global_transform: spark_math::Mat4::IDENTITY,
                        parent: None,
                        children: Vec::new(),
                        components: Vec::new(),
                    };
                    node.components
                        .push(Box::new(spark_core::scene::MeshComponent {
                            vertex_count: 0,
                            index_count: 0,
                            first_index: 0,
                            vertex_offset: 0,
                            texture_handle: None,
                            material_index: None,
                            bounding_radius: 1.0,
                            skin_index: None,
                        }));
                    self.node_to_add = Some((parent, node));
                }
                NodeType::Light => {
                    let mut node = Node {
                        name: "New Light".to_string(),
                        visible: true,
                        locked: false,
                        is_dirty: true,
                        local_transform: spark_math::Mat4::IDENTITY,
                        global_transform: spark_math::Mat4::IDENTITY,
                        parent: None,
                        children: Vec::new(),
                        components: Vec::new(),
                    };
                    node.components
                        .push(Box::new(spark_core::scene::LightComponent {
                            light_type: spark_core::scene::LightType::Point,
                            color: spark_math::Vec3::ONE,
                            intensity: 1.0,
                            range: 10.0,
                            spot_inner_angle: 30.0,
                            spot_outer_angle: 45.0,
                        }));
                    self.node_to_add = Some((parent, node));
                }
                NodeType::Sprite => {
                    let mut node = Node {
                        name: "New Sprite".to_string(),
                        visible: true,
                        locked: false,
                        is_dirty: true,
                        local_transform: spark_math::Mat4::IDENTITY,
                        global_transform: spark_math::Mat4::IDENTITY,
                        parent: None,
                        children: Vec::new(),
                        components: Vec::new(),
                    };
                    node.components
                        .push(Box::new(spark_core::scene::SpriteComponent {
                            texture_handle: None,
                            color: [1.0, 1.0, 1.0, 1.0],
                            flip_x: false,
                            flip_y: false,
                            size: spark_math::Vec2::new(1.0, 1.0),
                        }));
                    self.node_to_add = Some((parent, node));
                }
            }
        }

        if let Some(key) = self.node_to_delete.take() {
            self.execute_command(
                Box::new(DeleteNodeCommand {
                    node_key: key,
                    node: None,
                    parent_key: None,
                    index_in_parent: 0,
                }),
                scene,
            );
            if self.selected_node == Some(key) {
                self.selected_node = None;
            }
        }

        if let Some((parent, node)) = self.node_to_add.take() {
            self.execute_command(
                Box::new(AddNodeCommand {
                    parent_key: parent,
                    node: Some(node),
                    added_key: None,
                }),
                scene,
            );
        }

        if let Some((node_key, comp_idx)) = self.component_to_remove.take() {
            self.execute_command(
                Box::new(RemoveComponentCommand {
                    node_key,
                    component_index: comp_idx,
                    removed_component: None,
                }),
                scene,
            );
        }

        if let Some((node_key, comp)) = self.component_to_add.take() {
            self.execute_command(
                Box::new(AddComponentCommand {
                    node_key,
                    component: Some(comp),
                    component_index: None,
                }),
                scene,
            );
        }
    }

    fn draw_menu_bar(
        &mut self,
        scene: &mut Scene,
        resource_manager: &mut spark_core::resource::ResourceManager,
        asset_manager: &mut spark_core::asset::AssetManager,
        renderer: &mut spark_renderer::Renderer,
    ) {
        let ctx = self.egui_ctx.clone();
        egui::TopBottomPanel::top("menu_bar").show(&ctx, |ui| {
            ui.horizontal(|ui| {
                egui::menu::bar(ui, |ui| {
                    ui.menu_button("File", |ui| {
                        if ui.button("New").clicked() {
                            *scene = Scene::new();
                            ui.close_menu();
                        }
                        if ui.button("Open").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("Spark Scene", &["json"])
                                .pick_file()
                            {
                                if let Ok(new_scene) = Scene::load_from_file(
                                    path.to_str().expect("Scene path is not valid UTF-8"),
                                ) {
                                    *scene = new_scene;
                                }
                            }
                            ui.close_menu();
                        }
                        if ui.button("Import glTF").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("glTF", &["gltf", "glb"])
                                .pick_file()
                            {
                                if let Err(e) = resource_manager.load_scene(
                                    path,
                                    scene,
                                    renderer,
                                    asset_manager,
                                ) {
                                    log::error!("Failed to import glTF: {}", e);
                                }
                            }
                            ui.close_menu();
                        }
                        if ui.button("Save").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("Spark Scene", &["json"])
                                .save_file()
                            {
                                let _ = scene.save_to_file(
                                    path.to_str().expect("Save path is not valid UTF-8"),
                                );
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
                    ui.menu_button("View", |ui| {
                        ui.checkbox(&mut self.show_hierarchy, "Hierarchy");
                        ui.checkbox(&mut self.show_inspector, "Inspector");
                        ui.checkbox(&mut self.show_bottom_panel, "Bottom Panel");
                    });
                    ui.menu_button("Create", |ui| {
                        if ui.button("Empty Node").clicked() {
                            let node = Node {
                                name: "Empty Node".to_string(),
                                visible: true,
                                locked: false,
                                is_dirty: true,
                                local_transform: spark_math::Mat4::IDENTITY,
                                global_transform: spark_math::Mat4::IDENTITY,
                                parent: None,
                                children: Vec::new(),
                                components: Vec::new(),
                            };
                            self.node_to_add = Some((scene.root, node));
                            ui.close_menu();
                        }
                        ui.separator();
                        if ui.button("Cube").clicked() {
                            self.add_primitive_cube(scene);
                            ui.close_menu();
                        }
                        if ui.button("Sphere").clicked() {
                            self.add_primitive_sphere(scene);
                            ui.close_menu();
                        }
                        if ui.button("Plane").clicked() {
                            self.add_primitive_plane(scene);
                            ui.close_menu();
                        }
                        ui.separator();
                        if ui.button("Point Light").clicked() {
                            self.add_default_light(scene);
                            ui.close_menu();
                        }
                        if ui.button("Directional Light").clicked() {
                            self.add_directional_light(scene);
                            ui.close_menu();
                        }
                    });
                });
            });
        });
    }

    fn draw_toolbar(&mut self, scene: &mut Scene) {
        let ctx = self.egui_ctx.clone();
        egui::TopBottomPanel::top("toolbar").show(&ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(
                    &mut self.gizmo_mode,
                    egui_gizmo::GizmoMode::Translate,
                    "⬈ Move",
                );
                ui.selectable_value(
                    &mut self.gizmo_mode,
                    egui_gizmo::GizmoMode::Rotate,
                    "⟲ Rotate",
                );
                ui.selectable_value(
                    &mut self.gizmo_mode,
                    egui_gizmo::GizmoMode::Scale,
                    "⤢ Scale",
                );

                ui.separator();
                ui.toggle_value(&mut self.gizmo_local, "Local");
                ui.toggle_value(&mut self.snap_enabled, "Snap");
                if self.snap_enabled {
                    ui.add(
                        egui::DragValue::new(&mut self.snap_distance)
                            .speed(0.1)
                            .clamp_range(0.0..=10.0),
                    );
                }

                ui.separator();

                let (play_label, play_color) = if self.sim_state == SimulationState::Playing {
                    ("⏸ Pause", egui::Color32::KHAKI)
                } else {
                    ("▶ Play", egui::Color32::LIGHT_GREEN)
                };

                if ui
                    .button(egui::RichText::new(play_label).color(play_color))
                    .clicked()
                {
                    if self.sim_state == SimulationState::Stopped {
                        let json = serde_json::to_string(scene)
                            .expect("Failed to serialize scene for snapshot");
                        self.scene_snapshot = Some(
                            serde_json::from_str(&json)
                                .expect("Failed to deserialize scene snapshot"),
                        );
                    }
                    self.sim_state = if self.sim_state == SimulationState::Playing {
                        SimulationState::Paused
                    } else {
                        SimulationState::Playing
                    };
                }

                if ui
                    .button(egui::RichText::new("⏹ Stop").color(egui::Color32::LIGHT_RED))
                    .clicked()
                {
                    if let Some(snapshot) = self.scene_snapshot.take() {
                        *scene = snapshot;
                    }
                    self.sim_state = SimulationState::Stopped;
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("⟳").on_hover_text("Redo").clicked() {
                        self.redo(scene);
                    }
                    if ui.button("⟲").on_hover_text("Undo").clicked() {
                        self.undo(scene);
                    }
                });
            });
        });
    }

    fn draw_status_bar(&mut self, fps: f32) {
        let ctx = self.egui_ctx.clone();
        egui::TopBottomPanel::bottom("status_bar").show(&ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("Status: {}", self.status_message));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!("FPS: {:.1}", fps));
                    ui.separator();
                    ui.label("Spark Engine v0.1.0");
                });
            });
        });
    }
}
