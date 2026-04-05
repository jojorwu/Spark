use egui::{Context, Ui, Visuals};
use egui_gizmo::{Gizmo, GizmoMode};
use egui_winit::State;
use spark_math::Vec4Swizzles;
use winit::event::WindowEvent;
use winit::window::Window;

use spark_core::scene::{Node, NodeKey, Scene};

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
        } else if let Some(_key) = self.added_key {
             // Redo: we need to find a way to re-add exactly the same node if it was removed
             // For simplicity in this engine, we'll just store the node when it's not in the scene
        }
    }
    fn undo(&mut self, scene: &mut Scene) {
        if let Some(key) = self.added_key.take() {
            if let Some(node) = scene.nodes.remove(key) {
                // Remove from parent
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
                    self.index_in_parent = parent.children.iter().position(|&k| k == self.node_key).unwrap_or(0);
                    parent.children.retain(|&k| k != self.node_key);
                }
            }
            self.node = scene.nodes.remove(self.node_key);
        }
    }
    fn undo(&mut self, scene: &mut Scene) {
        if let Some(node) = self.node.take() {
            let key = scene.nodes.insert(node);
            // After re-inserting, we need to update the node_key for future redo
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
    pub sim_state: SimulationState,
    pub scene_snapshot: Option<Scene>,
    pub show_hierarchy: bool,
    pub show_inspector: bool,
    pub show_bottom_panel: bool,
    pub status_message: String,
    pub hierarchy_force_state: Option<bool>, // Some(true) to expand all, Some(false) to collapse all
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
        visuals.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, egui::Color32::from_gray(180));
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
            sim_state: SimulationState::Stopped,
            scene_snapshot: None,
            show_hierarchy: true,
            show_inspector: true,
            show_bottom_panel: true,
            status_message: "Ready".to_string(),
            hierarchy_force_state: None,
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

    fn draw_console_tab(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.log_filter_info, "Info");
            ui.checkbox(&mut self.log_filter_warn, "Warn");
            ui.checkbox(&mut self.log_filter_error, "Error");
            ui.separator();
            ui.label("🔍");
            ui.text_edit_singleline(&mut self.log_search);
            if ui.button("✖").clicked() {
                self.log_search.clear();
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button("Clear").clicked() {
                    self.logs.lock().unwrap().clear();
                }
            });
        });
        ui.separator();
        egui::ScrollArea::vertical()
            .stick_to_bottom(true)
            .show(ui, |ui| {
                let logs = self.logs.lock().unwrap();
                for log in logs.iter() {
                    let (color, visible) = if log.contains("ERROR") {
                        (egui::Color32::LIGHT_RED, self.log_filter_error)
                    } else if log.contains("WARN") {
                        (egui::Color32::KHAKI, self.log_filter_warn)
                    } else {
                        (egui::Color32::LIGHT_GRAY, self.log_filter_info)
                    };

                    let matches_search = self.log_search.is_empty() || log.to_lowercase().contains(&self.log_search.to_lowercase());

                    if visible && matches_search {
                        ui.label(egui::RichText::new(log).color(color).monospace());
                    }
                }
            });
    }

    fn draw_assets_tab(
        &mut self,
        ui: &mut Ui,
        scene: &mut Scene,
        resource_manager: &mut spark_core::resource::ResourceManager,
        renderer: &mut spark_renderer::Renderer,
        asset_manager: &mut spark_core::asset::AssetManager,
    ) {
        let mut asset_to_load = None;
        let mut dir_to_set = None;

        ui.horizontal(|ui| {
            if ui.button("⬅").on_hover_text("Up").clicked() {
                if let Some(parent) = self.asset_current_dir.parent() {
                    if parent.starts_with("assets") {
                        dir_to_set = Some(parent.to_path_buf());
                    }
                }
            }
            ui.label(format!("Path: {}", self.asset_current_dir.display()));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label("🔍");
                ui.text_edit_singleline(&mut self.asset_search);
            });
        });
        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| {
            let entries = std::fs::read_dir(&self.asset_current_dir);
            if let Ok(entries) = entries {
                let mut sorted_entries: Vec<_> = entries.filter_map(|e| e.ok()).collect();
                sorted_entries.sort_by_key(|e| (!e.path().is_dir(), e.file_name()));

                for entry in sorted_entries {
                    let path = entry.path();
                    let label = path.file_name().unwrap().to_string_lossy();

                    if !self.asset_search.is_empty()
                        && !label.to_lowercase().contains(&self.asset_search.to_lowercase())
                    {
                        continue;
                    }

                    ui.horizontal(|ui| {
                        if path.is_dir() {
                            if ui.selectable_label(false, format!("📁 {}", label)).clicked() {
                                dir_to_set = Some(path.to_path_buf());
                            }
                        } else {
                            let is_gltf = path
                                .extension()
                                .is_some_and(|ext| ext == "gltf" || ext == "glb");
                            let is_img = path.extension().is_some_and(|ext| ext == "png" || ext == "jpg");
                            let icon = if is_gltf {
                                "📦"
                            } else if is_img {
                                "🖼"
                            } else {
                                "📄"
                            };

                            if ui.selectable_label(false, format!("{} {}", icon, label)).clicked() && is_gltf {
                                asset_to_load = Some(path.to_path_buf());
                            }
                        }
                    });
                }
            }
        });

        if let Some(dir) = dir_to_set {
            self.asset_current_dir = dir;
        }
        if let Some(path) = asset_to_load {
            resource_manager.load_scene(path, scene, renderer, asset_manager);
        }
    }

    fn draw_materials_tab(&mut self, ui: &mut Ui, asset_manager: &mut spark_core::asset::AssetManager, resource_manager: &mut spark_core::resource::ResourceManager) {
        ui.horizontal(|ui| {
            ui.label("🔍 Search Materials:");
            ui.text_edit_singleline(&mut self.material_search);
        });
        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| {
            let mat_indices: Vec<_> = (0..asset_manager.materials.assets_len()).collect();
            for idx in mat_indices {
                let handle = spark_core::resource::Handle::new(idx as u32);
                if let Some(mat) = asset_manager.materials.get_mut(handle) {
                    if !self.material_search.is_empty()
                        && !mat.name.to_lowercase().contains(&self.material_search.to_lowercase())
                    {
                        continue;
                    }

                    ui.collapsing(format!("Material: {}", mat.name), |ui| {
                        self.draw_material_editor(ui, mat, idx, asset_manager, resource_manager);
                    });
                }
            }
        });
    }

    fn draw_material_editor(&mut self, ui: &mut Ui, mat: &mut spark_core::asset::Material, idx: usize, asset_manager: &mut spark_core::asset::AssetManager, resource_manager: &mut spark_core::resource::ResourceManager) {
        ui.horizontal(|ui| {
            ui.label("Name:");
            ui.text_edit_singleline(&mut mat.name);
        });
        ui.horizontal(|ui| {
            ui.label("Albedo:");
            ui.color_edit_button_rgba_unmultiplied(&mut mat.albedo_factor);
        });
        ui.horizontal(|ui| {
            ui.label("Metallic:");
            ui.add(egui::Slider::new(&mut mat.metallic_factor, 0.0..=1.0));
        });
        ui.horizontal(|ui| {
            ui.label("Roughness:");
            ui.add(egui::Slider::new(&mut mat.roughness_factor, 0.0..=1.0));
        });
        ui.horizontal(|ui| {
            ui.label("Emissive:");
            ui.color_edit_button_rgba_unmultiplied(&mut mat.emissive_factor);
        });
        ui.checkbox(&mut mat.is_transparent, "Transparent");

        ui.horizontal(|ui| {
            ui.label("Albedo Texture:");
            let tex_name = mat.albedo_texture.map(|h| {
                asset_manager.texture_path_map.iter()
                    .find(|(_, &handle)| handle == h)
                    .map(|(path, _)| path.file_name().unwrap().to_string_lossy().into_owned())
                    .unwrap_or_else(|| format!("Texture ID: {}", h.id()))
            }).unwrap_or_else(|| "None".to_string());

            egui::ComboBox::from_id_source(format!("mat_tex_{}", idx))
                .selected_text(tex_name)
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut mat.albedo_texture, None, "None");
                    for t_idx in 0..resource_manager.gpu_textures.assets_len() {
                        let h = spark_core::resource::Handle::new(t_idx as u32);
                        let name = asset_manager.texture_path_map.iter()
                            .find(|(_, &handle)| handle == h)
                            .map(|(path, _)| path.file_name().unwrap().to_string_lossy().into_owned())
                            .unwrap_or_else(|| format!("ID: {}", t_idx));
                        ui.selectable_value(&mut mat.albedo_texture, Some(h), name);
                    }
                });
        });
    }

    fn draw_component_list(
        &mut self,
        ui: &mut Ui,
        node: &mut Node,
        asset_manager: &mut spark_core::asset::AssetManager,
        selected_key: NodeKey,
    ) {
        ui.horizontal(|ui| {
            ui.heading("Components");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.menu_button("✚ Add", |ui| {
                    if ui.button("Mesh").clicked() {
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
                        ui.close_menu();
                    }
                    if ui.button("Light").clicked() {
                        node.components
                            .push(Box::new(spark_core::scene::LightComponent {
                                light_type: spark_core::scene::LightType::Point,
                                color: spark_math::Vec3::ONE,
                                intensity: 1.0,
                                range: 10.0,
                                spot_inner_angle: 30.0,
                                spot_outer_angle: 45.0,
                            }));
                        ui.close_menu();
                    }
                    if ui.button("Camera").clicked() {
                        node.components
                            .push(Box::new(spark_core::scene::CameraComponent {
                                fov: 45.0,
                                near: 0.1,
                                far: 100.0,
                                orthographic: false,
                                ortho_size: 5.0,
                            }));
                        ui.close_menu();
                    }
                    if ui.button("Sprite").clicked() {
                        node.components
                            .push(Box::new(spark_core::scene::SpriteComponent {
                                texture_handle: None,
                                color: [1.0, 1.0, 1.0, 1.0],
                                flip_x: false,
                                flip_y: false,
                                size: spark_math::Vec2::new(1.0, 1.0),
                            }));
                        ui.close_menu();
                    }
                });
            });
        });

        let mut to_remove = None;
        for (idx, component) in node.components.iter_mut().enumerate() {
            ui.add_space(4.0);
            let header = match component.as_any().type_id() {
                t if t == std::any::TypeId::of::<spark_core::scene::MeshComponent>() => "Mesh",
                t if t == std::any::TypeId::of::<spark_core::scene::LightComponent>() => "Light",
                t if t == std::any::TypeId::of::<spark_core::scene::CameraComponent>() => "Camera",
                t if t == std::any::TypeId::of::<spark_core::scene::SpriteComponent>() => "Sprite",
                _ => "Unknown",
            };

            let response = ui.collapsing(header, |ui| {
                self.draw_component_editor(ui, component, asset_manager);
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
        if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::Z)) {
            if ctx.input(|i| i.modifiers.shift) {
                self.redo(scene);
            } else {
                self.undo(scene);
            }
        }
        if ctx.input(|i| i.modifiers.command && i.key_pressed(egui::Key::Y)) {
            self.redo(scene);
        }

        self.draw_menu_bar(scene, resource_manager, asset_manager, renderer);
        self.draw_toolbar(scene);
        self.draw_status_bar(fps);

        if self.show_bottom_panel {
            self.draw_bottom_panel(scene, resource_manager, asset_manager, renderer, project, fps);
        }
        if self.show_hierarchy {
            self.draw_hierarchy_panel(scene);
        }
        if self.show_inspector {
            self.draw_inspector_panel(scene, renderer, asset_manager);
        }

        if let Some(key) = self.node_to_duplicate.take() {
            if let Some(new_key) = scene.duplicate_node(key) {
                self.selected_node = Some(new_key);
            }
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
            if let Some(node) = scene.nodes.get_mut(node_key) {
                if comp_idx < node.components.len() {
                    node.components.remove(comp_idx);
                }
            }
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
                                if let Ok(new_scene) = Scene::load_from_file(path.to_str().unwrap())
                                {
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
                                resource_manager.load_scene(path, scene, renderer, asset_manager);
                            }
                            ui.close_menu();
                        }
                        if ui.button("Save").clicked() {
                            if let Some(path) = rfd::FileDialog::new()
                                .add_filter("Spark Scene", &["json"])
                                .save_file()
                            {
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
                    ui.menu_button("View", |ui| {
                        ui.checkbox(&mut self.show_hierarchy, "Hierarchy");
                        ui.checkbox(&mut self.show_inspector, "Inspector");
                        ui.checkbox(&mut self.show_bottom_panel, "Bottom Panel");
                    });
                });
            });
        });
    }

    fn draw_toolbar(&mut self, scene: &mut Scene) {
        let ctx = self.egui_ctx.clone();
        egui::TopBottomPanel::top("toolbar").show(&ctx, |ui| {
            ui.horizontal(|ui| {
                // Gizmo Tools
                ui.selectable_value(&mut self.gizmo_mode, egui_gizmo::GizmoMode::Translate, "⬈ Move");
                ui.selectable_value(&mut self.gizmo_mode, egui_gizmo::GizmoMode::Rotate, "⟲ Rotate");
                ui.selectable_value(&mut self.gizmo_mode, egui_gizmo::GizmoMode::Scale, "⤢ Scale");

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

                // Simulation Controls
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
                        let json = serde_json::to_string(scene).unwrap();
                        self.scene_snapshot = Some(serde_json::from_str(&json).unwrap());
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

    /// Отрисовывает нижнюю панель с вкладками консоли, ассетов, материалов и настроек.
    /// Draws the bottom panel containing console, assets, materials, and settings tabs.
    fn draw_bottom_panel(
        &mut self,
        scene: &mut Scene,
        resource_manager: &mut spark_core::resource::ResourceManager,
        asset_manager: &mut spark_core::asset::AssetManager,
        renderer: &mut spark_renderer::Renderer,
        project: &mut spark_core::Project,
        fps: f32,
    ) {
        let ctx = self.egui_ctx.clone();
        egui::TopBottomPanel::bottom("bottom_panel").show(&ctx, |ui| {
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.active_bottom_tab, BottomTab::Console, "📝 Console");
                ui.selectable_value(&mut self.active_bottom_tab, BottomTab::Assets, "📁 Assets");
                ui.selectable_value(&mut self.active_bottom_tab, BottomTab::Materials, "🎨 Materials");
                ui.selectable_value(&mut self.active_bottom_tab, BottomTab::Settings, "⚙ Settings");
                ui.selectable_value(
                    &mut self.active_bottom_tab,
                    BottomTab::Statistics,
                    "📊 Statistics",
                );
            });
            ui.separator();

            match self.active_bottom_tab {
                BottomTab::Console => self.draw_console_tab(ui),
                BottomTab::Assets => self.draw_assets_tab(ui, scene, resource_manager, renderer, asset_manager),
                BottomTab::Materials => self.draw_materials_tab(ui, asset_manager, resource_manager),
                BottomTab::Settings => {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.heading("Post-Processing");
                        ui.horizontal(|ui| {
                            ui.label("Manual Exposure:");
                            ui.add(egui::Slider::new(
                                &mut renderer.settings.exposure,
                                0.1..=10.0,
                            ));
                        });
                        ui.horizontal(|ui| {
                            ui.checkbox(
                                &mut renderer.settings.enable_auto_exposure,
                                "Auto Exposure",
                            );
                            if renderer.settings.enable_auto_exposure {
                                ui.label("Min:");
                                ui.add(
                                    egui::DragValue::new(&mut renderer.settings.auto_exposure_min)
                                        .speed(0.1),
                                );
                                ui.label("Max:");
                                ui.add(
                                    egui::DragValue::new(&mut renderer.settings.auto_exposure_max)
                                        .speed(0.1),
                                );
                                ui.label("Speed:");
                                ui.add(
                                    egui::DragValue::new(
                                        &mut renderer.settings.auto_exposure_speed,
                                    )
                                    .speed(0.1),
                                );
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Gamma:");
                            ui.add(egui::Slider::new(&mut renderer.settings.gamma, 1.0..=3.0));
                        });
                        ui.separator();

                        ui.horizontal(|ui| {
                            ui.checkbox(&mut renderer.settings.enable_bloom, "Bloom");
                            if renderer.settings.enable_bloom {
                                ui.label("Threshold:");
                                ui.add(egui::Slider::new(
                                    &mut renderer.settings.bloom_threshold,
                                    0.1..=2.0,
                                ));
                                ui.label("Intensity:");
                                ui.add(egui::Slider::new(
                                    &mut renderer.settings.bloom_intensity,
                                    0.0..=2.0,
                                ));
                            }
                        });

                        ui.horizontal(|ui| {
                            ui.checkbox(
                                &mut renderer.settings.enable_color_grading,
                                "Color Grading",
                            );
                            if renderer.settings.enable_color_grading {
                                ui.label("LUT Index:");
                                ui.add(
                                    egui::DragValue::new(&mut renderer.settings.lut_index)
                                        .clamp_range(-1..=15),
                                );
                            }
                        });

                        ui.horizontal(|ui| {
                            ui.label("Vignette Intensity:");
                            ui.add(egui::Slider::new(
                                &mut renderer.settings.vignette_intensity,
                                0.0..=1.0,
                            ));
                            ui.label("Smoothness:");
                            ui.add(egui::Slider::new(
                                &mut renderer.settings.vignette_smoothness,
                                0.0..=1.0,
                            ));
                        });

                        ui.horizontal(|ui| {
                            ui.label("Chromatic Aberration:");
                            ui.add(egui::Slider::new(
                                &mut renderer.settings.chromatic_aberration,
                                0.0..=0.01,
                            ));
                        });

                        ui.horizontal(|ui| {
                            ui.label("Film Grain:");
                            ui.add(egui::Slider::new(
                                &mut renderer.settings.film_grain,
                                0.0..=0.1,
                            ));
                        });

                        ui.horizontal(|ui| {
                            ui.checkbox(&mut renderer.settings.enable_dof, "Depth of Field");
                            if renderer.settings.enable_dof {
                                ui.label("Distance:");
                                ui.add(egui::Slider::new(
                                    &mut renderer.settings.dof_focus_distance,
                                    0.1..=50.0,
                                ));
                                ui.label("Range:");
                                ui.add(egui::Slider::new(
                                    &mut renderer.settings.dof_focus_range,
                                    0.1..=20.0,
                                ));
                                ui.label("Size:");
                                ui.add(egui::Slider::new(
                                    &mut renderer.settings.dof_bokeh_size,
                                    1.0..=20.0,
                                ));
                            }
                        });

                        ui.separator();
                        ui.heading("Environmental Effects");
                        ui.checkbox(&mut renderer.settings.enable_volumetric, "Volumetric Fog");
                        if renderer.settings.enable_volumetric {
                            ui.horizontal(|ui| {
                                ui.label("Fog Color:");
                                ui.color_edit_button_rgb(&mut renderer.settings.fog_color);
                            });
                            ui.horizontal(|ui| {
                                ui.label("Density:");
                                ui.add(egui::Slider::new(
                                    &mut renderer.settings.fog_density,
                                    0.0..=0.1,
                                ));
                                ui.label("Height Falloff:");
                                ui.add(egui::Slider::new(
                                    &mut renderer.settings.fog_height_falloff,
                                    0.0..=1.0,
                                ));
                            });
                        }

                        ui.separator();
                        ui.heading("General Features");
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut renderer.settings.enable_shadows, "Shadows");
                            if renderer.settings.enable_shadows {
                                ui.label("PCF:");
                                egui::ComboBox::from_id_source("shadow_pcf")
                                    .selected_text(if renderer.settings.shadow_pcf_samples == 0 {
                                        "Hard"
                                    } else {
                                        "Soft (3x3)"
                                    })
                                    .show_ui(ui, |ui| {
                                        ui.selectable_value(
                                            &mut renderer.settings.shadow_pcf_samples,
                                            0,
                                            "Hard",
                                        );
                                        ui.selectable_value(
                                            &mut renderer.settings.shadow_pcf_samples,
                                            1,
                                            "Soft (3x3)",
                                        );
                                    });
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut renderer.settings.enable_ssao, "SSAO");
                            if renderer.settings.enable_ssao {
                                ui.label("Radius:");
                                ui.add(egui::Slider::new(
                                    &mut renderer.settings.ssao_radius,
                                    0.1..=2.0,
                                ));
                                ui.label("Strength:");
                                ui.add(egui::Slider::new(
                                    &mut renderer.settings.ssao_strength,
                                    0.1..=5.0,
                                ));
                            }
                        });
                        ui.checkbox(&mut renderer.settings.enable_taa, "TAA");
                        ui.checkbox(&mut renderer.settings.enable_grid, "Ground Grid");
                        ui.checkbox(&mut renderer.settings.enable_ibl, "IBL");

                        ui.separator();
                        ui.heading("Physics");
                        ui.horizontal(|ui| {
                            ui.label("Gravity:");
                            ui.add(
                                egui::DragValue::new(&mut project.physics_settings.gravity.x)
                                    .speed(0.1)
                                    .prefix("X:"),
                            );
                            ui.add(
                                egui::DragValue::new(&mut project.physics_settings.gravity.y)
                                    .speed(0.1)
                                    .prefix("Y:"),
                            );
                            ui.add(
                                egui::DragValue::new(&mut project.physics_settings.gravity.z)
                                    .speed(0.1)
                                    .prefix("Z:"),
                            );
                        });
                        ui.horizontal(|ui| {
                            ui.label("Sim Freq (Hz):");
                            ui.add(egui::Slider::new(
                                &mut project.physics_settings.simulation_frequency,
                                10.0..=240.0,
                            ));
                        });

                        ui.separator();
                        ui.heading("Advanced Features");
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut renderer.settings.enable_ssr, "SSR");
                            if renderer.settings.enable_ssr {
                                ui.label("Steps:");
                                ui.add(egui::DragValue::new(&mut renderer.settings.ssr_max_steps));
                                ui.label("Step:");
                                ui.add(
                                    egui::DragValue::new(&mut renderer.settings.ssr_step)
                                        .speed(0.01),
                                );
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut renderer.settings.enable_ssgi, "SSGI");
                            if renderer.settings.enable_ssgi {
                                ui.label("Intensity:");
                                ui.add(egui::Slider::new(
                                    &mut renderer.settings.ssgi_intensity,
                                    0.0..=2.0,
                                ));
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut renderer.settings.enable_motion_blur, "Motion Blur");
                            if renderer.settings.enable_motion_blur {
                                ui.label("Strength:");
                                ui.add(egui::Slider::new(
                                    &mut renderer.settings.motion_blur_strength,
                                    0.0..=1.0,
                                ));
                            }
                        });

                        ui.separator();
                        ui.heading("Ray Tracing");
                        ui.horizontal(|ui| {
                            ui.checkbox(
                                &mut renderer.settings.enable_rt_reflections,
                                "RT Reflections",
                            );
                            ui.checkbox(&mut renderer.settings.enable_rt_shadows, "RT Shadows");
                        });
                        ui.horizontal(|ui| {
                            ui.checkbox(
                                &mut renderer.settings.enable_rt_ao,
                                "RT Ambient Occlusion",
                            );
                            ui.checkbox(
                                &mut renderer.settings.enable_rt_gi,
                                "RT Global Illumination",
                            );
                        });
                    });
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
                ui.label("🔍");
                ui.text_edit_singleline(&mut self.hierarchy_search);
                if ui.button("✖").clicked() {
                    self.hierarchy_search.clear();
                }
            });

            ui.horizontal(|ui| {
                if ui.button("Add Mesh").clicked() {
                    self.add_default_mesh(scene);
                }
                if ui.button("Add Light").clicked() {
                    self.add_default_light(scene);
                }
                if ui.button("Add Sprite").clicked() {
                    self.add_default_sprite(scene);
                }
            });

            ui.horizontal(|ui| {
                if ui.button("Expand All").clicked() {
                    self.hierarchy_force_state = Some(true);
                }
                if ui.button("Collapse All").clicked() {
                    self.hierarchy_force_state = Some(false);
                }
            });

            ui.separator();

            let search = self.hierarchy_search.clone();
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.draw_node_tree_recursive(
                    ui,
                    scene,
                    scene.root,
                    &search,
                );
            });

        self.hierarchy_force_state = None;

            if let Some(selected_key) = self.selected_node {
                if ui.button("Delete Selected").clicked() {
                    self.delete_node(scene, selected_key);
                    self.selected_node = None;
                }
            }
        });
    }

    fn add_default_mesh(&mut self, scene: &mut Scene) {
        let mut new_node = Node {
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
        new_node
            .components
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
        self.node_to_add = Some((scene.root, new_node));
    }

    fn add_default_light(&mut self, scene: &mut Scene) {
        let mut new_node = Node {
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
        new_node
            .components
            .push(Box::new(spark_core::scene::LightComponent {
                light_type: spark_core::scene::LightType::Point,
                color: spark_math::Vec3::ONE,
                intensity: 1.0,
                range: 10.0,
                spot_inner_angle: 30.0,
                spot_outer_angle: 45.0,
            }));
        self.node_to_add = Some((scene.root, new_node));
    }

    fn add_default_sprite(&mut self, scene: &mut Scene) {
        let mut new_node = Node {
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
        new_node
            .components
            .push(Box::new(spark_core::scene::SpriteComponent {
                texture_handle: None,
                color: [1.0, 1.0, 1.0, 1.0],
                flip_x: false,
                flip_y: false,
                size: spark_math::Vec2::new(1.0, 1.0),
            }));
        self.node_to_add = Some((scene.root, new_node));
    }

    fn delete_node(&mut self, scene: &mut Scene, key: NodeKey) {
        if let Some(node) = scene.nodes.get(key) {
            if let Some(parent_key) = node.parent {
                if let Some(parent) = scene.nodes.get_mut(parent_key) {
                    parent.children.retain(|&k| k != key);
                }
            }
        }
        scene.nodes.remove(key);
    }

    /// Отрисовывает панель инспектора для выбранного узла сцены.
    /// Draws the inspector panel for the currently selected scene node.
    fn draw_inspector_panel(
        &mut self,
        scene: &mut Scene,
        _renderer: &mut spark_renderer::Renderer,
        asset_manager: &mut spark_core::asset::AssetManager,
    ) {
        let ctx = self.egui_ctx.clone();
        egui::SidePanel::right("inspector").show(&ctx, |ui| {
            ui.heading("Inspector");
            ui.separator();
            if let Some(selected_key) = self.selected_node {
                let mut changed_transform = None;
                if let Some(node) = scene.nodes.get_mut(selected_key) {
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.label("Name:");
                            ui.text_edit_singleline(&mut node.name);
                        });
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut node.visible, "Visible");
                            ui.checkbox(&mut node.locked, "Locked");
                        });
                    });

                    ui.add_space(8.0);
                    ui.group(|ui| {
                        changed_transform = self.draw_transform_editor(ui, node);
                    });

                    ui.add_space(8.0);
                    ui.group(|ui| {
                        self.draw_component_list(ui, node, asset_manager, selected_key);
                    });
                    ui.separator();
                    ui.label("Gizmo Mode");
                    ui.horizontal(|ui| {
                        ui.selectable_value(
                            &mut self.gizmo_mode,
                            egui_gizmo::GizmoMode::Translate,
                            "T",
                        );
                        ui.selectable_value(
                            &mut self.gizmo_mode,
                            egui_gizmo::GizmoMode::Rotate,
                            "R",
                        );
                        ui.selectable_value(
                            &mut self.gizmo_mode,
                            egui_gizmo::GizmoMode::Scale,
                            "S",
                        );
                    });
                }

                if let Some((old, new)) = changed_transform {
                    self.execute_command(
                        Box::new(TransformCommand {
                            node_key: selected_key,
                            old_transform: old,
                            new_transform: new,
                        }),
                        scene,
                    );
                }
            } else {
                ui.label("Select a node to inspect");
            }
        });
    }

    fn draw_vec3_editor(ui: &mut Ui, label: &str, vec: &mut spark_math::Vec3) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label(label);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.style_mut().spacing.item_spacing.x = 4.0;
                let z_resp = ui.add(egui::DragValue::new(&mut vec.z).speed(0.1).prefix("Z:"));
                if z_resp.changed() {
                    changed = true;
                }
                let y_resp = ui.add(egui::DragValue::new(&mut vec.y).speed(0.1).prefix("Y:"));
                if y_resp.changed() {
                    changed = true;
                }
                let x_resp = ui.add(egui::DragValue::new(&mut vec.x).speed(0.1).prefix("X:"));
                if x_resp.changed() {
                    changed = true;
                }
            });
        });
        changed
    }

    fn draw_transform_editor(
        &mut self,
        ui: &mut Ui,
        node: &mut Node,
    ) -> Option<(spark_math::Mat4, spark_math::Mat4)> {
        ui.vertical_centered(|ui| {
            ui.heading("Transform");
        });
        let (mut scale, mut rotation, mut translation) =
            node.local_transform.to_scale_rotation_translation();

        let mut changed = false;
        let initial_transform = node.local_transform;

        changed |= Self::draw_vec3_editor(ui, "Position", &mut translation);

        let mut euler = rotation.to_euler(spark_math::EulerRot::XYZ);
        let mut deg_euler = spark_math::Vec3::new(
            euler.0.to_degrees(),
            euler.1.to_degrees(),
            euler.2.to_degrees(),
        );
        if Self::draw_vec3_editor(ui, "Rotation", &mut deg_euler) {
            euler.0 = deg_euler.x.to_radians();
            euler.1 = deg_euler.y.to_radians();
            euler.2 = deg_euler.z.to_radians();
            rotation =
                spark_math::Quat::from_euler(spark_math::EulerRot::XYZ, euler.0, euler.1, euler.2);
            changed = true;
        }

        changed |= Self::draw_vec3_editor(ui, "Scale", &mut scale);

        if changed {
            node.local_transform =
                spark_math::Mat4::from_scale_rotation_translation(scale, rotation, translation);
        }

        if ui.input(|i| i.pointer.any_released()) && changed {
            return Some((initial_transform, node.local_transform));
        }
        None
    }

    fn draw_component_editor(
        &mut self,
        ui: &mut Ui,
        component: &mut Box<dyn spark_core::scene::Component>,
        asset_manager: &mut spark_core::asset::AssetManager,
    ) {
        let any = component.as_any_mut();
        if let Some(light) = any.downcast_mut::<spark_core::scene::LightComponent>() {
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label("Type:");
                    egui::ComboBox::from_id_source("light_type")
                        .selected_text(format!("{:?}", light.light_type))
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut light.light_type,
                                spark_core::scene::LightType::Directional,
                                "Directional",
                            );
                            ui.selectable_value(
                                &mut light.light_type,
                                spark_core::scene::LightType::Point,
                                "Point",
                            );
                            ui.selectable_value(
                                &mut light.light_type,
                                spark_core::scene::LightType::Spot,
                                "Spot",
                            );
                        });
                });
                ui.horizontal(|ui| {
                    ui.label("Color:");
                    ui.color_edit_button_rgb(light.color.as_mut());
                });
                ui.horizontal(|ui| {
                    ui.label("Intensity:");
                    ui.add(
                        egui::DragValue::new(&mut light.intensity)
                            .speed(0.1)
                            .clamp_range(0.0..=f32::MAX),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Range:");
                    ui.add(
                        egui::DragValue::new(&mut light.range)
                            .speed(0.1)
                            .clamp_range(0.0..=f32::MAX),
                    );
                });
                if let spark_core::scene::LightType::Spot = light.light_type {
                    ui.horizontal(|ui| {
                        ui.label("Inner Angle:");
                        ui.add(egui::Slider::new(&mut light.spot_inner_angle, 0.0..=180.0));
                    });
                    ui.horizontal(|ui| {
                        ui.label("Outer Angle:");
                        ui.add(egui::Slider::new(&mut light.spot_outer_angle, 0.0..=180.0));
                    });
                }
            });
        } else if let Some(mesh) = any.downcast_mut::<spark_core::scene::MeshComponent>() {
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label("Radius:");
                    ui.add(egui::DragValue::new(&mut mesh.bounding_radius).speed(0.1));
                });
                ui.horizontal(|ui| {
                    ui.label("Material:");
                    let mat_name = mesh.material_index.and_then(|idx| {
                        asset_manager.materials.get(spark_core::resource::Handle::new(idx)).map(|m| m.name.clone())
                    }).unwrap_or_else(|| "None".to_string());

                    egui::ComboBox::from_id_source(format!("mesh_mat_{:?}", mesh as *const _))
                        .selected_text(mat_name)
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut mesh.material_index, None, "None");
                            for idx in 0..asset_manager.materials.assets_len() {
                                if let Some(mat) = asset_manager.materials.get(spark_core::resource::Handle::new(idx as u32)) {
                                    ui.selectable_value(&mut mesh.material_index, Some(idx as u32), &mat.name);
                                }
                            }
                        });
                });
            });
        } else if let Some(camera) = any.downcast_mut::<spark_core::scene::CameraComponent>() {
            ui.vertical(|ui| {
                ui.checkbox(&mut camera.orthographic, "Orthographic");
                if camera.orthographic {
                    ui.horizontal(|ui| {
                        ui.label("Ortho Size:");
                        ui.add(
                            egui::DragValue::new(&mut camera.ortho_size)
                                .speed(0.1)
                                .clamp_range(0.0..=f32::MAX),
                        );
                    });
                } else {
                    ui.horizontal(|ui| {
                        ui.label("FOV:");
                        ui.add(egui::Slider::new(&mut camera.fov, 1.0..=179.0));
                    });
                }
                ui.horizontal(|ui| {
                    ui.label("Near:");
                    ui.add(
                        egui::DragValue::new(&mut camera.near)
                            .speed(0.01)
                            .clamp_range(0.0..=f32::MAX),
                    );
                });
                ui.horizontal(|ui| {
                    ui.label("Far:");
                    ui.add(
                        egui::DragValue::new(&mut camera.far)
                            .speed(1.0)
                            .clamp_range(0.0..=f32::MAX),
                    );
                });
            });
        } else if let Some(sprite) = any.downcast_mut::<spark_core::scene::SpriteComponent>() {
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label("Color:");
                    ui.color_edit_button_rgba_unmultiplied(&mut sprite.color);
                });
                ui.horizontal(|ui| {
                    ui.label("Size:");
                    ui.add(
                        egui::DragValue::new(&mut sprite.size.x)
                            .speed(0.1)
                            .prefix("W:"),
                    );
                    ui.add(
                        egui::DragValue::new(&mut sprite.size.y)
                            .speed(0.1)
                            .prefix("H:"),
                    );
                });
                ui.horizontal(|ui| {
                    ui.checkbox(&mut sprite.flip_x, "Flip X");
                    ui.checkbox(&mut sprite.flip_y, "Flip Y");
                });
            });
        }
    }

    fn draw_node_tree_recursive(
        &mut self,
        ui: &mut egui::Ui,
        scene: &mut Scene,
        node_key: NodeKey,
        search: &str,
    ) {
        let (label, children, icon) = if let Some(node) = scene.nodes.get(node_key) {
            let icon = if node
                .components
                .iter()
                .any(|c| c.as_any().is::<spark_core::scene::CameraComponent>())
            {
                "🎥"
            } else if node
                .components
                .iter()
                .any(|c| c.as_any().is::<spark_core::scene::LightComponent>())
            {
                "💡"
            } else if node
                .components
                .iter()
                .any(|c| c.as_any().is::<spark_core::scene::SpriteComponent>())
            {
                "🖼"
            } else if node
                .components
                .iter()
                .any(|c| c.as_any().is::<spark_core::scene::MeshComponent>())
            {
                "📦"
            } else {
                "⭕"
            };
            (node.name.clone(), node.children.clone(), icon)
        } else {
            return;
        };

        let is_selected = Some(node_key) == self.selected_node;
        let matches_search =
            search.is_empty() || label.to_lowercase().contains(&search.to_lowercase());

        if matches_search {
            ui.horizontal(|ui| {
                ui.label(icon);

                let mut visible = true;
                let mut locked = false;
                if let Some(node) = scene.nodes.get_mut(node_key) {
                    visible = node.visible;
                    locked = node.locked;
                }

                if ui.button(if visible { "👁" } else { "👓" }).clicked() {
                    if let Some(node) = scene.nodes.get_mut(node_key) {
                        node.visible = !node.visible;
                    }
                }
                if ui.button(if locked { "🔒" } else { "🔓" }).clicked() {
                    if let Some(node) = scene.nodes.get_mut(node_key) {
                        node.locked = !node.locked;
                    }
                }

                let response = ui.selectable_label(is_selected, &label);

                let mut delete_requested = false;
                response.context_menu(|ui| {
                    if ui.button("Duplicate").clicked() {
                        self.node_to_duplicate = Some(node_key);
                        ui.close_menu();
                    }
                    if ui.button("Delete").clicked() {
                        delete_requested = true;
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Add Child Mesh").clicked() {
                        self.node_to_add_child = Some((node_key, NodeType::Mesh));
                        ui.close_menu();
                    }
                    if ui.button("Add Child Light").clicked() {
                        self.node_to_add_child = Some((node_key, NodeType::Light));
                        ui.close_menu();
                    }
                    if ui.button("Add Child Sprite").clicked() {
                        self.node_to_add_child = Some((node_key, NodeType::Sprite));
                        ui.close_menu();
                    }
                });

                if delete_requested {
                    self.node_to_delete = Some(node_key);
                }

                if response.clicked() {
                    self.selected_node = Some(node_key);
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
                            if k == dragged_key {
                                can_reparent = false;
                                break;
                            }
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
            });
        }

        for &child_key in &children {
            let id = ui.make_persistent_id(child_key);
            if let Some(force) = self.hierarchy_force_state {
                egui::collapsing_header::CollapsingState::load_with_default_open(ui.ctx(), id, !force).set_open(force);
            }

            ui.indent(node_key, |ui| {
                self.draw_node_tree_recursive(
                    ui,
                    scene,
                    child_key,
                    search,
                );
            });
        }
    }

    pub fn draw_viewport(
        &mut self,
        scene: &mut Scene,
        renderer: &mut spark_renderer::Renderer,
        _fps: f32,
    ) {
        if let Some(texture_id) = self.viewport_texture_id {
            let ctx = self.egui_ctx.clone();
            egui::CentralPanel::default()
                .frame(egui::Frame::none().fill(egui::Color32::BLACK))
                .show(&ctx, |ui| {
                    // Performance Overlay (now just extra info)
                    let painter = ui.painter();
                    let rect = ui.max_rect();
                    painter.rect_filled(
                        egui::Rect::from_min_size(
                            rect.min + egui::vec2(10.0, 10.0),
                            egui::vec2(150.0, 80.0),
                        ),
                        5.0,
                        egui::Color32::from_black_alpha(150),
                    );
                    ui.put(
                        egui::Rect::from_min_size(
                            rect.min + egui::vec2(20.0, 20.0),
                            egui::vec2(130.0, 60.0),
                        ),
                        egui::Label::new(
                            egui::RichText::new(format!(
                                "Viewport\nObjects: {}\nTris: TODO\nDraw Calls: TODO",
                                renderer.last_object_count
                            ))
                            .color(egui::Color32::WHITE)
                            .size(12.0),
                        ),
                    );

                    // Keyboard shortcuts
                    if ui.input(|i| i.key_pressed(egui::Key::T)) {
                        self.gizmo_mode = egui_gizmo::GizmoMode::Translate;
                    }
                    if ui.input(|i| i.key_pressed(egui::Key::R)) {
                        self.gizmo_mode = egui_gizmo::GizmoMode::Rotate;
                    }
                    if ui.input(|i| i.key_pressed(egui::Key::S)) {
                        self.gizmo_mode = egui_gizmo::GizmoMode::Scale;
                    }

                    if ui.input(|i| i.key_pressed(egui::Key::D) && i.modifiers.command) {
                        if let Some(key) = self.selected_node {
                            self.node_to_duplicate = Some(key);
                        }
                    }

                    if ui.input(|i| i.key_pressed(egui::Key::F)) {
                        if let Some(key) = self.selected_node {
                            if let Some(node) = scene.nodes.get(key) {
                                let pos = node.global_transform.w_axis.xyz();
                                let forward = spark_math::Vec3::new(
                                    self.camera_rot.x.cos() * self.camera_rot.y.cos(),
                                    self.camera_rot.y.sin(),
                                    self.camera_rot.x.sin() * self.camera_rot.y.cos(),
                                ).normalize();
                                self.camera_pos = pos - forward * 5.0;
                            }
                        }
                    }

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
                        )
                        .normalize();
                        let right = forward.cross(spark_math::Vec3::Y).normalize();

                        if ui.input(|i| i.key_down(egui::Key::W)) {
                            self.camera_pos += forward * speed;
                        }
                        if ui.input(|i| i.key_down(egui::Key::S)) {
                            self.camera_pos -= forward * speed;
                        }
                        if ui.input(|i| i.key_down(egui::Key::A)) {
                            self.camera_pos -= right * speed;
                        }
                        if ui.input(|i| i.key_down(egui::Key::D)) {
                            self.camera_pos += right * speed;
                        }

                        scene.last_view_matrix = spark_math::Mat4::look_at_rh(
                            self.camera_pos,
                            self.camera_pos + forward,
                            spark_math::Vec3::Y,
                        );
                    }

                    // Picking
                    if response.clicked_by(egui::PointerButton::Primary) {
                        if let Some(pointer_pos) = ui.input(|i| i.pointer.interact_pos()) {
                            if rect.contains(pointer_pos) {
                                let normalized_x = (pointer_pos.x - rect.min.x) / rect.width();
                                let normalized_y = (pointer_pos.y - rect.min.y) / rect.height();

                                let view = scene.last_view_matrix;

                                let mut fov = 45.0f32.to_radians();
                                let mut near = 0.1;
                                let mut far = 100.0;
                                let mut ortho = false;
                                let mut ortho_size = 5.0;

                                // Try to find active camera component
                                for node in scene.nodes.values() {
                                    for comp in &node.components {
                                        if let Some(cam) = comp.as_any().downcast_ref::<spark_core::scene::CameraComponent>() {
                                            fov = cam.fov.to_radians();
                                            near = cam.near;
                                            far = cam.far;
                                            ortho = cam.orthographic;
                                            ortho_size = cam.ortho_size;
                                        }
                                    }
                                }

                                let projection = if ortho {
                                    let aspect = size.x / size.y;
                                    spark_math::Mat4::orthographic_rh(
                                        -ortho_size * aspect,
                                        ortho_size * aspect,
                                        -ortho_size,
                                        ortho_size,
                                        near,
                                        far,
                                    )
                                } else {
                                    spark_math::Mat4::perspective_rh(fov, size.x / size.y, near, far)
                                };
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
                        let (view, projection, model, parent_key, locked) = {
                            let node = scene.nodes.get(selected_key).unwrap();
                            let view = scene.last_view_matrix;

                            let mut fov = 45.0f32.to_radians();
                            let mut near = 0.1;
                            let mut far = 100.0;
                            let mut ortho = false;
                            let mut ortho_size = 5.0;

                            for node in scene.nodes.values() {
                                for comp in &node.components {
                                    if let Some(cam) = comp.as_any().downcast_ref::<spark_core::scene::CameraComponent>() {
                                        fov = cam.fov.to_radians();
                                        near = cam.near;
                                        far = cam.far;
                                        ortho = cam.orthographic;
                                        ortho_size = cam.ortho_size;
                                    }
                                }
                            }

                            let projection = if ortho {
                                let aspect = size.x / size.y;
                                spark_math::Mat4::orthographic_rh(
                                    -ortho_size * aspect,
                                    ortho_size * aspect,
                                    -ortho_size,
                                    ortho_size,
                                    near,
                                    far,
                                )
                            } else {
                                spark_math::Mat4::perspective_rh(fov, size.x / size.y, near, far)
                            };
                            (
                                view,
                                projection,
                                node.global_transform,
                                node.parent,
                                node.locked,
                            )
                        };

                        if locked {
                            ui.label(
                                egui::RichText::new("Node is locked").color(egui::Color32::YELLOW),
                            );
                        }

                        let gizmo = Gizmo::new("scene_gizmo")
                            .view_matrix(view.to_cols_array_2d().into())
                            .projection_matrix(projection.to_cols_array_2d().into())
                            .model_matrix(model.to_cols_array_2d().into())
                            .mode(self.gizmo_mode)
                            .orientation(if self.gizmo_local {
                                egui_gizmo::GizmoOrientation::Local
                            } else {
                                egui_gizmo::GizmoOrientation::Global
                            })
                            .snapping(self.snap_enabled)
                            .snap_distance(self.snap_distance)
                            .snap_angle(self.snap_distance.to_radians())
                            .snap_scale(self.snap_distance)
                            .viewport(rect);

                        if let Some(response) = gizmo.interact(ui) {
                            if locked {
                                return;
                            }
                            if self.initial_gizmo_transform.is_none() {
                                self.initial_gizmo_transform =
                                    Some(scene.nodes.get(selected_key).unwrap().local_transform);
                            }

                            let m = response.transform();
                            let new_model = spark_math::Mat4::from_cols_array_2d(&[
                                m.x.into(),
                                m.y.into(),
                                m.z.into(),
                                m.w.into(),
                            ]);

                            let parent_global_inv = if let Some(pk) = parent_key {
                                scene
                                    .nodes
                                    .get(pk)
                                    .map(|p| p.global_transform.inverse())
                                    .unwrap_or(spark_math::Mat4::IDENTITY)
                            } else {
                                spark_math::Mat4::IDENTITY
                            };

                            let new_local = parent_global_inv * new_model;

                            if let Some(node) = scene.nodes.get_mut(selected_key) {
                                node.local_transform = new_local;
                            }
                        } else if ui.input(|i| i.pointer.any_released()) {
                            if let Some(old_transform) = self.initial_gizmo_transform.take() {
                                if let Some(node) = scene.nodes.get(selected_key) {
                                    let new_transform = node.local_transform;
                                    if old_transform != new_transform {
                                        self.execute_command(
                                            Box::new(TransformCommand {
                                                node_key: selected_key,
                                                old_transform,
                                                new_transform,
                                            }),
                                            scene,
                                        );
                                    }
                                }
                            }
                        }
                    }
                });
        }
    }
}
