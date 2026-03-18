use egui::{Context, Visuals};
use egui_winit::State;
use winit::window::Window;
use winit::event::WindowEvent;

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
                        ui.close_menu();
                    }
                    if ui.button("Open").clicked() {
                        ui.close_menu();
                    }
                    if ui.button("Save").clicked() {
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
                    ui.text_edit_singleline(&mut node.name);
                    ui.separator();
                    ui.label("Transform (Local)");
                    // Simplified: just show that we can access it
                    ui.label(format!("Matrix: {:?}", node.local_transform));
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

    pub fn draw_viewport(&mut self) {
        if let Some(texture_id) = self.viewport_texture_id {
            egui::Window::new("Viewport").show(&self.egui_ctx, |ui| {
                let size = ui.available_size();
                ui.image(egui::load::SizedTexture::new(texture_id, size));
            });
        }
    }
}
