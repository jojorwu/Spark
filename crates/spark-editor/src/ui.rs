use egui::{Context, Visuals};
use egui_winit::State;
use winit::window::Window;
use winit::event::WindowEvent;

use spark_core::scene::{Scene, NodeKey};

pub struct EditorUI {
    pub egui_ctx: Context,
    pub egui_state: State,
    pub selected_node: Option<NodeKey>,
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
                        ui.close_menu();
                    }
                    if ui.button("Redo").clicked() {
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
            Self::draw_node_tree(ui, scene, scene.root, &mut self.selected_node);
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
}
