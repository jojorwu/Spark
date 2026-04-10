use crate::ui::{DeleteNodeCommand, EditorUI, NodeType};
use spark_core::scene::{Node, NodeKey, Scene};

impl EditorUI {
    pub fn draw_hierarchy_panel(&mut self, scene: &mut Scene) {
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
                self.draw_node_tree_recursive(ui, scene, scene.root, &search);
            });

            self.hierarchy_force_state = None;

            if let Some(selected_key) = self.selected_node {
                if ui.button("Delete Selected").clicked() {
                    self.execute_command(
                        Box::new(DeleteNodeCommand {
                            node_key: selected_key,
                            node: None,
                            parent_key: None,
                            index_in_parent: 0,
                        }),
                        scene,
                    );
                    self.selected_node = None;
                }
            }
        });
    }

    pub fn draw_node_tree_recursive(
        &mut self,
        ui: &mut egui::Ui,
        scene: &mut Scene,
        node_key: NodeKey,
        search: &str,
    ) {
        let (label, children, icon) = if let Some(node) = scene.nodes.get(node_key) {
            let icon = self.get_node_icon(node);
            (node.name.clone(), node.children.clone(), icon)
        } else {
            return;
        };

        let is_selected = Some(node_key) == self.selected_node;
        let matches_search =
            search.is_empty() || label.to_lowercase().contains(&search.to_lowercase());

        if matches_search {
            self.draw_node_item(ui, scene, node_key, &label, is_selected, icon);
        }

        for &child_key in &children {
            let id = ui.make_persistent_id(child_key);
            if let Some(force) = self.hierarchy_force_state {
                egui::collapsing_header::CollapsingState::load_with_default_open(
                    ui.ctx(),
                    id,
                    !force,
                )
                .set_open(force);
            }

            ui.indent(node_key, |ui| {
                self.draw_node_tree_recursive(ui, scene, child_key, search);
            });
        }
    }

    pub fn get_node_icon(&self, node: &Node) -> &'static str {
        if node
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
        }
    }

    pub fn draw_node_item(
        &mut self,
        ui: &mut egui::Ui,
        scene: &mut Scene,
        node_key: NodeKey,
        label: &str,
        is_selected: bool,
        icon: &str,
    ) {
        ui.horizontal(|ui| {
            ui.label(icon);

            let (visible, locked) = if let Some(node) = scene.nodes.get_mut(node_key) {
                (node.visible, node.locked)
            } else {
                (true, false)
            };

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

            let response = ui.selectable_label(is_selected, label);

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

    pub fn add_default_mesh(&mut self, scene: &mut Scene) {
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

    pub fn add_default_light(&mut self, scene: &mut Scene) {
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

    pub fn add_default_sprite(&mut self, scene: &mut Scene) {
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

    pub fn add_primitive_cube(&mut self, scene: &mut Scene) {
        let mut new_node = Node {
            name: "Cube".to_string(),
            ..Default::default()
        };
        new_node
            .components
            .push(Box::new(spark_core::scene::MeshComponent {
                vertex_count: 24,
                index_count: 36,
                first_index: 0,
                vertex_offset: 0,
                texture_handle: None,
                material_index: Some(0),
                bounding_radius: 1.0,
                skin_index: None,
            }));
        self.node_to_add = Some((scene.root, new_node));
    }

    pub fn add_primitive_plane(&mut self, scene: &mut Scene) {
        let mut new_node = Node {
            name: "Plane".to_string(),
            ..Default::default()
        };
        // Cube has 24 vertices, 36 indices
        new_node
            .components
            .push(Box::new(spark_core::scene::MeshComponent {
                vertex_count: 4,
                index_count: 6,
                first_index: 36,
                vertex_offset: 24,
                texture_handle: None,
                material_index: Some(0),
                bounding_radius: 10.0,
                skin_index: None,
            }));
        self.node_to_add = Some((scene.root, new_node));
    }

    pub fn add_primitive_sphere(&mut self, scene: &mut Scene) {
        let mut new_node = Node {
            name: "Sphere".to_string(),
            ..Default::default()
        };
        // Plane has 4 vertices, 6 indices
        // Total indices before sphere: 36 + 6 = 42
        // Total vertices before sphere: 24 + 4 = 28
        let segments = 32;
        let v_count = (segments + 1) * (segments + 1);
        let i_count = segments * segments * 6;
        new_node
            .components
            .push(Box::new(spark_core::scene::MeshComponent {
                vertex_count: v_count,
                index_count: i_count,
                first_index: 42,
                vertex_offset: 28,
                texture_handle: None,
                material_index: Some(0),
                bounding_radius: 0.5,
                skin_index: None,
            }));
        self.node_to_add = Some((scene.root, new_node));
    }

    pub fn add_directional_light(&mut self, scene: &mut Scene) {
        let mut new_node = Node {
            name: "Directional Light".to_string(),
            ..Default::default()
        };
        new_node
            .components
            .push(Box::new(spark_core::scene::LightComponent {
                light_type: spark_core::scene::LightType::Directional,
                color: spark_math::Vec3::ONE,
                intensity: 5.0,
                range: 0.0,
                spot_inner_angle: 0.0,
                spot_outer_angle: 0.0,
            }));
        self.node_to_add = Some((scene.root, new_node));
    }
}
