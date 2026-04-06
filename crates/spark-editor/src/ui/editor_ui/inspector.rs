use crate::ui::{EditorUI, TransformCommand};
use egui::Ui;
use spark_core::scene::{Node, NodeKey, Scene};

impl EditorUI {
    pub fn draw_inspector_panel(
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

    pub fn draw_transform_editor(
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

    pub fn draw_vec3_editor(ui: &mut Ui, label: &str, vec: &mut spark_math::Vec3) -> bool {
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

    pub fn draw_component_list(
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

    pub fn draw_component_editor(
        &mut self,
        ui: &mut Ui,
        component: &mut Box<dyn spark_core::scene::Component>,
        asset_manager: &mut spark_core::asset::AssetManager,
    ) {
        let any = component.as_any_mut();
        if let Some(light) = any.downcast_mut::<spark_core::scene::LightComponent>() {
            self.draw_light_editor(ui, light);
        } else if let Some(mesh) = any.downcast_mut::<spark_core::scene::MeshComponent>() {
            self.draw_mesh_editor(ui, mesh, asset_manager);
        } else if let Some(camera) = any.downcast_mut::<spark_core::scene::CameraComponent>() {
            self.draw_camera_editor(ui, camera);
        } else if let Some(sprite) = any.downcast_mut::<spark_core::scene::SpriteComponent>() {
            self.draw_sprite_editor(ui, sprite);
        }
    }

    pub fn draw_light_editor(&self, ui: &mut Ui, light: &mut spark_core::scene::LightComponent) {
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
    }

    pub fn draw_mesh_editor(
        &self,
        ui: &mut Ui,
        mesh: &mut spark_core::scene::MeshComponent,
        asset_manager: &spark_core::asset::AssetManager,
    ) {
        ui.vertical(|ui| {
            ui.horizontal(|ui| {
                ui.label("Radius:");
                ui.add(egui::DragValue::new(&mut mesh.bounding_radius).speed(0.1));
            });
            ui.horizontal(|ui| {
                ui.label("Material:");
                let mat_name = mesh
                    .material_index
                    .and_then(|idx| {
                        asset_manager
                            .materials
                            .get(spark_core::resource::Handle::new(idx))
                            .map(|m| m.name.clone())
                    })
                    .unwrap_or_else(|| "None".to_string());

                egui::ComboBox::from_id_source(format!("mesh_mat_{:?}", mesh as *const _))
                    .selected_text(mat_name)
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut mesh.material_index, None, "None");
                        for idx in 0..asset_manager.materials.assets_len() {
                            if let Some(mat) = asset_manager
                                .materials
                                .get(spark_core::resource::Handle::new(idx as u32))
                            {
                                ui.selectable_value(
                                    &mut mesh.material_index,
                                    Some(idx as u32),
                                    &mat.name,
                                );
                            }
                        }
                    });
            });
        });
    }

    pub fn draw_camera_editor(&self, ui: &mut Ui, camera: &mut spark_core::scene::CameraComponent) {
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
    }

    pub fn draw_sprite_editor(&self, ui: &mut Ui, sprite: &mut spark_core::scene::SpriteComponent) {
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
