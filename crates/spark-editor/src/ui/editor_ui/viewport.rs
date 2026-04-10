use crate::ui::{EditorUI, TransformCommand};
use egui_gizmo::Gizmo;
use spark_core::scene::Scene;
use spark_math::Vec4Swizzles;
use std::sync::atomic::Ordering;

impl EditorUI {
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
                            egui::vec2(140.0, 70.0),
                        ),
                        egui::Label::new(
                            egui::RichText::new(format!(
                                "Viewport\nObjects: {}\nTris: {}\nDraw Calls: {}",
                                renderer.last_object_count,
                                renderer.last_triangle_count.load(Ordering::Relaxed),
                                renderer.last_draw_calls.load(Ordering::Relaxed),
                            ))
                            .color(egui::Color32::WHITE)
                            .size(12.0),
                        ),
                    );

                    // View Options Menu
                    ui.put(
                        egui::Rect::from_min_size(
                            rect.max - egui::vec2(120.0, 40.0),
                            egui::vec2(110.0, 30.0),
                        ),
                        |ui: &mut egui::Ui| {
                            ui.menu_button("👁 View Options", |ui| {
                                ui.checkbox(&mut renderer.settings.enable_shadows, "Shadows");
                                ui.checkbox(&mut renderer.settings.enable_ssao, "SSAO");
                                ui.checkbox(&mut renderer.settings.enable_bloom, "Bloom");
                                ui.checkbox(&mut renderer.settings.enable_taa, "TAA");
                                ui.checkbox(&mut renderer.settings.enable_grid, "Grid");
                            }).response
                        },
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
                                )
                                .normalize();
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
                                        if let Some(cam) = comp
                                            .as_any()
                                            .downcast_ref::<spark_core::scene::CameraComponent>(
                                        ) {
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
                                    spark_math::Mat4::perspective_rh(
                                        fov,
                                        size.x / size.y,
                                        near,
                                        far,
                                    )
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
                            let node = scene
                                .nodes
                                .get(selected_key)
                                .expect("Selected node missing from scene");
                            let view = scene.last_view_matrix;

                            let mut fov = 45.0f32.to_radians();
                            let mut near = 0.1;
                            let mut far = 100.0;
                            let mut ortho = false;
                            let mut ortho_size = 5.0;

                            for node in scene.nodes.values() {
                                for comp in &node.components {
                                    if let Some(cam) =
                                        comp.as_any()
                                            .downcast_ref::<spark_core::scene::CameraComponent>()
                                    {
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
                                self.initial_gizmo_transform = Some(
                                    scene
                                        .nodes
                                        .get(selected_key)
                                        .expect("Selected node missing from scene")
                                        .local_transform,
                                );
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
