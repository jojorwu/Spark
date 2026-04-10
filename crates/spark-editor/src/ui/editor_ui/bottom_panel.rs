use crate::ui::{BottomTab, EditorUI};
use egui::Ui;
use spark_core::scene::Scene;

impl EditorUI {
    pub fn draw_bottom_panel(
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
                ui.selectable_value(
                    &mut self.active_bottom_tab,
                    BottomTab::Console,
                    "📝 Console",
                );
                ui.selectable_value(&mut self.active_bottom_tab, BottomTab::Assets, "📁 Assets");
                ui.selectable_value(
                    &mut self.active_bottom_tab,
                    BottomTab::Materials,
                    "🎨 Materials",
                );
                ui.selectable_value(
                    &mut self.active_bottom_tab,
                    BottomTab::Settings,
                    "⚙ Settings",
                );
                ui.selectable_value(
                    &mut self.active_bottom_tab,
                    BottomTab::Statistics,
                    "📊 Statistics",
                );
            });
            ui.separator();

            match self.active_bottom_tab {
                BottomTab::Console => self.draw_console_tab(ui),
                BottomTab::Assets => {
                    self.draw_assets_tab(ui, scene, resource_manager, renderer, asset_manager)
                }
                BottomTab::Materials => {
                    self.draw_materials_tab(ui, asset_manager, resource_manager)
                }
                BottomTab::Settings => self.draw_settings_tab(ui, renderer, project),
                BottomTab::Statistics => self.draw_statistics_tab(ui, renderer, fps),
            }
        });
    }

    pub fn draw_console_tab(&mut self, ui: &mut Ui) {
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
                    self.logs.lock().expect("Failed to lock logs").clear();
                }
            });
        });
        ui.separator();
        egui::ScrollArea::vertical()
            .stick_to_bottom(true)
            .show(ui, |ui| {
                let logs = self.logs.lock().expect("Failed to lock logs");
                for log in logs.iter() {
                    let (color, visible) = if log.contains("ERROR") {
                        (egui::Color32::LIGHT_RED, self.log_filter_error)
                    } else if log.contains("WARN") {
                        (egui::Color32::KHAKI, self.log_filter_warn)
                    } else {
                        (egui::Color32::LIGHT_GRAY, self.log_filter_info)
                    };

                    let matches_search = self.log_search.is_empty()
                        || log.to_lowercase().contains(&self.log_search.to_lowercase());

                    if visible && matches_search {
                        ui.label(egui::RichText::new(log).color(color).monospace());
                    }
                }
            });
    }

    pub fn draw_assets_tab(
        &mut self,
        ui: &mut Ui,
        scene: &mut Scene,
        resource_manager: &mut spark_core::resource::ResourceManager,
        renderer: &mut spark_renderer::Renderer,
        asset_manager: &mut spark_core::asset::AssetManager,
    ) {
        if let Some((old_path, mut new_name)) = self.asset_rename_state.take() {
            let mut close = false;
            egui::Window::new("Rename Asset")
                .collapsible(false)
                .resizable(false)
                .show(ui.ctx(), |ui| {
                    ui.horizontal(|ui| {
                        ui.label("New Name:");
                        ui.text_edit_singleline(&mut new_name);
                    });
                    ui.horizontal(|ui| {
                        if ui.button("Rename").clicked() {
                            let new_path = old_path.with_file_name(&new_name);
                            if let Err(e) = std::fs::rename(&old_path, new_path) {
                                log::error!("Failed to rename asset: {}", e);
                            }
                            close = true;
                        }
                        if ui.button("Cancel").clicked() {
                            close = true;
                        }
                    });
                });
            if !close {
                self.asset_rename_state = Some((old_path, new_name));
            }
        }

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
                    let label = path
                        .file_name()
                        .expect("Asset entry has no file name")
                        .to_string_lossy();

                    if !self.asset_search.is_empty()
                        && !label
                            .to_lowercase()
                            .contains(&self.asset_search.to_lowercase())
                    {
                        continue;
                    }

                    ui.horizontal(|ui| {
                        let response = if path.is_dir() {
                            ui.selectable_label(false, format!("📁 {}", label))
                        } else {
                            let is_gltf = path
                                .extension()
                                .is_some_and(|ext| ext == "gltf" || ext == "glb");
                            let is_img = path
                                .extension()
                                .is_some_and(|ext| ext == "png" || ext == "jpg");
                            let icon = if is_gltf {
                                "📦"
                            } else if is_img {
                                "🖼"
                            } else {
                                "📄"
                            };
                            ui.selectable_label(false, format!("{} {}", icon, label))
                        };

                        if response.clicked() {
                            if path.is_dir() {
                                dir_to_set = Some(path.to_path_buf());
                            } else {
                                let is_gltf = path
                                    .extension()
                                    .is_some_and(|ext| ext == "gltf" || ext == "glb");
                                if is_gltf {
                                    asset_to_load = Some(path.to_path_buf());
                                }
                            }
                        }

                        response.context_menu(|ui| {
                            if ui.button("Rename").clicked() {
                                self.asset_rename_state = Some((
                                    path.clone(),
                                    path.file_name().unwrap().to_string_lossy().into_owned(),
                                ));
                                ui.close_menu();
                            }
                            if ui.button("Delete").clicked() {
                                if path.is_dir() {
                                    let _ = std::fs::remove_dir_all(&path);
                                } else {
                                    let _ = std::fs::remove_file(&path);
                                }
                                ui.close_menu();
                            }
                            if !path.is_dir() {
                                let is_img = path
                                    .extension()
                                    .is_some_and(|ext| ext == "png" || ext == "jpg");
                                if is_img && ui.button("Create Material from Texture").clicked() {
                                    let mat_path = path.with_extension("json");
                                    let name = path.file_stem().unwrap().to_string_lossy();
                                    let mat = spark_core::asset::Material {
                                        name: name.into_owned(),
                                        albedo_factor: [1.0, 1.0, 1.0, 1.0],
                                        metallic_factor: 0.0,
                                        roughness_factor: 0.5,
                                        emissive_factor: [0.0, 0.0, 0.0, 1.0],
                                        is_transparent: false,
                                        albedo_texture: None, // Will be linked manually for now or I can try to find handle
                                        normal_texture: None,
                                        metallic_roughness_texture: None,
                                    };
                                    if let Ok(json) = serde_json::to_string_pretty(&mat) {
                                        if let Err(e) = std::fs::write(&mat_path, json) {
                                            log::error!("Failed to create material file: {}", e);
                                        } else {
                                            self.status_message = format!(
                                                "Material created: {:?}",
                                                mat_path.file_name().unwrap()
                                            );
                                        }
                                    }
                                    ui.close_menu();
                                }
                            }
                        });
                    });
                }
            }
        });

        if let Some(dir) = dir_to_set {
            self.asset_current_dir = dir;
        }
        if let Some(path) = asset_to_load {
            if let Err(e) = resource_manager.load_scene(path, scene, renderer, asset_manager) {
                log::error!("Failed to load scene: {}", e);
            }
        }
    }

    pub fn draw_materials_tab(
        &mut self,
        ui: &mut Ui,
        asset_manager: &mut spark_core::asset::AssetManager,
        resource_manager: &mut spark_core::resource::ResourceManager,
    ) {
        ui.horizontal(|ui| {
            ui.label("🔍 Search Materials:");
            ui.text_edit_singleline(&mut self.material_search);
        });
        ui.separator();

        egui::ScrollArea::vertical().show(ui, |ui| {
            let mat_indices: Vec<_> = (0..asset_manager.materials.assets_len()).collect();
            for idx in mat_indices {
                let handle = spark_core::resource::Handle::new(idx as u32);

                let mut matches = true;
                if let Some(mat) = asset_manager.materials.get(handle) {
                    if !self.material_search.is_empty()
                        && !mat
                            .name
                            .to_lowercase()
                            .contains(&self.material_search.to_lowercase())
                    {
                        matches = false;
                    }
                } else {
                    matches = false;
                }

                if matches {
                    let mat_name = asset_manager
                        .materials
                        .get(handle)
                        .expect("Material handle invalid")
                        .name
                        .clone();
                    ui.collapsing(format!("Material: {}", mat_name), |ui| {
                        if let Some(mat) = asset_manager.materials.get_mut(handle) {
                            Self::draw_material_editor_static(
                                ui,
                                mat,
                                idx,
                                &asset_manager.texture_path_map,
                                resource_manager,
                            );
                        }
                    });
                }
            }
        });
    }

    pub fn draw_material_editor_static(
        ui: &mut Ui,
        mat: &mut spark_core::asset::Material,
        idx: usize,
        texture_path_map: &std::collections::HashMap<
            std::path::PathBuf,
            spark_core::resource::Handle<spark_renderer::vulkan::texture::Texture>,
        >,
        resource_manager: &mut spark_core::resource::ResourceManager,
    ) {
        let mut changed = false;

        ui.horizontal(|ui| {
            ui.label("Name:");
            changed |= ui.text_edit_singleline(&mut mat.name).changed();
        });

        ui.group(|ui| {
            ui.label(egui::RichText::new("Basic Properties").strong());
            ui.horizontal(|ui| {
                ui.label("Albedo:");
                changed |= ui
                    .color_edit_button_rgba_unmultiplied(&mut mat.albedo_factor)
                    .changed();
            });
            ui.horizontal(|ui| {
                ui.label("Emissive:");
                changed |= ui
                    .color_edit_button_rgba_unmultiplied(&mut mat.emissive_factor)
                    .changed();
            });
            changed |= ui
                .checkbox(&mut mat.is_transparent, "Transparent")
                .changed();
        });

        ui.add_space(4.0);

        ui.group(|ui| {
            ui.label(egui::RichText::new("PBR Parameters").strong());
            ui.horizontal(|ui| {
                ui.label("Metallic:");
                changed |= ui
                    .add(egui::Slider::new(&mut mat.metallic_factor, 0.0..=1.0))
                    .changed();
            });
            ui.horizontal(|ui| {
                ui.label("Roughness:");
                changed |= ui
                    .add(egui::Slider::new(&mut mat.roughness_factor, 0.0..=1.0))
                    .changed();
            });
        });

        ui.add_space(4.0);

        ui.group(|ui| {
            ui.label(egui::RichText::new("Textures").strong());
            changed |= Self::draw_texture_assignment(
                ui,
                "Albedo Texture:",
                &mut mat.albedo_texture,
                idx,
                "albedo",
                texture_path_map,
                resource_manager,
            );
            changed |= Self::draw_texture_assignment(
                ui,
                "Normal Texture:",
                &mut mat.normal_texture,
                idx,
                "normal",
                texture_path_map,
                resource_manager,
            );
            changed |= Self::draw_texture_assignment(
                ui,
                "Metallic/Roughness Texture:",
                &mut mat.metallic_roughness_texture,
                idx,
                "mr",
                texture_path_map,
                resource_manager,
            );
        });

        if changed {
            ui.ctx().request_repaint();
        }
    }

    fn draw_texture_assignment(
        ui: &mut Ui,
        label: &str,
        texture_handle: &mut Option<
            spark_core::asset::Handle<spark_renderer::vulkan::texture::Texture>,
        >,
        mat_idx: usize,
        tex_type: &str,
        texture_path_map: &std::collections::HashMap<
            std::path::PathBuf,
            spark_core::resource::Handle<spark_renderer::vulkan::texture::Texture>,
        >,
        resource_manager: &mut spark_core::resource::ResourceManager,
    ) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            ui.label(label);
            let tex_name = texture_handle
                .map(|h| {
                    texture_path_map
                        .iter()
                        .find(|(_, &handle)| handle == h)
                        .map(|(path, _)| {
                            path.file_name()
                                .expect("Texture path has no file name")
                                .to_string_lossy()
                                .into_owned()
                        })
                        .unwrap_or_else(|| format!("Texture ID: {}", h.id()))
                })
                .unwrap_or_else(|| "None".to_string());

            let response =
                egui::ComboBox::from_id_source(format!("mat_tex_{}_{}", mat_idx, tex_type))
                    .selected_text(tex_name)
                    .show_ui(ui, |ui| {
                        let mut local_changed = false;
                        local_changed |=
                            ui.selectable_value(texture_handle, None, "None").changed();
                        for t_idx in 0..resource_manager.gpu_textures.assets_len() {
                            let h = spark_core::resource::Handle::new(t_idx as u32);
                            let name = texture_path_map
                                .iter()
                                .find(|(_, &handle)| handle == h)
                                .map(|(path, _)| {
                                    path.file_name()
                                        .expect("Texture path has no file name")
                                        .to_string_lossy()
                                        .into_owned()
                                })
                                .unwrap_or_else(|| format!("ID: {}", t_idx));
                            local_changed |=
                                ui.selectable_value(texture_handle, Some(h), name).changed();
                        }
                        local_changed
                    });
            if let Some(inner) = response.inner {
                changed |= inner;
            }
        });
        changed
    }

    pub fn draw_settings_tab(
        &mut self,
        ui: &mut Ui,
        renderer: &mut spark_renderer::Renderer,
        project: &mut spark_core::Project,
    ) {
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
                ui.checkbox(&mut renderer.settings.enable_auto_exposure, "Auto Exposure");
                if renderer.settings.enable_auto_exposure {
                    ui.label("Min:");
                    ui.add(
                        egui::DragValue::new(&mut renderer.settings.auto_exposure_min).speed(0.1),
                    );
                    ui.label("Max:");
                    ui.add(
                        egui::DragValue::new(&mut renderer.settings.auto_exposure_max).speed(0.1),
                    );
                    ui.label("Speed:");
                    ui.add(
                        egui::DragValue::new(&mut renderer.settings.auto_exposure_speed).speed(0.1),
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
                ui.checkbox(&mut renderer.settings.enable_color_grading, "Color Grading");
                if renderer.settings.enable_color_grading {
                    ui.label("LUT Index:");
                    ui.add(
                        egui::DragValue::new(&mut renderer.settings.lut_index).clamp_range(-1..=15),
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
                    ui.label("PCF Kernel:");
                    egui::ComboBox::from_id_source("shadow_pcf")
                        .selected_text(if renderer.settings.shadow_pcf_samples == 0 {
                            "Hard"
                        } else {
                            "Soft (5x5)"
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut renderer.settings.shadow_pcf_samples,
                                0,
                                "Hard (No Filter)",
                            );
                            ui.selectable_value(
                                &mut renderer.settings.shadow_pcf_samples,
                                1,
                                "Soft (High Quality 5x5)",
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
                    ui.label("Bias:");
                    ui.add(egui::Slider::new(
                        &mut renderer.settings.ssao_bias,
                        0.001..=0.5,
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
                    ui.add(egui::DragValue::new(&mut renderer.settings.ssr_step).speed(0.01));
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
                ui.checkbox(&mut renderer.settings.enable_rt_ao, "RT Ambient Occlusion");
                ui.checkbox(
                    &mut renderer.settings.enable_rt_gi,
                    "RT Global Illumination",
                );
            });
        });
    }

    pub fn draw_statistics_tab(
        &mut self,
        ui: &mut Ui,
        renderer: &mut spark_renderer::Renderer,
        fps: f32,
    ) {
        ui.horizontal(|ui| {
            ui.label("Camera Speed:");
            ui.add(egui::Slider::new(&mut self.camera_speed, 0.01..=1.0));
        });
        ui.separator();
        ui.label(format!("FPS: {:.1}", fps));
        ui.label(format!("Active Objects: {}", renderer.last_object_count));
        ui.label(format!(
            "Draw Calls: {}",
            renderer
                .last_draw_calls
                .load(std::sync::atomic::Ordering::Relaxed)
        ));
        ui.label(format!(
            "Triangles: {}",
            renderer
                .last_triangle_count
                .load(std::sync::atomic::Ordering::Relaxed)
        ));
        ui.label("GPU Memory: TODO");
    }
}
