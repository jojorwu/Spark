use std::path::{Path, PathBuf};
use crate::resource::ResourceManager;
use crate::asset::AssetManager;
use crate::scene::Scene;
use spark_renderer::Renderer;

pub struct GltfLoader;

impl GltfLoader {
    pub fn load_scene(
        rm: &mut ResourceManager,
        am: &mut AssetManager,
        path: PathBuf,
        scene_tree: &mut Scene,
        renderer: &Renderer,
    ) {
        use rayon::prelude::*;
        log::info!("Loading glTF scene: {:?}", path);
        let (doc, buffers, images) = gltf::import(&path).expect("Failed to load glTF");
        let parent_dir = path.parent().unwrap_or_else(|| Path::new("")).to_path_buf();

        // Pre-load images in parallel
        let loaded_images: Vec<_> = images
            .par_iter()
            .map(|data| {
                image::load_from_memory(&data.pixels).unwrap_or_else(|_| {
                    image::DynamicImage::ImageRgba8(image::RgbaImage::new(1, 1))
                })
            })
            .collect();

        for img in loaded_images {
            am.textures.add(img);
        }

        let default_scene = doc.default_scene().or(doc.scenes().next());
        if let Some(scene) = default_scene {
            for node in scene.nodes() {
                Self::process_node(
                    rm,
                    am,
                    node,
                    &buffers,
                    scene_tree,
                    scene_tree.root,
                    renderer,
                    &parent_dir,
                );
            }
        }
    }

    fn process_node(
        rm: &mut ResourceManager,
        am: &mut AssetManager,
        node: gltf::Node,
        buffers: &[gltf::buffer::Data],
        scene_tree: &mut Scene,
        parent: crate::scene::NodeKey,
        renderer: &Renderer,
        parent_dir: &Path,
    ) {
        use crate::scene::{Component, MeshComponent, Node};
        use spark_math::{Mat4, Quat, Vec3, Vec4};

        let (translation, rotation, scale) = node.transform().decomposed();
        let local_transform = Mat4::from_scale_rotation_translation(
            Vec3::from_array(scale),
            Quat::from_array(rotation),
            Vec3::from_array(translation),
        );

        let mut components: Vec<Box<dyn Component>> = Vec::new();

        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                use spark_renderer::vertex::Vertex;
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                let positions = match reader.read_positions() {
                    Some(p) => p.collect::<Vec<_>>(),
                    None => continue,
                };
                let v_offset = rm.all_vertices.len() as i32;
                let i_start = rm.all_indices.len() as u32;

                let mut max_dist_sq = 0.0f32;
                let normals = reader.read_normals().map(|n| n.collect::<Vec<_>>());
                let tangents = reader.read_tangents().map(|t| t.collect::<Vec<_>>());
                let tex_coords = reader
                    .read_tex_coords(0)
                    .map(|t| t.into_f32().collect::<Vec<_>>());

                for i in 0..positions.len() {
                    let p = positions[i];
                    let dist_sq = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
                    if dist_sq > max_dist_sq {
                        max_dist_sq = dist_sq;
                    }

                    let n = if let Some(ref normals) = normals {
                        spark_math::Vec3::from_array(normals[i])
                    } else {
                        spark_math::Vec3::Y
                    };

                    let tc = if let Some(ref tex_coords) = tex_coords {
                        spark_math::Vec2::from_array(tex_coords[i])
                    } else {
                        spark_math::Vec2::ZERO
                    };

                    let tan = if let Some(ref tangents) = tangents {
                        spark_math::Vec3::new(tangents[i][0], tangents[i][1], tangents[i][2])
                    } else {
                        spark_math::Vec3::X
                    };

                    rm.all_vertices.push(Vertex::pack(
                        spark_math::Vec3::from_array(p),
                        n,
                        tc,
                        spark_math::Vec3::ONE,
                        tan,
                    ));
                }

                let index_count = if let Some(indices) = reader.read_indices() {
                    let idxs: Vec<u32> = indices.into_u32().collect();
                    let count = idxs.len() as u32;
                    rm.all_indices.extend(idxs);
                    count
                } else {
                    positions.len() as u32
                };

                let gltf_mat = primitive.material();
                let pbr = gltf_mat.pbr_metallic_roughness();

                let mut mat_ssbo = spark_renderer::MaterialDataSSBO {
                    albedo_factor: Vec4::from_array(pbr.base_color_factor()),
                    emissive_factor: Vec4::from_array([
                        gltf_mat.emissive_factor()[0],
                        gltf_mat.emissive_factor()[1],
                        gltf_mat.emissive_factor()[2],
                        1.0,
                    ]),
                    metallic_factor: pbr.metallic_factor(),
                    roughness_factor: pbr.roughness_factor(),
                    alpha_cutoff: gltf_mat.alpha_cutoff().unwrap_or(0.5),
                    flags: 0,
                    albedo_texture: -1,
                    normal_texture: -1,
                    metallic_roughness_texture: -1,
                    emissive_texture: -1,
                    occlusion_texture: -1,
                    padding: [0; 3],
                };

                if let Some(tex) = pbr.base_color_texture() {
                    if let gltf::image::Source::Uri { uri, .. } = tex.texture().source().source() {
                        let handle = rm.upload_texture(&parent_dir.join(uri), renderer, am);
                        if let Some(tex) = rm.gpu_textures.get(handle) {
                            mat_ssbo.albedo_texture = tex.bindless_index as i32;
                        }
                    }
                }
                if let Some(tex) = gltf_mat.normal_texture() {
                    if let gltf::image::Source::Uri { uri, .. } = tex.texture().source().source() {
                        let handle = rm.upload_texture(&parent_dir.join(uri), renderer, am);
                        if let Some(tex) = rm.gpu_textures.get(handle) {
                            mat_ssbo.normal_texture = tex.bindless_index as i32;
                        }
                    }
                }
                if let Some(tex) = pbr.metallic_roughness_texture() {
                    if let gltf::image::Source::Uri { uri, .. } = tex.texture().source().source() {
                        let handle = rm.upload_texture(&parent_dir.join(uri), renderer, am);
                        if let Some(tex) = rm.gpu_textures.get(handle) {
                            mat_ssbo.metallic_roughness_texture = tex.bindless_index as i32;
                        }
                    }
                }
                if let Some(tex) = gltf_mat.emissive_texture() {
                    if let gltf::image::Source::Uri { uri, .. } = tex.texture().source().source() {
                        let handle = rm.upload_texture(&parent_dir.join(uri), renderer, am);
                        if let Some(tex) = rm.gpu_textures.get(handle) {
                            mat_ssbo.emissive_texture = tex.bindless_index as i32;
                        }
                    }
                }
                if let Some(tex) = gltf_mat.occlusion_texture() {
                    if let gltf::image::Source::Uri { uri, .. } = tex.texture().source().source() {
                        let handle = rm.upload_texture(&parent_dir.join(uri), renderer, am);
                        if let Some(tex) = rm.gpu_textures.get(handle) {
                            mat_ssbo.occlusion_texture = tex.bindless_index as i32;
                        }
                    }
                }

                rm.all_materials_ssbo.push(mat_ssbo);
                let mat_idx = (rm.all_materials_ssbo.len() - 1) as u32;

                am.materials.add(crate::asset::Material {
                    name: gltf_mat.name().unwrap_or("Unnamed Material").to_string(),
                    albedo_factor: pbr.base_color_factor(),
                    emissive_factor: [
                        gltf_mat.emissive_factor()[0],
                        gltf_mat.emissive_factor()[1],
                        gltf_mat.emissive_factor()[2],
                        1.0,
                    ],
                    metallic_factor: pbr.metallic_factor(),
                    roughness_factor: pbr.roughness_factor(),
                    albedo_texture: None, // Simplified
                    normal_texture: None,
                    metallic_roughness_texture: None,
                    is_transparent: gltf_mat.alpha_mode() == gltf::material::AlphaMode::Blend,
                });

                let mesh_comp = MeshComponent {
                    vertex_count: positions.len() as u32,
                    index_count,
                    first_index: i_start,
                    vertex_offset: v_offset,
                    texture_handle: None,
                    material_index: Some(mat_idx),
                    bounding_radius: max_dist_sq.sqrt(),
                    skin_index: node.skin().map(|s| s.index() as u32),
                };
                components.push(Box::new(mesh_comp) as Box<dyn Component>);
                rm.needs_upload = true;
            }
        }

        let spark_node = Node {
            name: node.name().unwrap_or("Unnamed Node").to_string(),
            visible: true,
            locked: false,
            is_dirty: true,
            local_transform,
            global_transform: Mat4::IDENTITY,
            parent: None,
            children: Vec::new(),
            components,
        };

        let key = scene_tree.add_node(parent, spark_node);
        for child in node.children() {
            Self::process_node(rm, am, child, buffers, scene_tree, key, renderer, parent_dir);
        }
    }
}
