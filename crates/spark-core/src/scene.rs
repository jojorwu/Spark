use slotmap::{SlotMap, new_key_type};
use spark_math::{Mat4, Vec4Swizzles};
use serde::{Serialize, Deserialize};
use std::sync::Mutex;

new_key_type! {
    pub struct NodeKey;
}

#[typetag::serde(tag = "type")]
pub trait Component: Send + Sync {
    fn on_init(&mut self, _node: NodeKey, _scene: &mut Scene) {}
    fn on_update(&mut self, _node: NodeKey, _scene: &mut Scene, _delta: f32) {}
    fn as_any(&self) -> &dyn std::any::Any;
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
    fn clone_box(&self) -> Box<dyn Component>;
}

#[derive(Serialize, Deserialize, Clone)]
pub struct MeshComponent {
    pub vertex_count: u32,
    pub index_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub texture_handle: Option<crate::resource::Handle<spark_renderer::vulkan::texture::Texture>>,
    pub material_index: Option<u32>,
    pub bounding_radius: f32,
}

#[typetag::serde]
impl Component for MeshComponent {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn clone_box(&self) -> Box<dyn Component> { Box::new(self.clone()) }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct CameraComponent {
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

#[typetag::serde]
impl Component for CameraComponent {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn clone_box(&self) -> Box<dyn Component> { Box::new(self.clone()) }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct LightComponent {
    pub light_type: LightType,
    pub color: spark_math::Vec3,
    pub intensity: f32,
    pub range: f32,
}

#[typetag::serde]
impl Component for LightComponent {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn clone_box(&self) -> Box<dyn Component> { Box::new(self.clone()) }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum LightType {
    Directional,
    Point,
}

#[derive(Serialize, Deserialize)]
pub struct Node {
    pub name: String,
    pub local_transform: Mat4,
    #[serde(skip)]
    pub global_transform: Mat4,
    pub parent: Option<NodeKey>,
    pub children: Vec<NodeKey>,
    pub components: Vec<Box<dyn Component>>,
}

impl Clone for Node {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            local_transform: self.local_transform,
            global_transform: self.global_transform,
            parent: self.parent,
            children: self.children.clone(),
            components: self.components.iter().map(|c| c.clone_box()).collect(),
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct Scene {
    pub nodes: SlotMap<NodeKey, Node>,
    pub root: NodeKey,
    #[serde(skip)]
    pub last_view_matrix: Mat4,
}

type TextureHandle = crate::resource::Handle<spark_renderer::vulkan::texture::Texture>;

pub type RenderableData = (Mat4, u32, u32, u32, i32, Option<TextureHandle>, Option<u32>, f32);
pub type InstancedKey = (u32, u32, i32, Option<TextureHandle>, Option<u32>, u32);

struct SceneDataCollector {
    renderables: Vec<RenderableData>,
    instanced: std::collections::HashMap<InstancedKey, Vec<Mat4>>,
    lights: Vec<(Mat4, LightType, spark_math::Vec3, f32, f32)>,
}

impl SceneDataCollector {
    fn new() -> Self {
        Self {
            renderables: Vec::new(),
            instanced: std::collections::HashMap::new(),
            lights: Vec::new(),
        }
    }

    fn merge(&mut self, other: SceneDataCollector) {
        self.renderables.extend(other.renderables);
        self.lights.extend(other.lights);
        for (key, transforms) in other.instanced {
            self.instanced.entry(key).or_default().extend(transforms);
        }
    }
}

impl Scene {
    pub fn new() -> Self {
        let mut nodes = SlotMap::with_key();
        let root = nodes.insert(Node {
            name: "Root".to_string(),
            local_transform: Mat4::IDENTITY,
            global_transform: Mat4::IDENTITY,
            parent: None,
            children: Vec::new(),
            components: Vec::new(),
        });

        Self { nodes, root, last_view_matrix: Mat4::IDENTITY }
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

impl Scene {

    pub fn remove_node(&mut self, key: NodeKey) {
        let (children, parent_key) = if let Some(node) = self.nodes.get(key) {
            (node.children.clone(), node.parent)
        } else {
            return;
        };

        for child in children {
            self.remove_node(child);
        }

        if let Some(pk) = parent_key {
            if let Some(parent) = self.nodes.get_mut(pk) {
                parent.children.retain(|&k| k != key);
            }
        }

        self.nodes.remove(key);
    }

    pub fn add_node(&mut self, parent: NodeKey, mut node: Node) -> NodeKey {
        node.parent = Some(parent);
        let key = self.nodes.insert(node);
        if let Some(parent_node) = self.nodes.get_mut(parent) {
            parent_node.children.push(key);
        }
        self.update_all_transforms();
        key
    }

    pub fn update_all_transforms(&mut self) {
        use rayon::prelude::*;

        let mut level = vec![self.root];
        let mut parent_globals = std::collections::HashMap::new();
        parent_globals.insert(self.root, Mat4::IDENTITY);

        while !level.is_empty() {
            let next_level: Mutex<Vec<NodeKey>> = Mutex::new(Vec::new());

            // Parallel update of the current level
            level.par_iter().for_each(|&key| {
                // Safety: We ensure that we only access nodes in the current level
                // and their parents (which were updated in the previous level).
                // To do this strictly safely in Rust, we'd need a more complex structure.
                // For now, we use a controlled raw pointer approach for performance.
                unsafe {
                    let scene_ptr = self as *const Scene as *mut Scene;
                    if let Some(node) = (*scene_ptr).nodes.get_mut(key) {
                        let parent_global = if let Some(pk) = node.parent {
                            (*scene_ptr).nodes.get(pk).map(|p| p.global_transform).unwrap_or(Mat4::IDENTITY)
                        } else {
                            Mat4::IDENTITY
                        };

                        node.global_transform = parent_global * node.local_transform;

                        // Detect camera
                        for component in &node.components {
                            if component.as_any().is::<CameraComponent>() {
                                (*scene_ptr).last_view_matrix = node.global_transform.inverse();
                            }
                        }

                        next_level.lock().unwrap().extend(&node.children);
                    }
                }
            });

            level = next_level.into_inner().unwrap();
        }
    }

    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let mut scene: Scene = serde_json::from_str(&content)?;
        scene.update_all_transforms();
        Ok(scene)
    }

    pub fn pick_node(&self, ray: &spark_math::Ray) -> Option<(NodeKey, f32)> {
        let mut closest_node = None;
        let mut min_t = f32::MAX;

        for (key, node) in &self.nodes {
            let mut radius = 0.5f32;
            let mut has_bounds = false;

            for component in &node.components {
                if let Some(mesh) = component.as_any().downcast_ref::<MeshComponent>() {
                    radius = mesh.bounding_radius;
                    has_bounds = true;
                } else if component.as_any().is::<LightComponent>() || component.as_any().is::<CameraComponent>() {
                    has_bounds = true;
                }
            }

            if !has_bounds { continue; }

            let center = node.global_transform.w_axis.xyz();
            if let Some(t) = ray.intersect_sphere(center, radius) {
                if t < min_t {
                    min_t = t;
                    closest_node = Some(key);
                }
            }
        }

        closest_node.map(|key| (key, min_t))
    }

    /// Collects a Packet for the renderer.
    pub fn collect_frame_packet(&self, frustum: Option<&spark_math::Frustum>, resource_manager: &crate::resource::ResourceManager) -> spark_renderer::resource::FramePacket {
        use rayon::prelude::*;
        let data = self.collect_data_recursive(self.root, frustum);

        // Parallel processing of renderables
        let (opaque_meshes, transparent_meshes): (Vec<_>, Vec<_>) = data.renderables.par_iter().map(|r| {
            let mat_idx = r.6.unwrap_or(0);
            let is_transparent = resource_manager.all_materials.get(mat_idx as usize).is_some_and(|m| (m.flags & 1) != 0);

            let draw = spark_renderer::resource::MeshDraw {
                model: r.0,
                vertex_count: r.1,
                index_count: r.2,
                first_index: r.3,
                vertex_offset: r.4,
                material_index: mat_idx,
                bounding_radius: r.7,
            };
            (draw, is_transparent)
        }).partition(|(_, is_trans)| !*is_trans);

        // Strip the boolean from the partition result
        let mut opaque_meshes: Vec<_> = opaque_meshes.into_iter().map(|(d, _)| d).collect();
        let mut transparent_meshes: Vec<_> = transparent_meshes.into_iter().map(|(d, _)| d).collect();

        // Parallel sort transparent meshes back-to-front
        let view_pos = self.last_view_matrix.inverse().w_axis.xyz();
        transparent_meshes.par_sort_by(|a, b| {
            let dist_a = (a.model.w_axis.xyz() - view_pos).length_squared();
            let dist_b = (b.model.w_axis.xyz() - view_pos).length_squared();
            dist_b.partial_cmp(&dist_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Parallel processing of instanced data
        let instanced_results: Vec<_> = data.instanced.par_iter().flat_map(|((ic, fi, vo, _tex, mat_idx, br_bits), transforms)| {
            let br = f32::from_bits(*br_bits);
            let midx = mat_idx.unwrap_or(0);
            let is_transparent = resource_manager.all_materials.get(midx as usize).is_some_and(|m| (m.flags & 1) != 0);

            transforms.iter().map(move |&t| {
                let draw = spark_renderer::resource::MeshDraw {
                    model: t,
                    vertex_count: 0,
                    index_count: *ic,
                    first_index: *fi,
                    vertex_offset: *vo,
                    material_index: midx,
                    bounding_radius: br,
                };
                (draw, is_transparent)
            }).collect::<Vec<_>>()
        }).collect();

        for (draw, is_trans) in instanced_results {
            if is_trans {
                transparent_meshes.push(draw);
            } else {
                opaque_meshes.push(draw);
            }
        }

        let lights = data.lights.into_iter().map(|(t, _type, color, intensity, _range)| {
            let translation = spark_math::Vec3::new(t.w_axis.x, t.w_axis.y, t.w_axis.z);
            spark_renderer::resource::LightDraw {
                position: translation,
                color,
                intensity,
            }
        }).collect();

        spark_renderer::resource::FramePacket {
            view_matrix: self.last_view_matrix,
            projection_matrix: spark_math::Mat4::IDENTITY, // Placeholder, usually set by render_phase
            opaque_meshes,
            transparent_meshes,
            lights,
        }
    }

    fn collect_data_recursive(
        &self,
        node_key: NodeKey,
        frustum: Option<&spark_math::Frustum>,
    ) -> SceneDataCollector {
        let mut data = SceneDataCollector::new();
        if let Some(node) = self.nodes.get(node_key) {
            for component in &node.components {
                let any = component.as_any();
                if let Some(mesh) = any.downcast_ref::<MeshComponent>() {
                    let visible = if let Some(f) = frustum {
                        let translation = spark_math::Vec3::new(node.global_transform.w_axis.x, node.global_transform.w_axis.y, node.global_transform.w_axis.z);
                        f.intersects_sphere(translation, mesh.bounding_radius)
                    } else {
                        true
                    };
                    if visible {
                        if mesh.material_index.is_some() {
                            data.instanced.entry((mesh.index_count, mesh.first_index, mesh.vertex_offset, mesh.texture_handle, mesh.material_index, (mesh.bounding_radius).to_bits())).or_default().push(node.global_transform);
                        } else {
                            data.renderables.push((
                                node.global_transform,
                                mesh.vertex_count,
                                mesh.index_count,
                                mesh.first_index,
                                mesh.vertex_offset,
                                mesh.texture_handle,
                                mesh.material_index,
                                mesh.bounding_radius,
                            ));
                        }
                    }
                } else if let Some(light) = any.downcast_ref::<LightComponent>() {
                    data.lights.push((node.global_transform, light.light_type, light.color, light.intensity, light.range));
                }
            }

            if !node.children.is_empty() {
                if node.children.len() > 1 {
                    data.merge(self.collect_data_parallel(&node.children, frustum));
                } else {
                    data.merge(self.collect_data_recursive(node.children[0], frustum));
                }
            }
        }
        data
    }

    fn collect_data_parallel(
        &self,
        children: &[NodeKey],
        frustum: Option<&spark_math::Frustum>,
    ) -> SceneDataCollector {
        if children.len() <= 1 {
            return self.collect_data_recursive(children[0], frustum);
        }

        let mid = children.len() / 2;
        let (left, right) = children.split_at(mid);

        let (mut left_data, right_data) = rayon::join(
            || self.collect_data_parallel(left, frustum),
            || self.collect_data_parallel(right, frustum)
        );

        left_data.merge(right_data);
        left_data
    }

    pub fn pick_node_parallel(&self, ray: &spark_math::Ray) -> Option<(NodeKey, f32)> {
        use rayon::prelude::*;
        let nodes: Vec<_> = self.nodes.iter().collect();
        nodes.into_par_iter()
            .filter_map(|(key, node)| {
                let mut radius = 0.5f32;
                let mut has_bounds = false;

                for component in &node.components {
                    if let Some(mesh) = component.as_any().downcast_ref::<MeshComponent>() {
                        radius = mesh.bounding_radius;
                        has_bounds = true;
                    } else if component.as_any().is::<LightComponent>() || component.as_any().is::<CameraComponent>() {
                        has_bounds = true;
                    }
                }

                if !has_bounds { return None; }

                let center = spark_math::Vec3::new(node.global_transform.w_axis.x, node.global_transform.w_axis.y, node.global_transform.w_axis.z);
                ray.intersect_sphere(center, radius).map(|t| (key, t))
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }
}
