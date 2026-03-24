use slotmap::{SlotMap, new_key_type};
use spark_math::{Mat4, Vec4Swizzles, Vec2};
use serde::{Serialize, Deserialize};

new_key_type! {
    pub struct NodeKey;
}

#[typetag::serde(tag = "type")]
pub trait Component: Send + Sync {
    fn on_init(&mut self, _node: NodeKey, _scene: &mut Scene) {}
    fn on_update(&mut self, _node: NodeKey, _ctx: &crate::FrameContext) {}
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
pub struct SpriteComponent {
    pub texture_handle: Option<crate::resource::Handle<spark_renderer::vulkan::texture::Texture>>,
    pub color: [f32; 4],
    pub flip_x: bool,
    pub flip_y: bool,
    pub size: Vec2,
}

#[typetag::serde]
impl Component for SpriteComponent {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn clone_box(&self) -> Box<dyn Component> { Box::new(self.clone()) }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct CameraComponent {
    pub fov: f32,
    pub near: f32,
    pub far: f32,
    pub orthographic: bool,
    pub ortho_size: f32,
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
    pub spot_inner_angle: f32,
    pub spot_outer_angle: f32,
}

#[typetag::serde]
impl Component for LightComponent {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn clone_box(&self) -> Box<dyn Component> { Box::new(self.clone()) }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum LightType {
    Directional,
    Point,
    Spot,
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
    lights: Vec<(Mat4, LightType, spark_math::Vec3, f32, f32, f32, f32)>,
    sprites: Vec<(Mat4, Option<TextureHandle>, [f32; 4], Vec2)>,
}

impl SceneDataCollector {
    fn new() -> Self {
        Self {
            renderables: Vec::new(),
            instanced: std::collections::HashMap::new(),
            lights: Vec::new(),
            sprites: Vec::new(),
        }
    }

    fn merge(&mut self, other: SceneDataCollector) {
        self.renderables.extend(other.renderables);
        self.lights.extend(other.lights);
        self.sprites.extend(other.sprites);
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
        let mut levels = Vec::new();
        let mut current_level = vec![self.root];

        while !current_level.is_empty() {
            let mut next_level = Vec::new();
            for &key in &current_level {
                if let Some(node) = self.nodes.get(key) {
                    next_level.extend(node.children.iter().copied());
                }
            }
            levels.push(current_level);
            current_level = next_level;
        }

        for level in levels {
            use rayon::prelude::*;
            level.into_par_iter().for_each(|key| {
                // Safety: We process level by level. All parents for the current level
                // have already been updated in the previous level. Nodes within the same level
                // do not depend on each other's global_transform.
                unsafe {
                    let scene_ptr = self as *const Scene as *mut Scene;
                    let scene = &mut *scene_ptr;

                    let parent_global = if let Some(node) = scene.nodes.get(key) {
                        node.parent.and_then(|pk| scene.nodes.get(pk)).map(|p| p.global_transform).unwrap_or(Mat4::IDENTITY)
                    } else {
                        Mat4::IDENTITY
                    };

                    if let Some(node) = scene.nodes.get_mut(key) {
                        node.global_transform = parent_global * node.local_transform;
                        let current_global = node.global_transform;

                        for component in &node.components {
                            if component.as_any().is::<CameraComponent>() {
                                scene.last_view_matrix = current_global.inverse();
                            }
                        }
                    }
                }
            });
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
                } else if component.as_any().is::<SpriteComponent>() {
                    radius = 0.5; // Default for sprites
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

        // Pre-calculate transparency to avoid repeat resource_manager lookups
        let (mut opaque_meshes, mut transparent_meshes): (Vec<_>, Vec<_>) = rayon::join(
            || {
                data.renderables.par_iter().filter_map(|r| {
                    let mat_idx = r.6.unwrap_or(0);
                    let is_transparent = resource_manager.all_materials.get(mat_idx as usize).is_some_and(|m| (m.flags & 1) != 0);
                    if !is_transparent {
                        Some(spark_renderer::resource::MeshDraw {
                            model: r.0,
                            vertex_count: r.1,
                            index_count: r.2,
                            first_index: r.3,
                            vertex_offset: r.4,
                            material_index: mat_idx,
                            bounding_radius: r.7,
                        })
                    } else {
                        None
                    }
                }).collect()
            },
            || {
                data.renderables.par_iter().filter_map(|r| {
                    let mat_idx = r.6.unwrap_or(0);
                    let is_transparent = resource_manager.all_materials.get(mat_idx as usize).is_some_and(|m| (m.flags & 1) != 0);
                    if is_transparent {
                        Some(spark_renderer::resource::MeshDraw {
                            model: r.0,
                            vertex_count: r.1,
                            index_count: r.2,
                            first_index: r.3,
                            vertex_offset: r.4,
                            material_index: mat_idx,
                            bounding_radius: r.7,
                        })
                    } else {
                        None
                    }
                }).collect()
            }
        );

        // Parallel processing of instanced data
        let instanced_results: Vec<Vec<(spark_renderer::resource::MeshDraw, bool)>> = data.instanced.par_iter().map(|((ic, fi, vo, _tex, mat_idx, br_bits), transforms)| {
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
            }).collect()
        }).collect();

        for batch in instanced_results {
            for (draw, is_trans) in batch {
                if is_trans {
                    transparent_meshes.push(draw);
                } else {
                    opaque_meshes.push(draw);
                }
            }
        }

        // Parallel sort transparent meshes back-to-front
        let view_pos = self.last_view_matrix.inverse().w_axis.xyz();
        transparent_meshes.par_sort_by(|a, b| {
            let dist_a = (a.model.w_axis.xyz() - view_pos).length_squared();
            let dist_b = (b.model.w_axis.xyz() - view_pos).length_squared();
            dist_b.partial_cmp(&dist_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        let lights = data.lights.into_par_iter().map(|(t, light_type, color, intensity, range, spot_inner, spot_outer)| {
            let position = spark_math::Vec3::new(t.w_axis.x, t.w_axis.y, t.w_axis.z);
            let direction = -spark_math::Vec3::new(t.z_axis.x, t.z_axis.y, t.z_axis.z).normalize();
            spark_renderer::resource::LightDraw {
                position,
                direction,
                color,
                intensity,
                range,
                light_type: match light_type {
                    LightType::Directional => 0,
                    LightType::Point => 1,
                    LightType::Spot => 2,
                },
                spot_angles: [spot_inner.cos(), spot_outer.cos()],
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
                    data.lights.push((node.global_transform, light.light_type, light.color, light.intensity, light.range, light.spot_inner_angle.to_radians(), light.spot_outer_angle.to_radians()));
                } else if let Some(sprite) = any.downcast_ref::<SpriteComponent>() {
                     data.sprites.push((node.global_transform, sprite.texture_handle, sprite.color, sprite.size));
                }
            }

            if !node.children.is_empty() {
                // Heuristic: Use parallel collection only for branches with many children
                if node.children.len() > 100 {
                    data.merge(self.collect_data_parallel(&node.children, frustum));
                } else {
                    for &child_key in &node.children {
                        data.merge(self.collect_data_recursive(child_key, frustum));
                    }
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
                    } else if component.as_any().is::<SpriteComponent>() {
                        radius = 0.5;
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
