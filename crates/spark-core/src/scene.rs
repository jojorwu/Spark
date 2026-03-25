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
    pub visible: bool,
    pub locked: bool,
    pub is_dirty: bool,
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
            visible: self.visible,
            locked: self.locked,
            is_dirty: self.is_dirty,
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
    #[serde(skip)]
    pub component_registry: std::collections::HashMap<std::any::TypeId, Vec<NodeKey>>,
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
            visible: true,
            locked: false,
            is_dirty: true,
            local_transform: Mat4::IDENTITY,
            global_transform: Mat4::IDENTITY,
            parent: None,
            children: Vec::new(),
            components: Vec::new(),
        });

        Self {
            nodes,
            root,
            last_view_matrix: Mat4::IDENTITY,
            component_registry: std::collections::HashMap::new(),
        }
    }

    pub fn rebuild_component_registry(&mut self) {
        self.component_registry.clear();
        for (key, node) in &self.nodes {
            for component in &node.components {
                let type_id = component.as_any().type_id();
                self.component_registry.entry(type_id).or_default().push(key);
            }
        }
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

pub struct Query<'a> {
    scene: &'a Scene,
    matches: Option<std::collections::HashSet<NodeKey>>,
}

impl<'a> Query<'a> {
    pub fn with<T: 'static>(mut self) -> Self {
        let tid = std::any::TypeId::of::<T>();
        if let Some(nodes) = self.scene.component_registry.get(&tid) {
            let set: std::collections::HashSet<_> = nodes.iter().copied().collect();
            if let Some(ref mut matches) = self.matches {
                matches.retain(|k| set.contains(k));
            } else {
                self.matches = Some(set);
            }
        } else {
            self.matches = Some(std::collections::HashSet::new());
        }
        self
    }

    pub fn without<T: 'static>(mut self) -> Self {
        let tid = std::any::TypeId::of::<T>();
        if let Some(nodes) = self.scene.component_registry.get(&tid) {
            if let Some(ref mut matches) = self.matches {
                for k in nodes {
                    matches.remove(k);
                }
            } else {
                let mut all: std::collections::HashSet<_> = self.scene.nodes.keys().collect();
                for k in nodes {
                    all.remove(k);
                }
                self.matches = Some(all);
            }
        }
        self
    }

    pub fn build(self) -> Vec<NodeKey> {
        self.matches.map(|m| m.into_iter().collect()).unwrap_or_else(|| self.scene.nodes.keys().collect())
    }
}

impl Scene {
    pub fn query(&self) -> Query {
        Query { scene: self, matches: None }
    }

    pub fn query_components<T: 'static>(&self) -> Vec<NodeKey> {
        self.component_registry.get(&std::any::TypeId::of::<T>()).cloned().unwrap_or_default()
    }

    pub fn update_components(&mut self, delta: f32, renderer: *mut spark_renderer::Renderer, resource_manager: *mut crate::resource::ResourceManager, project: &crate::Project, task_system: &crate::task::TaskSystem, resources: &crate::resource_container::Resources) {
        let ctx = crate::FrameContext {
            scene: self as *mut Scene,
            renderer,
            resource_manager,
            project,
            resources,
            task_system,
            delta,
            event_proxy: crate::systems_events::events::EventProxy {
                events: &[],
                outgoing: &std::sync::Mutex::new(Vec::new()),
            },
            input: &crate::input::InputManager::new(),
            command_queue: &crate::command::CommandQueue::new(),
            event_bus: &crate::event_bus::EventBus::new(),
        };

        let nodes: Vec<NodeKey> = self.nodes.keys().collect();
        for key in nodes {
             // We need to avoid simultaneous borrow and modification if components modify the scene.
             // For now, components use CommandQueue which is fine.
             let components_count = self.nodes.get(key).map(|n| n.components.len()).unwrap_or(0);
             for i in 0..components_count {
                 let component = &mut self.nodes.get_mut(key).unwrap().components[i];
                 component.on_update(key, &ctx);
             }
        }
    }

    pub fn remove_node(&mut self, key: NodeKey) {
        let (children, parent_key, components_types) = if let Some(node) = self.nodes.get(key) {
            let types: Vec<_> = node.components.iter().map(|c| c.as_any().type_id()).collect();
            (node.children.clone(), node.parent, types)
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

        // Update registry
        for tid in components_types {
            if let Some(list) = self.component_registry.get_mut(&tid) {
                list.retain(|&k| k != key);
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

        // Update registry
        for component in &self.nodes[key].components {
            let type_id = component.as_any().type_id();
            self.component_registry.entry(type_id).or_default().push(key);
        }

        self.update_all_transforms();
        key
    }

    pub fn update_all_transforms(&mut self) {
        self.update_transform_recursive(self.root, Mat4::IDENTITY, false);
    }

    fn update_transform_recursive(&mut self, key: NodeKey, parent_global: Mat4, parent_dirty: bool) {
        let (dirty, global) = if let Some(node) = self.nodes.get_mut(key) {
            let dirty = node.is_dirty || parent_dirty;
            if dirty {
                node.global_transform = parent_global * node.local_transform;
                node.is_dirty = false;
            }
            (dirty, node.global_transform)
        } else {
            return;
        };

        if dirty {
             for component in &self.nodes[key].components {
                if component.as_any().is::<CameraComponent>() {
                    self.last_view_matrix = global.inverse();
                }
            }
        }

        let children = self.nodes[key].children.clone();
        for child in children {
            self.update_transform_recursive(child, global, dirty);
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
                    let is_transparent = resource_manager.materials.get(crate::resource::Handle::new(mat_idx)).is_some_and(|m| m.is_transparent);
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
                    let is_transparent = resource_manager.materials.get(crate::resource::Handle::new(mat_idx)).is_some_and(|m| m.is_transparent);
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
            let is_transparent = resource_manager.materials.get(crate::resource::Handle::new(midx)).is_some_and(|m| m.is_transparent);

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
