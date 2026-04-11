use serde::{Deserialize, Serialize};
use slotmap::{new_key_type, SlotMap};
use spark_math::{Mat4, Vec2, Vec4Swizzles};

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
    pub skin_index: Option<u32>,
}

#[typetag::serde]
impl Component for MeshComponent {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Component> {
        Box::new(self.clone())
    }
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
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Component> {
        Box::new(self.clone())
    }
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
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Component> {
        Box::new(self.clone())
    }
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
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Component> {
        Box::new(self.clone())
    }
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

impl Default for Node {
    fn default() -> Self {
        Self {
            name: "New Node".to_string(),
            visible: true,
            locked: false,
            is_dirty: true,
            local_transform: Mat4::IDENTITY,
            global_transform: Mat4::IDENTITY,
            parent: None,
            children: Vec::new(),
            components: Vec::new(),
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
    #[serde(skip)]
    pub nodes_version: std::sync::atomic::AtomicU64,
    #[serde(skip)]
    pub cached_layers: Vec<Vec<NodeKey>>,
    #[serde(skip)]
    pub layers_version: u64,
}

type TextureHandle = crate::resource::Handle<spark_renderer::vulkan::texture::Texture>;

pub type RenderableData = (
    Mat4,
    u32,
    u32,
    u32,
    i32,
    Option<TextureHandle>,
    Option<u32>,
    f32,
);
pub type InstancedKey = (u32, u32, i32, Option<TextureHandle>, Option<u32>, u32);

struct SceneDataCollector {
    renderables: Vec<RenderableData>,
    instanced: Vec<(InstancedKey, Mat4)>,
    lights: Vec<(Mat4, LightType, spark_math::Vec3, f32, f32, f32, f32)>,
    sprites: Vec<(Mat4, Option<TextureHandle>, [f32; 4], Vec2)>,
}

impl SceneDataCollector {
    fn new() -> Self {
        Self {
            renderables: Vec::new(),
            instanced: Vec::new(),
            lights: Vec::new(),
            sprites: Vec::new(),
        }
    }

    fn merge(&mut self, mut other: SceneDataCollector) {
        self.renderables.append(&mut other.renderables);
        self.lights.append(&mut other.lights);
        self.sprites.append(&mut other.sprites);
        self.instanced.append(&mut other.instanced);
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
            nodes_version: std::sync::atomic::AtomicU64::new(1),
            cached_layers: Vec::new(),
            layers_version: 0,
        }
    }

    pub fn rebuild_component_registry(&mut self) {
        self.component_registry.clear();
        for (key, node) in &self.nodes {
            for component in &node.components {
                let type_id = component.as_any().type_id();
                self.component_registry
                    .entry(type_id)
                    .or_default()
                    .push(key);
            }
        }
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

impl spark_renderer::RenderableScene for Scene {
    fn get_active_camera_matrices(
        &self,
        extent: spark_renderer::ash::vk::Extent2D,
    ) -> Option<(spark_math::Mat4, spark_math::Mat4, f32, f32)> {
        for node in self.nodes.values() {
            for component in &node.components {
                if let Some(camera) = component.as_any().downcast_ref::<CameraComponent>() {
                    let view = node.global_transform.inverse();
                    let projection = if camera.orthographic {
                        let aspect = extent.width as f32 / extent.height as f32;
                        let size = camera.ortho_size;
                        spark_math::Mat4::orthographic_rh(
                            -size * aspect,
                            size * aspect,
                            -size,
                            size,
                            camera.near,
                            camera.far,
                        )
                    } else {
                        spark_math::Mat4::perspective_rh(
                            camera.fov.to_radians(),
                            extent.width as f32 / extent.height as f32,
                            camera.near,
                            camera.far,
                        )
                    };
                    return Some((view, projection, camera.near, camera.far));
                }
            }
        }
        None
    }

    fn collect_frame_packet(
        &self,
        frustum: Option<&spark_math::Frustum>,
        asset_manager: &dyn spark_renderer::RenderableAssetManager,
    ) -> spark_renderer::resource::FramePacket {
        self.collect_frame_packet_internal(frustum, asset_manager)
    }
}

pub struct Query<'a> {
    scene: &'a Scene,
    matches: Option<std::collections::HashSet<NodeKey>>,
}

impl<'a> Query<'a> {
    /// Filters the query to only include nodes that have component `T`.
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

    /// Filters the query to exclude nodes that have component `T`.
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

    /// Executes the query and returns the matching node keys.
    pub fn execute(self) -> Vec<NodeKey> {
        if let Some(matches) = self.matches {
            matches.into_iter().collect()
        } else {
            self.scene.nodes.keys().collect()
        }
    }
}

impl Scene {
    pub fn query(&self) -> Query<'_> {
        Query {
            scene: self,
            matches: None,
        }
    }

    pub fn query_components<T: 'static>(&self) -> Vec<NodeKey> {
        self.component_registry
            .get(&std::any::TypeId::of::<T>())
            .cloned()
            .unwrap_or_default()
    }

    pub fn remove_node(&mut self, key: NodeKey) {
        let (children, parent_key, components_types) = if let Some(node) = self.nodes.get(key) {
            let types: Vec<_> = node
                .components
                .iter()
                .map(|c| c.as_any().type_id())
                .collect();
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
        self.nodes_version
            .fetch_add(1, std::sync::atomic::Ordering::Release);
    }

    pub fn add_node(&mut self, parent: NodeKey, mut node: Node) -> NodeKey {
        node.parent = Some(parent);
        let key = self.nodes.insert(node);
        if let Some(parent_node) = self.nodes.get_mut(parent) {
            parent_node.children.push(key);
        }

        self.nodes_version
            .fetch_add(1, std::sync::atomic::Ordering::Release);

        // Update registry
        for component in &self.nodes[key].components {
            let type_id = component.as_any().type_id();
            self.component_registry
                .entry(type_id)
                .or_default()
                .push(key);
        }

        self.update_all_transforms();
        key
    }

    /// Adds a component to a specific node.
    pub fn add_component<T: Component + 'static>(&mut self, node_key: NodeKey, component: T) {
        if let Some(node) = self.nodes.get_mut(node_key) {
            let type_id = std::any::TypeId::of::<T>();
            node.components.push(Box::new(component));
            self.component_registry
                .entry(type_id)
                .or_default()
                .push(node_key);
            self.nodes_version
                .fetch_add(1, std::sync::atomic::Ordering::Release);
        }
    }

    /// Removes all components of type `T` from a specific node.
    pub fn remove_components<T: 'static>(&mut self, node_key: NodeKey) {
        if let Some(node) = self.nodes.get_mut(node_key) {
            let type_id = std::any::TypeId::of::<T>();
            node.components.retain(|c| c.as_any().type_id() != type_id);
            if let Some(list) = self.component_registry.get_mut(&type_id) {
                list.retain(|&k| k != node_key);
            }
            self.nodes_version
                .fetch_add(1, std::sync::atomic::Ordering::Release);
        }
    }

    pub fn duplicate_node(&mut self, key: NodeKey) -> Option<NodeKey> {
        let node_to_clone = self.nodes.get(key)?.clone();
        let parent = node_to_clone.parent;

        self.nodes_version
            .fetch_add(1, std::sync::atomic::Ordering::Release);

        let new_key = self.nodes.insert(Node {
            name: format!("{} (Copy)", node_to_clone.name),
            visible: node_to_clone.visible,
            locked: node_to_clone.locked,
            is_dirty: true,
            local_transform: node_to_clone.local_transform,
            global_transform: node_to_clone.global_transform,
            parent,
            children: Vec::new(),
            components: node_to_clone
                .components
                .iter()
                .map(|c| c.clone_box())
                .collect(),
        });

        if let Some(pk) = parent {
            if let Some(p_node) = self.nodes.get_mut(pk) {
                p_node.children.push(new_key);
            }
        }

        for component in &self.nodes[new_key].components {
            let type_id = component.as_any().type_id();
            self.component_registry
                .entry(type_id)
                .or_default()
                .push(new_key);
        }

        let children_to_clone = node_to_clone.children.clone();
        for child_key in children_to_clone {
            if let Some(new_child_key) = self.duplicate_node_rec(child_key, new_key) {
                self.nodes[new_key].children.push(new_child_key);
            }
        }

        self.update_all_transforms();
        Some(new_key)
    }

    fn duplicate_node_rec(&mut self, key: NodeKey, new_parent: NodeKey) -> Option<NodeKey> {
        let node_to_clone = self.nodes.get(key)?.clone();

        self.nodes_version
            .fetch_add(1, std::sync::atomic::Ordering::Release);

        let new_key = self.nodes.insert(Node {
            name: node_to_clone.name.clone(),
            visible: node_to_clone.visible,
            locked: node_to_clone.locked,
            is_dirty: true,
            local_transform: node_to_clone.local_transform,
            global_transform: node_to_clone.global_transform,
            parent: Some(new_parent),
            children: Vec::new(),
            components: node_to_clone
                .components
                .iter()
                .map(|c| c.clone_box())
                .collect(),
        });

        for component in &self.nodes[new_key].components {
            let type_id = component.as_any().type_id();
            self.component_registry
                .entry(type_id)
                .or_default()
                .push(new_key);
        }

        let children_to_clone = node_to_clone.children.clone();
        for child_key in children_to_clone {
            if let Some(new_child_key) = self.duplicate_node_rec(child_key, new_key) {
                self.nodes[new_key].children.push(new_child_key);
            }
        }

        Some(new_key)
    }

    /// Обновляет глобальные трансформации для всех узлов сцены.
    /// Использует поуровневый параллелизм для корректного распространения изменений.
    ///
    /// Updates all global transforms in the scene.
    /// Uses level-based parallelism to ensure correct parent-child propagation.
    pub fn update_all_transforms(&mut self) {
        use rayon::prelude::*;

        let current_version = self
            .nodes_version
            .load(std::sync::atomic::Ordering::Acquire);
        if current_version != self.layers_version {
            let mut layers = Vec::new();
            let mut current_layer = vec![self.root];

            while !current_layer.is_empty() {
                let mut next_layer = Vec::new();
                for &key in &current_layer {
                    if let Some(node) = self.nodes.get(key) {
                        next_layer.extend(node.children.iter().copied());
                    }
                }
                layers.push(current_layer);
                current_layer = next_layer;
            }
            self.cached_layers = layers;
            self.layers_version = current_version;
        }

        // Итерация по уровням иерархии (от корня к листьям).
        // Iterate through hierarchy levels (root to leaves).
        for layer in &self.cached_layers {
            let nodes_ptr = &self.nodes as *const SlotMap<NodeKey, Node> as usize;
            layer.into_par_iter().for_each(|&key| unsafe {
                // SAFETY: We process the scene hierarchy layer by layer. Since each node belongs to exactly
                // one layer and we process layers sequentially from root to leaves, we guarantee that
                // parent transforms are already updated before children.
                // Parallel access within a layer is safe because each node key in the layer is unique,
                // ensuring disjoint mutable access to node data.
                let nodes = &*(nodes_ptr as *const SlotMap<NodeKey, Node>);
                let node = nodes
                    .get(key)
                    .expect("Node not found in SlotMap during transform update");
                let (parent_global, parent_dirty) = if let Some(parent_key) = node.parent {
                    let p = nodes
                        .get(parent_key)
                        .expect("Parent node not found during transform update");
                    (p.global_transform, p.is_dirty)
                } else {
                    (Mat4::IDENTITY, false)
                };

                if node.is_dirty || parent_dirty {
                    let node_ptr = node as *const Node as *mut Node;
                    (*node_ptr).global_transform = parent_global * (*node_ptr).local_transform;
                    (*node_ptr).is_dirty = true;
                }
            });
        }

        // Final pass to clear dirty flags and update view matrix
        for node in self.nodes.values_mut() {
            if node.is_dirty {
                for component in &node.components {
                    if component.as_any().is::<CameraComponent>() {
                        self.last_view_matrix = node.global_transform.inverse();
                    }
                }
                node.is_dirty = false;
            }
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
                } else if component.as_any().is::<LightComponent>()
                    || component.as_any().is::<CameraComponent>()
                {
                    has_bounds = true;
                } else if component.as_any().is::<SpriteComponent>() {
                    radius = 0.5; // Default for sprites
                    has_bounds = true;
                }
            }

            if !has_bounds {
                continue;
            }

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
    pub fn collect_frame_packet(
        &self,
        frustum: Option<&spark_math::Frustum>,
        asset_manager: &crate::asset::AssetManager,
    ) -> spark_renderer::resource::FramePacket {
        self.collect_frame_packet_internal(frustum, asset_manager)
    }

    fn collect_frame_packet_internal(
        &self,
        frustum: Option<&spark_math::Frustum>,
        asset_manager: &dyn spark_renderer::RenderableAssetManager,
    ) -> spark_renderer::resource::FramePacket {
        let mut data = SceneDataCollector::new();
        self.collect_data_recursive(self.root, frustum, &mut data);

        let (mut opaque_meshes, mut transparent_meshes) =
            self.classify_renderables(&data.renderables, asset_manager);

        self.process_instanced_data(
            data.instanced,
            asset_manager,
            &mut opaque_meshes,
            &mut transparent_meshes,
        );

        self.sort_transparent_meshes_back_to_front(&mut transparent_meshes);

        let lights = self.process_lights(data.lights);

        spark_renderer::resource::FramePacket {
            view_matrix: self.last_view_matrix,
            projection_matrix: spark_math::Mat4::IDENTITY,
            opaque_meshes,
            transparent_meshes,
            lights,
        }
    }

    fn process_lights(
        &self,
        lights: Vec<(Mat4, LightType, spark_math::Vec3, f32, f32, f32, f32)>,
    ) -> Vec<spark_renderer::resource::LightDraw> {
        self.convert_light_data(lights)
    }

    /// Classifies individual renderables into opaque and transparent lists.
    fn classify_renderables(
        &self,
        renderables: &[RenderableData],
        asset_manager: &dyn spark_renderer::RenderableAssetManager,
    ) -> (
        Vec<spark_renderer::resource::MeshDraw>,
        Vec<spark_renderer::resource::MeshDraw>,
    ) {
        use rayon::prelude::*;

        rayon::join(
            || {
                renderables
                    .par_iter()
                    .filter_map(|r| {
                        let mat_idx = r.6.unwrap_or(0);
                        if !asset_manager.is_material_transparent(mat_idx) {
                            Some(spark_renderer::resource::MeshDraw {
                                model: r.0,
                                vertex_count: r.1,
                                index_count: r.2,
                                first_index: r.3,
                                vertex_offset: r.4,
                                material_index: mat_idx,
                                bounding_radius: r.7,
                                mesh_id: r.3,
                            })
                        } else {
                            None
                        }
                    })
                    .collect()
            },
            || {
                renderables
                    .par_iter()
                    .filter_map(|r| {
                        let mat_idx = r.6.unwrap_or(0);
                        if asset_manager.is_material_transparent(mat_idx) {
                            Some(spark_renderer::resource::MeshDraw {
                                model: r.0,
                                vertex_count: r.1,
                                index_count: r.2,
                                first_index: r.3,
                                vertex_offset: r.4,
                                material_index: mat_idx,
                                bounding_radius: r.7,
                                mesh_id: r.3,
                            })
                        } else {
                            None
                        }
                    })
                    .collect()
            },
        )
    }

    /// Processes instanced data, classifying results into opaque and transparent lists.
    fn process_instanced_data(
        &self,
        mut instanced: Vec<(InstancedKey, Mat4)>,
        asset_manager: &dyn spark_renderer::RenderableAssetManager,
        opaque_meshes: &mut Vec<spark_renderer::resource::MeshDraw>,
        transparent_meshes: &mut Vec<spark_renderer::resource::MeshDraw>,
    ) {
        use rayon::prelude::*;

        instanced.par_sort_unstable_by_key(|&(key, _)| key);

        let results: Vec<Vec<(spark_renderer::resource::MeshDraw, bool)>> = instanced
            .par_chunk_by(|a, b| a.0 == b.0)
            .map(|chunk| {
                let (ic, fi, vo, _tex, mat_idx, br_bits) = chunk[0].0;
                let br = f32::from_bits(br_bits);
                let midx = mat_idx.unwrap_or(0);
                let is_transparent = asset_manager.is_material_transparent(midx);

                chunk
                    .iter()
                    .map(move |&(_, t)| {
                        let draw = spark_renderer::resource::MeshDraw {
                            model: t,
                            vertex_count: 0,
                            index_count: ic,
                            first_index: fi,
                            vertex_offset: vo,
                            material_index: midx,
                            bounding_radius: br,
                            mesh_id: fi,
                        };
                        (draw, is_transparent)
                    })
                    .collect()
            })
            .collect();

        for batch in results {
            for (draw, is_trans) in batch {
                if is_trans {
                    transparent_meshes.push(draw);
                } else {
                    opaque_meshes.push(draw);
                }
            }
        }
    }

    /// Sorts transparent meshes back-to-front based on the last view position.
    fn sort_transparent_meshes_back_to_front(
        &self,
        meshes: &mut [spark_renderer::resource::MeshDraw],
    ) {
        use rayon::prelude::*;

        let view_pos = self.last_view_matrix.inverse().w_axis.xyz();
        meshes.par_sort_by(|a, b| {
            let dist_a = (a.model.w_axis.xyz() - view_pos).length_squared();
            let dist_b = (b.model.w_axis.xyz() - view_pos).length_squared();
            dist_b
                .partial_cmp(&dist_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Converts light data collected from the scene into a format suitable for the renderer.
    fn convert_light_data(
        &self,
        lights: Vec<(Mat4, LightType, spark_math::Vec3, f32, f32, f32, f32)>,
    ) -> Vec<spark_renderer::resource::LightDraw> {
        use rayon::prelude::*;

        lights
            .into_par_iter()
            .map(
                |(t, light_type, color, intensity, range, spot_inner, spot_outer)| {
                    let position = spark_math::Vec3::new(t.w_axis.x, t.w_axis.y, t.w_axis.z);
                    let direction =
                        -spark_math::Vec3::new(t.z_axis.x, t.z_axis.y, t.z_axis.z).normalize();
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
                },
            )
            .collect()
    }

    fn collect_data_recursive(
        &self,
        node_key: NodeKey,
        frustum: Option<&spark_math::Frustum>,
        data: &mut SceneDataCollector,
    ) {
        if let Some(node) = self.nodes.get(node_key) {
            for component in &node.components {
                let any = component.as_any();
                if let Some(mesh) = any.downcast_ref::<MeshComponent>() {
                    let visible = if let Some(f) = frustum {
                        let gt = node.global_transform;
                        let translation =
                            spark_math::Vec3::new(gt.w_axis.x, gt.w_axis.y, gt.w_axis.z);
                        f.intersects_sphere(translation, mesh.bounding_radius)
                    } else {
                        true
                    };
                    if visible {
                        let gt = node.global_transform;
                        if mesh.material_index.is_some() {
                            data.instanced.push((
                                (
                                    mesh.index_count,
                                    mesh.first_index,
                                    mesh.vertex_offset,
                                    mesh.texture_handle,
                                    mesh.material_index,
                                    (mesh.bounding_radius).to_bits(),
                                ),
                                gt,
                            ));
                        } else {
                            data.renderables.push((
                                gt,
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
                    let gt = node.global_transform;
                    data.lights.push((
                        gt,
                        light.light_type,
                        light.color,
                        light.intensity,
                        light.range,
                        light.spot_inner_angle.to_radians(),
                        light.spot_outer_angle.to_radians(),
                    ));
                } else if let Some(sprite) = any.downcast_ref::<SpriteComponent>() {
                    let gt = node.global_transform;
                    data.sprites
                        .push((gt, sprite.texture_handle, sprite.color, sprite.size));
                }
            }

            if !node.children.is_empty() {
                // Heuristic: Use parallel collection only for branches with many children
                if node.children.len() > 100 {
                    data.merge(self.collect_data_parallel(&node.children, frustum));
                } else {
                    for &child_key in &node.children {
                        self.collect_data_recursive(child_key, frustum, data);
                    }
                }
            }
        }
    }

    fn collect_data_parallel(
        &self,
        children: &[NodeKey],
        frustum: Option<&spark_math::Frustum>,
    ) -> SceneDataCollector {
        if children.len() <= 1 {
            let mut data = SceneDataCollector::new();
            self.collect_data_recursive(children[0], frustum, &mut data);
            return data;
        }

        let mid = children.len() / 2;
        let (left, right) = children.split_at(mid);

        let (mut left_data, right_data) = rayon::join(
            || self.collect_data_parallel(left, frustum),
            || self.collect_data_parallel(right, frustum),
        );

        left_data.merge(right_data);
        left_data
    }

    pub fn pick_node_parallel(&self, ray: &spark_math::Ray) -> Option<(NodeKey, f32)> {
        use rayon::prelude::*;
        let nodes: Vec<_> = self.nodes.iter().collect();
        nodes
            .into_par_iter()
            .filter_map(|(key, node)| {
                let mut radius = 0.5f32;
                let mut has_bounds = false;

                for component in &node.components {
                    if let Some(mesh) = component.as_any().downcast_ref::<MeshComponent>() {
                        radius = mesh.bounding_radius;
                        has_bounds = true;
                    } else if component.as_any().is::<LightComponent>()
                        || component.as_any().is::<CameraComponent>()
                    {
                        has_bounds = true;
                    } else if component.as_any().is::<SpriteComponent>() {
                        radius = 0.5;
                        has_bounds = true;
                    }
                }

                if !has_bounds {
                    return None;
                }

                let center = node.global_transform.w_axis.xyz();
                ray.intersect_sphere(center, radius).map(|t| (key, t))
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }
}
