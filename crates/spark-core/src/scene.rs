use slotmap::{SlotMap, new_key_type};
use spark_math::{Mat4, Vec4Swizzles};
use rayon::prelude::*;
use serde::{Serialize, Deserialize};

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

struct SceneDataCollector {
    renderables: Vec<(Mat4, u32, u32, u32, i32, Option<TextureHandle>, Option<u32>, f32)>,
    instanced: std::collections::HashMap<(u32, u32, i32, Option<TextureHandle>, Option<u32>, u32), Vec<Mat4>>,
    lights: Vec<(Mat4, LightType, spark_math::Vec3, f32, f32)>,
}

impl SceneDataCollector {
    fn merge(mut self, other: Self) -> Self {
        self.renderables.extend(other.renderables);
        for (key, transforms) in other.instanced {
            self.instanced.entry(key).or_default().extend(transforms);
        }
        self.lights.extend(other.lights);
        self
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

    pub fn add_node(&mut self, parent: NodeKey, mut node: Node) -> NodeKey {
        node.parent = Some(parent);
        let key = self.nodes.insert(node);
        if let Some(parent_node) = self.nodes.get_mut(parent) {
            parent_node.children.push(key);
        }
        self.update_transforms(key);
        key
    }

    pub fn update_transforms(&mut self, start_node: NodeKey) {
        let parent_global = if let Some(node) = self.nodes.get(start_node) {
            if let Some(parent_key) = node.parent {
                self.nodes.get(parent_key).map(|p| p.global_transform).unwrap_or(Mat4::IDENTITY)
            } else {
                Mat4::IDENTITY
            }
        } else {
            return;
        };

        self.update_transforms_recursive(start_node, parent_global);
    }

    fn update_transforms_recursive(&mut self, node_key: NodeKey, parent_global: Mat4) {
        if let Some(node) = self.nodes.get_mut(node_key) {
            node.global_transform = parent_global * node.local_transform;
            let current_global = node.global_transform;

            for component in &node.components {
                if component.as_any().is::<CameraComponent>() {
                    self.last_view_matrix = current_global.inverse();
                }
            }

            // To avoid cloning, we'd need an iterative approach with a stack or a more complex borrow
            // For now, cloning the keys (not the nodes) is relatively cheap.
            let children = node.children.clone();
            for child_key in children {
                self.update_transforms_recursive(child_key, current_global);
            }
        }
    }

    /// Updates global transforms for all nodes in the scene tree and caches the active camera's view matrix.
    pub fn update_all_transforms(&mut self) {
        // For simple parent-child propagation, sequential is usually fine.
        // But for true multi-threading with Rayon, we'd need a more decoupled approach.
        // For now, we'll keep the recursive update but ensure we can scale.
        self.update_transforms(self.root);
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
    pub fn collect_frame_packet(&self, frustum: Option<&spark_math::Frustum>) -> spark_renderer::resource::FramePacket {
        let data = self.collect_data_parallel(self.root, frustum);

        let mut meshes = Vec::new();
        for r in data.renderables {
            meshes.push(spark_renderer::resource::MeshDraw {
                model: r.0,
                vertex_count: r.1,
                index_count: r.2,
                first_index: r.3,
                vertex_offset: r.4,
                material_index: r.6.unwrap_or(0),
                bounding_radius: r.7,
            });
        }

        for ((ic, fi, vo, _tex, mat_idx, br_bits), transforms) in data.instanced {
            let br = f32::from_bits(br_bits);
            for t in transforms {
                meshes.push(spark_renderer::resource::MeshDraw {
                    model: t,
                    vertex_count: 0, // Not needed for indirect
                    index_count: ic,
                    first_index: fi,
                    vertex_offset: vo,
                    material_index: mat_idx.unwrap_or(0),
                    bounding_radius: br,
                });
            }
        }

        let lights = data.lights.into_iter().map(|(t, _type, color, intensity, _range)| {
            spark_renderer::resource::LightDraw {
                position: t.w_axis.xyz(),
                color,
                intensity,
            }
        }).collect();

        spark_renderer::resource::FramePacket {
            view_matrix: self.last_view_matrix,
            meshes,
            lights,
        }
    }

    fn collect_data_parallel(
        &self,
        node_key: NodeKey,
        frustum: Option<&spark_math::Frustum>,
    ) -> SceneDataCollector {
        let mut data = SceneDataCollector {
            renderables: Vec::new(),
            instanced: std::collections::HashMap::new(),
            lights: Vec::new(),
        };

        if let Some(node) = self.nodes.get(node_key) {
            for component in &node.components {
                let any = component.as_any();
                if let Some(mesh) = any.downcast_ref::<MeshComponent>() {
                    let visible = if let Some(f) = frustum {
                        let translation = node.global_transform.w_axis.xyz();
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
                let children_data: SceneDataCollector = node.children.par_iter()
                    .map(|&child_key| self.collect_data_parallel(child_key, frustum))
                    .reduce(|| SceneDataCollector {
                        renderables: Vec::new(),
                        instanced: std::collections::HashMap::new(),
                        lights: Vec::new(),
                    }, |a, b| a.merge(b));

                data.renderables.extend(children_data.renderables);
                for (key, transforms) in children_data.instanced {
                    data.instanced.entry(key).or_default().extend(transforms);
                }
                data.lights.extend(children_data.lights);
            }
        }
        data
    }
}
