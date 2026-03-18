use slotmap::{SlotMap, new_key_type};
use spark_math::{Mat4, Vec4Swizzles};
use rayon::prelude::*;

new_key_type! { pub struct NodeKey; }

pub enum NodeData {
    None,
    Mesh {
        vertex_count: u32,
        index_count: u32,
        first_index: u32,
        vertex_offset: i32,
        texture_id: Option<u32>,
        vertex_buffer_id: Option<u32>,
        bounding_radius: f32,
    },
    Camera {
        fov: f32,
        near: f32,
        far: f32,
    },
    Light {
        light_type: LightType,
        color: spark_math::Vec3,
        intensity: f32,
        range: f32, // For point lights
    },
}

#[derive(Clone, Copy, Debug)]
pub enum LightType {
    Directional,
    Point,
}

pub struct Node {
    pub name: String,
    pub local_transform: Mat4,
    pub global_transform: Mat4,
    pub parent: Option<NodeKey>,
    pub children: Vec<NodeKey>,
    pub data: NodeData,
}

pub struct Scene {
    pub nodes: SlotMap<NodeKey, Node>,
    pub root: NodeKey,
    pub last_view_matrix: Mat4,
}

struct SceneDataCollector {
    renderables: Vec<(Mat4, u32, u32, u32, i32, Option<u32>, Option<u32>)>,
    instanced: std::collections::HashMap<(u32, u32, i32, Option<u32>, u32), Vec<Mat4>>,
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
            data: NodeData::None,
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

            if let NodeData::Camera { .. } = node.data {
                self.last_view_matrix = current_global.inverse();
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

    /// Collects renderables, instanced meshes, and lights from the scene tree, performing frustum culling.
    /// This method is parallelized using Rayon for high performance in large scenes.
    pub fn collect_render_data(
        &self,
        frustum: Option<&spark_math::Frustum>,
    ) -> (
        Vec<(Mat4, u32, u32, u32, i32, Option<u32>, Option<u32>)>,
        Vec<(u32, u32, i32, Option<u32>, u32, Vec<Mat4>)>,
        Mat4,
        Vec<(Mat4, LightType, spark_math::Vec3, f32, f32)>
    ) {
        let data = self.collect_data_parallel(self.root, frustum);

        let instanced_data = data.instanced.into_iter()
            .map(|((ic, fi, vo, tex_id, vb_id), transforms)| (ic, fi, vo, tex_id, vb_id, transforms))
            .collect();

        (data.renderables, instanced_data, self.last_view_matrix, data.lights)
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
            match &node.data {
                NodeData::Mesh {
                    vertex_count,
                    index_count,
                    first_index,
                    vertex_offset,
                    texture_id,
                    vertex_buffer_id,
                    bounding_radius,
                } => {
                    let visible = if let Some(f) = frustum {
                        let translation = node.global_transform.w_axis.xyz();
                        f.intersects_sphere(translation, *bounding_radius)
                    } else {
                        true
                    };
                    if visible {
                        if let Some(vb_id) = vertex_buffer_id {
                            data.instanced.entry((*index_count, *first_index, *vertex_offset, *texture_id, *vb_id)).or_default().push(node.global_transform);
                        } else {
                            data.renderables.push((
                                node.global_transform,
                                *vertex_count,
                                *index_count,
                                *first_index,
                                *vertex_offset,
                                *texture_id,
                                *vertex_buffer_id,
                            ));
                        }
                    }
                }
                NodeData::Light { light_type, color, intensity, range } => {
                    data.lights.push((node.global_transform, *light_type, *color, *intensity, *range));
                }
                _ => {}
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
