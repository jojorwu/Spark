use slotmap::{SlotMap, new_key_type};
use spark_math::{Mat4, Vec4Swizzles};

new_key_type! { pub struct NodeKey; }

pub enum NodeData {
    None,
    Mesh {
        vertex_count: u32,
        texture_id: Option<String>,
        vertex_buffer_id: Option<u32>,
        bounding_radius: f32,
    },
    Camera {
        fov: f32,
        near: f32,
        far: f32,
    },
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

        Self { nodes, root }
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

            // To avoid cloning, we'd need an iterative approach with a stack or a more complex borrow
            // For now, cloning the keys (not the nodes) is relatively cheap.
            let children = node.children.clone();
            for child_key in children {
                self.update_transforms_recursive(child_key, current_global);
            }
        }
    }

    pub fn update_all_transforms(&mut self) {
        // For simple parent-child propagation, sequential is usually fine.
        // But for true multi-threading with Rayon, we'd need a more decoupled approach.
        // For now, we'll keep the recursive update but ensure we can scale.
        self.update_transforms(self.root);
    }

    pub fn collect_render_data(
        &self,
        frustum: Option<&spark_math::Frustum>,
    ) -> (Vec<(Mat4, u32, Option<String>, Option<u32>)>, Vec<(u32, Vec<Mat4>)>, Mat4) {
        let mut renderables = Vec::new();
        let mut instanced = std::collections::HashMap::new();
        let mut view_matrix = Mat4::IDENTITY;
        self.collect_data_recursive(self.root, &mut renderables, &mut instanced, &mut view_matrix, frustum);

        let instanced_data = instanced.into_iter().map(|(vb_id, transforms)| (vb_id, transforms)).collect();

        (renderables, instanced_data, view_matrix)
    }

    fn collect_data_recursive(
        &self,
        node_key: NodeKey,
        renderables: &mut Vec<(Mat4, u32, Option<String>, Option<u32>)>,
        instanced: &mut std::collections::HashMap<u32, Vec<Mat4>>,
        view_matrix: &mut Mat4,
        frustum: Option<&spark_math::Frustum>,
    ) {
        if let Some(node) = self.nodes.get(node_key) {
            match &node.data {
                NodeData::Mesh {
                    vertex_count,
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
                            instanced.entry(*vb_id).or_insert_with(Vec::new).push(node.global_transform);
                        } else {
                            renderables.push((
                                node.global_transform,
                                *vertex_count,
                                texture_id.clone(),
                                *vertex_buffer_id,
                            ));
                        }
                    }
                }
                NodeData::Camera { .. } => {
                    *view_matrix = node.global_transform.inverse();
                }
                _ => {}
            }
            for child in &node.children {
                self.collect_data_recursive(*child, renderables, instanced, view_matrix, frustum);
            }
        }
    }
}
