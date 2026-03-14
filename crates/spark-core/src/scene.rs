use slotmap::{SlotMap, new_key_type};
use spark_math::Mat4;

new_key_type! { pub struct NodeKey; }

pub enum NodeData {
    None,
    Mesh {
        vertex_count: u32,
        // In the future, this will hold a reference to a GPU buffer handle
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
        let (parent_global, children) = if let Some(node) = self.nodes.get(start_node) {
            let parent_global = if let Some(parent_key) = node.parent {
                self.nodes.get(parent_key).map(|p| p.global_transform).unwrap_or(Mat4::IDENTITY)
            } else {
                Mat4::IDENTITY
            };
            (parent_global, node.children.clone())
        } else {
            return;
        };

        if let Some(node) = self.nodes.get_mut(start_node) {
            node.global_transform = parent_global * node.local_transform;
        }

        for child in children {
            self.update_transforms(child);
        }
    }

    pub fn update_all_transforms(&mut self) {
        self.update_transforms(self.root);
    }

    pub fn collect_render_data(&self) -> (Vec<(Mat4, u32)>, Mat4) {
        let mut renderables = Vec::new();
        let mut view_matrix = Mat4::IDENTITY;
        self.collect_data_recursive(self.root, &mut renderables, &mut view_matrix);
        (renderables, view_matrix)
    }

    fn collect_data_recursive(
        &self,
        node_key: NodeKey,
        renderables: &mut Vec<(Mat4, u32)>,
        view_matrix: &mut Mat4,
    ) {
        if let Some(node) = self.nodes.get(node_key) {
            match node.data {
                NodeData::Mesh { vertex_count } => {
                    renderables.push((node.global_transform, vertex_count));
                }
                NodeData::Camera { .. } => {
                    *view_matrix = node.global_transform.inverse();
                }
                _ => {}
            }
            for child in &node.children {
                self.collect_data_recursive(*child, renderables, view_matrix);
            }
        }
    }
}
