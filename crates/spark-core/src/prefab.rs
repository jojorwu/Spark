use serde::{Serialize, Deserialize};
use crate::scene::{Node, NodeKey, Scene};

#[derive(Serialize, Deserialize, Clone)]
pub struct Prefab {
    pub root_node: Node,
}

impl Prefab {
    pub fn instantiate(&self, scene: &mut Scene, parent: NodeKey) -> NodeKey {
        let node = self.root_node.clone();
        scene.add_node(parent, node)
    }

    pub fn from_node(scene: &Scene, key: NodeKey) -> Option<Self> {
        scene.nodes.get(key).map(|node| {
            Self {
                root_node: node.clone(),
            }
        })
    }

    pub fn save_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load_from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let prefab: Prefab = serde_json::from_str(&content)?;
        Ok(prefab)
    }
}
