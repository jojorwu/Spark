use crate::scene::{Node, NodeKey, Component};
use crate::resource::ResourceManager;
use crate::scene::Scene;

pub trait Command: Send + Sync {
    fn apply(&mut self, scene: &mut Scene, resource_manager: &mut ResourceManager);
}

pub struct AddNodeCommand {
    pub parent: NodeKey,
    pub node: Option<Node>,
}

impl Command for AddNodeCommand {
    fn apply(&mut self, scene: &mut Scene, _rm: &mut ResourceManager) {
        if let Some(node) = self.node.take() {
            scene.add_node(self.parent, node);
        }
    }
}

pub struct RemoveNodeCommand {
    pub key: NodeKey,
}

impl Command for RemoveNodeCommand {
    fn apply(&mut self, scene: &mut Scene, _rm: &mut ResourceManager) {
        // Simple implementation: remove from parent and then from map
        if let Some(node) = scene.nodes.get(self.key) {
            if let Some(parent_key) = node.parent {
                if let Some(parent) = scene.nodes.get_mut(parent_key) {
                    parent.children.retain(|&k| k != self.key);
                }
            }
        }
        scene.nodes.remove(self.key);
    }
}

pub struct AddComponentCommand {
    pub node_key: NodeKey,
    pub component: Option<Box<dyn Component>>,
}

impl Command for AddComponentCommand {
    fn apply(&mut self, scene: &mut Scene, _rm: &mut ResourceManager) {
        if let Some(node) = scene.nodes.get_mut(self.node_key) {
            if let Some(comp) = self.component.take() {
                node.components.push(comp);
            }
        }
    }
}

pub struct CommandQueue {
    commands: Vec<Box<dyn Command>>,
}

impl CommandQueue {
    pub fn new() -> Self {
        Self { commands: Vec::new() }
    }

    pub fn push<C: Command + 'static>(&mut self, command: C) {
        self.commands.push(Box::new(command));
    }

    pub fn execute_all(&mut self, scene: &mut Scene, resource_manager: &mut ResourceManager) {
        for mut command in self.commands.drain(..) {
            command.apply(scene, resource_manager);
        }
    }
}

impl Default for CommandQueue {
    fn default() -> Self {
        Self::new()
    }
}
