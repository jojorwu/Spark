use crate::resource::ResourceManager;
use crate::scene::Scene;
use crate::scene::{Component, Node, NodeKey};

pub trait Command: Send + Sync {
    fn apply(&mut self, scene: &mut Scene, resource_manager: &mut ResourceManager);
}

unsafe impl Send for AddNodeCommand {}
unsafe impl Sync for AddNodeCommand {}

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

pub struct TransformCommand {
    pub node: NodeKey,
    pub transform: spark_math::Mat4,
    pub relative: bool,
}

impl Command for TransformCommand {
    fn apply(&mut self, scene: &mut Scene, _rm: &mut ResourceManager) {
        if let Some(node) = scene.nodes.get_mut(self.node) {
            if self.relative {
                node.local_transform *= self.transform;
            } else {
                node.local_transform = self.transform;
            }
            node.is_dirty = true;
        }
    }
}

pub struct RemoveNodeCommand {
    pub key: NodeKey,
}

impl Command for RemoveNodeCommand {
    fn apply(&mut self, scene: &mut Scene, _rm: &mut ResourceManager) {
        scene.remove_node(self.key);
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

use std::sync::Mutex;

pub struct CommandQueue {
    commands: Mutex<Vec<Box<dyn Command>>>,
}

impl CommandQueue {
    pub fn new() -> Self {
        Self {
            commands: Mutex::new(Vec::new()),
        }
    }

    pub fn push<C: Command + 'static>(&self, command: C) {
        self.commands.lock().unwrap().push(Box::new(command));
    }

    pub fn execute_all(&self, scene: &mut Scene, resource_manager: &mut ResourceManager) {
        let mut commands = {
            let mut guard = self.commands.lock().unwrap();
            std::mem::take(&mut *guard)
        };
        for mut command in commands.drain(..) {
            command.apply(scene, resource_manager);
        }
    }
}

impl Default for CommandQueue {
    fn default() -> Self {
        Self::new()
    }
}
