use std::collections::HashMap;
use std::path::PathBuf;
use image::DynamicImage;

pub struct ResourceManager {
    pub textures: HashMap<PathBuf, DynamicImage>,
    pub scenes: HashMap<PathBuf, gltf::Document>,
}

impl ResourceManager {
    pub fn new() -> Self {
        Self {
            textures: HashMap::new(),
            scenes: HashMap::new(),
        }
    }

    pub fn load_scene(&mut self, path: PathBuf, scene_tree: &mut crate::scene::Scene) {
        log::info!("Loading glTF scene: {:?}", path);
        let (doc, buffers, _) = gltf::import(path).expect("Failed to load glTF");

        for gltf_scene in doc.scenes() {
            for node in gltf_scene.nodes() {
                self.process_gltf_node(node, &buffers, scene_tree, scene_tree.root);
            }
        }
    }

    fn process_gltf_node(
        &self,
        node: gltf::Node,
        buffers: &[gltf::buffer::Data],
        scene_tree: &mut crate::scene::Scene,
        parent: crate::scene::NodeKey,
    ) {
        use crate::scene::{Node, NodeData};
        use spark_math::{Mat4, Vec3, Quat};

        let (translation, rotation, scale) = node.transform().decomposed();
        let local_transform = Mat4::from_scale_rotation_translation(
            Vec3::from_array(scale),
            Quat::from_array(rotation),
            Vec3::from_array(translation),
        );

        let data = if let Some(mesh) = node.mesh() {
            let mut vertex_count = 0;
            for primitive in mesh.primitives() {
                let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                if let Some(pos_iter) = reader.read_positions() {
                    vertex_count += pos_iter.count() as u32;
                }
            }
            NodeData::Mesh { vertex_count, texture_id: None, vertex_buffer_id: None }
        } else {
            NodeData::None
        };

        let spark_node = Node {
            name: node.name().unwrap_or("Unnamed Node").to_string(),
            local_transform,
            global_transform: Mat4::IDENTITY,
            parent: None,
            children: Vec::new(),
            data,
        };

        let key = scene_tree.add_node(parent, spark_node);

        for child in node.children() {
            self.process_gltf_node(child, buffers, scene_tree, key);
        }
    }

    pub fn load_texture(&mut self, path: PathBuf) -> &DynamicImage {
        self.textures.entry(path.clone()).or_insert_with(|| {
            log::info!("Loading texture: {:?}", path);
            image::open(path).expect("Failed to load texture")
        })
    }
}
