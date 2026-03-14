use std::collections::HashMap;
use std::path::PathBuf;
use image::DynamicImage;

pub struct ResourceManager {
    textures: HashMap<PathBuf, DynamicImage>,
    scenes: HashMap<PathBuf, gltf::Document>,
}

impl ResourceManager {
    pub fn new() -> Self {
        Self {
            textures: HashMap::new(),
            scenes: HashMap::new(),
        }
    }

    pub fn load_scene(&mut self, path: PathBuf) -> &gltf::Document {
        self.scenes.entry(path.clone()).or_insert_with(|| {
            log::info!("Loading glTF scene: {:?}", path);
            let (doc, _, _) = gltf::import(path).expect("Failed to load glTF");
            doc
        })
    }

    pub fn load_texture(&mut self, path: PathBuf) -> &DynamicImage {
        self.textures.entry(path.clone()).or_insert_with(|| {
            log::info!("Loading texture: {:?}", path);
            image::open(path).expect("Failed to load texture")
        })
    }
}
