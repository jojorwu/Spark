use image::DynamicImage;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

#[derive(Debug, Serialize, Deserialize)]
pub struct Handle<T> {
    id: u32,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Handle<T> {}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T> Eq for Handle<T> {}

impl<T> std::hash::Hash for Handle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl<T> Handle<T> {
    pub fn new(id: u32) -> Self {
        Self {
            id,
            _marker: PhantomData,
        }
    }
    pub fn id(&self) -> u32 {
        self.id
    }
}

pub struct AssetStorage<T> {
    assets: Vec<T>,
}

impl<T> AssetStorage<T> {
    pub fn new() -> Self {
        Self { assets: Vec::new() }
    }
    pub fn add(&mut self, asset: T) -> Handle<T> {
        let id = self.assets.len() as u32;
        self.assets.push(asset);
        Handle::new(id)
    }
    pub fn get(&self, handle: Handle<T>) -> Option<&T> {
        self.assets.get(handle.id as usize)
    }
    pub fn get_mut(&mut self, handle: Handle<T>) -> Option<&mut T> {
        self.assets.get_mut(handle.id as usize)
    }
    pub fn assets_len(&self) -> usize {
        self.assets.len()
    }
}

impl<T> Default for AssetStorage<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Material {
    pub name: String,
    pub albedo_factor: [f32; 4],
    pub emissive_factor: [f32; 4],
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub albedo_texture: Option<Handle<spark_renderer::vulkan::texture::Texture>>,
    pub normal_texture: Option<Handle<spark_renderer::vulkan::texture::Texture>>,
    pub metallic_roughness_texture: Option<Handle<spark_renderer::vulkan::texture::Texture>>,
    pub is_transparent: bool,
}

impl std::fmt::Debug for Material {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Material")
            .field("name", &self.name)
            .finish()
    }
}

pub struct AssetManager {
    pub textures: AssetStorage<DynamicImage>,
    pub materials: AssetStorage<Material>,
    pub texture_path_map: HashMap<PathBuf, Handle<spark_renderer::vulkan::texture::Texture>>,
}

impl AssetManager {
    pub fn new() -> Self {
        Self {
            textures: AssetStorage::new(),
            materials: AssetStorage::new(),
            texture_path_map: HashMap::new(),
        }
    }

    pub fn load_gltf(
        &mut self,
        path: &Path,
        scene: &mut crate::scene::Scene,
        renderer: &spark_renderer::Renderer,
        resource_manager: &mut crate::resource::ResourceManager,
    ) {
        crate::gltf_loader::GltfLoader::load_scene(
            resource_manager,
            self,
            path.to_path_buf(),
            scene,
            renderer,
        );
    }
}

impl Default for AssetManager {
    fn default() -> Self {
        Self::new()
    }
}
