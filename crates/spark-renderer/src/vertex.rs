use ash::vk;
use spark_math::{Vec2, Vec3};
use std::mem;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub pos: Vec3,
    pub normal: Vec3,
    pub color: Vec3,
    pub tex_coord: Vec2,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct InstanceData {
    pub model: spark_math::Mat4,
}

impl InstanceData {
    pub fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(1)
            .stride(mem::size_of::<InstanceData>() as u32)
            .input_rate(vk::VertexInputRate::INSTANCE)
    }

    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 4] {
        [
            // Mat4 takes 4 locations
            vk::VertexInputAttributeDescription::default()
                .binding(1)
                .location(4)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescription::default()
                .binding(1)
                .location(5)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(16),
            vk::VertexInputAttributeDescription::default()
                .binding(1)
                .location(6)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(32),
            vk::VertexInputAttributeDescription::default()
                .binding(1)
                .location(7)
                .format(vk::Format::R32G32B32A32_SFLOAT)
                .offset(48),
        ]
    }
}

impl Vertex {
    pub fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 4] {
        [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(12), // normal
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(2)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(24), // color
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(3)
                .format(vk::Format::R32G32_SFLOAT)
                .offset(36), // tex_coord
        ]
    }
}
