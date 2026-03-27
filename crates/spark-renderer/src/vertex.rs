use ash::vk;
use spark_math::Vec3;
use std::mem;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub padding: f32,
    pub normal: u32,    // Packed 10_10_10_2
    pub tex_coord: u32, // Packed 16_16
    pub color: u32,     // Packed 8_8_8_8
    pub tangent: u32,   // Packed 10_10_10_2
}

impl Vertex {
    pub fn pack(
        pos: Vec3,
        normal: Vec3,
        tex_coord: spark_math::Vec2,
        color: Vec3,
        tangent: Vec3,
    ) -> Self {
        let pack_10_10_10_2 = |n: Vec3| -> u32 {
            let n = n.normalize();
            let x = ((n.x * 0.5 + 0.5) * 1023.0) as u32;
            let y = ((n.y * 0.5 + 0.5) * 1023.0) as u32;
            let z = ((n.z * 0.5 + 0.5) * 1023.0) as u32;
            x | (y << 10) | (z << 20)
        };

        let pack_tc = |tc: spark_math::Vec2| -> u32 {
            let x = (tc.x * 65535.0) as u32 & 0xFFFF;
            let y = (tc.y * 65535.0) as u32 & 0xFFFF;
            x | (y << 16)
        };

        let pack_color = |c: Vec3| -> u32 {
            let r = (c.x * 255.0) as u32 & 0xFF;
            let g = (c.y * 255.0) as u32 & 0xFF;
            let b = (c.z * 255.0) as u32 & 0xFF;
            r | (g << 8) | (b << 16) | (255 << 24)
        };

        Self {
            pos: pos.to_array(),
            padding: 0.0,
            normal: pack_10_10_10_2(normal),
            tex_coord: pack_tc(tex_coord),
            color: pack_color(color),
            tangent: pack_10_10_10_2(tangent),
        }
    }

    pub fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(mem::size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)
    }

    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 5] {
        [
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::A2B10G10R10_UNORM_PACK32)
                .offset(16),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(2)
                .format(vk::Format::R16G16_UNORM)
                .offset(20),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(3)
                .format(vk::Format::R8G8B8A8_UNORM)
                .offset(24),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(4)
                .format(vk::Format::A2B10G10R10_UNORM_PACK32)
                .offset(28),
        ]
    }
}
