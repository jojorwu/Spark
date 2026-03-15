pub use glam::*;
pub use glam::f32::Quat;
pub use glam::Vec4Swizzles;

pub struct Frustum {
    pub planes: [Vec4; 6],
}

impl Frustum {
    pub fn from_matrix(m: Mat4) -> Self {
        let mut planes = [Vec4::ZERO; 6];
        // Left
        planes[0] = m.row(3) + m.row(0);
        // Right
        planes[1] = m.row(3) - m.row(0);
        // Bottom
        planes[2] = m.row(3) + m.row(1);
        // Top
        planes[3] = m.row(3) - m.row(1);
        // Near
        planes[4] = m.row(3) + m.row(2);
        // Far
        planes[5] = m.row(3) - m.row(2);

        for plane in &mut planes {
            *plane /= plane.xyz().length();
        }

        Self { planes }
    }

    pub fn intersects_sphere(&self, center: Vec3, radius: f32) -> bool {
        for plane in &self.planes {
            if plane.xyz().dot(center) + plane.w + radius < 0.0 {
                return false;
            }
        }
        true
    }
}
