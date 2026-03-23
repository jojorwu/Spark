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
            let xyz = Vec3::new(plane.x, plane.y, plane.z);
            let length = xyz.length();
            *plane /= length;
        }

        Self { planes }
    }

    pub fn intersects_sphere(&self, center: Vec3, radius: f32) -> bool {
        for plane in &self.planes {
            let xyz = Vec3::new(plane.x, plane.y, plane.z);
            if xyz.dot(center) + plane.w + radius < 0.0 {
                return false;
            }
        }
        true
    }
}

pub struct Ray {
    pub origin: Vec3,
    pub direction: Vec3,
}

impl Ray {
    pub fn new(origin: Vec3, direction: Vec3) -> Self {
        Self {
            origin,
            direction: direction.normalize(),
        }
    }

    pub fn intersect_sphere(&self, center: Vec3, radius: f32) -> Option<f32> {
        let oc = self.origin - center;
        let a = self.direction.dot(self.direction);
        let b = 2.0 * oc.dot(self.direction);
        let c = oc.dot(oc) - radius * radius;
        let discriminant = b * b - 4.0 * a * c;

        if discriminant < 0.0 {
            None
        } else {
            let t = (-b - discriminant.sqrt()) / (2.0 * a);
            if t > 0.0 {
                Some(t)
            } else {
                let t = (-b + discriminant.sqrt()) / (2.0 * a);
                if t > 0.0 {
                    Some(t)
                } else {
                    None
                }
            }
        }
    }
}

pub use glam::Vec4Swizzles as _;
