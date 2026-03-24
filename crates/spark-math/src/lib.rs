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

    pub fn intersects_aabb(&self, aabb: &AABB) -> bool {
        for plane in &self.planes {
            let mut p = aabb.min;
            if plane.x >= 0.0 { p.x = aabb.max.x; }
            if plane.y >= 0.0 { p.y = aabb.max.y; }
            if plane.z >= 0.0 { p.z = aabb.max.z; }

            let xyz = Vec3::new(plane.x, plane.y, plane.z);
            if xyz.dot(p) + plane.w < 0.0 {
                return false;
            }
        }
        true
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct AABB {
    pub min: Vec3,
    pub max: Vec3,
}

impl AABB {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    pub fn from_points(points: &[Vec3]) -> Self {
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        for &p in points {
            min = min.min(p);
            max = max.max(p);
        }
        Self { min, max }
    }

    pub fn transform(&self, m: Mat4) -> Self {
        let corners = [
            m.transform_point3(Vec3::new(self.min.x, self.min.y, self.min.z)),
            m.transform_point3(Vec3::new(self.max.x, self.min.y, self.min.z)),
            m.transform_point3(Vec3::new(self.min.x, self.max.y, self.min.z)),
            m.transform_point3(Vec3::new(self.max.x, self.max.y, self.min.z)),
            m.transform_point3(Vec3::new(self.min.x, self.min.y, self.max.z)),
            m.transform_point3(Vec3::new(self.max.x, self.min.y, self.max.z)),
            m.transform_point3(Vec3::new(self.min.x, self.max.y, self.max.z)),
            m.transform_point3(Vec3::new(self.max.x, self.max.y, self.max.z)),
        ];
        Self::from_points(&corners)
    }

    pub fn merge(&self, other: &AABB) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
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
