pub use glam::f32::Quat;
pub use glam::Vec4Swizzles;
pub use glam::*;

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
            // Using the dot product of the plane normal and the sphere center
            // plus the plane's W component (distance from origin).
            // plane.xyz is the normal, plane.w is the distance.
            if plane.xyz().dot(center) + plane.w + radius < 0.0 {
                return false;
            }
        }
        true
    }

    pub fn intersects_aabb(&self, min: Vec3, max: Vec3) -> bool {
        for plane in &self.planes {
            let mut p = min;
            if plane.x >= 0.0 {
                p.x = max.x;
            }
            if plane.y >= 0.0 {
                p.y = max.y;
            }
            if plane.z >= 0.0 {
                p.z = max.z;
            }

            if plane.xyz().dot(p) + plane.w < 0.0 {
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
        let b = oc.dot(self.direction);
        let c = oc.dot(oc) - radius * radius;
        let discriminant = b * b - c;

        if discriminant < 0.0 {
            return None;
        }

        let sqrt_d = discriminant.sqrt();
        let t = -b - sqrt_d;
        if t > 0.0 {
            return Some(t);
        }
        let t = -b + sqrt_d;
        if t > 0.0 {
            return Some(t);
        }
        None
    }

    pub fn intersect_aabb(&self, min: Vec3, max: Vec3) -> Option<(f32, f32)> {
        let mut tmin = (min.x - self.origin.x) / self.direction.x;
        let mut tmax = (max.x - self.origin.x) / self.direction.x;

        if tmin > tmax {
            std::mem::swap(&mut tmin, &mut tmax);
        }

        let mut tymin = (min.y - self.origin.y) / self.direction.y;
        let mut tymax = (max.y - self.origin.y) / self.direction.y;

        if tymin > tymax {
            std::mem::swap(&mut tymin, &mut tymax);
        }

        if (tmin > tymax) || (tymin > tmax) {
            return None;
        }

        if tymin > tmin {
            tmin = tymin;
        }
        if tymax < tmax {
            tmax = tymax;
        }

        let mut tzmin = (min.z - self.origin.z) / self.direction.z;
        let mut tzmax = (max.z - self.origin.z) / self.direction.z;

        if tzmin > tzmax {
            std::mem::swap(&mut tzmin, &mut tzmax);
        }

        if (tmin > tzmax) || (tzmin > tmax) {
            return None;
        }

        if tzmin > tmin {
            tmin = tzmin;
        }
        if tzmax < tmax {
            tmax = tzmax;
        }

        Some((tmin, tmax))
    }
}

pub use glam::Vec4Swizzles as _;
