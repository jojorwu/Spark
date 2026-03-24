use crate::scene::{Component, NodeKey};
use serde::{Serialize, Deserialize};
use spark_math::{Vec3, Vec4Swizzles};

#[derive(Serialize, Deserialize, Clone)]
pub struct RigidBody {
    pub velocity: Vec3,
    pub mass: f32,
    pub use_gravity: bool,
}

#[typetag::serde]
impl Component for RigidBody {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn clone_box(&self) -> Box<dyn Component> { Box::new(self.clone()) }
    fn on_update(&mut self, node_key: NodeKey, ctx: &crate::FrameContext) {
        if self.use_gravity {
            let gravity = ctx.project.physics_settings.gravity;
            self.velocity += gravity * ctx.delta;
        }

        unsafe {
            let scene = ctx.scene_mut();
            if let Some(node) = scene.nodes.get_mut(node_key) {
                let translation = node.local_transform.w_axis.xyz() + self.velocity * ctx.delta;
                node.local_transform.w_axis.x = translation.x;
                node.local_transform.w_axis.y = translation.y;
                node.local_transform.w_axis.z = translation.z;
            }
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SphereCollider {
    pub radius: f32,
    pub offset: Vec3,
}

#[typetag::serde]
impl Component for SphereCollider {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
    fn clone_box(&self) -> Box<dyn Component> { Box::new(self.clone()) }
}

pub struct PhysicsSystem;

impl crate::System for PhysicsSystem {
    fn name(&self) -> &str { "PhysicsSystem" }
    fn update(&mut self, _ctx: &crate::FrameContext) {
        // Advanced collision detection logic would go here
    }
}
