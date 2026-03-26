use crate::scene::{Component, NodeKey};
use serde::{Deserialize, Serialize};
use spark_math::Vec3;

#[derive(Serialize, Deserialize, Clone)]
pub struct RigidBody {
    pub velocity: Vec3,
    pub mass: f32,
    pub use_gravity: bool,
}

#[typetag::serde]
impl Component for RigidBody {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Component> {
        Box::new(self.clone())
    }
    fn on_update(&mut self, node_key: NodeKey, ctx: &crate::FrameContext) {
        if self.use_gravity {
            let gravity = ctx.project.physics_settings.gravity;
            self.velocity += gravity * ctx.delta;
        }

        ctx.command_queue.push(crate::command::TransformCommand {
            node: node_key,
            transform: spark_math::Mat4::from_translation(self.velocity * ctx.delta),
            relative: true,
        });
    }
}

pub struct CollisionEvent {
    pub node_a: NodeKey,
    pub node_b: NodeKey,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SphereCollider {
    pub radius: f32,
    pub offset: Vec3,
}

#[typetag::serde]
impl Component for SphereCollider {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Component> {
        Box::new(self.clone())
    }
}

pub struct PhysicsSystem;

impl crate::System for PhysicsSystem {
    fn name(&self) -> &str {
        "PhysicsSystem"
    }
    fn dependencies(&self) -> Vec<&'static str> {
        vec!["TimeSystem"]
    }
    fn resource_access(&self) -> crate::ResourceAccess {
        crate::ResourceAccess::new().with_scene(crate::Access::Write)
    }
    fn update(&mut self, _ctx: &crate::FrameContext) {
        // Advanced collision detection logic would go here
    }
}
