use crate::{System, FrameContext, InitContext};
use crate::resource_container::Resources;

pub struct Time {
    pub delta: f32,
    pub elapsed: f32,
}

pub struct TimeSystem;

impl System for TimeSystem {
    fn name(&self) -> &str { "TimeSystem" }
    fn on_init(&mut self, ctx: &mut InitContext) {
        ctx.resources.insert(Time { delta: 0.0, elapsed: 0.0 });
    }
    fn update(&mut self, ctx: &FrameContext) {
        if let Some(time_res) = ctx.get_resource::<Time>() {
            let mut time = time_res.write().unwrap();
            let time = time.downcast_mut::<Time>().unwrap();
            time.delta = ctx.delta;
            time.elapsed += ctx.delta;
        }
    }
}
