use spark_core::scene::{Scene, NodeKey};
use spark_math::Vec3;

#[no_mangle]
pub extern "C" fn spark_node_set_position(scene: *mut Scene, node_key: u64, x: f32, y: f32, z: f32) {
    let _scene = unsafe { &mut *scene };
    // Simplified: map u64 back to NodeKey if possible, or use a handle map.
    // For this demonstration, we assume the u64 is a valid index or raw key.
    log::info!("C# calling spark_node_set_position: node={}, pos=({}, {}, {})", node_key, x, y, z);
}
