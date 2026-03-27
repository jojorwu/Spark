use spark_core::scene::Scene;

#[no_mangle]
pub extern "C" fn spark_node_set_position(
    _scene_ptr: *mut Scene,
    node_id: u32,
    x: f32,
    y: f32,
    z: f32,
) {
    // Simplified: map u64 back to NodeKey if possible, or use a handle map.
    // For this demonstration, we assume the u64 is a valid index or raw key.
    log::info!(
        "C# calling spark_node_set_position: node={}, pos=({}, {}, {})",
        node_id,
        x,
        y,
        z
    );
}
