use spark_core::scene::*;
use spark_math::*;

#[test]
fn test_scene_node_management() {
    let mut scene = Scene::new();
    let node = Node {
        name: "TestNode".to_string(),
        local_transform: Mat4::IDENTITY,
        global_transform: Mat4::IDENTITY,
        local_aabb: AABB::default(),
        global_aabb: AABB::default(),
        parent: None,
        children: Vec::new(),
        components: Vec::new(),
    };

    let key = scene.add_node(scene.root, node);
    assert_eq!(scene.nodes.len(), 2); // Root + TestNode

    let fetched = scene.nodes.get(key).unwrap();
    assert_eq!(fetched.name, "TestNode");
}

#[test]
fn test_transform_propagation() {
    let mut scene = Scene::new();
    let parent_node = Node {
        name: "Parent".to_string(),
        local_transform: Mat4::from_translation(Vec3::new(1.0, 0.0, 0.0)),
        global_transform: Mat4::IDENTITY,
        local_aabb: AABB::default(),
        global_aabb: AABB::default(),
        parent: None,
        children: Vec::new(),
        components: Vec::new(),
    };
    let parent_key = scene.add_node(scene.root, parent_node);

    let child_node = Node {
        name: "Child".to_string(),
        local_transform: Mat4::from_translation(Vec3::new(0.0, 1.0, 0.0)),
        global_transform: Mat4::IDENTITY,
        local_aabb: AABB::default(),
        global_aabb: AABB::default(),
        parent: None,
        children: Vec::new(),
        components: Vec::new(),
    };
    let child_key = scene.add_node(parent_key, child_node);

    scene.update_all_transforms();

    let child = scene.nodes.get(child_key).unwrap();
    let expected_pos = Vec3::new(1.0, 1.0, 0.0);
    assert!((child.global_transform.w_axis.xyz() - expected_pos).length() < 0.0001);
}
