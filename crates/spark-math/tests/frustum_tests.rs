use spark_math::*;

#[test]
fn test_frustum_sphere_intersection() {
    let projection = Mat4::perspective_rh(45.0f32.to_radians(), 1.0, 0.1, 100.0);
    let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO, Vec3::Y);
    let frustum = Frustum::from_matrix(projection * view);

    // Sphere inside
    assert!(frustum.intersects_sphere(Vec3::ZERO, 1.0));

    // Sphere way outside
    assert!(!frustum.intersects_sphere(Vec3::new(0.0, 0.0, 20.0), 1.0));

    // Sphere just outside but radius makes it intersect
    assert!(frustum.intersects_sphere(Vec3::new(0.0, 0.0, 10.0), 1.0));
}
