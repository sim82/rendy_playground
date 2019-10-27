pub fn perspective_projection(a: f32, fov: f32, zn: f32, zf: f32) -> nalgebra::Matrix4<f32> {
    let f = 1f32 / (fov * 0.5f32).tan();

    nalgebra::Matrix4::new(
        f / a,
        0f32,
        0f32,
        0f32,
        0f32,
        -f,
        0f32,
        0f32,
        0f32,
        0f32,
        zf / (zn - zf),
        -1f32,
        0f32,
        0f32,
        (zn * zf) / (zn - zf),
        0f32,
    )
}
