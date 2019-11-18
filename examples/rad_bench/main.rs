use nalgebra::Vector3;
use rand::prelude::*;
use rand::SeedableRng;
use random_color::RandomColor;
use rendy_playground::{crystal, crystal::Vec3};
use std::sync::mpsc::{channel, sync_channel, Receiver, Sender};
use std::time::{Duration, Instant};
trait ToRgbVec3 {
    fn to_rgb_vec3(&mut self) -> Vec3;
}

impl ToRgbVec3 for RandomColor {
    fn to_rgb_vec3(&mut self) -> Vec3 {
        let a = self.to_rgb_array();
        Vec3::new(
            a[0] as f32 / 255f32,
            a[1] as f32 / 255f32,
            a[2] as f32 / 255f32,
        )
    }
}
fn main() {
    unsafe {
        // don't need / want denormals -> flush to zero
        core::arch::x86_64::_MM_SET_FLUSH_ZERO_MODE(core::arch::x86_64::_MM_FLUSH_ZERO_ON);
    }
    let bm = crystal::read_map("hidden_ramp.txt").expect("could not read file");

    let mut planes = crystal::PlanesSep::new();
    planes.create_planes(&bm);
    let planes_copy: Vec<crystal::Plane> = planes.planes_iter().cloned().collect();

    let mut scene = crystal::rads::Scene::new(planes, bm);
    let mut last_stat = Instant::now();
    let mut rng: StdRng = SeedableRng::seed_from_u64(12345);

    for c in &mut scene.diffuse {
        *c = Vec3::new(
            rng.gen_range(0.0, 1.0),
            rng.gen_range(0.0, 1.0),
            rng.gen_range(0.0, 1.0),
        );
    }
    for c in &mut scene.emit {
        *c = Vec3::new(
            rng.gen_range(0.0, 1.0),
            rng.gen_range(0.0, 1.0),
            rng.gen_range(0.0, 1.0),
        );
    }
    let mut sum = 0f32;
    loop {
        scene.do_rad();
        let sum_new = scene.rad_front.r.iter().sum::<f32>()
            + scene.rad_front.g.iter().sum::<f32>()
            + scene.rad_front.b.iter().sum::<f32>();
        if sum != sum_new {
            println!("sum: {}", sum);
            sum = sum_new;
        }
        let d_time = last_stat.elapsed();
        if d_time >= Duration::from_secs(1) {
            let pintss = scene.pints as f64
                / (d_time.as_secs() as f64 + d_time.subsec_nanos() as f64 * 1e-9);
            scene.pints = 0;

            println!("pint/s: {:e}", pintss);
            // log::info!("bounces/s: {:e}", pintss);
            last_stat = Instant::now();
        }
    }
}
