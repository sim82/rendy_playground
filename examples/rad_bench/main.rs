use nalgebra::Vector3;
use rendy_playground::crystal;
use std::sync::mpsc::{channel, sync_channel, Receiver, Sender};
use std::time::{Duration, Instant};

fn main() {
    let bm = crystal::read_map("hidden_ramp.txt").expect("could not read file");

    let mut planes = crystal::PlanesSep::new();
    planes.create_planes(&bm);
    let planes_copy: Vec<crystal::Plane> = planes.planes_iter().cloned().collect();

    let mut scene = crystal::rads::Scene::new(planes, bm);
    let mut last_stat = Instant::now();
    loop {
        scene.do_rad();
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
