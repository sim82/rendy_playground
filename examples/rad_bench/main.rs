use nalgebra::Vector3;
use rendy_playground::{crystal, crystal::misc::RadWorker};
use std::sync::mpsc::{channel, sync_channel, Receiver, Sender};
fn main() {
    let bm = crystal::read_map("hidden_ramp.txt").expect("could not read file");

    let mut planes = crystal::PlanesSep::new();
    planes.create_planes(&bm);
    let planes_copy: Vec<crystal::Plane> = planes.planes_iter().cloned().collect();

    let (tx, rx) = channel();
    let (tx_sync, rx_sync) = channel(); // used as semaphore to sync with thread start
    let (script_lines_sink, script_lines_source) = channel();

    let rad_worker = RadWorker::start(
        crystal::rads::Scene::new(planes, bm),
        vec![Vector3::new(0.0, 0.0, 0.0); planes_copy.len()],
        rx,
        tx_sync,
        script_lines_sink,
    );

    rx_sync.recv().unwrap();

    while let Ok(color) = rad_worker.rx.recv() {
        // println!("color");
    }
}
