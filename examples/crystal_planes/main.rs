use clap::{App, Arg};
use rand::prelude::*;
use rendy_playground::{
    crystal,
    crystal::rad::Scene,
    crystal::{Bitmap, PlanesSep, Point3, Point3i, Vec3},
    script,
};
use std::sync::mpsc::{channel, sync_channel, Receiver, Sender};
use std::thread::{spawn, JoinHandle};
use std::time::{Duration, Instant};

type Color = Vec3;

enum GameEvent {
    UpdateLightPos(Point3),
    Stop,
}

struct RadWorker {
    rx: Receiver<std::vec::Vec<Color>>,

    join_handle: JoinHandle<()>,
    binding_tx: Sender<script::BindingAction>,
}
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> Vec3 {
    let mut hh = h;
    if hh >= 360.0 {
        hh = 0.0;
    }
    hh /= 60.0;
    let i = hh as i32; //.into();
    let ff = hh - i as f32;
    let p = v * (1.0 - s);
    let q = v * (1.0 - (s * ff));
    let t = v * (1.0 - (s * (1.0 - ff)));
    match i {
        0 => Vec3::new(v, t, p),
        1 => Vec3::new(q, v, p),
        2 => Vec3::new(p, v, t),
        3 => Vec3::new(p, q, v),
        4 => Vec3::new(t, p, v),
        _ => Vec3::new(v, p, q),
    }
}
impl RadWorker {
    fn start(
        mut scene: Scene,
        // colors_buffer_pool: std::vec::Vec<Color>, //CpuBufferPool<Color>,
        mut colors_cpu: Vec<Color>,
        rx_event: Receiver<GameEvent>,
        tx_sync: Sender<()>,
        script_lines_sink: Sender<String>,
    ) -> RadWorker {
        let (tx, rx) = sync_channel(2);
        let (btx, brx) = channel();

        // let scene = Arc::new(scene);
        // let scene_thread = scene.clone();
        let join_handle = spawn(move || {
            let mut binding_dispatcher = script::BindingDispatcher::new(brx);

            let mut light_pos = crystal::Point3::new(120f32, 32f32, 80f32);
            let mut light_update = false;
            let mut last_stat = Instant::now();
            let mut do_stop = false;

            let light_mode = script::ValueWatch::new();
            binding_dispatcher.bind_value("light_mode", light_mode.clone());

            let light_pos_watch = script::ValueWatch::new();
            binding_dispatcher.bind_value("light_pos", light_pos_watch.clone());
            // let light_mode = Rc::new(RefCell::new(0));
            // let mut last_light_mode = -1;
            // binding_dispatcher.bind_i32("light_mode", light_mode.clone());
            tx_sync.send(()).unwrap();
            // let mut offs = 0;
            while !do_stop {
                binding_dispatcher.dispatch();

                if let Some(light_mode) = light_mode.borrow_mut().get_update::<i32>() {
                    match light_mode {
                        1 => {
                            let mut rng = thread_rng();

                            let color1 = hsv_to_rgb(rng.gen_range(0.0, 180.0), 1.0, 1.0);
                            let color2 = hsv_to_rgb(rng.gen_range(180.0, 360.0), 1.0, 1.0);
                            let color3 = hsv_to_rgb(rng.gen_range(0.0, 180.0), 1.0, 1.0);
                            let color4 = hsv_to_rgb(rng.gen_range(180.0, 360.0), 1.0, 1.0);

                            for (i, plane) in scene.planes.planes_iter().enumerate() {
                                scene.diffuse[i] = Vec3::new(1f32, 1f32, 1f32);

                                let up = plane.cell + crystal::Dir::ZxPos.get_normal::<i32>();
                                let not_edge = (&scene.bitmap as &Bitmap).get(up);

                                scene.emit[i] = if not_edge {
                                    Vec3::new(0.0, 0.0, 0.0)
                                } else {
                                    match plane.dir {
                                        crystal::Dir::YzPos => color1,
                                        crystal::Dir::YzNeg => color2,
                                        crystal::Dir::XyPos => color3,
                                        crystal::Dir::XyNeg => color4,
                                        // crystal::Dir::XyPos | crystal::Dir::XyNeg => {
                                        //     Vector3::new(0.8f32, 0.8f32, 0.8f32)
                                        // }
                                        _ => Vec3::new(0.0, 0.0, 0.0),
                                        // let color = hsv_to_rgb(rng.gen_range(0.0, 360.0), 1.0, 1.0); //random::<f32>(), 1.0, 1.0);
                                        // scene.diffuse[i] = Vector3::new(color.0, color.1, color.2);
                                    }
                                }
                            }
                        }
                        2 => {
                            let mut rng = thread_rng();

                            let color1 = hsv_to_rgb(rng.gen_range(0.0, 180.0), 1.0, 1.0);
                            // let color1 = Vector3::new(1f32, 0.5f32, 0f32);
                            let color2 = hsv_to_rgb(rng.gen_range(180.0, 360.0), 1.0, 1.0);

                            for (i, plane) in scene.planes.planes_iter().enumerate() {
                                scene.diffuse[i] = Vec3::new(1f32, 1f32, 1f32);
                                scene.emit[i] = if (plane.cell.y) % 3 != 0 {
                                    Vec3::new(0.0, 0.0, 0.0)
                                } else {
                                    match plane.dir {
                                        crystal::Dir::XyPos => color1,
                                        crystal::Dir::XyNeg => color2,
                                        // crystal::Dir::XyPos | crystal::Dir::XyNeg => {
                                        //     Vector3::new(0.8f32, 0.8f32, 0.8f32)
                                        // }
                                        _ => Vec3::new(0.0, 0.0, 0.0),
                                        // let color = hsv_to_rgb(rng.gen_range(0.0, 360.0), 1.0, 1.0); //random::<f32>(), 1.0, 1.0);
                                        // scene.diffuse[i] = Vector3::new(color.0, color.1, color.2);
                                    }
                                }
                            }
                        }
                        3 => {
                            let mut rng = thread_rng();

                            for i in 0..scene.planes.num_planes() {
                                // seriously, there is no Vec.fill?
                                scene.diffuse[i] = Vec3::new(1f32, 1f32, 1f32);
                                scene.emit[i] = Vec3::new(0.0, 0.0, 0.0);
                            }

                            let num_dots = 1000;
                            for _ in 0..num_dots {
                                let i = rng.gen_range(0, scene.planes.num_planes());
                                scene.emit[i] = hsv_to_rgb(rng.gen_range(0.0, 360.0), 1.0, 1.0);
                            }
                        }
                        _ => {}
                    }
                }

                if let Some(pos) = light_pos_watch.borrow_mut().get_update::<Point3>() {
                    light_pos = pos;
                    light_update = true;
                    // }
                    // GameEvent::DoAction1 => {

                    // let color1 = hsv_to_rgb(rng.gen_range(0.0, 360.0), 1.0, 1.0);
                    let color1 = Vec3::new(1f32, 0.5f32, 0f32);
                    // let color2 = hsv_to_rgb(rng.gen_range(0.0, 360.0), 1.0, 1.0);
                    let color2 = Vec3::new(0f32, 1f32, 0f32);
                    for (i, plane) in scene.planes.planes_iter().enumerate() {
                        if ((plane.cell.y) / 2) % 2 == 1 {
                            continue;
                        }
                        scene.diffuse[i] = match plane.dir {
                            crystal::Dir::XyPos => color1,
                            crystal::Dir::XyNeg => color2,
                            crystal::Dir::YzPos | crystal::Dir::YzNeg => {
                                Vec3::new(0.8f32, 0.8f32, 0.8f32)
                            }
                            _ => Vec3::new(1f32, 1f32, 1f32),
                            // let color = hsv_to_rgb(rng.gen_range(0.0, 360.0), 1.0, 1.0); //random::<f32>(), 1.0, 1.0);
                            // scene.diffuse[i] = Vector3::new(color.0, color.1, color.2);
                        }
                    }
                }

                while let Ok(event) = rx_event.try_recv() {
                    match event {
                        GameEvent::Stop => do_stop = true,
                        _ => (),
                    }
                }

                if light_update {
                    scene.clear_emit();
                    scene.apply_light(light_pos, Vec3::new(1f32, 0.8f32, 0.6f32));
                    light_update = false;
                }
                println!("do_rad");

                scene.do_rad();
                for (i, plane) in scene.planes.planes_iter().enumerate() {
                    for v in plane.vertices.iter() {
                        colors_cpu[*v as usize] = Vec3::new(
                            scene.rad_front.r[i],
                            scene.rad_front.g[i],
                            scene.rad_front.b[i],
                        );
                    }
                }
                // let chunk = colors_buffer_pool
                //     .chunk(colors_cpu.iter().cloned())
                //     .unwrap();

                let chunk = colors_cpu.clone();
                // println!("size: {} -> {}", old_cap, colors_buffer_pool.capacity());

                if tx.send(chunk).is_err() {
                    println!("send failed.");
                }
                // println!("send");

                let d_time = last_stat.elapsed();
                if d_time >= Duration::from_secs(1) {
                    let pintss = scene.pints as f64
                        / (d_time.as_secs() as f64 + d_time.subsec_nanos() as f64 * 1e-9);
                    scene.pints = 0;

                    println!("pint/s: {:e}", pintss);
                    // log::info!("bounces/s: {:e}", pintss);

                    script_lines_sink
                        .send(format!("set rad_bps {:e}", pintss))
                        .expect("script_lines_sink send failed");

                    last_stat = Instant::now();
                }
            }
        });
        RadWorker {
            rx: rx,
            join_handle: join_handle,
            binding_tx: btx,
        }
    }
}

fn main() {
    env_logger::Builder::from_default_env()
        .filter_module("crystal_planes", log::LevelFilter::Trace)
        .init();

    let matches = App::new("crystal_planes")
        .version("1.0")
        .about("Realime Radiosity test")
        .arg(
            Arg::with_name("timed")
                .help("use time-based frame sync")
                .long("timed"),
        )
        .arg(
            Arg::with_name("threads")
                .help("set number of rayon threads")
                .long("threads")
                .takes_value(true),
        )
        .get_matches();

    let timed = matches.is_present("timed");
    if let Some(threads) = matches.value_of("threads") {
        if let Ok(num_threads) = threads.parse::<usize>() {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build_global()
                .unwrap();
        }
    }

    unsafe {
        // don't need / want denormals -> flush to zero
        core::arch::x86_64::_MM_SET_FLUSH_ZERO_MODE(core::arch::x86_64::_MM_FLUSH_ZERO_ON);
    }
    {
        let bm = crystal::read_map("hidden_ramp.txt").expect("could not read file");

        let mut planes = PlanesSep::new();
        planes.create_planes(&bm);
        // planes.print();
        let mut scene = Scene::new(planes, bm);
        scene.print_stat();

        let mut colors_cpu = Vec::new();
        // let colors_buffer_pool;

        let planes = &scene.planes;

        let mut x: Vec<(&Point3i, i32)> = planes.vertex_iter().collect();
        x.sort_by_key(|(_, v)| *v);
        let scale = 0.25f32;

        let vertices: Vec<_> = x
            .iter()
            .map(|(plane, _)| {
                Vec3::new(
                    plane.x as f32 * scale,
                    plane.y as f32 * scale,
                    plane.z as f32 * scale,
                )
            })
            .collect();

        let indices: Vec<_> = planes
            .planes_iter()
            .flat_map(|plane| {
                vec![
                    plane.vertices[0] as u32,
                    plane.vertices[1] as u32,
                    plane.vertices[2] as u32,
                    plane.vertices[0] as u32,
                    plane.vertices[2] as u32,
                    plane.vertices[3] as u32,
                ]
                //[plane[0], plane[1], plane[2]]
            })
            .collect();
        // let (vertex_buffer, vb_future) = ImmutableBuffer::from_iter(
        //     vertices.iter().cloned(),
        //     BufferUsage::all(),
        //     vk_state.queue(),
        // )
        // .unwrap();

        // let mut colors_tmp = Vec::new();
        for _ in 0..vertices.len() {
            colors_cpu.push(Vec3::new(
                rand::random::<f32>(),
                rand::random::<f32>(),
                rand::random::<f32>(),
            ));
        }

        let (tx, rx) = channel();
        let (tx_sync, rx_sync) = channel(); // used as semaphore to sync with thread start
        let (script_lines_sink, script_lines_source) = channel();
        let rad_worker = RadWorker::start(
            scene, // colors_buffer_pool,
            colors_cpu,
            rx,
            tx_sync,
            script_lines_sink,
        );
        rx_sync.recv().unwrap();

        loop {
            if let Ok(_buf) = rad_worker.rx.recv() {
                // println!("receive");
                // self.colors_buffer_gpu = Some(Arc::new(buf));
            }
        }

        rad_worker.join_handle.join().unwrap();
        // panic!("exit");
    }
}
