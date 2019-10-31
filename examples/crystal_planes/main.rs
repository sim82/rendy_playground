use clap::{App, Arg};
use rendy_playground::{
    crystal,
    crystal::rad::Scene,
    crystal::{Bitmap, PlanesSep, Point3, Point3i, Vec3},
};

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

        loop {
            scene.do_rad();
        }
        // panic!("exit");
    }
}
