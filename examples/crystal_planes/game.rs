use rendy_playground::{
    crystal,
    crystal::rads::Scene,
    crystal::{Bitmap, Color, PlanesSep, Point3, Point3i, Vec3},
    script,
};
use std::{
    sync::mpsc::{channel, sync_channel, Receiver, Sender},
    thread::{spawn, JoinHandle},
    time::{Duration, Instant},
};
pub enum GameEvent {
    UpdateLightPos(Point3),
    SubscribeFrontBuffer(Sender<std::vec::Vec<Color>>),
    Stop,
}

pub struct MainLoop {
    pub tx_game_event: Sender<GameEvent>,
    join_handle: JoinHandle<()>,
}

impl MainLoop {
    pub fn start(scene: Scene) -> Self {
        let (tx_game_event, rx_game_event) = channel();

        let join_handle = spawn(move || {
            let mut do_stop = false;
            let mut tx_front_buffer = None;
            while !do_stop {
                match rx_game_event.recv().unwrap() {
                    GameEvent::UpdateLightPos(light_pos) => {
                        scene.clear_emit();
                        scene.apply_light(light_pos, Vec3::new(1f32, 0.8f32, 0.6f32));
                    }
                    GameEvent::SubscribeFrontBuffer(sender) => {
                        tx_front_buffer = Some(sender);
                    }
                    GameEvent::Stop => do_stop = true,
                }
            }

            scene.do_rad();
            if let Some(tx_front_buffer) = tx_front_buffer {
                let colors_cpu = vec![];
                colors_cpu.reserve(scene.planes.planes.len());
                for (i, _) in scene.planes.planes_iter().enumerate() {
                    colors_cpu.push(Vec3::new(
                        scene.rad_front.r[i],
                        scene.rad_front.g[i],
                        scene.rad_front.b[i],
                    ));
                }
                tx_front_buffer.send(colors_cpu).unwrap();
            }
        });

        MainLoop {
            tx_game_event,
            join_handle,
        }
    }
    pub fn join(self) {
        self.join_handle.join();
    }
}
