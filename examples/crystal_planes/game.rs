use rendy_playground::{
    crystal,
    crystal::rads::Scene,
    crystal::{Bitmap, Color, PlanesSep, Point3, Point3i, Vec3},
    script,
};
use std::{
    // sync::mpsc::{channel, sync_channel, Receiver, Sender},
    thread::{spawn, JoinHandle},
    time::{Duration, Instant},
};
pub enum GameEvent {
    UpdateLightPos(Point3),
    SubscribeFrontBuffer(std::sync::mpsc::Sender<std::vec::Vec<Color>>),
    Stop,
}
use async_std::{
    sync::{channel, Receiver, Sender},
    task,
};

pub struct MainLoop {
    tx_game_event: Sender<GameEvent>,
    join_handle: JoinHandle<()>,
}

impl MainLoop {
    pub fn start(mut scene: Scene) -> Self {
        let (tx_game_event, rx_game_event) = channel(64);
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
                crystal::Dir::YzPos | crystal::Dir::YzNeg => Vec3::new(0.8f32, 0.8f32, 0.8f32),
                _ => Vec3::new(1f32, 1f32, 1f32),
                // let color = hsv_to_rgb(rng.gen_range(0.0, 360.0), 1.0, 1.0); //random::<f32>(), 1.0, 1.0);
                // scene.diffuse[i] = Vector3::new(color.0, color.1, color.2);
            }
        }
        let join_handle = spawn(move || {
            task::block_on(MainLoop::start_async(scene, rx_game_event));
        });
        MainLoop {
            tx_game_event,
            join_handle,
        }
    }
    async fn start_async(mut scene: Scene, rx_game_event: Receiver<GameEvent>) {
        let mut do_stop = false;
        let mut tx_front_buffer = None;
        let mut light_pos = Point3::new(0f32, 0f32, 0f32);
        let mut light_update = true;
        while !do_stop {
            match rx_game_event.recv().await.unwrap() {
                GameEvent::UpdateLightPos(light_pos_new) => {
                    light_pos = light_pos_new;
                    light_update = true;
                }
                GameEvent::SubscribeFrontBuffer(sender) => {
                    tx_front_buffer = Some(sender);
                }
                GameEvent::Stop => do_stop = true,
            }

            if light_update {
                scene.clear_emit();
                scene.apply_light(light_pos, Vec3::new(1f32, 0.8f32, 0.6f32));
            }
            scene.do_rad();
            let sum = scene.rad_front.r.iter().sum::<f32>()
                + scene.rad_front.g.iter().sum::<f32>()
                + scene.rad_front.b.iter().sum::<f32>();
            println!("sum: {}", sum);
            if let Some(ref mut tx_front_buffer) = tx_front_buffer {
                let mut colors_cpu = vec![];
                colors_cpu.reserve(scene.planes.planes.len());
                for (i, _) in scene.planes.planes_iter().enumerate() {
                    colors_cpu.push(Vec3::new(
                        scene.rad_front.r[i],
                        scene.rad_front.g[i],
                        scene.rad_front.b[i],
                    ));
                }
                // println!("send fornt buffer");
                tx_front_buffer.send(colors_cpu).unwrap();
            }
        }
    }

    pub fn send_game_event(&self, event: GameEvent) {
        task::block_on(async move { self.tx_game_event.send(event).await })
    }

    pub fn join(self) {
        self.join_handle.join();
    }
}
