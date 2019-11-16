//!
//! adapted from the rendy meshes demo
//!

#![cfg_attr(
    not(any(feature = "dx12", feature = "metal", feature = "vulkan")),
    allow(unused)
)]

// #[cfg(feature = "dx12")]
// use gfx_backend_dx12::Backend;

// #[cfg(feature = "metal")]
// use gfx_backend_metal::Backend;

// #[cfg(feature = "vulkan")]
use gfx_backend_vulkan::Backend;
use rand::prelude::*;
use rendy::shader::SpirvReflection;
use rendy_playground::crystal;
use std::sync::mpsc::{channel, sync_channel, Receiver, Sender};
use {
    genmesh::generators::{IndexedPolygon, SharedVertex},
    rand::distributions::{Distribution, Uniform},
    rendy::{
        command::{DrawIndexedCommand, QueueId, RenderPassEncoder},
        factory::{Config, Factory},
        graph::{render::*, GraphBuilder, GraphContext, NodeBuffer, NodeImage},
        hal::{self, adapter::PhysicalDevice as _, device::Device as _},
        init::winit::{
            event::{Event, WindowEvent},
            event_loop::{ControlFlow, EventLoop},
            window::WindowBuilder,
        },
        init::AnyWindowedRendy,
        memory::Dynamic,
        mesh::{Mesh, Model, PosColorNorm},
        resource::{Buffer, BufferInfo, DescriptorSet, DescriptorSetLayout, Escape, Handle},
        shader::{ShaderKind, SourceLanguage, SourceShaderInfo, SpirvShader},
    },
    std::{cmp::min, mem::size_of, time},
};
use {
    genmesh::Triangulate, nalgebra::Vector3, random_color::RandomColor, rendy::mesh::Position,
    rendy_playground::player,
};
mod render;
use render::{
    Camera, MeshRenderPipeline, MeshRenderPipelineDesc, PerInstance, PerInstanceConst,
    ProfileTimer, Scene,
};
pub mod rad;
use rad::RadWorker;

pub mod game;

fn main() {
    env_logger::Builder::from_default_env()
        .filter_module("meshes", log::LevelFilter::Trace)
        .init();

    let mut event_loop = EventLoop::new();

    let window = WindowBuilder::new()
        .with_inner_size((960, 640).into())
        .with_title("Rendy example");

    let config: Config = Default::default();
    let rendy = AnyWindowedRendy::init_auto(&config, window, &event_loop).unwrap();

    rendy::with_any_windowed_rendy!((rendy)
        use back; (mut factory, mut families, surface, window) => {

        let mut graph_builder = GraphBuilder::<Backend, Scene<Backend>>::new();

        let size = window.inner_size().to_physical(window.hidpi_factor());
        let window_kind = hal::image::Kind::D2(size.width as u32, size.height as u32, 1, 1);
        let aspect = size.width / size.height;

        let depth = graph_builder.create_image(
            window_kind,
            1,
            hal::format::Format::D32Sfloat,
            Some(hal::command::ClearValue {
                depth_stencil: hal::command::ClearDepthStencil {
                    depth: 1.0,
                    stencil: 0,
                },
            }),
        );

        let pass = graph_builder.add_node(
            MeshRenderPipeline::builder()
                .into_subpass()
                .with_color_surface()
                .with_depth_stencil(depth)
                .into_pass()
                .with_surface(
                    surface,
                    hal::window::Extent2D {
                        width: size.width as _,
                        height: size.height as _,
                    },
                    Some(hal::command::ClearValue {
                        color: hal::command::ClearColor {
                            float32: [0.5, 0.5, 1.0, 1.0],
                        },
                    }),
                ),
        );

        let bm = crystal::read_map("hidden_ramp.txt").expect("could not read file");

        let mut planes = crystal::PlanesSep::new();
        planes.create_planes(&bm);
        let planes_copy : Vec<crystal::Plane> = planes.planes_iter().cloned().collect();



        let (tx_rad_buffer, rx_rad_buffer) = channel();
        let mut scene = Scene {
            camera: Camera {
                proj: nalgebra::Perspective3::new(aspect as f32, 3.1415 / 4.0, 1.0, 200.0)
                    .to_homogeneous(),
                view: nalgebra::Projective3::identity() * nalgebra::Translation3::new(0.0, 0.0, 10.0),
            },
            object_mesh: None,
            per_instance: vec![],
            per_instance_const: vec![],
            rx_rad_buffer: rx_rad_buffer,
        };

        let mut rc = RandomColor::new();
        rc.luminosity(random_color::Luminosity::Bright);
        println!("planes: {}", planes_copy.len());
        for i in 0..std::cmp::min(render::NUM_INSTANCES as usize,planes_copy.len()) {
            let color = rc.to_rgb_array();
            let point = planes_copy[i].cell;
            let dir = match planes_copy[i].dir {
                crystal::Dir::ZxPos => 4,
                crystal::Dir::ZxNeg => 5,
                crystal::Dir::YzPos => 2,
                crystal::Dir::YzNeg => 3,
                crystal::Dir::XyPos => 0,
                crystal::Dir::XyNeg => 1,
            };
            scene.per_instance_const.push(PerInstanceConst{
                translate: nalgebra::Vector3::new(point[0] as f32 * 0.25, point[1] as f32 * 0.25, point[2] as f32 * 0.25),
                dir : dir,
            });
            scene.per_instance.push(PerInstance{
                color : nalgebra::Vector3::new(
                    color[0] as f32 / 255.0,
                    color[1] as f32 / 255.0,
                    color[2] as f32 / 255.0,
                ),
                pad : 0,
            });
        }

        let graph = graph_builder
        .build(&mut factory, &mut families, &scene)
        .unwrap();

        let icosphere = genmesh::generators::Plane::new();
        let indices: Vec<_> =
            genmesh::Vertices::vertices(icosphere.indexed_polygon_iter().triangulate())
                .map(|i| i as u32)
                .collect();

        println!("indices: {}", indices.len());
        let vertices: Vec<_> = icosphere
            .shared_vertex_iter()
            .map(|v| Position(v.pos.into()))
            .collect();
        println!("vertices: {}", vertices.len());
        for v in &vertices {
            println!("vert: {:?}", v);
        }
        scene.object_mesh = Some(
            Mesh::<Backend>::builder()
                .with_indices(&indices[..])
                .with_vertices(&vertices[..])
                .build(graph.node_queue(pass), &factory)
                .unwrap(),
        );

        let started = time::Instant::now();

        let mut frames = 0u64..;

        let mut checkpoint = started;
        let mut player_state = player::State::new();
        let mut event_manager = player::EventManager::new();
        let mut graph = Some(graph);

        let mut main_loop = game::MainLoop::start(crystal::rads::Scene::new(planes, bm));
        main_loop.tx_game_event.send(game::GameEvent::SubscribeFrontBuffer(tx_rad_buffer));

        let mut main_loop = Some(main_loop);

        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Poll;
            match event {
                Event::WindowEvent { event, .. } => match event {
                    WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                    _ => event_manager.window_event(event)
                },
                Event::EventsCleared => {
                    if event_manager.should_close() {
                        *control_flow = ControlFlow::Exit;
                    }
                    factory.maintain(&mut families);


                    player_state.apply_events(event_manager.input_events());
                    scene.camera = Camera {
                        proj: rendy_playground::math::perspective_projection(
                            aspect as f32,
                            3.1415 / 4.0,
                            1.0,
                            200.0,
                        ),
                        view: player_state.get_view_matrix(),
                    };

                    if let Some(ref mut graph) = graph {
                        let pt = ProfileTimer::start("graph.run");
                        graph.run(&mut factory, &mut families, &scene);
                    }

                    let elapsed = checkpoint.elapsed();
                    if (checkpoint.elapsed() >= std::time::Duration::from_secs(5))
                    {
                        checkpoint = time::Instant::now();
                    }
                }
                _ => {}
            }
            if *control_flow == ControlFlow::Exit {
                println!("waiting for MainLoop ...");
                if let Some(main_loop) = main_loop.take() {
                    main_loop.tx_game_event.send(game::GameEvent::Stop).unwrap();
                    main_loop.join();
                }
                println!("done.");
                if let Some(graph) = graph.take() {
                    graph.dispose(&mut factory, &scene);
                }
                drop(scene.object_mesh.take());
                println!("end.");
            }
        });
    });
}
