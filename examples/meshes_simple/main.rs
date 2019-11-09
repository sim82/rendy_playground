//!
//! adapted from the rendy meshes demo
//!

#![cfg_attr(
    not(any(feature = "dx12", feature = "metal", feature = "vulkan")),
    allow(unused)
)]
use {
    genmesh::{
        generators::{IndexedPolygon, SharedVertex},
        Triangulate,
    },
    rand::distributions::{Distribution, Uniform},
    rendy::{
        command::{DrawIndexedCommand, QueueId, RenderPassEncoder},
        factory::{Config, Factory},
        graph::{
            present::PresentNode, render::*, GraphBuilder, GraphContext, NodeBuffer, NodeImage,
        },
        hal::{self, Device as _, PhysicalDevice as _},
        memory::Dynamic,
        mesh::{Mesh, Model, PosColor},
        resource::{Buffer, BufferInfo, DescriptorSet, DescriptorSetLayout, Escape, Handle},
        shader::{ShaderKind, SourceLanguage, SourceShaderInfo, SpirvShader},
        wsi::winit::{Event, EventsLoop, WindowBuilder, WindowEvent},
    },
    rendy_playground::player,
    std::{cmp::min, mem::size_of, time},
};

use rendy::shader::SpirvReflection;

#[cfg(feature = "dx12")]
type Backend = rendy::dx12::Backend;

#[cfg(feature = "metal")]
type Backend = rendy::metal::Backend;

#[cfg(feature = "vulkan")]
type Backend = rendy::vulkan::Backend;

lazy_static::lazy_static! {
    static ref VERTEX: SpirvShader = SourceShaderInfo::new(
        include_str!("shader.vert"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/examples/meshes_simple/shader.vert").into(),
        ShaderKind::Vertex,
        SourceLanguage::GLSL,
        "main",
    ).precompile().unwrap();

    static ref FRAGMENT: SpirvShader = SourceShaderInfo::new(
        include_str!("shader.frag"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/examples/meshes_simple/shader.frag").into(),
        ShaderKind::Fragment,
        SourceLanguage::GLSL,
        "main",
    ).precompile().unwrap();

    static ref SHADERS: rendy::shader::ShaderSetBuilder = rendy::shader::ShaderSetBuilder::default()
        .with_vertex(&*VERTEX).unwrap()
        .with_fragment(&*FRAGMENT).unwrap();

    static ref SHADER_REFLECTION: SpirvReflection = SHADERS.reflect().unwrap();
}

#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct UniformArgs {
    proj: nalgebra::Matrix4<f32>,
    view: nalgebra::Matrix4<f32>,
}

#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct PerInstance {
    translate: nalgebra::Vector3<f32>,
}

#[derive(Debug)]
struct Camera {
    view: nalgebra::Projective3<f32>,
    // proj: nalgebra::Perspective3<f32>,
    proj: nalgebra::Matrix4<f32>,
}

#[derive(Debug)]
struct Scene<B: hal::Backend> {
    camera: Camera,
    object_mesh: Option<Mesh<B>>,
}

const UNIFORM_SIZE: u64 = size_of::<UniformArgs>() as u64;
const NUM_INSTANCES: u64 = 64;
const PER_INSTANCE_SIZE: u64 = size_of::<PerInstance>() as u64;

const fn buffer_frame_size(align: u64) -> u64 {
    ((UNIFORM_SIZE + PER_INSTANCE_SIZE * NUM_INSTANCES - 1) / align + 1) * align
}
const fn uniform_offset(index: usize, align: u64) -> u64 {
    buffer_frame_size(align) * index as u64
}

#[derive(Debug, Default)]
struct MeshRenderPipelineDesc;

#[derive(Debug)]
struct MeshRenderPipeline<B: hal::Backend> {
    align: u64,
    buffer: Escape<Buffer<B>>,
    sets: Vec<Escape<DescriptorSet<B>>>,
}

impl<B> SimpleGraphicsPipelineDesc<B, Scene<B>> for MeshRenderPipelineDesc
where
    B: hal::Backend,
{
    type Pipeline = MeshRenderPipeline<B>;

    fn load_shader_set(
        &self,
        factory: &mut Factory<B>,
        _scene: &Scene<B>,
    ) -> rendy_shader::ShaderSet<B> {
        SHADERS.build(factory, Default::default()).unwrap()
    }

    fn vertices(
        &self,
    ) -> Vec<(
        Vec<hal::pso::Element<hal::format::Format>>,
        hal::pso::ElemStride,
        hal::pso::VertexInputRate,
    )> {
        return vec![
            SHADER_REFLECTION
                .attributes(&["position", "color"])
                .unwrap()
                .gfx_vertex_input_desc(hal::pso::VertexInputRate::Vertex),
            SHADER_REFLECTION
                .attributes(&["translate"])
                .unwrap()
                .gfx_vertex_input_desc(hal::pso::VertexInputRate::Instance(1)),
        ];
    }

    fn layout(&self) -> Layout {
        return SHADER_REFLECTION.layout().unwrap();
    }

    fn build<'a>(
        self,
        ctx: &GraphContext<B>,
        factory: &mut Factory<B>,
        _queue: QueueId,
        _scene: &Scene<B>,
        buffers: Vec<NodeBuffer>,
        images: Vec<NodeImage>,
        set_layouts: &[Handle<DescriptorSetLayout<B>>],
    ) -> Result<MeshRenderPipeline<B>, failure::Error> {
        assert!(buffers.is_empty());
        assert!(images.is_empty());
        assert_eq!(set_layouts.len(), 1);

        let frames = ctx.frames_in_flight as _;
        let align = factory
            .physical()
            .limits()
            .min_uniform_buffer_offset_alignment;

        let buffer = factory
            .create_buffer(
                BufferInfo {
                    size: buffer_frame_size(align) * frames as u64,
                    usage: hal::buffer::Usage::UNIFORM
                        | hal::buffer::Usage::INDIRECT
                        | hal::buffer::Usage::VERTEX,
                },
                Dynamic,
            )
            .unwrap();

        let mut sets = Vec::new();
        for index in 0..frames {
            unsafe {
                let set = factory
                    .create_descriptor_set(set_layouts[0].clone())
                    .unwrap();
                factory.write_descriptor_sets(Some(hal::pso::DescriptorSetWrite {
                    set: set.raw(),
                    binding: 0,
                    array_offset: 0,
                    descriptors: Some(hal::pso::Descriptor::Buffer(
                        buffer.raw(),
                        Some(uniform_offset(index, align))
                            ..Some(uniform_offset(index, align) + UNIFORM_SIZE),
                    )),
                }));
                sets.push(set);
            }
        }

        Ok(MeshRenderPipeline {
            align,
            buffer,
            sets,
        })
    }
}

impl<B> SimpleGraphicsPipeline<B, Scene<B>> for MeshRenderPipeline<B>
where
    B: hal::Backend,
{
    type Desc = MeshRenderPipelineDesc;

    fn prepare(
        &mut self,
        factory: &Factory<B>,
        _queue: QueueId,
        _set_layouts: &[Handle<DescriptorSetLayout<B>>],
        index: usize,
        scene: &Scene<B>,
    ) -> PrepareResult {
        // println!("index: {}", index);
        unsafe {
            factory
                .upload_visible_buffer(
                    &mut self.buffer,
                    uniform_offset(index, self.align),
                    &[UniformArgs {
                        // proj: scene.camera.proj.to_homogeneous(),
                        proj: scene.camera.proj,
                        view: scene.camera.view.to_homogeneous(),
                    }],
                )
                .unwrap()
        };

        PrepareResult::DrawReuse
    }

    fn draw(
        &mut self,
        layout: &B::PipelineLayout,
        mut encoder: RenderPassEncoder<'_, B>,
        index: usize,
        scene: &Scene<B>,
    ) {
        unsafe {
            encoder.bind_graphics_descriptor_sets(
                layout,
                0,
                Some(self.sets[index].raw()),
                std::iter::empty(),
            );

            let vertex = [SHADER_REFLECTION
                .attributes(&["position", "color"])
                .unwrap()];

            scene
                .object_mesh
                .as_ref()
                .unwrap()
                .bind(0, &vertex, &mut encoder)
                .unwrap();

            encoder.draw_indexed(0..scene.object_mesh.as_ref().unwrap().len(), 0 as i32, 0..1)
        }
    }

    fn dispose(self, _factory: &mut Factory<B>, _scene: &Scene<B>) {}
}

#[cfg(any(feature = "dx12", feature = "metal", feature = "vulkan"))]
fn main() {
    env_logger::Builder::from_default_env()
        .filter_module("meshes", log::LevelFilter::Trace)
        .init();

    let config: Config = Default::default();

    let (mut factory, mut families): (Factory<Backend>, _) = rendy::factory::init(config).unwrap();

    let mut event_loop = EventsLoop::new();

    let window = WindowBuilder::new()
        .with_title("Rendy example")
        .build(&event_loop)
        .unwrap();

    event_loop.poll_events(|_| ());

    let surface = factory.create_surface(&window);

    let mut graph_builder = GraphBuilder::<Backend, Scene<Backend>>::new();

    let size = window
        .get_inner_size()
        .unwrap()
        .to_physical(window.get_hidpi_factor());
    let window_kind = hal::image::Kind::D2(size.width as u32, size.height as u32, 1, 1);
    let aspect = size.width / size.height;

    let color = graph_builder.create_image(
        window_kind,
        1,
        factory.get_surface_format(&surface),
        Some(hal::command::ClearValue::Color([0.5, 0.5, 1.0, 1.0].into())),
    );

    let depth = graph_builder.create_image(
        window_kind,
        1,
        hal::format::Format::D16Unorm,
        Some(hal::command::ClearValue::DepthStencil(
            hal::command::ClearDepthStencil(1.0, 0),
        )),
    );

    let pass = graph_builder.add_node(
        MeshRenderPipeline::builder()
            .into_subpass()
            .with_color(color)
            .with_depth_stencil(depth)
            .into_pass(),
    );

    let present_builder = PresentNode::builder(&factory, surface, color).with_dependency(pass);

    let frames = present_builder.image_count();

    graph_builder.add_node(present_builder);

    let mut scene = Scene {
        camera: Camera {
            proj: nalgebra::Perspective3::new(aspect as f32, 3.1415 / 4.0, 1.0, 200.0)
                .to_homogeneous(),
            view: nalgebra::Projective3::identity() * nalgebra::Translation3::new(0.0, 0.0, 10.0),
        },
        object_mesh: None,
    };
    println!(
        "crap: {:?}",
        nalgebra::Perspective3::new(aspect as f32, 3.1415 / 4.0, 1.0, 200.0).to_homogeneous()
    );
    println!(
        "good: {:?}",
        rendy_playground::math::perspective_projection(aspect as f32, 3.1415 / 4.0, 1.0, 200.0,)
    );
    log::info!("{:#?}", scene);

    let mut graph = graph_builder
        .with_frames_in_flight(frames)
        .build(&mut factory, &mut families, &scene)
        .unwrap();

    // let icosphere = genmesh::generators::IcoSphere::subdivide(3);
    // let icosphere = genmesh::generators::Torus::new(1f32, 0.5f32, 32, 32);
    let icosphere = genmesh::generators::Plane::new();
    // icosphere.
    let indices: Vec<_> =
        genmesh::Vertices::vertices(icosphere.indexed_polygon_iter().triangulate())
            .map(|i| i as u32)
            .collect();

    println!("indices: {}", indices.len());
    let vertices: Vec<_> = icosphere
        .shared_vertex_iter()
        .map(|v| PosColor {
            position: v.pos.into(),
            color: [
                (v.pos.x + 1.0) / 2.0,
                (v.pos.y + 1.0) / 2.0,
                (v.pos.z + 1.0) / 2.0,
                1.0,
            ]
            .into(),
            // normal: v.normal.into(),
        })
        .collect();
    println!("vertices: {}", vertices.len());
    for v in &vertices {
        println!("vert: {:?}", v.position);
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
    let mut rng = rand::thread_rng();
    // let rxy = Uniform::new(-1.0, 1.0);
    // let rz = Uniform::new(0.0, 185.0);

    let mut checkpoint = started;
    let mut player_state = player::State::new();
    let mut event_manager = player::EventManager::new();
    while !event_manager.should_close() {
        factory.maintain(&mut families);
        player_state.apply_events(event_manager.poll_events(&mut event_loop));
        scene.camera = Camera {
            // proj: nalgebra::Perspective3::new(aspect as f32, 3.1415 / 4.0, 1.0, 200.0),
            proj: rendy_playground::math::perspective_projection(
                aspect as f32,
                3.1415 / 4.0,
                1.0,
                200.0,
            ),
            view: player_state.get_view_matrix(),
        };

        graph.run(&mut factory, &mut families, &scene);
        let elapsed = checkpoint.elapsed();
    }

    graph.dispose(&mut factory, &scene);
}

#[cfg(not(any(feature = "dx12", feature = "metal", feature = "vulkan")))]
fn main() {
    panic!("Specify feature: { dx12, metal, vulkan }");
}
