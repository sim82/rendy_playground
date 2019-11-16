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
pub struct UniformArgs {
    pub proj: nalgebra::Matrix4<f32>,
    pub view: nalgebra::Matrix4<f32>,
    pub model: [nalgebra::Matrix4<f32>; 6],
}

#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct PerInstanceConst {
    pub translate: nalgebra::Vector3<f32>,
    pub dir: u32,
}

#[derive(Clone, Copy, Debug)]
#[repr(C, align(16))]
pub struct PerInstance {
    pub color: nalgebra::Vector3<f32>,
    pub pad: u32,
}

#[derive(Debug)]
pub struct Camera {
    pub view: nalgebra::Projective3<f32>,
    // proj: nalgebra::Perspective3<f32>,
    pub proj: nalgebra::Matrix4<f32>,
}

const UNIFORM_SIZE: u64 = size_of::<UniformArgs>() as u64;
pub const NUM_INSTANCES: u64 = 1024 * 1024;
const PER_INSTANCE_CONST_SIZE: u64 = size_of::<PerInstanceConst>() as u64;
const PER_INSTANCE_SIZE: u64 = size_of::<PerInstance>() as u64;

const fn align_to(s: u64, align: u64) -> u64 {
    ((s - 1) / align + 1) * align
}
const fn buffer_const_size(align: u64) -> u64 {
    align_to(PER_INSTANCE_CONST_SIZE * NUM_INSTANCES, align)
}
const fn buffer_frame_size(align: u64) -> u64 {
    align_to(UNIFORM_SIZE + PER_INSTANCE_SIZE * NUM_INSTANCES, align)
}
const fn buffer_size(align: u64, frames: u64) -> u64 {
    buffer_const_size(align) + buffer_frame_size(align) * frames
}
const fn uniform_offset(index: usize, align: u64) -> u64 {
    buffer_const_size(align) + buffer_frame_size(align) * index as u64
}
const fn per_instance_offset(index: usize, align: u64) -> u64 {
    uniform_offset(index, align) + UNIFORM_SIZE
}

#[derive(Debug, Default)]
pub struct MeshRenderPipelineDesc;

#[derive(Debug)]
pub struct MeshRenderPipeline<B: hal::Backend> {
    align: u64,
    buffer: Escape<Buffer<B>>,
    sets: Vec<Escape<DescriptorSet<B>>>,
}

pub struct ProfileTimer {
    label: std::string::String,
    start: std::time::Instant,
}

impl ProfileTimer {
    pub fn start(label: &str) -> Self {
        ProfileTimer {
            label: label.into(),
            start: std::time::Instant::now(),
        }
    }
}

impl Drop for ProfileTimer {
    fn drop(&mut self) {
        // println!("{}: {:?}", self.label, self.start.elapsed());
    }
}

pub struct Scene<B: hal::Backend> {
    pub camera: Camera,
    pub object_mesh: Option<Mesh<B>>,
    pub per_instance_const: Vec<PerInstanceConst>,
    pub per_instance: Vec<PerInstance>,
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
                .attributes(&["position"])
                .unwrap()
                .gfx_vertex_input_desc(hal::pso::VertexInputRate::Vertex),
            SHADER_REFLECTION
                .attributes(&["translate", "dir"])
                .unwrap()
                .gfx_vertex_input_desc(hal::pso::VertexInputRate::Instance(1)),
            SHADER_REFLECTION
                .attributes(&["color", "pad"])
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
        scene: &Scene<B>,
        buffers: Vec<NodeBuffer>,
        images: Vec<NodeImage>,
        set_layouts: &[Handle<DescriptorSetLayout<B>>],
    ) -> Result<MeshRenderPipeline<B>, rendy_core::hal::pso::CreationError> {
        assert!(buffers.is_empty());
        assert!(images.is_empty());
        assert_eq!(set_layouts.len(), 1);

        let frames = ctx.frames_in_flight as _;
        let align = factory
            .physical()
            .limits()
            .min_uniform_buffer_offset_alignment;

        let mut buffer = factory
            .create_buffer(
                BufferInfo {
                    size: buffer_size(align, frames) as u64,
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
                        Some(uniform_offset(index as usize, align))
                            ..Some(uniform_offset(index as usize, align) + UNIFORM_SIZE),
                    )),
                }));
                sets.push(set);
            }
        }

        if !scene.per_instance_const.is_empty() {
            unsafe {
                factory
                    .upload_visible_buffer(&mut buffer, 0, &scene.per_instance_const[..])
                    .expect("update const buffer failed")
            };
        }

        Ok(MeshRenderPipeline {
            align,
            buffer,
            sets,
        })
    }
}

fn model_transform() -> nalgebra::Matrix4<f32> {
    let rot = nalgebra::UnitQuaternion::identity();
    nalgebra::Similarity3::from_parts(Vector3::new(0.5, 0.5, 0.0).into(), rot, 0.5).into()
}

fn model_transform2() -> [nalgebra::Matrix4<f32>; 6] {
    let z_pos = nalgebra::UnitQuaternion::identity();
    let z_neg = nalgebra::UnitQuaternion::face_towards(
        &Vector3::new(0.0, 0.0, -1.0),
        &Vector3::new(0.0, 1.0, 0.0),
    );
    let x_pos = nalgebra::UnitQuaternion::face_towards(
        &Vector3::new(1.0, 0.0, 0.0),
        &Vector3::new(0.0, 1.0, 0.0),
    );
    let x_neg = nalgebra::UnitQuaternion::face_towards(
        &Vector3::new(-1.0, 0.0, 0.0),
        &Vector3::new(0.0, 1.0, 0.0),
    );
    let y_pos = nalgebra::UnitQuaternion::face_towards(
        &Vector3::new(0.0, 1.0, 0.0),
        &Vector3::new(0.0, 0.0, 1.0),
    );
    let y_neg = nalgebra::UnitQuaternion::face_towards(
        &Vector3::new(0.0, -1.0, 0.0),
        &Vector3::new(0.0, 0.0, -1.0),
    );
    // let unit = 0.125;
    let unit = 0.125;
    let scale = 0.125;
    [
        nalgebra::Similarity3::from_parts(Vector3::new(0.0, 0.0, unit).into(), z_pos, scale).into(),
        nalgebra::Similarity3::from_parts(Vector3::new(0.0, 0.0, -unit).into(), z_neg, scale)
            .into(),
        nalgebra::Similarity3::from_parts(Vector3::new(unit, 0.0, 0.0).into(), x_pos, scale).into(),
        nalgebra::Similarity3::from_parts(Vector3::new(-unit, 0.0, 0.0).into(), x_neg, scale)
            .into(),
        nalgebra::Similarity3::from_parts(Vector3::new(0.0, unit, 0.0).into(), y_pos, scale).into(),
        nalgebra::Similarity3::from_parts(Vector3::new(0.0, -unit, 0.0).into(), y_neg, scale)
            .into(),
    ]
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
        let pt = ProfileTimer::start("prepare");

        unsafe {
            factory
                .upload_visible_buffer(
                    &mut self.buffer,
                    uniform_offset(index, self.align),
                    &[UniformArgs {
                        proj: scene.camera.proj,
                        view: scene.camera.view.to_homogeneous(),
                        model: model_transform2(),
                    }],
                )
                .unwrap()
        };
        if !scene.per_instance.is_empty() {
            unsafe {
                factory
                    .upload_visible_buffer(
                        &mut self.buffer,
                        per_instance_offset(index, self.align),
                        &scene.per_instance[..],
                    )
                    .unwrap()
            };
        }
        PrepareResult::DrawReuse
    }

    fn draw(
        &mut self,
        layout: &B::PipelineLayout,
        mut encoder: RenderPassEncoder<'_, B>,
        index: usize,
        scene: &Scene<B>,
    ) {
        println!("draw");

        unsafe {
            encoder.bind_graphics_descriptor_sets(
                layout,
                0,
                Some(self.sets[index].raw()),
                std::iter::empty(),
            );

            let vertex = [SHADER_REFLECTION.attributes(&["position"]).unwrap()];

            scene
                .object_mesh
                .as_ref()
                .unwrap()
                .bind(0, &vertex, &mut encoder)
                .unwrap();
            encoder.bind_vertex_buffers(1, std::iter::once((self.buffer.raw(), 0)));
            encoder.bind_vertex_buffers(
                2,
                std::iter::once((self.buffer.raw(), per_instance_offset(index, self.align))),
            );
            encoder.draw_indexed(
                0..scene.object_mesh.as_ref().unwrap().len(),
                0 as i32,
                0..scene.per_instance.len() as u32,
            )
        }
    }

    fn dispose(self, _factory: &mut Factory<B>, _scene: &Scene<B>) {}
}
