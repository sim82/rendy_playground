use super::{ffs, util, PlanesSep};
#[allow(unused_imports)]
use super::{Bitmap, BlockMap, DisplayWrap, Point3, Point3i, Vec3, Vec3i};
use rayon::prelude::*;
use std::time::Instant;

pub struct RadBuffer {
    pub r: Vec<f32>,
    pub g: Vec<f32>,
    pub b: Vec<f32>,
}
type RadSlice<'a> = (&'a [f32], &'a [f32], &'a [f32]);
type MutRadSlice<'a> = (&'a mut [f32], &'a mut [f32], &'a mut [f32]);

impl RadBuffer {
    fn new(size: usize) -> RadBuffer {
        RadBuffer {
            r: vec![0f32; size],
            g: vec![0f32; size],
            b: vec![0f32; size],
        }
    }

    pub fn slice(&self, i: std::ops::Range<usize>) -> RadSlice<'_> {
        (&self.r[i.clone()], &self.g[i.clone()], &self.b[i.clone()])
    }
    pub fn slice_mut(&mut self, i: std::ops::Range<usize>) -> MutRadSlice<'_> {
        (
            &mut self.r[i.clone()],
            &mut self.g[i.clone()],
            &mut self.b[i.clone()],
        )
    }
    // this is a bit redundant, but found no better way since SliceIndex is non-copy and thus cannot be used for indexing multiple Vecs
    pub fn slice_full(&self) -> RadSlice<'_> {
        (&self.r[..], &self.g[..], &self.b[..])
    }
    pub fn slice_full_mut(&mut self) -> MutRadSlice<'_> {
        (&mut self.r[..], &mut self.g[..], &mut self.b[..])
    }

    pub fn chunks_mut(&mut self, size: usize) -> impl Iterator<Item = MutRadSlice<'_>> {
        itertools::izip!(
            self.r.chunks_mut(size),
            self.g.chunks_mut(size),
            self.b.chunks_mut(size)
        )
    }

    fn chunks_mut2(
        &mut self,
        size: usize,
    ) -> (
        impl Iterator<Item = &mut [f32]>,
        impl Iterator<Item = &mut [f32]>,
        impl Iterator<Item = &mut [f32]>,
    ) {
        (
            self.r.chunks_mut(size),
            self.g.chunks_mut(size),
            self.b.chunks_mut(size),
        )
    }
}

pub struct Scene {
    pub planes: PlanesSep,
    pub bitmap: BlockMap,
    pub emit: Vec<Vec3>,
    pub extents: Vec<Vec<ffs::Extent>>,
    pub rad_front: RadBuffer,
    pub rad_back: RadBuffer,
    pub diffuse: Vec<Vec3>,
    pub pints: usize,
}

fn vec_mul(v1: &Vec3, v2: &Vec3) -> Vec3 {
    Vec3::new(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z)
}

impl Scene {
    pub fn new(planes: PlanesSep, bitmap: BlockMap) -> Self {
        let filename = "extents.bin";

        let extents = if let Some(extents) = ffs::load_extents(filename) {
            extents
        } else {
            let formfactors = ffs::split_formfactors(ffs::setup_formfactors(&planes, &bitmap));
            let extents = ffs::to_extents(&formfactors);
            ffs::write_extents(filename, &extents);
            println!("wrote {}", filename);
            extents
        };

        let start = Instant::now();
        println!("blocks done: {:?}", start.elapsed());

        Scene {
            emit: vec![Vec3::new(0.0, 0.0, 0.0); planes.num_planes()],
            rad_front: RadBuffer::new(planes.num_planes()),
            rad_back: RadBuffer::new(planes.num_planes()),
            extents: extents,
            diffuse: vec![Vec3::new(1f32, 1f32, 1f32); planes.num_planes()],
            planes: planes,
            bitmap: bitmap,
            pints: 0,
        }
    }

    pub fn clear_emit(&mut self) {
        for v in self.emit.iter_mut() {
            *v = Vec3::new(0.0, 0.0, 0.0);
        }
    }

    pub fn apply_light(&mut self, pos: Point3, color: Vec3) {
        let light_pos = Point3i::new(pos.x as i32, pos.y as i32, pos.z as i32);
        for (i, plane) in self.planes.planes_iter().enumerate() {
            let trace_pos = plane.cell + plane.dir.get_normal();

            let d = (pos - Point3::new(trace_pos.x as f32, trace_pos.y as f32, trace_pos.z as f32))
                .normalize();

            // normalize: make directional light
            let len = d.magnitude();
            // d /= len;
            let dot = nalgebra::Matrix::dot(&d, &plane.dir.get_normal());

            //self.emit[i] = Vec3::zero(); //new(0.2, 0.2, 0.2);
            let diff_color = self.diffuse[i];
            if !util::occluded(light_pos, trace_pos, &self.bitmap) && dot > 0f32 {
                // println!("light");
                self.emit[i] +=
                    vec_mul(&diff_color, &color) * dot * (5f32 / (2f32 * 3.1415f32 * len * len));
            }
        }
    }

    pub fn do_rad(&mut self) {
        self.do_rad_blocks();
    }

    pub fn do_rad_blocks(&mut self) {
        // let start = Instant::now();

        std::mem::swap(&mut self.rad_front, &mut self.rad_back);
        // self.rad_front.copy

        assert!(self.rad_front.r.len() == self.extents.len());
        let mut front = RadBuffer::new(0);
        std::mem::swap(&mut self.rad_front, &mut front);

        let num_chunks = 32;
        let chunk_size = self.extents.len() / num_chunks;
        let extents_split = self.extents.chunks(chunk_size).collect::<Vec<_>>();
        let emit_split = self.emit.chunks(chunk_size).collect::<Vec<_>>();
        let diffuse_split = self.diffuse.chunks(chunk_size).collect::<Vec<_>>();

        let (r_split, g_split, b_split) = front.chunks_mut2(chunk_size);
        let mut tmp = itertools::izip!(
            r_split,
            g_split,
            b_split,
            extents_split,
            emit_split,
            diffuse_split
        )
        .collect::<Vec<_>>();

        self.pints += tmp
            .par_iter_mut()
            // .iter_mut()
            .map(
                |(ref mut r, ref mut g, ref mut b, extents, emit, diffuse)| {
                    RadWorkblockScalar {
                        src: self.rad_back.slice_full(),
                        dest: (r, g, b),
                        extents,
                        emit,
                        diffuse,
                    }
                    .do_iter()
                },
            )
            .sum::<usize>();

        std::mem::swap(&mut self.rad_front, &mut front);
    }

    pub fn print_stat(&self) {}
}

struct RadWorkblockScalar<'a> {
    src: RadSlice<'a>,
    dest: MutRadSlice<'a>,
    extents: &'a [Vec<ffs::Extent>],
    emit: &'a [Vec3],
    diffuse: &'a [Vec3],
}

impl RadWorkblockScalar<'_> {
    pub fn do_iter(&mut self) -> usize {
        let mut pints: usize = 0;

        for (i, extents) in self.extents.iter().enumerate() {
            let mut rad_r = 0f32;
            let mut rad_g = 0f32;
            let mut rad_b = 0f32;
            let diffuse = self.diffuse[i as usize];

            // let RadBuffer { r, g, b } = &self.rad_back;
            let (r, g, b) = self.src;
            for ffs::Extent { start, ffs } in extents {
                for (j, ff) in ffs.iter().enumerate() {
                    unsafe {
                        rad_r += r.get_unchecked(j + *start as usize) * diffuse.x * *ff;
                        rad_g += g.get_unchecked(j + *start as usize) * diffuse.y * *ff;
                        rad_b += b.get_unchecked(j + *start as usize) * diffuse.z * *ff;
                    }
                }
                pints += ffs.len();
            }

            self.dest.0[i as usize] = self.emit[i as usize].x + rad_r;
            self.dest.1[i as usize] = self.emit[i as usize].y + rad_g;
            self.dest.2[i as usize] = self.emit[i as usize].z + rad_b;
        }

        pints
    }
}
