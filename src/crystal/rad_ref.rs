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
    /// Utility for making specifically aligned vectors
    pub fn aligned_vector<T>(len: usize, align: usize) -> Vec<T> {
        let t_size = std::mem::size_of::<T>();
        let t_align = std::mem::align_of::<T>();
        let layout = if t_align >= align {
            std::alloc::Layout::from_size_align(t_size * len, t_align).unwrap()
        } else {
            std::alloc::Layout::from_size_align(t_size * len, align).unwrap()
        };
        unsafe {
            let mem = std::alloc::alloc(layout);
            assert_eq!((mem as usize) % 16, 0);
            Vec::<T>::from_raw_parts(mem as *mut T, len, len)
        }
    }

    pub fn aligned_vector_init<T: Copy>(len: usize, align: usize, init: T) -> Vec<T> {
        let mut v = Self::aligned_vector::<T>(len, align);
        for x in v.iter_mut() {
            *x = init;
        }
        v
    }

    fn new(size: usize) -> RadBuffer {
        RadBuffer {
            r: Self::aligned_vector_init(size, 64, 0f32),
            g: Self::aligned_vector_init(size, 64, 0f32),
            b: Self::aligned_vector_init(size, 64, 0f32),
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

        Scene {
            emit: vec![Vec3::new(0.0, 0.0, 0.0); planes.num_planes()],
            // rad_front: vec![Vec3::zero(); planes.num_planes()],
            // rad_back: vec![Vec3::zero(); planes.num_planes()],
            rad_front: RadBuffer::new(planes.num_planes()),
            rad_back: RadBuffer::new(planes.num_planes()),
            extents: extents,
            //ff: formfactors,
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
        self.do_rad_extents();
    }

    pub fn do_rad_extents(&mut self) {
        std::mem::swap(&mut self.rad_front, &mut self.rad_back);

        for (i, extents) in self.extents.iter().enumerate() {
            let mut rad_r = 0f32;
            let mut rad_g = 0f32;
            let mut rad_b = 0f32;
            let diffuse = self.diffuse[i as usize];

            let RadBuffer { r, g, b } = &self.rad_back;
            for ffs::Extent { start, ffs } in extents {
                for (j, ff) in ffs.iter().enumerate() {
                    unsafe {
                        rad_r += r.get_unchecked(j + *start as usize) * diffuse.x * *ff;
                        rad_g += g.get_unchecked(j + *start as usize) * diffuse.y * *ff;
                        rad_b += b.get_unchecked(j + *start as usize) * diffuse.z * *ff;
                    }
                }
                self.pints += ffs.len();
            }

            self.rad_front.r[i as usize] = self.emit[i as usize].x + rad_r;
            self.rad_front.g[i as usize] = self.emit[i as usize].y + rad_g;
            self.rad_front.b[i as usize] = self.emit[i as usize].z + rad_b;
        }
    }

    pub fn print_stat(&self) {
        // println!("write blocks");

        // for blocklist in &self.blocks {
        //     blocklist.print_stat();
        // }

        // let ff_size: usize = self.extents.iter().map(|x| x.ffs.len() * 4).sum();

        // let color_size = self.rad_front.r.len() * 3 * 4 * 2;

        // println!("working set:\nff: {}\ncolor: {}", ff_size, color_size);
    }
}
