use nalgebra::Matrix4;
use nalgebra::{
    Isometry3, Point3, Projective3, RealField, Rotation3, Translation3, UnitQuaternion, Vector3,
    Vector4,
};
use winit::event::{ElementState, Event, KeyboardInput, VirtualKeyCode, WindowEvent};

#[derive(Copy, Clone)]
pub enum InputEvent {
    Key(VirtualKeyCode, ElementState),
    KeyFocus(bool),
    Character(char),
    PointerDelta(f32, f32),
}

pub struct FlyModel {
    lon: f32,
    lat: f32,
    pos: Point3<f32>,
}
pub struct InputState {
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub action1: bool,
    pub action2: bool,
    pub run: bool,
    pub d_lon: f32,
    pub d_lat: f32,

    pub z_neg: bool,
    pub z_pos: bool,
    pub x_neg: bool,
    pub x_pos: bool,
}
fn deg_to_rad(f: f32) -> f32 {
    // f * (f32::pi() / 180.0)
    f.to_radians()
}
impl InputState {
    pub fn new() -> Self {
        InputState {
            forward: false,
            backward: false,
            left: false,
            right: false,
            action1: false,
            action2: false,
            run: false,
            d_lon: 0f32,
            d_lat: 0f32,
            z_neg: false,
            z_pos: false,
            x_neg: false,
            x_pos: false,
        }
    }

    pub fn delta_lon(&mut self) -> f32 {
        let ret = self.d_lon;
        self.d_lon = 0f32;
        ret
    }

    pub fn delta_lat(&mut self) -> f32 {
        let ret = self.d_lat;
        self.d_lat = 0f32;
        ret
    }

    pub fn apply(&mut self, ev: InputEvent) {
        match ev {
            InputEvent::Key(keycode, state) => {
                let down = state == ElementState::Pressed;

                match keycode {
                    VirtualKeyCode::W => self.forward = down,
                    VirtualKeyCode::S => self.backward = down,
                    VirtualKeyCode::A => self.left = down,
                    VirtualKeyCode::D => self.right = down,
                    VirtualKeyCode::I => self.z_neg = down,
                    VirtualKeyCode::K => self.z_pos = down,
                    VirtualKeyCode::J => self.x_neg = down,
                    VirtualKeyCode::L => self.x_pos = down,
                    VirtualKeyCode::Q => self.action1 = down,
                    VirtualKeyCode::E => self.action2 = down,
                    VirtualKeyCode::LShift | VirtualKeyCode::RShift => self.run = down,
                    _ => {}
                }
            }
            InputEvent::PointerDelta(x, y) => {
                self.d_lon += deg_to_rad(x);
                self.d_lat += deg_to_rad(y);
            }
            _ => (),
        }
    }

    pub fn apply_all(&mut self, events: Vec<InputEvent>) {
        for ev in events {
            self.apply(ev);
        }
    }
}

impl FlyModel {
    pub fn new(pos: Point3<f32>, lon: f32, lat: f32) -> Self {
        FlyModel {
            lon: lon,
            lat: lat,
            pos: pos,
        }
    }

    pub fn apply_delta_lon(&mut self, d: f32) {
        self.lon += deg_to_rad(-d * 10.0); // rhs coordinate system -> positive lat means turn left
    }
    pub fn apply_delta_lat(&mut self, d: f32) {
        self.lat = num_traits::clamp(
            self.lat - deg_to_rad(d * 10.0),
            deg_to_rad(-90.0),
            deg_to_rad(90.0),
        );
    }
    pub fn get_rotation_lon(&self) -> UnitQuaternion<f32> {
        nalgebra::UnitQuaternion::from_axis_angle(&Vector3::y_axis(), self.lon).into()
    }
    pub fn get_rotation_lat(&self) -> UnitQuaternion<f32> {
        nalgebra::UnitQuaternion::from_axis_angle(&Vector3::x_axis(), self.lat).into()
    }
    pub fn get_rotation(&self) -> UnitQuaternion<f32> {
        self.get_rotation_lon() * self.get_rotation_lat()
    }
    pub fn apply_move_forward(&mut self, d: f32) {
        let forward = Vector3::new(0.0, 0.0, -d);
        self.pos += self.get_rotation() * forward;
    }
    pub fn apply_move_right(&mut self, d: f32) {
        let right = Vector3::new(d, 0.0, 0.0);
        self.pos += self.get_rotation() * self.get_rotation_lat() * right;
    }
    pub fn get_view_matrix(&self) -> Projective3<f32> {
        // (self.get_rotation_lat(true) * self.get_rotation_lon(true) * self.get_translation(true)).invert().unwrap()
        if let Some(mat) =
            (nalgebra::Projective3::identity() * self.get_translation() * self.get_rotation())
                .try_inverse()
        {
            mat
        } else {
            panic!("matrix invert failed");
        }
    }
    pub fn get_translation(&self) -> Translation3<f32> {
        // Matrix4::from_translation( Point3::<f32>::to_vec(self.pos) )
        nalgebra::Translation3::from(self.pos.coords)
    }
}

pub struct State {
    player_model: FlyModel,
    input_state: InputState,
}

impl State {
    pub fn new() -> Self {
        State {
            player_model: FlyModel::new(Point3::origin(), 0.0, 0.0),
            input_state: InputState::new(),
        }
    }

    pub fn apply_events(&mut self, events: Vec<InputEvent>) {
        self.input_state.apply_all(events);
        self.player_model
            .apply_delta_lon(self.input_state.delta_lon());
        self.player_model
            .apply_delta_lat(self.input_state.delta_lat());

        const FORWARD_VEL: f32 = 1.0 / 60.0 * 2.0;
        let boost = if self.input_state.run { 3.0 } else { 1.0 };
        if self.input_state.forward {
            self.player_model.apply_move_forward(FORWARD_VEL * boost);
        }
        if self.input_state.backward {
            self.player_model.apply_move_forward(-FORWARD_VEL * boost);
        }
        if self.input_state.left {
            self.player_model.apply_move_right(-FORWARD_VEL * boost);
        }
        if self.input_state.right {
            self.player_model.apply_move_right(FORWARD_VEL * boost);
        }

        // println!("pos: {:?}", self.player_model.pos);
    }
    pub fn get_view_matrix(&self) -> Projective3<f32> {
        self.player_model.get_view_matrix()
    }
}

pub struct EventManager /*LOL*/ {
    old_pos: Option<winit::dpi::LogicalPosition>,
    should_close: bool,
    input_events: Vec<InputEvent>,
}

impl EventManager {
    pub fn new() -> Self {
        EventManager {
            old_pos: None,
            should_close: false,
            input_events: vec![],
        }
    }
    pub fn should_close(&self) -> bool {
        self.should_close
    }
    pub fn window_event(&mut self, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => self.should_close = true,
            WindowEvent::KeyboardInput {
                input:
                    KeyboardInput {
                        state,
                        virtual_keycode: Some(keycode),
                        ..
                    },
                ..
            } => {
                self.input_events.push(InputEvent::Key(keycode, state));
                match keycode {
                    VirtualKeyCode::F3 => self.should_close = true,
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position: pos, .. } => {
                if let Some(op) = self.old_pos {
                    self.input_events.push(InputEvent::PointerDelta(
                        (pos.x - op.x) as f32,
                        (pos.y - op.y) as f32,
                    ));
                }

                self.old_pos = Some(pos);
            }
            _ => (),
        }
    }
    pub fn input_events(&mut self) -> Vec<InputEvent> {
        self.input_events.split_off(0)
    }
}
