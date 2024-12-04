use std::f32::consts::FRAC_PI_2;
use std::time::Duration;

use cgmath::*;
use winit::dpi::PhysicalPosition;
use winit::event::*;
use winit::keyboard::KeyCode;

const SAFE_FRAC_PI_2: f32 = FRAC_PI_2 - 0.0001;

#[derive(Debug)]
pub struct Camera {
    pub position: Point3<f32>,
    yaw: Rad<f32>,
    pitch: Rad<f32>,
}

impl Camera {
    pub fn new<V: Into<Point3<f32>>, Y: Into<Rad<f32>>, P: Into<Rad<f32>>>(
        position: V,
        yaw: Y,
        pitch: P,
    ) -> Self {
        Self {
            position: position.into(),
            yaw: yaw.into(),
            pitch: pitch.into(),
        }
    }

    pub fn calc_matrix(&self) -> Matrix4<f32> {
        let (sin_pitch, cos_pitch) = self.pitch.0.sin_cos();
        let (sin_yaw, cos_yaw) = self.yaw.0.sin_cos();

        Matrix4::look_to_rh(
            self.position,
            Vector3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw).normalize(),
            Vector3::unit_y(),
        )
    }
}

pub struct Projection {
    aspect: f32,
    fovy: Rad<f32>,
    znear: f32,
    zfar: f32,
}

impl Projection {
    pub fn new<F: Into<Rad<f32>>>(width: u32, height: u32, fovy: F, znear: f32, zfar: f32) -> Self {
        Self {
            aspect: width as f32 / height as f32,
            fovy: fovy.into(),
            znear,
            zfar,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.aspect = width as f32 / height as f32;
    }

    pub fn calc_matrix(&self) -> Matrix4<f32> {
        perspective(self.fovy, self.aspect, self.znear, self.zfar)
    }
}

#[derive(Debug)]
pub struct CameraController {
    amount_left: f32,
    amount_right: f32,
    amount_forward: f32,
    amount_backward: f32,
    amount_up: f32,
    amount_down: f32,
    rotate_horizontal: f32,
    rotate_vertical: f32,
    scroll: f32,
    scrolling_out: bool,
    scrolling_in: bool,
    speed: f32,
    sensitivity: f32,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            amount_left: 0.0,
            amount_right: 0.0,
            amount_forward: 0.0,
            amount_backward: 0.0,
            amount_up: 0.0,
            amount_down: 0.0,
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
            scroll: 0.0,
            scrolling_out: false,
            scrolling_in: false,
            speed,
            sensitivity,
        }
    }

    pub fn process_gamepad(&mut self, event: gilrs::EventType) {
        match event {
            gilrs::EventType::AxisChanged(gilrs::Axis::LeftStickX, amount, _) => {
                if amount < -0.4 {
                    self.amount_left = 1.0;
                } else if amount > 0.4 {
                    self.amount_right = 1.0;
                } else {
                    self.amount_left = 0.0;
                    self.amount_right = 0.0;
                }
            }
            gilrs::EventType::AxisChanged(gilrs::Axis::LeftStickY, amount, _) => {
                if amount < -0.4 {
                    self.amount_backward = 1.0;
                } else if amount > 0.4 {
                    self.amount_forward = 1.0;
                } else {
                    self.amount_backward = 0.0;
                    self.amount_forward = 0.0;
                }
            }
            gilrs::EventType::ButtonPressed(gilrs::Button::LeftTrigger, _)
            | gilrs::EventType::ButtonPressed(gilrs::Button::DPadDown, _) => {
                self.amount_down = 1.0;
            }
            gilrs::EventType::ButtonReleased(gilrs::Button::LeftTrigger, _)
            | gilrs::EventType::ButtonReleased(gilrs::Button::DPadDown, _) => {
                self.amount_down = 0.0;
            }
            gilrs::EventType::ButtonPressed(gilrs::Button::RightTrigger, _)
            | gilrs::EventType::ButtonPressed(gilrs::Button::DPadUp, _) => {
                self.amount_up = 1.0;
            }
            gilrs::EventType::ButtonReleased(gilrs::Button::RightTrigger, _)
            | gilrs::EventType::ButtonReleased(gilrs::Button::DPadUp, _) => {
                self.amount_up = 0.0;
            }

            gilrs::EventType::AxisChanged(gilrs::Axis::RightStickX, amount, _) => {
                if amount.abs() > 0.4 {
                    self.rotate_horizontal = amount as f32 * 10.0;
                } else {
                    self.rotate_horizontal = 0.0;
                }
            }
            gilrs::EventType::AxisChanged(gilrs::Axis::RightStickY, amount, _) => {
                if amount.abs() > 0.4 {
                    self.rotate_vertical = -amount as f32 * 10.0;
                } else {
                    self.rotate_vertical = 0.0;
                }
            }
            gilrs::EventType::ButtonPressed(gilrs::Button::LeftTrigger2, _)
            | gilrs::EventType::ButtonPressed(gilrs::Button::DPadLeft, _) => {
                self.scrolling_out = true;
            }
            gilrs::EventType::ButtonReleased(gilrs::Button::LeftTrigger2, _)
            | gilrs::EventType::ButtonReleased(gilrs::Button::DPadLeft, _) => {
                self.scrolling_out = false;
            }
            gilrs::EventType::ButtonPressed(gilrs::Button::RightTrigger2, _)
            | gilrs::EventType::ButtonPressed(gilrs::Button::DPadRight, _) => {
                self.scrolling_in = true;
            }
            gilrs::EventType::ButtonReleased(gilrs::Button::RightTrigger2, _)
            | gilrs::EventType::ButtonReleased(gilrs::Button::DPadRight, _) => {
                self.scrolling_in = false;
            }
            _ => {}
        }
    }

    pub fn process_keyboard(&mut self, key: KeyCode, state: ElementState) -> bool {
        let amount = if state == ElementState::Pressed {
            1.0
        } else {
            0.0
        };
        match key {
            KeyCode::KeyW | KeyCode::ArrowUp => {
                self.amount_forward = amount;
                true
            }
            KeyCode::KeyS | KeyCode::ArrowDown => {
                self.amount_backward = amount;
                true
            }
            KeyCode::KeyA | KeyCode::ArrowLeft => {
                self.amount_left = amount;
                true
            }
            KeyCode::KeyD | KeyCode::ArrowRight => {
                self.amount_right = amount;
                true
            }
            KeyCode::Space => {
                self.amount_up = amount;
                true
            }
            KeyCode::ShiftLeft => {
                self.amount_down = amount;
                true
            }
            _ => false,
        }
    }

    pub fn process_mouse(&mut self, mouse_dx: f64, mouse_dy: f64) {
        self.rotate_horizontal = mouse_dx as f32;
        self.rotate_vertical = mouse_dy as f32;
    }

    pub fn process_scroll(&mut self, delta: &MouseScrollDelta) {
        self.scroll = -match delta {
            // I'm assuming a line is about 100 pixels
            MouseScrollDelta::LineDelta(_, scroll) => scroll * 100.0,
            MouseScrollDelta::PixelDelta(PhysicalPosition { y: scroll, .. }) => *scroll as f32,
        };
    }

    pub fn update_camera(&mut self, camera: &mut Camera, dt: Duration) {
        let dt = dt.as_secs_f32();

        // Move forward/backward and left/right
        let (yaw_sin, yaw_cos) = camera.yaw.0.sin_cos();
        let forward = Vector3::new(yaw_cos, 0.0, yaw_sin).normalize();
        let right = Vector3::new(-yaw_sin, 0.0, yaw_cos).normalize();
        camera.position += forward * (self.amount_forward - self.amount_backward) * self.speed * dt;
        camera.position += right * (self.amount_right - self.amount_left) * self.speed * dt;

        // Move in/out (aka. "zoom")
        // Note: this isn't an actual zoom. The camera's position
        // changes when zooming. I've added this to make it easier
        // to get closer to an object you want to focus on.
        let (pitch_sin, pitch_cos) = camera.pitch.0.sin_cos();
        let scrollward =
            Vector3::new(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin).normalize();
        camera.position += scrollward * self.scroll * self.speed * self.sensitivity * dt;
        if self.scrolling_out {
            self.scroll -= 0.2;
        } else if self.scrolling_in {
            self.scroll += 0.2;
        } else {
            self.scroll = 0.0;
        }

        // Move up/down. Since we don't use roll, we can just
        // modify the y coordinate directly.
        camera.position.y += (self.amount_up - self.amount_down) * self.speed * dt;

        // Rotate
        camera.yaw += Rad(self.rotate_horizontal) * self.sensitivity * dt;
        camera.pitch += Rad(-self.rotate_vertical) * self.sensitivity * dt;

        // Keep the camera's angle from going too high/low.
        if camera.pitch < -Rad(SAFE_FRAC_PI_2) {
            camera.pitch = -Rad(SAFE_FRAC_PI_2);
        } else if camera.pitch > Rad(SAFE_FRAC_PI_2) {
            camera.pitch = Rad(SAFE_FRAC_PI_2);
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    view_position: [f32; 4],
    view: [[f32; 4]; 4],
    view_proj: [[f32; 4]; 4],
    inv_proj: [[f32; 4]; 4],
    inv_view: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_position: [0.0; 4],
            view: cgmath::Matrix4::identity().into(),
            view_proj: cgmath::Matrix4::identity().into(),
            inv_proj: cgmath::Matrix4::identity().into(),
            inv_view: cgmath::Matrix4::identity().into(),
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera, projection: &Projection) {
        // self.view_position = camera.position.to_homogeneous().into();
        // self.view_proj = (projection.calc_matrix() * camera.calc_matrix()).into();
        self.view_position = camera.position.to_homogeneous().into();
        let proj = projection.calc_matrix();
        let view = camera.calc_matrix();
        let view_proj = proj * view;
        self.view = view.into();
        self.view_proj = view_proj.into();
        self.inv_proj = proj.invert().unwrap().into();
        self.inv_view = view.transpose().into();
    }
}
