import numpy as np
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.clock import Clock
from plyer import accelerometer, gyroscope
from imu_ekf import IMUErrorStateEKF
from config import Config

class SensorApp(App):
    def build(self):
        # Initialize EKF
        self.ekf = IMUErrorStateEKF(Config)

        # Buffers for smoothing
        self.buffer_len = 10
        self.accel_buffer = np.zeros((self.buffer_len, 3))
        self.gyro_buffer = np.zeros((self.buffer_len, 3))

        # UI
        layout = GridLayout(cols=1, padding=10, spacing=6)
        self.accel_label = Label()
        self.gyro_label = Label()
        self.angle_label = Label()
        self.posi_label = Label()
        self.vel_label = Label()
        self.status_label = Label()
        for w in [self.accel_label, self.gyro_label, self.angle_label,
                  self.posi_label, self.vel_label, self.status_label]:
            layout.add_widget(w)

        accelerometer.enable()
        gyroscope.enable()
        Clock.schedule_interval(self.update_sensors, 1.0 / 60.0)
        return layout

    def safe_vector(self, data):
        try:
            if data is None or len(data) != 3:
                return None
            arr = np.array(data, dtype=float)
            return None if np.any(np.isnan(arr)) else arr
        except Exception:
            return None

    def update_sensors(self, dt):
        accel_raw = self.safe_vector(accelerometer.acceleration) 
        gyro_raw = self.safe_vector(gyroscope.rotation)
        if accel_raw is None or gyro_raw is None:
            self.status_label.text = "Waiting for sensor data..."
            return

        # Smooth data
        self.accel_buffer = np.roll(self.accel_buffer, -1, axis=0)
        self.accel_buffer[-1] = accel_raw
        self.gyro_buffer = np.roll(self.gyro_buffer, -1, axis=0)
        self.gyro_buffer[-1] = gyro_raw
        accel_smooth = np.mean(self.accel_buffer, axis=0) 
        gyro_smooth = np.mean(self.gyro_buffer, axis=0)

        # EKF predict
        self.ekf.predict(accel_smooth, gyro_smooth, dt)
        state = self.ekf.get_state()

        # Orientation
        q = state["q"]
        roll, pitch, yaw = q.as_euler("xyz", degrees=True)

        # World linear acceleration (gravity-free)
        R_bw = q.as_matrix()
        a_unbiased = accel_smooth - self.ekf.ba
        a_world = R_bw @ a_unbiased
        a_world_no_grav = a_world + self.ekf.g

        # ZUPT detection
        if np.linalg.norm(a_world_no_grav) < 0.5 and np.linalg.norm(gyro_smooth) < 0.01:
            self.ekf.update_zupt()
            motion_status = "Stationary (ZUPT applied)"
        else:
            motion_status = "In motion"

        # Position & velocity
        p = state["p"]
        v = state["v"]

        # Update UI
        self.accel_label.text = f"Accel (world, no g): {a_world_no_grav.round(3)}"
        self.gyro_label.text = f"Gyro: {gyro_smooth.round(3)}"
        self.angle_label.text = f"Orientation (deg):\nRoll={roll:.1f}, Pitch={pitch:.1f}, Yaw={yaw:.1f}"
        self.posi_label.text = f"Position (m):\nX={p[0]:.3f}, Y={p[1]:.3f}, Z={p[2]:.3f}"
        self.vel_label.text = f"Velocity (m/s):\nVx={v[0]:.3f}, Vy={v[1]:.3f}, Vz={v[2]:.3f}"
        self.status_label.text = motion_status

    def on_stop(self):
        accelerometer.disable()
        gyroscope.disable()


if __name__ == "__main__":
    SensorApp().run()
