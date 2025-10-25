import numpy as np
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.clock import Clock
from plyer import accelerometer, gyroscope
from imu_ekf import IMUErrorStateEKF
from config import Config  # Make sure this is a class, not a dict
from scipy.fft import fft, ifft,fftfreq


class SensorApp(App):
    def build(self):
        # EKF initialization
        self.startup_time = 0.5
        self.elapsed_time = 0.0
        self.warming_up = True
        self.ekf = IMUErrorStateEKF(Config())  # ✅ instantiate config class

        # Buffers
        self.buffer_len = 11
        self.accel_buffer = np.zeros((self.buffer_len, 3))
        self.gyro_buffer = np.zeros((self.buffer_len, 3))
        self.sum_a = np.zeros(3)

        # UI setup
        layout = GridLayout(cols=1, padding=10, spacing=6)
        self.accel_label = Label()
        self.gyro_label = Label()
        self.angle_label = Label()
        self.posi_label = Label()
        self.vel_label = Label()
        self.status_label = Label()

        for widget in [
            self.accel_label,
            self.gyro_label,
            self.angle_label,
            self.posi_label,
            self.vel_label,
            self.status_label,
        ]:
            layout.add_widget(widget)

        accelerometer.enable()
        gyroscope.enable()
        Clock.schedule_interval(self.update_sensors, 1.0 / 50.0)
        return layout

    def fft_filter(self, window, T=1 / 50.0):
        window = np.asarray(window)
        if window.size == 0:
            return np.zeros((0, 3))
        freqs=fftfreq(len(window),1/50)
      
        yf = fft(window, axis=0)
        mask = (
            (np.real(yf) <= 1.0)
            & (np.real(yf) >= -1.0) 
            
            & (np.imag(yf) >= -1.0)
            & (np.imag(yf) <= 1.0)
            
           # | (np.real(yf) >= -0.5) 
            #| (np.real(yf) <= 0.5) 
        )
        mask2 = (freqs >= -9) & (freqs <= 9)
        mask2 = mask2[:, np.newaxis]  # shape becomes (10, 1)
        yf = np.where(mask & mask2, yf, 0 + 0j)
        return np.real(ifft(yf, axis=0))

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

        # Buffer update
        self.accel_buffer = np.roll(self.accel_buffer, -1, axis=0)
        self.accel_buffer[-1] = accel_raw
        self.gyro_buffer = np.roll(self.gyro_buffer, -1, axis=0)
        self.gyro_buffer[-1] = gyro_raw

        # Warmup check
        self.elapsed_time += dt
        if self.warming_up:
            if self.elapsed_time >= self.startup_time:
                self.warming_up = False
                self.status_label.text = "Sensor warmup complete"
            else:
                self.status_label.text = f"Warming up... {self.elapsed_time:.2f}s"
            return

        # Filtering
        diffs = np.diff(self.accel_buffer, axis=0)
        filtered = self.fft_filter(diffs)
        mask = np.abs(filtered) > 1e-1
        filtered_masked = np.where(mask, self.accel_buffer[1:], 0.0)
        accel_smooth = np.mean(filtered_masked, axis=0)
        gyro_smooth = np.mean(self.gyro_buffer, axis=0)

        # EKF prediction
        self.ekf.predict(accel_smooth, gyro_smooth, 1/50.0)
        state = self.ekf.get_state()

        # Orientation
        q = state["q"]
        roll, pitch, yaw = q.as_euler("xyz", degrees=True)

        # Gravity-free acceleration
        a_world_no_grav = self.ekf.get_accel_world_no_gravity()
        
        # Suppose accel_change is a 3-element vector, e.g. np.array([ax, ay, az])
        accel_change = np.sum(np.diff(filtered_masked, axis=0), axis=0) # or whatever makes sense
        
        # Compute magnitude
        magnitude = np.linalg.norm(accel_change)
        
        # Apply condition: only sum if magnitude is between 1 and 10
        if 1 < magnitude < 10:
            self.sum_a += accel_change
        else:
            # Add zero (no change)
            self.sum_a += np.zeros_like(accel_change)

        # ZUPT detection
        if np.linalg.norm(a_world_no_grav) < 1.5 and np.linalg.norm(gyro_smooth) < 0.5:
            self.ekf.update_zupt_xyz()
            motion_status = "Stationary (ZUPT applied)"
        else:
            motion_status = "In motion"

        # Position & velocity
        p = state["p"]
        v = state["v"]

        # UI update
        self.accel_label.text = f"Accel (world, no g): {a_world_no_grav.round(3)}\n{self.sum_a}"
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
