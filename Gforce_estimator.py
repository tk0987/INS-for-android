import numpy as np
from kivy.app import App
from kivy.uix.label import Label
from kivy.clock import Clock
from plyer import accelerometer, gyroscope
import time

class IMUCalibrationApp(App):
    def build(self):
        self.status_label = Label(text="Keep the phone stationary.\nCollecting IMU data...")
        self.accel_data = []
        self.gyro_data = []
        self.calibration_duration = 120  # seconds
        self.start_time = time.time()

        accelerometer.enable()
        gyroscope.enable()
        Clock.schedule_interval(self.read_sensors, 1.0 / 60.0)
        return self.status_label

    def read_sensors(self, dt):
        acc = accelerometer.acceleration
        gyro = gyroscope.rotation

        if acc is not None and gyro is not None:
            try:
                acc = [float(x) for x in acc]
                gyro = [float(x) for x in gyro]
                self.accel_data.append(acc)
                self.gyro_data.append(gyro)
            except (ValueError, TypeError):
                pass

        if time.time() - self.start_time >= self.calibration_duration:
            Clock.unschedule(self.read_sensors)
            if len(self.accel_data) > 0 and len(self.gyro_data) > 0:
                self.compute_and_save_noise()
            self.stop()

    def compute_and_save_noise(self):
        acc_arr = np.array(self.accel_data)
        gyro_arr = np.array(self.gyro_data)

        # Compute g-force magnitude for each sample
        g_magnitudes = np.linalg.norm(acc_arr, axis=1)  # ||a|| = sqrt(ax² + ay² + az²)

        # Compute mean and standard deviation
        g_mean = np.mean(g_magnitudes)
        g_std = np.std(g_magnitudes)

        # Gyroscope variance as before
        var_gyro = np.var(gyro_arr, axis=0).mean()
        var_bg = var_gyro * 0.1  # rough bias instability

        # Save results
        with open("imu_gforce_params.txt", "w") as f:
            f.write("Estimated IMU noise parameters (g-force magnitude):\n")
            f.write(f"g_mean  = {g_mean:.6f}  # m/s²\n")
            f.write(f"g_std   = {g_std:.6f}  # m/s²\n")
            f.write(f"var_gyro = {var_gyro:.6f}  # (rad/s)^2\n")
            f.write(f"var_bg   = {var_bg:.6f}  # gyroscope bias instability\n")

if __name__ == "__main__":
    IMUCalibrationApp().run()
