import numpy as np
from kivy.app import App
from kivy.uix.label import Label
from kivy.clock import Clock
from plyer import accelerometer, gyroscope
import time

class IMUCalibrationApp(App):
    def build(self):
        # Label to keep Kivy window alive
        self.status_label = Label(text="Keep the phone stationary.\nCollecting IMU data...")
        
        # Buffers to store raw IMU data
        self.accel_data = []
        self.gyro_data = []
        
        # Calibration duration in seconds
        self.calibration_duration = 120  # 2 minutes, adjust as needed
        self.start_time = time.time()
        
        accelerometer.enable()
        gyroscope.enable()
        
        # Schedule sensor reading at 60 Hz
        Clock.schedule_interval(self.read_sensors, 1.0 / 60.0)
        return self.status_label

    def read_sensors(self, dt):
        # Read sensors
        acc = accelerometer.acceleration
        gyro = gyroscope.rotation

        # Only append if both are valid
        if acc is not None and gyro is not None:
            # Convert to floats
            try:
                acc = [float(x) for x in acc]
                gyro = [float(x) for x in gyro]
                self.accel_data.append(acc)
                self.gyro_data.append(gyro)
            except (ValueError, TypeError):
                pass  # skip invalid readings

        # Stop after calibration duration
        if time.time() - self.start_time >= self.calibration_duration:
            Clock.unschedule(self.read_sensors)
            if len(self.accel_data) > 0 and len(self.gyro_data) > 0:
                self.compute_and_save_noise()
            self.stop()

    def compute_and_save_noise(self):
        acc_arr = np.array(self.accel_data)
        gyro_arr = np.array(self.gyro_data)
        
        # Estimate gravity from mean acceleration
        g_vec = np.mean(acc_arr, axis=0)
        acc_no_g = acc_arr - g_vec.reshape(1, 3)
        
        # Compute EKF noise parameters
        var_acc = np.var(acc_no_g, axis=0).mean()
        var_gyro = np.var(gyro_arr, axis=0).mean()
        
        # Rough bias instability estimates (fraction of noise variance)
        var_ba = var_acc * 0.1
        var_bg = var_gyro * 0.1
        
        # Save to text file
        with open("imu_noise_params.txt", "w") as f:
            f.write("Estimated IMU noise parameters:\n")
            f.write(f"var_acc  = {var_acc:.6f}  # (m/s²)^2\n")
            f.write(f"var_gyro = {var_gyro:.6f}  # (rad/s)^2\n")
            f.write(f"var_ba   = {var_ba:.6f}  # accelerometer bias instability\n")
            f.write(f"var_bg   = {var_bg:.6f}  # gyroscope bias instability\n")

if __name__ == "__main__":
    IMUCalibrationApp().run()
