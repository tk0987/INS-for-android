#import numpy as np
#from scipy import ndimage
#from kivy.app import App
#from kivy.uix.boxlayout import BoxLayout
#from kivy.uix.label import Label
#from kivy.uix.image import Image
#from kivy.uix.camera import Camera
#from kivy.graphics import PushMatrix, PopMatrix, Rotate
#from kivy.graphics.texture import Texture
#from kivy.uix.button import Button
#import traceback
#from vis_odo import relative_pose_from_frames
#from kivy.clock import Clock
#from plyer import accelerometer, gyroscope
#from imu_ekf import IMUErrorStateEKF
#from config import Config  # Make sure this is a class, not a dict
#from scipy.fft import fft, ifft,fftfreq
# ---------- SURF-like feature extraction ----------
#def fast_hessian_determinant(img, sigma):
#    Lxx = ndimage.gaussian_filter(img, sigma=sigma, order=(2, 0))
#    Lyy = ndimage.gaussian_filter(img, sigma=sigma, order=(0, 2))
#    Lxy = ndimage.gaussian_filter(img, sigma=sigma, order=(1, 1))
#    return Lxx * Lyy - (0.9 * Lxy) ** 2

#def detect_keypoints(img, sigmas=(1.5, 3.0, 6.0, 12.0), percentile=99.5):
#    keypoints = []
#    for sigma in sigmas:
#        det = fast_hessian_determinant(img, sigma)
#        maxima = (det == ndimage.maximum_filter(det, size=3))
#        thresh = np.percentile(det, percentile)
#        mask = (det > thresh) & maxima
#        ys, xs = np.nonzero(mask)
#        keypoints.extend([(int(y), int(x), float(sigma)) for y, x in zip(ys, xs)])
#    return keypoints, None

#def haar_wavelet_descriptors(img, keypoints, patch_size=20, grid=4):
#    dx = ndimage.convolve(img, np.array([[-1, 1]]), mode='reflect')
#    dy = ndimage.convolve(img, np.array([[-1], [1]]), mode='reflect')
#    descriptors = []
#    step = patch_size // grid
#    for y, x, sigma in keypoints:
#        half = patch_size // 2
#        if y - half < 0 or x - half < 0 or y + half >= img.shape[0] or x + half >= img.shape[1]:
#            continue
#        patch_dx = dx[y - half:y + half, x - half:x + half]
#        patch_dy = dy[y - half:y + half, x - half:x + half]
#        patch_dx = patch_dx[:step * grid, :step * grid].reshape(grid, step, grid, step)
#        patch_dy = patch_dy[:step * grid, :step * grid].reshape(grid, step, grid, step)
#        sum_dx = patch_dx.sum(axis=(1, 3))
#        sum_dy = patch_dy.sum(axis=(1, 3))
#        desc = np.hstack([sum_dx.flatten(), sum_dy.flatten(),
#                          np.abs(sum_dx).flatten(), np.abs(sum_dy).flatten()])
#        desc /= (np.linalg.norm(desc) + 1e-6)
#        descriptors.append(desc.astype(np.float32))
#    return np.array(descriptors, dtype=np.float32)

#def surf_features(img, sigmas=(1.5, 3.0, 6.0, 12.0), percentile=99.5, patch_size=20):
#    keypoints, _ = detect_keypoints(img, sigmas, percentile)
#    descriptors = haar_wavelet_descriptors(img, keypoints, patch_size)
#    return keypoints, descriptors

# ---------- Rotated Camera ----------
#class RotatedCamera(Camera):
#    def __init__(self, **kwargs):
#        super().__init__(**kwargs)
#        with self.canvas.before:
#            PushMatrix()
#            self.rot = Rotate(angle=-90, origin=self.center)
#        with self.canvas.after:
#            PopMatrix()
#        self.bind(pos=self._update_origin, size=self._update_origin)

#    def _update_origin(self, *args):
#        self.rot.origin = self.center

# ---------- Main App ----------
#class CameraARCoreApp(App):
#    def build(self):
#        self.prev_gray = None
#        self.prev_kps = None
#        self.prev_desc = None

#        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

#        # Top labels
#        label_box = BoxLayout(orientation='vertical', size_hint_y=None, height=450, spacing=5)
#        self.pose_label = Label(text="📌 Camera started", size_hint_y=None, height=40)
#        self.R_label = Label(text="🔄 R: Not yet estimated", size_hint_y=None, height=260)
#        self.t_label = Label(text="➡️ t: Not yet estimated", size_hint_y=None, height=60)
#        self.orient_label = Label(text="➡️ Orientation: Not yet estimated", size_hint_y=None, height=60)
#        label_box.add_widget(self.pose_label)
#        label_box.add_widget(self.R_label)
#        label_box.add_widget(self.t_label)
#        main_layout.add_widget(label_box)

#        # Camera preview
#        self.cam_widget = RotatedCamera(index=0, resolution=(640, 480), play=True)
#        main_layout.add_widget(self.cam_widget)

#        # Captured / processed image below camera
#        self.output_img = Image(size_hint_y=None, height=300)
#        main_layout.add_widget(self.output_img)

#        # Capture button at the bottom
#        btn = Button(text="📸 Capture Frame + Run SURF", size_hint_y=None, height=100)
#        btn.bind(on_press=self.capture_frame)
#        main_layout.add_widget(btn)

#        return main_layout

#    def capture_frame(self, instance):
#        try:
#            tex = self.cam_widget.texture
#            if not tex:
#                self.pose_label.text = "No frame available yet!"
#                return

#            w, h = tex.size
#            buf = tex.pixels
#            frame = np.frombuffer(buf, np.uint8).reshape(h, w, 4)

#            # Rotate & flip vertically before processing
#            frame = np.rot90(frame, k=3)
#            frame = np.flipud(frame)

#            gray = (0.2989 * frame[:, :, 0] +
#                    0.5870 * frame[:, :, 1] +
#                    0.1140 * frame[:, :, 2]).astype(np.float32) / 255.0

#            # SURF feature extraction
#            keypoints, descriptors = surf_features(gray)
#            self.pose_label.text = f"Detected {len(keypoints)} keypoints"

#            # Visualization
#            vis = np.dstack([gray, gray, gray])
#            vis = np.clip(vis * 255, 0, 255).astype(np.uint8)
#            for y, x, s in keypoints[:500]:
#                y0, y1 = max(0, y-2), min(vis.shape[0], y+3)
#                x0, x1 = max(0, x-2), min(vis.shape[1], x+3)
#                vis[y0:y1, x0:x1] = [255, 0, 0]

#            buf = vis.flatten()
#            texture = Texture.create(size=(vis.shape[1], vis.shape[0]), colorfmt='rgb')
#            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
#            self.output_img.texture = texture

#            # VO computation
#            if self.prev_gray is None:
#                self.prev_gray = gray
#                self.prev_kps = [(kp[1], kp[0]) for kp in keypoints]
#                self.prev_desc = descriptors
#            else:
#                focal = 23888
#                K = np.array([[focal, 0, w / 2.0],
#                              [0, focal, h / 2.0],
#                              [0, 0, 1.0]])
#                curr_kps = [(kp[1], kp[0]) for kp in keypoints]
#                R, t, mask, pts3d = relative_pose_from_frames(
#                    self.prev_kps, self.prev_desc,
#                    curr_kps, descriptors,
#                    K, ransac_iters=2000, ransac_thresh_px=1.0
#                )
#                if R is not None and t is not None:
#                    self.R_label.text = f"R:\n{np.array2string(R, precision=2)}"
#                    self.t_label.text = f"t:\n{np.array2string(t, precision=4)}"
#                else:
#                    self.R_label.text = "R: Estimation failed"
#                    self.t_label.text = "t: Estimation failed"

#                self.prev_gray = gray
#                self.prev_kps = curr_kps
#                self.prev_desc = descriptors

#        except Exception as e:
#            self.pose_label.text = f"Error: {str(e)}"
#            traceback.print_exc()

#if __name__ == '__main__':
#    CameraARCoreApp().run()
import numpy as np
from scipy import ndimage
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.camera import Camera
from kivy.graphics import PushMatrix, PopMatrix, Rotate
from kivy.graphics.texture import Texture
from kivy.uix.button import Button
from kivy.clock import Clock
import traceback

from plyer import accelerometer, gyroscope
from imu_ekf import IMUErrorStateEKF
from config import Config
from scipy.fft import fft, ifft, fftfreq
from vis_odo import relative_pose_from_frames
from surf import surf_features

# ---------- Sensor & EKF utilities ----------
def safe_vector(data):
    try:
        if data is None or len(data) != 3:
            return None
        arr = np.array(data, dtype=float)
        return None if np.any(np.isnan(arr)) else arr
    except Exception:
        return None


def fft_filter(window, sample_rate=50.0):
    window = np.asarray(window)
    if window.size == 0:
        return np.zeros((0, 3))
    freqs = fftfreq(len(window), 1 / sample_rate)
    yf = fft(window, axis=0)
    mask = (freqs >= -9) & (freqs <= 9)
    yf = np.where(mask[:, np.newaxis], yf, 0)
    return np.real(ifft(yf, axis=0))


# ---------- Rotated Camera ----------
class RotatedCamera(Camera):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas.before:
            PushMatrix()
            self.rot = Rotate(angle=-90, origin=self.center)
        with self.canvas.after:
            PopMatrix()
        self.bind(pos=self._update_origin, size=self._update_origin)

    def _update_origin(self, *args):
        self.rot.origin = self.center

class CameraARCoreApp(App):
    def build(self):
        self.prev_gray = None
        self.prev_kps = None
        self.prev_desc = None
        self.prev_R_imu = None


        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)
        label_box = BoxLayout(orientation='vertical', size_hint_y=None, height=450, spacing=5)
        self.pose_label = Label(text=" Camera started", size_hint_y=None, height=40)
        self.R_label = Label(text=" R: Not yet estimated", size_hint_y=None, height=260)
        self.t_label = Label(text="️ t: Not yet estimated", size_hint_y=None, height=60)
        self.orient_label = Label(text=" Orientation: Initializing...", size_hint_y=None, height=60)
        label_box.add_widget(self.pose_label)
        label_box.add_widget(self.R_label)
        label_box.add_widget(self.t_label)
        label_box.add_widget(self.orient_label)
        main_layout.add_widget(label_box)
        self.cam_widget = RotatedCamera(index=0, resolution=(640, 480), play=True)
        main_layout.add_widget(self.cam_widget)
        self.output_img = Image(size_hint_y=None, height=300)
        main_layout.add_widget(self.output_img)
        btn = Button(text=" Capture Frame + Run SURF", size_hint_y=None, height=100)
        btn.bind(on_press=self.capture_frame)
        main_layout.add_widget(btn)
        self.ekf = IMUErrorStateEKF(Config())
        self.accel_buffer = np.zeros((10, 3))
        self.gyro_buffer = np.zeros((10, 3))
        accelerometer.enable()
        gyroscope.enable()
        Clock.schedule_interval(self.update_orientation, 1 / 50.0)
        return main_layout
    def update_orientation(self, dt):
        accel = safe_vector(accelerometer.acceleration)
        gyro = safe_vector(gyroscope.rotation)
        if accel is None or gyro is None:
            self.orient_label.text = " Waiting for IMU data..."
            return
        self.accel_buffer = np.roll(self.accel_buffer, -1, axis=0)
        self.accel_buffer[-1] = accel
        self.gyro_buffer = np.roll(self.gyro_buffer, -1, axis=0)
        self.gyro_buffer[-1] = gyro
        accel_filtered = np.mean(fft_filter(self.accel_buffer), axis=0)
        gyro_filtered = np.mean(self.gyro_buffer, axis=0)
        self.ekf.predict(accel_filtered, gyro_filtered, 1/50.0)
        state = self.ekf.get_state()
        q = state["q"]
        roll, pitch, yaw = q.as_euler("xyz", degrees=True)
        a_world_no_grav = self.ekf.get_accel_world_no_gravity()
        if np.linalg.norm(a_world_no_grav) < 1.5 and np.linalg.norm(gyro_filtered) < 0.5:
            self.ekf.update_zupt_xyz()
        self.orient_label.text = f" Orientation:\nRoll={roll:.1f}°, Pitch={pitch:.1f}°, Yaw={yaw:.1f}°"
    def capture_frame(self, instance):
        try:
            tex = self.cam_widget.texture
            if not tex:
                self.pose_label.text = "No frame available yet!"
                return
    
            state = self.ekf.get_state()
            R_curr = state["q"].as_matrix()  # Current IMU rotation
    
            w, h = tex.size
            buf = tex.pixels
            frame = np.frombuffer(buf, np.uint8).reshape(h, w, 4)
            frame = np.rot90(frame, k=3)
            frame = np.flipud(frame)
            frame = np.fliplr(frame)
    
            gray = (0.2989 * frame[:, :, 0] +
                    0.5870 * frame[:, :, 1] +
                    0.1140 * frame[:, :, 2]).astype(np.float32) / 255.0
            keypoints, descriptors = surf_features(gray)
            self.pose_label.text = f"Detected {len(keypoints)} keypoints"
    
            vis = np.dstack([gray, gray, gray])
            vis = np.clip(vis * 255, 0, 255).astype(np.uint8)
            for y, x, s, a in keypoints[:500]:
                y0, y1 = max(0, y-2), min(vis.shape[0], y+3)
                x0, x1 = max(0, x-2), min(vis.shape[1], x+3)
                vis[y0:y1, x0:x1] = [255, 0, 0]
            buf = vis.flatten()
            texture = Texture.create(size=(vis.shape[1], vis.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.output_img.texture = texture
    
            if self.prev_gray is None or self.prev_R_imu is None:
                self.prev_gray = gray
                self.prev_kps = [(kp[1], kp[0]) for kp in keypoints]
                self.prev_desc = descriptors
                self.prev_R_imu = R_curr
            else:
                focal = 23888
                K = np.array([[focal, 0, w / 2.0],
                              [0, focal, h / 2.0],
                              [0, 0, 1.0]])
                curr_kps = [(kp[1], kp[0]) for kp in keypoints]
    
                # Compute relative rotation from IMU
                R_rel = R_curr @ self.prev_R_imu.T
    
                # Use modified pose estimation
                from vis_odo import relative_pose_from_frames
                R, t, mask, pts3d = relative_pose_from_frames(
                    self.prev_kps, self.prev_desc,
                    curr_kps, descriptors,
                    K, R_rel, ransac_iters=2000, ransac_thresh_px=1.5
                )
    
                if R is not None and t is not None:
                    self.R_label.text = f"R:\n{np.array2string(R, precision=2)}"
                    self.t_label.text = f"t:\n{np.array2string(t, precision=4)}"
                else:
                    self.R_label.text = "R: Estimation failed"
                    self.t_label.text = "t: Estimation failed"
    
                self.prev_gray = gray
                self.prev_kps = curr_kps
                self.prev_desc = descriptors
                self.prev_R_imu = R_curr
        except Exception as e:
            self.pose_label.text = f"Error: {str(e)}"
            traceback.print_exc()
    
    def on_stop(self):
        accelerometer.disable()
        gyroscope.disable()
if __name__ == '__main__':
    CameraARCoreApp().run()
