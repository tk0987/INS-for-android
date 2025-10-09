# imu_ekf.py
import numpy as np
from scipy.spatial.transform import Rotation as R


def skew(v):
    """Return 3x3 skew-symmetric matrix for vector v (3,)"""
    v = np.asarray(v).reshape(3)
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


class IMUErrorStateEKF:
    def __init__(self, config,
                 var_acc=0.004631, var_gyro=1e-6,
                 var_ba=2e-6, var_bg=1e-7,
                 g=np.array([0, 0, 9.828663]),
                 P0=None, x0=None):
        """
        Extended Kalman Filter for IMU error-state model.
        Works with both dict or Config class input.
        """

        # ✅ Support both dict and class-style config
        if isinstance(config, dict):
            self.config = config
        else:
            # convert Config class attributes into dict
            self.config = {k: getattr(config, k)
                           for k in dir(config)
                           if not k.startswith("__")}

        # load covariance matrices
            self.Q = getattr(config, "Q", np.eye(12))        # process noise
            self.R = getattr(config, "R_meas", np.eye(6))   # measurement noise
            self.P = getattr(config, "P0", np.eye(15)).copy()
            self.x = getattr(config, "x0", None).copy()

        # Nominal state initialization
        if x0 is None:
            self.q = R.from_quat([0, 0, 0, 1])
            self.p = np.zeros(3)
            self.v = np.zeros(3)
            self.ba = np.zeros(3)
            self.bg = np.zeros(3)
        else:
            # x0 expected as dict with q (Rotation or quat), p, v, ba, bg
            if isinstance(x0.get("q"), R):
                self.q = x0["q"]
            else:
                self.q = R.from_quat(x0.get("q", [0, 0, 0, 1]))
            self.p = np.asarray(x0.get("p", np.zeros(3))).astype(float)
            self.v = np.asarray(x0.get("v", np.zeros(3))).astype(float)
            self.ba = np.asarray(x0.get("ba", np.zeros(3))).astype(float)
            self.bg = np.asarray(x0.get("bg", np.zeros(3))).astype(float)

        # error covariance (15x15)
        if P0 is None:
            P0 = np.eye(15) 
            # enlarge pos/vel uncertainty
            P0[3:6, 3:6] = np.eye(3) * 10.0
            P0[6:9, 6:9] = np.eye(3) * 10.0
            P0[0:3, 0:3] = np.eye(3) * 10.0
        self.P = P0.astype(float)

        # continuous-time process noise covariances (power spectral densities)
        self.var_acc = float(var_acc)
        self.var_gyro = float(var_gyro)
        self.var_ba = float(var_ba)
        self.var_bg = float(var_bg)

        self.g = np.asarray(g, dtype=float)

    # ----------------------
    # Helpers
    # ----------------------
    def get_state(self):
        """Return nominal state as dict (quat as scipy Rotation)."""
        return {
            "q": self.q,
            "p": self.p.copy(),
            "v": self.v.copy(),
            "ba": self.ba.copy(),
            "bg": self.bg.copy()
        }

    def inject_error_and_reset(self, dx):
        """
        dx: 15-vector error state [dtheta(3), dp(3), dv(3), dba(3), dbg(3)]
        Apply to nominal state and reset error state to zero.
        """
        dtheta = dx[0:3]
        dp = dx[3:6]
        dv = dx[6:9]
        dba = dx[9:12]
        dbg = dx[12:15]

        # apply small rotation: q <- delta_q * q
        angle = np.linalg.norm(dtheta)
        if angle > 1e-12:
            axis = dtheta / angle
            dq = R.from_rotvec(axis * angle)
        else:
            dq = R.from_rotvec(dtheta)
        self.q = dq * self.q

        # additive corrections
        self.p += dp
        self.v += dv
        self.ba += dba
        self.bg += dbg

    # ----------------------
    # Prediction step
    # ----------------------
    def predict(self, a_meas, w_meas, dt):
        """Propagate nominal state and covariance."""
        a_meas = np.asarray(a_meas, dtype=float).reshape(3)
        w_meas = np.asarray(w_meas, dtype=float).reshape(3)

        omega = w_meas - self.bg  # rad/s
        rot_delta = R.from_rotvec(omega * dt)
        self.q = rot_delta * self.q

        # accel to world
        a_unbiased = a_meas - self.ba
        R_bw = self.q.as_matrix()
        a_world = R_bw @ a_unbiased
        a_world_with_g = a_world + self.g

        # integrate
        self.p = self.p + self.v * dt + 0.5 * a_world_with_g * dt**2
        self.v = self.v + a_world_with_g * dt

        # covariance propagation
        Fc = np.zeros((15, 15))
        Fc[0:3, 0:3] = -skew(omega)
        Fc[0:3, 12:15] = -np.eye(3)
        Fc[3:6, 6:9] = np.eye(3)
        Fc[6:9, 0:3] = -R_bw @ skew(a_unbiased)
        Fc[6:9, 9:12] = -R_bw

        Fd = np.eye(15) + Fc * dt

        Gc = np.zeros((15, 12))
        Gc[0:3, 3:6] = -np.eye(3)
        Gc[0:3, 9:12] = -np.eye(3)
        Gc[6:9, 0:3] = R_bw
        Gc[9:12, 6:9] = np.eye(3)
        Gc[12:15, 9:12] = np.eye(3)

        Qc = np.zeros((12, 12))
        Qc[0:3, 0:3] = np.eye(3) * self.var_acc
        Qc[3:6, 3:6] = np.eye(3) * self.var_gyro
        Qc[6:9, 6:9] = np.eye(3) * self.var_ba
        Qc[9:12, 9:12] = np.eye(3) * self.var_bg

        Qd = (Gc @ Qc @ Gc.T) * dt
        self.P = Fd @ self.P @ Fd.T + Qd

        # normalize quaternion
        qvec = self.q.as_quat()
        qnorm = np.linalg.norm(qvec)
        if qnorm > 0:
            self.q = R.from_quat(qvec / qnorm)

    # ----------------------
    # Measurement updates
    # ----------------------
    def update_position(self, p_meas, R_meas):
        """Position update."""
        H = np.zeros((3, 15))
        H[:, 3:6] = np.eye(3)
        z = np.asarray(p_meas).reshape(3)

        S = H @ self.P @ H.T + R_meas
        K = self.P @ H.T @ np.linalg.inv(S)
        y = z - self.p
        dx = K @ y

        self.inject_error_and_reset(dx)
        I = np.eye(15)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_meas @ K.T

    def update_zupt(self, cov_v=1e-2):
        """Zero-Velocity Update."""
        H = np.zeros((3, 15))
        H[:, 6:9] = np.eye(3)
        Rm = np.eye(3) * cov_v
        z = np.zeros(3)

        S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ np.linalg.inv(S)
        y = z - self.v
        dx = K @ y

        self.inject_error_and_reset(dx)
        I = np.eye(15)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ Rm @ K.T
