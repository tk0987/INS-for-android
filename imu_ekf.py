import numpy as np
from scipy.spatial.transform import Rotation as R


def skew(v):
    """Return 3x3 skew-symmetric matrix from a 3D vector."""
    v = np.asarray(v).reshape(3)
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


class IMUErrorStateEKF:
    def __init__(self, config,
                 var_acc=0.004631**2, var_gyro=1e-6,
                 var_ba=2e-6, var_bg=1e-7,
                 gravity=np.array([0, 0, 9.828663]),
                 P0=None, x0=None):
        """Initialize the IMU Error-State Extended Kalman Filter."""

        # Load config as dict
        self.config = config if isinstance(config, dict) else {
            k: getattr(config, k) for k in dir(config) if not k.startswith("__")
        }

        # Load noise and initial state
        self.Q = self.config.get("Q", np.eye(12))
        self.R = self.config.get("R_meas", np.eye(6))
        self.P = self.config.get("P0", np.eye(15)).copy()
        self.x = self.config.get("x0", None)

        # Initialize nominal state
        self.q = R.from_quat([0, 0, 0, 1])
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)

        if x0:
            self.q = x0["q"] if isinstance(x0.get("q"), R) else R.from_quat(x0.get("q", [0, 0, 0, 1]))
            self.p = np.asarray(x0.get("p", self.p)).astype(float)
            self.v = np.asarray(x0.get("v", self.v)).astype(float)
            self.ba = np.asarray(x0.get("ba", self.ba)).astype(float)
            self.bg = np.asarray(x0.get("bg", self.bg)).astype(float)

        # Initialize error covariance
        if P0 is None:
            P0 = np.eye(15)
            P0[0:3, 0:3] = P0[3:6, 3:6] = P0[6:9, 6:9] = np.eye(3) * 10.0
        self.P = P0.astype(float)

        # Noise parameters
        self.var_acc = var_acc
        self.var_gyro = var_gyro
        self.var_ba = var_ba
        self.var_bg = var_bg
        self.g = gravity
        self.a_world_with_g = np.zeros(3)

    def get_state(self):
        """Return current nominal state as a dictionary."""
        return {
            "q": self.q,
            "p": self.p.copy(),
            "v": self.v.copy(),
            "ba": self.ba.copy(),
            "bg": self.bg.copy()
        }

    def inject_error_and_reset(self, dx):
        """Inject error state into nominal state and reset error."""
        dtheta, dp, dv, dba, dbg = np.split(dx, 5)
        dq = R.from_rotvec(dtheta)
        self.q = dq * self.q
        self.p += dp
        self.v += dv
        self.ba += dba
        self.bg += dbg

    def predict(self, a_meas, w_meas, dt):
        """Propagate nominal state and error covariance forward in time."""
        a_meas = np.asarray(a_meas).reshape(3)
        w_meas = np.asarray(w_meas).reshape(3)

        omega = w_meas - self.bg
        self.q = R.from_rotvec(omega * dt) * self.q

        a_unbiased = a_meas - self.ba
        R_bw = self.q.as_matrix()
        a_world = R_bw @ a_unbiased
        self.a_world_with_g = a_world# - self.g

        self.p += self.v * dt + 0.5 * self.a_world_with_g * dt**2
        self.v += self.a_world_with_g * dt

        # Linearized dynamics
        Fc = np.zeros((15, 15))
        Fc[0:3, 0:3] = -skew(omega)
        Fc[0:3, 12:15] = -np.eye(3)
        Fc[3:6, 6:9] = np.eye(3)
        Fc[6:9, 0:3] = -R_bw @ skew(a_unbiased)
        Fc[6:9, 9:12] = -R_bw

        Fd = np.eye(15) + Fc * dt

        # Noise Jacobian
        Gc = np.zeros((15, 12))
        Gc[0:3, 3:6] = Gc[0:3, 9:12] = -np.eye(3)
        Gc[6:9, 0:3] = R_bw
        Gc[9:12, 6:9] = Gc[12:15, 9:12] = np.eye(3)

        # Process noise
        Qc = np.diag([
            self.var_acc, self.var_acc, self.var_acc,
            self.var_gyro, self.var_gyro, self.var_gyro,
            self.var_ba, self.var_ba, self.var_ba,
            self.var_bg, self.var_bg, self.var_bg
        ])
        Qd = Gc @ Qc @ Gc.T * dt
        self.P = Fd @ self.P @ Fd.T + Qd

        # Normalize quaternion
        qvec = self.q.as_quat()
        self.q = R.from_quat(qvec / np.linalg.norm(qvec))

    def update_position(self, p_meas, R_meas):
        """Update state using position measurement."""
        H = np.zeros((3, 15))
        H[:, 3:6] = np.eye(3)
        z = np.asarray(p_meas).reshape(3)

        S = H @ self.P @ H.T + R_meas
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ (z - self.p)

        self.inject_error_and_reset(dx)
        I = np.eye(15)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R_meas @ K.T

    def update_zupt_xyz(self, cov_v=1e0):
        """Zero-velocity update in all 3 axes."""
        H = np.zeros((3, 15))
        H[:, 6:9] = np.eye(3)
        Rm = np.eye(3) * cov_v
        z = np.zeros(3)

        S = H @ self.P @ H.T + Rm
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ (z - self.v)

        self.inject_error_and_reset(dx)
        I = np.eye(15)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ Rm @ K.T
        self.v[:] = 0.0  # enforce zero velocity
    def get_accel_world_no_gravity(self):
        """Return gravity-free world acceleration from last prediction step."""
        return self.a_world_with_g.copy()
