import numpy as np
from scipy.spatial.transform import Rotation as Rot

class Config:
    # Continuous process noise (PSD)
    var_acc=0.004631**2
    var_gyro = 0.000005
    var_ba = 0.000002
    var_bg = 0.00000001

    # Covariance matrices
    Q_accel = np.diag([0.1, 0.1, 0.1])
    Q_gyro = np.diag([0.01, 0.01, 0.01])
    Q = np.block([
        [Q_accel, np.zeros((3,3))],
        [np.zeros((3,3)), Q_gyro]
    ])

    R_accel = np.diag([var_acc,var_acc,var_acc])
    R_gyro = np.diag([5*10**-2.5,5*10**-2.5,5*10**-2.5])
    R_meas = np.block([
        [R_accel, np.zeros((3,3))],
        [np.zeros((3,3)), R_gyro]
    ])

    # Initial error covariance
    P0 = np.eye(15) * 1e-3

    # Initial nominal state
    x0 = {
        "q": Rot.from_quat([0, 0, 0, 1]),  # Identity quaternion
        "p": np.zeros(3),
        "v": np.zeros(3),
        "ba": np.zeros(3),
        "bg": np.zeros(3),
    }

    # Gravity
    g = np.array([0, 0, 9.81])
