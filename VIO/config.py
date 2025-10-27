import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.linalg import solve_discrete_are

class Config:
    # Continuous process noise (PSD)
    var_acc=0.004631**2
    var_gyro = 0.000005
    var_ba = 0.000002*12
    var_bg = 0.00000001

    # Covariance matrices
    R_accel = np.array([
        [0.282491373, -0.211905161, 0.001383682],
        [-0.211905161, 0.182725701, -0.001345789],
        [0.001383682, -0.001345789, 0.000159465]
    ]) 

    R_gyro = np.array([
        [0.004512037, -0.001263739, 0.000984814],
        [-0.001263739, 0.001543665, -0.000343729],
        [0.000984814, -0.000343729, 0.000633979]
    ]) 

    Q_accel = np.array([
        [14.0315, -3.4443, -1.5943],
        [-3.4443, 9.7637, 0.6543],
        [-1.5943, 0.6543, 5.2091]
    ]) 

    Q_gyro = np.array([
        [878.774, -70.378, 42.616],
        [-70.378, 1133.271, 221.973],
        [42.616, 221.973, 2202.27]
    ]) 
    Q = np.block([
        [Q_accel, np.zeros((3,3))],
        [np.zeros((3,3)), Q_gyro]
    ])

    R_meas = np.block([
        [R_accel, np.zeros((3,3))],
        [np.zeros((3,3)), R_gyro]
    ])

    # Initial error covariance
    P0 = np.eye(15)

    # Initial nominal state
    x0 = {
        "q": Rot.from_quat([0, 0, 0, 1]),  # Identity quaternion
        "p": np.zeros(3),
        "v": np.zeros(3),
        "ba": np.zeros(3),
        "bg": np.zeros(3),
    }

    # Gravity
    g = np.array([0, 0, 9.828663])
