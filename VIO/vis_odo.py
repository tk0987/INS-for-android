import numpy as np
from scipy import linalg
from scipy.spatial import cKDTree

EPS = 1e-12

# -------------------------
# 1. Mutual + ratio descriptor matcher (L2-normalized descriptors)
# -------------------------
def match_descriptors(desc1, desc2, ratio=0.8, dist_thresh=0.6):
    """
    Mutual nearest + Lowe ratio + absolute distance threshold.
    desc1: (N1, D), desc2: (N2, D) -- can be None/empty
    Returns: list of (idx1, idx2)
    """
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []

    # L2 normalize rows (safe even if already normalized)
    def l2_normalize_rows(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.maximum(norms, EPS)
        return X / norms

    d1 = l2_normalize_rows(desc1.astype(np.float64))
    d2 = l2_normalize_rows(desc2.astype(np.float64))

    # kd-trees both directions
    k = 2 if len(d2) >= 2 else 1
    tree12 = cKDTree(d2)
    d12, i12 = tree12.query(d1, k=k)

    k21 = 2 if len(d1) >= 2 else 1
    tree21 = cKDTree(d1)
    d21, i21 = tree21.query(d2, k=k21)

    # normalize shapes when k==1
    if k == 1:
        d12 = d12.reshape(-1, 1)
        i12 = i12.reshape(-1, 1)
    if k21 == 1:
        d21 = d21.reshape(-1, 1)
        i21 = i21.reshape(-1, 1)

    matches = []
    for i in range(len(d1)):
        # require two neighbors for ratio test
        if d12.shape[1] < 2:
            continue
        if d12[i, 0] < ratio * (d12[i, 1] + EPS) and d12[i, 0] < dist_thresh:
            j = int(i12[i, 0])
            # mutual check
            if int(i21[j, 0]) == i:
                matches.append((i, j))

    return matches

# -------------------------
# 2. Eight-point for normalized coords (returns Essential matrix)
# -------------------------
def eight_point_normalized(x1, x2, enforce_essential=True):
    """
    Eight-point algorithm assuming x1, x2 are normalized image coordinates (N,2).
    Returns E (3x3) in normalized-coordinate form (i.e., E such that x2^T E x1 = 0).
    """
    assert x1.shape == x2.shape and x1.shape[0] >= 8, "need at least 8 correspondences"
    N = x1.shape[0]
    x1x, x1y = x1[:, 0], x1[:, 1]
    x2x, x2y = x2[:, 0], x2[:, 1]

    A = np.column_stack([
        x2x * x1x, x2x * x1y, x2x,
        x2y * x1x, x2y * x1y, x2y,
        x1x,       x1y,       np.ones(N)
    ])

    _, _, Vt = linalg.svd(A)
    E_hat = Vt[-1].reshape(3, 3)

    # Enforce rank-2 (smallest singular value -> 0) robustly
    U, S, Vt2 = linalg.svd(E_hat)
    S = np.pad(S, (0, max(0, 3 - len(S))), constant_values=0)
    S[2] = 0.0
    E_rank2 = U @ np.diag(S) @ Vt2

    if enforce_essential:
        # enforce essential matrix singular values (a,a,0)
        Ue, Se, Vte = linalg.svd(E_rank2)
        Se = np.pad(Se, (0, max(0, 3 - len(Se))), constant_values=0)
        a = float((Se[0] + Se[1]) / 2.0) if (Se[0] + Se[1]) > 0 else 0.0
        E = Ue @ np.diag([a, a, 0.0]) @ Vte
    else:
        E = E_rank2

    return E

# -------------------------
# 3. Sampson distance (for E) -- operates on normalized homogeneous coords
# -------------------------
def sampson_distance(E, x1h, x2h):
    """
    x1h, x2h: (N,3) homogeneous normalized coordinates
    returns: (N,) Sampson distances
    """
    Ex1 = (E @ x1h.T)              # 3 x N
    Et_x2 = (E.T @ x2h.T)          # 3 x N
    x2tEx1 = np.sum(x2h * Ex1.T, axis=1)   # N
    num = x2tEx1**2
    den = Ex1[0]**2 + Ex1[1]**2 + Et_x2[0]**2 + Et_x2[1]**2 + EPS
    return num / den

# -------------------------
# 4. RANSAC for Essential matrix (using normalized coords and Sampson distance)
# -------------------------
def estimate_essential_ransac(pts1_px, pts2_px, K, thresh_px=1.0, max_iters=2000, prob=0.999):
    """
    pts1_px, pts2_px: (N,2) pixel coordinates (matched)
    K: camera intrinsics (3x3)
    Returns: E (3x3) in normalized coords, inlier_mask (N,)
    """
    N = len(pts1_px)
    if N < 8:
        return None, np.zeros(N, dtype=bool)

    # Normalize by K once
    Kinv = linalg.inv(K)
    ones = np.ones((N, 1))
    x1h = (Kinv @ np.hstack([pts1_px, ones]).T).T    # (N,3)
    x2h = (Kinv @ np.hstack([pts2_px, ones]).T).T

    x1 = x1h[:, :2]
    x2 = x2h[:, :2]

    # convert pixel threshold to normalized units approximately using focal length
    focal = (K[0, 0] + K[1, 1]) / 2.0
    thresh_norm = (thresh_px / (focal + EPS))**2  # Sampson is squared-dist-like, compare squared

    best_E = None
    best_inliers = None
    best_count = 0
    # adaptive iteration cap (optional)
    iters = 0
    max_trials = max_iters

    while iters < max_trials:
        iters += 1
        # sample 8 unique indices
        if N == 8:
            idx = np.arange(8)
        else:
            idx = np.random.choice(N, 8, replace=False)

        try:
            E_candidate = eight_point_normalized(x1[idx], x2[idx], enforce_essential=True)
        except Exception:
            continue

        # compute Sampson distances for all points
        ds = sampson_distance(E_candidate, x1h, x2h)   # (N,)
        inliers = ds < thresh_norm
        count = int(np.sum(inliers))

        if count > best_count:
            best_count = count
            best_E = E_candidate
            best_inliers = inliers
            # update adaptive max_trials using desired probability (optional)
            inlier_ratio = count / float(N)
            # avoid log(0)
            if inlier_ratio > 0:
                p_no_outliers = 1 - inlier_ratio**8
                p_no_outliers = np.clip(p_no_outliers, EPS, 1 - EPS)
                max_trials = min(max_iters, int(np.log(1 - prob) / np.log(p_no_outliers) + 1))
        # early exit if very good
        if best_count > 0.8 * N:
            break

    if best_E is None or best_inliers is None or np.sum(best_inliers) < 8:
        return None, np.zeros(N, dtype=bool)

    # refine E on inliers (recompute from normalized coords)
    try:
        E_refined = eight_point_normalized(x1[best_inliers], x2[best_inliers], enforce_essential=True)
    except AssertionError:
        E_refined = best_E

    return E_refined, best_inliers

# -------------------------
# 5. Decompose essential matrix
# -------------------------
def decompose_essential(E):
    U, _, Vt = linalg.svd(E)
    # ensure right-handedness
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    if np.linalg.det(Vt) < 0:
        Vt[-1, :] *= -1

    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])

    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]
    return [(R1,  t), (R1, -t), (R2,  t), (R2, -t)]

# -------------------------
# 6. Triangulation (normalized coords)
# -------------------------
def triangulate_point_normalized(p1, p2, P1, P2):
    """
    p1, p2: normalized (x,y) image coords (not pixels)
    P1, P2: 3x4 projection matrices in normalized coords
    returns 3D point (X,Y,Z) or NaNs if invalid
    """
    x1, y1 = p1
    x2, y2 = p2
    A = np.vstack([
        x1 * P1[2, :] - P1[0, :],
        y1 * P1[2, :] - P1[1, :],
        x2 * P2[2, :] - P2[0, :],
        y2 * P2[2, :] - P2[1, :]
    ])
    _, _, Vt = linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < EPS:
        return np.array([np.nan, np.nan, np.nan])
    return (X[:3] / X[3])

def triangulate_points_normalized(pts1_norm, pts2_norm, R, t):
    """
    pts1_norm, pts2_norm: (M,2) normalized coordinates
    R, t: pose of camera2 relative to camera1
    returns: (M,3) points in camera1 coordinates
    """
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])          # normalized
    P2 = np.hstack([R, t.reshape(3, 1)])
    pts3d = []
    for p1, p2 in zip(pts1_norm, pts2_norm):
        X = triangulate_point_normalized(p1, p2, P1, P2)
        pts3d.append(X)
    return np.array(pts3d)

# -------------------------
# 7. Choose correct R,t from decompositions (using normalized coords)
# -------------------------
def choose_rt_from_solutions(solutions, pts1_in_px, pts2_in_px, K):
    """
    solutions: list of (R, t)
    pts*_in_px: matched inlier pixel coordinates (M,2)
    K: intrinsics (used to normalize coordinates)
    """
    # normalize points once
    Kinv = linalg.inv(K)
    ones = np.ones((pts1_in_px.shape[0], 1))
    x1h = (Kinv @ np.hstack([pts1_in_px, ones]).T).T
    x2h = (Kinv @ np.hstack([pts2_in_px, ones]).T).T
    x1_norm = x1h[:, :2]
    x2_norm = x2h[:, :2]

    best = None
    best_pts3d = None
    best_count = -1
    for R, t in solutions:
        pts3d = triangulate_points_normalized(x1_norm, x2_norm, R, t)
        if np.any(np.isnan(pts3d)):
            continue
        z1 = pts3d[:, 2]
        X2 = (R @ pts3d.T) + t.reshape(3, 1)
        z2 = X2[2, :]
        count = int(np.sum((z1 > 1e-6) & (z2 > 1e-6)))
        if count > best_count:
            best = (R, t)
            best_pts3d = pts3d
            best_count = count
    return best, best_pts3d, best_count

# -------------------------
# 8. Main relative pose estimation function (integrated pipeline)
# -------------------------
def relative_pose_from_frames(kps1, desc1, kps2, desc2, K,
                              R_imu=np.eye(3),
                              ransac_iters=2000,
                              ransac_thresh_px=1.5,
                              use_imu_rotation=True,
                              imu_weight=0.9):
    """
    Estimate relative pose between two frames using visual matches,
    optionally constraining rotation with IMU estimate.
    """
    matches = match_descriptors(desc1, desc2, ratio=0.9)
    if len(matches) < 8:
        return None, None, None, None

    pts1 = np.array([kps1[i] for i, _ in matches])
    pts2 = np.array([kps2[j] for _, j in matches])

    E, inlier_mask = estimate_essential_ransac(
        pts1, pts2, K, thresh_px=ransac_thresh_px, max_iters=ransac_iters
    )
    if E is None:
        return None, None, None, None

    # ensure inlier_mask length matches matches
    if inlier_mask.shape[0] != pts1.shape[0]:
        return None, None, None, None

    pts1_in = pts1[inlier_mask]
    pts2_in = pts2[inlier_mask]
    if pts1_in.shape[0] < 8:
        return None, None, None, None

    # decompose E → multiple (R, t)
    solutions = decompose_essential(E)
    best, pts3d, count = choose_rt_from_solutions(solutions, pts1_in, pts2_in, K)
    if best is None:
        return None, None, None, None

    R_est, t_est = best

    # ---- 🔹 Optional IMU fusion ----
    if use_imu_rotation and not np.allclose(R_imu, np.eye(3)):
        # Blend R_est with R_imu using SVD-based correction
        # Convert to rotation vector form for smooth interpolation
        dR = R_imu.T @ R_est
        theta = np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
        if theta > 1e-6:
            axis = (1 / (2 * np.sin(theta))) * np.array([
                dR[2, 1] - dR[1, 2],
                dR[0, 2] - dR[2, 0],
                dR[1, 0] - dR[0, 1]
            ])
            R_correction = expm_so3(axis * theta * (1 - imu_weight))
            R_est = R_imu @ R_correction

    # ---- Build full inlier mask ----
    full_mask = np.zeros(len(matches), dtype=bool)
    full_mask[np.where(inlier_mask)[0]] = True

    return R_est, t_est, full_mask, pts3d


def expm_so3(omega):
    """Exponential map from so(3) to SO(3)."""
    theta = np.linalg.norm(omega)
    if theta < 1e-9:
        return np.eye(3)
    k = omega / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
