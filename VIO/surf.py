import numpy as np
from scipy import ndimage

# ---------- SURF-like feature extraction with orientation + scale normalization ----------

def fast_hessian_determinant(img, sigma):
    """Approximate Hessian determinant using Gaussian second derivatives."""
    Lxx = ndimage.gaussian_filter(img, sigma=sigma, order=(2, 0))
    Lyy = ndimage.gaussian_filter(img, sigma=sigma, order=(0, 2))
    Lxy = ndimage.gaussian_filter(img, sigma=sigma, order=(1, 1))
    return Lxx * Lyy - (0.9 * Lxy) ** 2


def detect_keypoints(img, sigmas=(1.5, 3.0, 6.0, 12.0), percentile=99.0):
    """Detect keypoints as local maxima in determinant-of-Hessian response."""
    keypoints = []
    for sigma in sigmas:
        det = fast_hessian_determinant(img, sigma)
        maxima = (det == ndimage.maximum_filter(det, size=3))
        thresh = np.percentile(det, percentile)
        mask = (det > thresh) & maxima
        ys, xs = np.nonzero(mask)
        for y, x in zip(ys, xs):
            keypoints.append((int(y), int(x), float(sigma)))
    return keypoints


def assign_orientation(img, keypoints, radius_factor=6, num_bins=36):
    """Estimate dominant orientation using Haar wavelet responses."""
    dx = ndimage.convolve(img, np.array([[-1, 1]]), mode='reflect')
    dy = ndimage.convolve(img, np.array([[-1], [1]]), mode='reflect')
    oriented_kps = []

    for y, x, sigma in keypoints:
        radius = int(radius_factor * sigma)
        if y - radius < 0 or x - radius < 0 or y + radius >= img.shape[0] or x + radius >= img.shape[1]:
            continue

        patch_dx = dx[y - radius:y + radius + 1, x - radius:x + radius + 1]
        patch_dy = dy[y - radius:y + radius + 1, x - radius:x + radius + 1]

        angles = np.arctan2(patch_dy, patch_dx)
        magnitudes = np.hypot(patch_dx, patch_dy)
        hist, bin_edges = np.histogram(angles, bins=num_bins, range=(-np.pi, np.pi), weights=magnitudes)
        dominant_angle = bin_edges[np.argmax(hist)]
        oriented_kps.append((y, x, sigma, dominant_angle))

    return oriented_kps


def haar_wavelet_descriptors(img, keypoints, patch_factor=6, grid=4):
    """Compute SURF-like descriptors with orientation and scale normalization."""
    dx = ndimage.convolve(img, np.array([[-1, 1]]), mode='reflect')
    dy = ndimage.convolve(img, np.array([[-1], [1]]), mode='reflect')
    descriptors = []

    for y, x, sigma, angle in keypoints:
        patch_size = int(patch_factor * sigma)
        half = patch_size // 2
        if y - half < 0 or x - half < 0 or y + half >= img.shape[0] or x + half >= img.shape[1]:
            continue

        # Extract patch
        patch_dx = dx[y - half:y + half, x - half:x + half]
        patch_dy = dy[y - half:y + half, x - half:x + half]

        # Rotate patch according to dominant orientation
        coords_y, coords_x = np.mgrid[-half:half, -half:half]
        rot_x = np.cos(angle) * coords_x + np.sin(angle) * coords_y
        rot_y = -np.sin(angle) * coords_x + np.cos(angle) * coords_y
        rot_dx = ndimage.map_coordinates(patch_dx, [half + rot_y, half + rot_x], order=1, mode='reflect')
        rot_dy = ndimage.map_coordinates(patch_dy, [half + rot_y, half + rot_x], order=1, mode='reflect')

        # Divide into grid (4×4 subregions)
        step = patch_size // grid
        sub_dx = rot_dx[:step * grid, :step * grid].reshape(grid, step, grid, step)
        sub_dy = rot_dy[:step * grid, :step * grid].reshape(grid, step, grid, step)

        # Compute Haar sums per subregion
        sum_dx = sub_dx.sum(axis=(1, 3))
        sum_dy = sub_dy.sum(axis=(1, 3))
        sum_abs_dx = np.abs(sub_dx).sum(axis=(1, 3))
        sum_abs_dy = np.abs(sub_dy).sum(axis=(1, 3))

        desc = np.hstack([sum_dx.flatten(), sum_dy.flatten(),
                          sum_abs_dx.flatten(), sum_abs_dy.flatten()])

        # Normalize
        desc /= (np.linalg.norm(desc) + 1e-6)
        descriptors.append(desc.astype(np.float32))

    return np.array(descriptors, dtype=np.float32)


def surf_features(img, sigmas=(1.5, 3.0, 6.0, 12.0), percentile=98.0):
    """Full SURF-like feature extraction pipeline."""
    keypoints = detect_keypoints(img, sigmas, percentile)
    oriented_kps = assign_orientation(img, keypoints)
    descriptors = haar_wavelet_descriptors(img, oriented_kps)
    return oriented_kps, descriptors
