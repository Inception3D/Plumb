"""Adapt native Pi3X predictions to the reconstruction geometry contract."""

import math

PI3X_INITIALIZATION_MODE = "sam3-sam3d-pi3x-physical-initialization-v1"
PI3X_PIXEL_LIMIT = 255000


def pi3x_model_size(source_size, pixel_limit):
    """Return the official Pi3X raster [H, W]: area-limited, multiples of 14."""
    h, w = source_size
    scale = math.sqrt(pixel_limit / (w * h))
    tw, th = w * scale, h * scale
    k, m = round(tw / 14), round(th / 14)
    while k * m * 196 > pixel_limit:
        if k / m > tw / th:
            k -= 1
        else:
            m -= 1
    return [max(1, m) * 14, max(1, k) * 14]


def pi3x_preprocess(source_size, model_size, pixel_limit):
    """Describe the official full-frame resize; nothing is cropped or padded.

    The resize aligns pixel centers, so model x = sx * source x + (sx - 1) / 2.
    crop_xywh carries that half-pixel offset in the shared x_model = sx * x - crop
    convention used by camera and mask mapping.
    """
    h, w = source_size
    mh, mw = model_size
    if [mh, mw] != pi3x_model_size(source_size, pixel_limit):
        raise ValueError("Pi3X raster does not match its resize preprocessing")
    sx, sy = mw / w, mh / h
    return {"method": "pi3x-full-frame-resize-v1", "source_size": [h, w],
            "model_size": [mh, mw], "resized_size": [mh, mw], "scale_xy": [sx, sy],
            "crop_xywh": [(1 - sx) / 2, (1 - sy) / 2, mw, mh]}


def square_pixel_intrinsics(rays, source_size, np):
    """Per-frame model-raster intrinsics with square source pixels.

    ``rays`` are Pi3X camera-frame ray directions [N, H, W, 3] on the model
    raster, a full-frame resize of the source. Decoded video frames have square
    pixels, so each frame gets one source focal length f, fitted by least
    squares to all rays with a centred principal point; on the model raster it
    becomes fx = f * sx and fy = f * sy. Fitting fx and fy independently lets
    learned ray errors make pixels anisotropic, which stretches the scene.
    """
    n, h, w, _ = rays.shape
    sx, sy = w / source_size[1], h / source_size[0]
    z = np.abs(rays[..., 2]) + 1e-10
    x, y = rays[..., 0] / z, rays[..., 1] / z
    u = (np.arange(w) - (w - 1) / 2.0)[None, None, :]
    v = (np.arange(h) - (h - 1) / 2.0)[None, :, None]
    focal = (((sx * x * u).sum((1, 2)) + (sy * y * v).sum((1, 2)))
             / (((sx * x) ** 2).sum((1, 2)) + ((sy * y) ** 2).sum((1, 2))))
    intrinsics = np.zeros((n, 3, 3))
    intrinsics[:, 0, 0], intrinsics[:, 1, 1] = focal * sx, focal * sy
    intrinsics[:, 0, 2], intrinsics[:, 1, 2], intrinsics[:, 2, 2] = (w - 1) / 2.0, (h - 1) / 2.0, 1.0
    return intrinsics


def load_depth(manifest, np):
    with np.load(manifest["predictions_path"], allow_pickle=False) as arrays:
        depth = np.asarray(arrays["depths"]).copy()
        if depth.ndim == 4 and depth.shape[-1] == 1:
            depth = depth[..., 0]
    return depth


def load_camera_depth(manifest, np):
    depth = load_depth(manifest, np)
    with np.load(manifest["predictions_path"], allow_pickle=False) as arrays:
        intrinsics = np.asarray(arrays["intrinsics"])
        poses = np.asarray(arrays["camera_to_world"])
        if poses.shape != (len(depth), 3, 4):
            raise ValueError("Pi3X camera_to_world must have shape [N,3,4]")
        c2w = np.broadcast_to(np.eye(4), (len(depth), 4, 4)).copy()
        c2w[:, :3] = poses
        extrinsics = np.linalg.inv(c2w)[:, :3]
    return depth, intrinsics, extrinsics


def source_depths(manifest, np):
    """Restore Pi3X depth to source pixels with the inverse full-frame resize."""
    depth = load_depth(manifest, np)
    from pathlib import Path
    from PIL import Image

    result = []
    for row, values in zip(manifest["frames"], depth, strict=True):
        source_size = row.get("source_size_hw")
        if source_size is None:
            with Image.open(Path(row["path"])) as image:
                source_size = [image.height, image.width]
        pi3x_preprocess(source_size, list(values.shape), manifest["pixel_limit"])
        h, w = source_size
        image = Image.fromarray(values.astype(np.float32))
        result.append(np.asarray(image.resize((w, h), Image.Resampling.BILINEAR)))
    return result


# Windowed inference for videos too long for one forward pass. Each window is a
# separate Pi3X world; later windows are mapped into the first window's world
# by the similarity that best aligns their shared (overlap) frames' points.
PI3X_WINDOW_FRAMES = 96
PI3X_WINDOW_OVERLAP = 8
PI3X_ALIGNMENT_CONFIDENCE = .05
PI3X_ALIGNMENT_POINTS = 200_000


def pi3x_windows(frame_count, window, overlap):
    """[start, end) ranges covering every frame; consecutive windows share ``overlap`` frames."""
    if not 1 <= overlap < window:
        raise ValueError("Pi3X window overlap must be positive and smaller than the window")
    if frame_count <= window:
        return [(0, frame_count)]
    ranges, start = [], 0
    while True:
        end = min(start + window, frame_count)
        ranges.append((start, end))
        if end == frame_count:
            return ranges
        start = end - overlap


def umeyama_sim3(source, target, weights, np):
    """Scale, rotation and translation minimizing sum w |s R x + t - y|^2 (rotation proper)."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    x, y = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    mx, my = w @ x, w @ y
    dx, dy = x - mx, y - my
    covariance = (dy * w[:, None]).T @ dx
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1
    rotation = u @ sign @ vt
    variance = float(w @ np.sum(dx * dx, axis=1))
    scale = float(np.trace(np.diag(singular) @ sign) / variance)
    translation = my - scale * rotation @ mx
    return scale, rotation, translation


def merge_pi3x_windows(windows, overlap, np, *, seed=0):
    """Map per-window predictions into the first window's world.

    ``windows`` is a list of (start, arrays) where arrays hold that window's
    depths, world_points, camera_to_world (3x4), intrinsics and depth_confidence
    in its own first-camera world. Returns merged arrays and alignment records.
    """
    start0, first = windows[0]
    if start0 != 0:
        raise ValueError("The first Pi3X window must start at frame 0")
    merged = {key: [value] for key, value in first.items()}
    end = len(first["depths"])
    records = [{"range": [0, end], "scale": 1., "rotation_deg": 0., "translation": [0., 0., 0.]}]
    generator = np.random.default_rng(seed)
    for start, arrays in windows[1:]:
        if start != end - overlap:
            raise ValueError("Pi3X windows must overlap their predecessor by the declared frames")
        previous_points = np.concatenate(merged["world_points"])[start:end]
        previous_confidence = np.concatenate(merged["depth_confidence"])[start:end]
        shared = (previous_confidence > PI3X_ALIGNMENT_CONFIDENCE) & (arrays["depth_confidence"][:overlap] > PI3X_ALIGNMENT_CONFIDENCE)
        index = np.flatnonzero(shared.reshape(-1))
        if len(index) < 100:
            raise ValueError("Pi3X window overlap has too few confident points to align")
        if len(index) > PI3X_ALIGNMENT_POINTS:
            index = generator.choice(index, PI3X_ALIGNMENT_POINTS, replace=False)
        source = arrays["world_points"][:overlap].reshape(-1, 3)[index]
        target = previous_points.reshape(-1, 3)[index]
        weight = np.sqrt(arrays["depth_confidence"][:overlap].reshape(-1)[index]
                         * previous_confidence.reshape(-1)[index])
        scale, rotation, translation = umeyama_sim3(source, target, weight, np)
        if not (math.isfinite(scale) and scale > 0):
            raise ValueError("Pi3X window alignment produced a non-positive scale")
        residual = np.linalg.norm(scale * source @ rotation.T + translation - target, axis=1)
        new = slice(overlap, None)
        poses = arrays["camera_to_world"][new]
        aligned_poses = np.concatenate([rotation @ poses[:, :, :3],
                                        (scale * poses[:, :, 3] @ rotation.T + translation)[..., None]], axis=2)
        merged["camera_to_world"].append(aligned_poses.astype(first["camera_to_world"].dtype))
        merged["world_points"].append((scale * arrays["world_points"][new] @ rotation.T + translation)
                                      .astype(first["world_points"].dtype))
        merged["depths"].append((scale * arrays["depths"][new]).astype(first["depths"].dtype))
        merged["intrinsics"].append(arrays["intrinsics"][new])
        merged["depth_confidence"].append(arrays["depth_confidence"][new])
        end = start + len(arrays["depths"])
        angle = math.degrees(math.acos(max(-1., min(1., (np.trace(rotation) - 1) / 2))))
        records.append({"range": [start, end], "scale": scale, "rotation_deg": angle,
                        "translation": translation.tolist(), "alignment_points": int(len(index)),
                        "median_residual": float(np.median(residual)),
                        "median_overlap_depth": float(np.median(np.linalg.norm(target, axis=1)))})
    return {key: np.concatenate(values) for key, values in merged.items()}, records
