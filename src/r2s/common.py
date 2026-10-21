"""Shared helpers: paths, JSON I/O (numpy-safe), the video timeline, and camera math.

Conventions of the r2s tools
- World frame of a camera file written by `pi3x camera`: z UP, the fitted support plane is z = 0. Units: the
  Pi3X units (approximately metres, a learned estimate) unless you rescale.
- OpenCV camera frame: x right, y down, z forward. MuJoCo camera frame: x right, y up, looks along -z.
  R_world_from_mjcam = R_world_from_cv @ diag(1, -1, -1).
- Camera files: pinhole, square pixels, principal point at the image centre, MuJoCo `fovy` (vertical field of
  view, degrees). Camera.mjcf() gives the matching <camera> element.
- Frame k of the video is at time timeline["t"][k] (decoded PTS, seconds, relative to the first frame).
- Moving camera: output/camera_track.json {"camera", "t", "pos" (n,3), "quat" (n,4; w x y z of the MuJoCo camera
  frame in the world)}, one pose per input frame; render-only (set_camera_pose before each frame is drawn).
- Analysis files go to analysis/ ; deliverables to output/.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

ANALYSIS = Path("analysis")
OUTPUT = Path("output")
VIDEO = Path("input/video.mp4")


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def _plain_keys(o):
    """numpy scalars as dict keys (e.g. {np.int64(3): ...}) are not JSON keys: make them Python numbers."""
    if isinstance(o, dict):
        return {(k.item() if isinstance(k, np.generic) else k): _plain_keys(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain_keys(v) for v in o]
    return o


def save_json(obj, path, indent=1):
    """save_json(obj, path); the order save_json(path, obj) is accepted too."""
    if isinstance(obj, (str, Path)) and not isinstance(path, (str, Path)):
        obj, path = path, obj
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import os
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")     # unique: concurrent writers do not collide
    tmp.write_text(json.dumps(_plain_keys(obj), cls=NumpyEncoder, indent=indent, ensure_ascii=False))
    tmp.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text())


def timeline(path=ANALYSIS / "timeline.json") -> dict:
    """analysis/timeline.json (summary, small) + analysis/frame_times.json (PTS of every frame) written by
    `r2s probe` / `r2s look`. Returns one dict with key "t" (list of seconds)."""
    p = Path(path)
    if not p.exists():
        from .video import probe
        return probe()
    tl = load_json(p)
    if "t" not in tl:
        tl["t"] = load_json(p.with_name("frame_times.json"))
    return tl


# ---------------------------------------------------------------- camera math
def K_from_fovy(fovy_deg, width, height):
    f = 0.5 * height / math.tan(math.radians(fovy_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])


def fovy_from_f(f_px, height):
    return math.degrees(2 * math.atan(0.5 * height / f_px))


FLIP = np.diag([1.0, -1.0, -1.0])


class Camera:
    """Pinhole camera in the z-up world. R = R_world_from_cv (3x3), pos = camera centre (world, m)."""

    def __init__(self, fovy, width, height, R, pos):
        self.fovy = float(fovy)
        self.width, self.height = int(width), int(height)
        self.R = np.asarray(R, float)
        self.pos = np.asarray(pos, float)

    @property
    def K(self):
        return K_from_fovy(self.fovy, self.width, self.height)

    @property
    def f(self):
        """Focal length in pixels (settable: changes fovy)."""
        return self.K[0, 0]

    @f.setter
    def f(self, f_px):
        self.fovy = fovy_from_f(float(f_px), self.height)

    # -- serialisation (analysis/camera.json)
    def to_dict(self, **extra):
        R_mj = self.R @ FLIP
        return dict(fovy=self.fovy, width=self.width, height=self.height, pos=self.pos.tolist(),
                    R_world_from_cv=self.R.tolist(),
                    mjcf_xyaxes=np.concatenate([R_mj[:, 0], R_mj[:, 1]]).tolist(), **extra)

    @classmethod
    def from_dict(cls, d):
        R = d.get("R_world_from_cv", d.get("R")) if isinstance(d, dict) else None
        if R is None or any(k not in d for k in ("fovy", "width", "height", "pos")):
            raise ValueError("not a camera file (needs fovy, width, height, R_world_from_cv, pos; written by "
                             "`pi3x camera` or Camera.save())")
        return cls(d["fovy"], d["width"], d["height"], R, d["pos"])

    @classmethod
    def load(cls, path=ANALYSIS / "camera.json"):
        if not Path(path).exists():
            raise SystemExit(f"no {path} (written by `python -m r2s pi3x camera` or Camera.save())")
        try:
            return cls.from_dict(load_json(path))
        except ValueError as e:
            raise SystemExit(f"{path}: {e}")

    def save(self, path=ANALYSIS / "camera.json", **extra):
        save_json(self.to_dict(**extra), path)

    def mjcf(self, name="video_cam"):
        """The MJCF <camera> element (worldbody child) that renders this camera's image."""
        ax = np.round(self.to_dict()["mjcf_xyaxes"], 6)
        return (f'<camera name="{name}" mode="fixed" pos="{" ".join(f"{x:.6g}" for x in self.pos)}" '
                f'xyaxes="{" ".join(f"{x:.6g}" for x in ax)}" fovy="{self.fovy:.6g}"/>')

    @classmethod
    def from_mujoco(cls, model, data, name, width, height):
        """Camera of a compiled MuJoCo model (after mj_forward) as an r2s Camera (principal point at the centre)."""
        import mujoco
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if i < 0:
            raise ValueError(f"no camera {name!r} in the model")
        R_mj = data.cam_xmat[i].reshape(3, 3)
        return cls(float(model.cam_fovy[i]), width, height, R_mj @ FLIP, data.cam_xpos[i].copy())

    @classmethod
    def look_at(cls, fovy, width, height, pos, target, up=(0, 0, 1)):
        pos, target, up = (np.asarray(x, float) for x in (pos, target, up))
        z = target - pos
        z /= np.linalg.norm(z)
        x = np.cross(z, up)
        if np.linalg.norm(x) < 1e-9:
            x = np.cross(z, [0, 1, 0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)                     # OpenCV y points down in the image
        return cls(fovy, width, height, np.stack([x, y, z], 1), pos)

    # -- geometry
    def world_to_cv(self, X):
        return (np.asarray(X, float) - self.pos) @ self.R

    def project(self, X):
        """World points (...,3) -> pixels (...,2); points behind the camera give nan."""
        Xc = self.world_to_cv(X)
        z = Xc[..., 2:3]
        uv = Xc[..., :2] / np.where(z > 1e-9, z, np.nan) * self.f + np.array([self.width / 2, self.height / 2])
        return uv

    def depth(self, X):
        return self.world_to_cv(X)[..., 2]

    def ray(self, uv):
        """Unit ray directions (world) through pixels uv (...,2)."""
        uv = np.asarray(uv, float)
        d = np.stack([(uv[..., 0] - self.width / 2) / self.f, (uv[..., 1] - self.height / 2) / self.f,
                      np.ones(uv.shape[:-1])], -1)
        d = d @ self.R.T
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def lift_to_height(self, uv, z=0.0):
        """Intersect pixel rays with the horizontal plane world-z = z (e.g. z = radius for a ball centre)."""
        d = self.ray(uv)
        s = (z - self.pos[2]) / d[..., 2]
        return self.pos + s[..., None] * d

    lift = lift_to_height                      # same function, shorter name

    def scaled(self, s):
        """Same image, world scaled by s about the world origin (camera moves, angles unchanged)."""
        return Camera(self.fovy, self.width, self.height, self.R, self.pos * s)


def load_camera_track(path, n=None):
    """camera_track.json (one pose of the preview camera per input frame) -> (camera name, pos (n,3), quat (n,4))."""
    ct = load_json(path)
    pos, quat = np.asarray(ct["pos"], float), np.asarray(ct["quat"], float)
    if pos.ndim != 2 or pos.shape[1] != 3 or quat.shape != (len(pos), 4) or (n is not None and len(pos) != n):
        raise ValueError(f"{path}: needs pos (n,3) and quat (n,4) for the {n} input frames")
    return ct.get("camera"), pos, quat


def set_camera_pose(data, cam_id, pos, quat):
    """Render-only: the pose update_scene reads for a fixed world camera (mj_step recomputes it, physics never reads it)."""
    import mujoco
    data.cam_xpos[cam_id] = pos
    mujoco.mju_quat2Mat(data.cam_xmat[cam_id], np.asarray(quat, float) / np.linalg.norm(quat))


def rot_z(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])


def quat_from_R(R):
    """Rotation matrix -> MuJoCo quaternion (w, x, y, z)."""
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1))
    return q


def euler_deg_to_quat(rx, ry, rz):
    """Extrinsic XYZ Euler angles in degrees -> (w,x,y,z)."""
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_euler("xyz", [rx, ry, rz], degrees=True).as_quat()
    return np.array([w, x, y, z])
