"""Read a Pi3X result (python -m r2s gpu pi3x): camera, support plane, depth and 3D points of masks.

Pi3X output (tools/pi3x_recon): per sampled frame depth, intrinsics, camera_to_world and world points in its own
world = first sampled camera (OpenCV axes), units approximately metres (a learned estimate). These commands
convert it to video pixels and to the z-up world of a camera file.

  python -m r2s pi3x camera [--dir analysis/pi3x] [--region x0 y0 x1 y1] [--scale S] [--out analysis/camera.json]
        RANSAC plane through the confident Pi3X points of frame 0 inside REGION (video pixels; default the lower
        45 % of the image), z-up world with that plane at z = 0 (origin where the optical axis meets it, else
        below the camera); focal = Pi3X's focal of that sample; positions x S. Prints height, pitch, inliers.
  python -m r2s pi3x camera-track [--camera analysis/camera.json] [--scale S] [--rotation-only] [--name video_cam]
                                  [-o output/camera_track.json]
        pose of that camera on every input frame from Pi3X's camera motion between its samples (frame 0 = the camera
        file's pose; slerp / linear in between): the render-only camera track of a moving camera.
  python -m r2s pi3x objects --sam3 analysis/sam3_ball [--frame 0] [--camera analysis/camera.json] [-o FILE]
        (more than 20 objects, e.g. `gpu members`: saved to analysis/pi3x_objects_<name>.json, summary printed)
        per SAM3 object id on that frame: median and 5-95 % extent of its Pi3X points (visible surface; the
        least confident fifth of the mask's points left out) in the camera file's world, and their median Pi3X
        confidence next to the image's.
  python -m r2s pi3x track --sam3 analysis/sam3_ball [--ids 0 2] [--camera analysis/camera.json]
        the same on every Pi3X sample frame -> analysis/pi3x_track_<sam3 name>.json
        {id: {"frames", "t", "world", "cam" (OpenCV camera frame of that sample), "pixels", "conf"}}
  python -m r2s pi3x plane --region x0 y0 x1 y1 [--camera analysis/camera.json]
        plane through the Pi3X points of a region (ramp, table top, wall): normal, slope vs world up, height of
        the region centre, in the camera file's world.
  python -m r2s pi3x depth --uv 512 400 [--frame 0]      Pi3X depth (its units) at video pixels
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .common import ANALYSIS, OUTPUT, Camera, fovy_from_f, load_json, save_json, timeline


class Pi3X:
    def __init__(self, out_dir=ANALYSIS / "pi3x"):
        self.dir = Path(out_dir)
        if not (self.dir / "manifest.json").exists():
            raise SystemExit(f"Pi3X result {self.dir} not ready: `python -m r2s gpu wait {self.dir}`")
        self.manifest = load_json(self.dir / "manifest.json")
        self.npz = np.load(self.dir / "predictions.npz", mmap_mode="r")
        tl = timeline()
        self.W, self.H = tl["width"], tl["height"]
        fr = self.manifest["frames"]
        self.frames = np.array([f["source_frame_index"] for f in fr], int)
        self.proc_hw = tuple(self.manifest["processed_size_hw"])
        self.sx = self.W / self.proc_hw[1]
        self.sy = self.H / self.proc_hw[0]
        K = np.asarray(self.npz["intrinsics"], float)
        self.fx_video = K[:, 0, 0] * self.sx
        self.fy_video = K[:, 1, 1] * self.sy
        c2w = np.asarray(self.npz["camera_to_world"], float)
        T = np.tile(np.eye(4), (len(c2w), 1, 1))
        T[:, :3, :4] = c2w
        self.cam0_from_cam = np.linalg.inv(T[0]) @ T

    def sample(self, frame):
        i = int(np.argmin(np.abs(self.frames - frame)))
        return i

    def f_video(self, i=0):
        return float(np.sqrt(self.fx_video[i] * self.fy_video[i]))

    def points_cam(self, i):
        d = np.asarray(self.npz["depths"][i], float)
        fx, fy = self.fx_video[i] / self.sx, self.fy_video[i] / self.sy
        K = np.asarray(self.npz["intrinsics"][i], float)
        h, w = d.shape
        u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        return np.stack([(u - K[0, 2]) / fx * d, (v - K[1, 2]) / fy * d, d], -1)

    def conf(self, i):
        return np.asarray(self.npz["depth_confidence"][i], float)

    def region_mask(self, region):
        h, w = self.proc_hw
        x0, y0, x1, y1 = region
        m = np.zeros((h, w), bool)
        m[int(max(0, y0 / self.sy)):int(min(h, y1 / self.sy)), int(max(0, x0 / self.sx)):int(min(w, x1 / self.sx))] = True
        return m

    def fit_plane(self, i=0, region=None, conf_pct=30, thresh_rel=0.01, iters=500, seed=0):
        P = self.points_cam(i)
        c = self.conf(i)
        region = region or (0, self.H * 0.55, self.W, self.H)
        m = self.region_mask(region)
        m &= c >= np.percentile(c[m], conf_pct)
        X = P[m]
        if len(X) < 50:
            raise ValueError("too few confident points in region")
        rng = np.random.default_rng(seed)
        X = X[rng.choice(len(X), min(len(X), 20000), replace=False)]
        th = thresh_rel * np.median(X[:, 2])
        best = None
        for _ in range(iters):
            a, b, cc = X[rng.choice(len(X), 3, replace=False)]
            n = np.cross(b - a, cc - a)
            if np.linalg.norm(n) < 1e-12:
                continue
            n /= np.linalg.norm(n)
            inl = np.abs((X - a) @ n) < th
            if best is None or inl.sum() > best.sum():
                best = inl
        Xi = X[best]
        cen = Xi.mean(0)
        n = np.linalg.svd(Xi - cen)[2][-1]
        d = float(-cen @ n)
        if d < 0:
            n, d = -n, -d                      # normal points towards the camera (= world up)
        return dict(normal=n, d=d, inlier_frac=float(best.mean()), n_points=int(best.sum()))

    def camera(self, region=None, scale=1.0, frame=0):
        i = self.sample(frame)
        pl = self.fit_plane(i, region)
        n, d = pl["normal"], pl["d"]
        z_w = n
        x_w = np.array([1.0, 0, 0]) - n[0] * n          # camera x projected onto the plane
        x_w /= np.linalg.norm(x_w)
        y_w = np.cross(z_w, x_w)
        R = np.stack([x_w, y_w, z_w], 1)                 # columns: world axes in cv coords
        R_world_from_cv = R.T
        foot = -d * n                                    # cv coords of the plane point below the camera
        ray = np.array([0, 0, 1.0])
        den = ray @ n
        hit = (-d / den) * ray if den < -1e-6 else None  # optical axis hits the plane in front
        origin = hit if hit is not None and np.linalg.norm(hit) < 20 * d else foot
        pos = R_world_from_cv @ (np.zeros(3) - origin)
        f = self.f_video(i)
        cam = Camera(fovy_from_f(f, self.H), self.W, self.H, R_world_from_cv, pos * scale)
        info = dict(source="pi3x", pi3x_dir=str(self.dir), frame=int(self.frames[i]), plane_region=list(region or
                    (0, self.H * 0.55, self.W, self.H)), plane_inlier_frac=pl["inlier_frac"], scale_applied=scale,
                    camera_height_m=float(pos[2] * scale), focal_px=f,
                    optical_axis_pitch_deg=float(np.degrees(np.arcsin(-R_world_from_cv[2, 2]))),
                    note="units = Pi3X units (learned, approximately metres) x scale_applied")
        return cam, info

    def depth_at(self, i, uv, r=2):
        d = np.asarray(self.npz["depths"][i], float)
        out = []
        for u, v in np.atleast_2d(uv):
            x, y = int(u / self.sx), int(v / self.sy)
            win = d[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
            out.append(float(np.median(win)) if win.size else np.nan)
        return np.array(out)


def camera_track(pi3x_dir, camera=ANALYSIS / "camera.json", scale=None, rotation_only=False, name="video_cam"):
    """Pose of the camera file's camera on every input frame from Pi3X's per-sample camera motion (frame 0 = the
    camera file's pose); rotation slerped, position linear between samples, held outside the sampled range."""
    from scipy.spatial.transform import Rotation, Slerp
    from .common import FLIP, quat_from_R
    p = Pi3X(pi3x_dir)
    info = load_json(camera)
    cam = Camera.from_dict(info)
    s = info.get("scale_applied", 1.0) if scale is None else scale
    order = np.argsort(p.frames)
    fr, C = p.frames[order].astype(float), p.cam0_from_cam[order]
    R = np.einsum("ij,njk->nik", cam.R, C[:, :3, :3])
    P = cam.pos + (np.zeros((len(C), 3)) if rotation_only else (C[:, :3, 3] * s) @ cam.R.T)
    tl = timeline()
    k = np.clip(np.arange(tl["n"], dtype=float), fr[0], fr[-1])
    Rk = Slerp(fr, Rotation.from_matrix(R))(k).as_matrix() if len(fr) > 1 else np.repeat(R, len(k), 0)
    Pk = np.stack([np.interp(k, fr, P[:, j]) for j in range(3)], 1)
    quat = np.array([quat_from_R(r @ FLIP) for r in Rk])
    rot = np.degrees(Rotation.from_matrix(np.einsum("ji,njk->nik", Rk[0], Rk)).magnitude())
    track = dict(camera=name, t=tl["t"], pos=Pk.round(7).tolist(), quat=quat.round(8).tolist(),
                 source=f"Pi3X {pi3x_dir} ({len(fr)} samples) x camera file {camera}, scale {s}"
                        + (", rotation only" if rotation_only else ""),
                 note="render-only pose of the preview camera per input frame (MuJoCo camera frame in the world)")
    stats = dict(samples=len(fr), max_rotation_deg=float(rot.max()), max_rotation_frame=int(rot.argmax()),
                 max_translation=float(np.linalg.norm(Pk - Pk[0], axis=1).max()), scale=s)
    return track, stats


def summarize(out_dir):
    p = Pi3X(out_dir)
    from scipy.spatial.transform import Rotation
    rot = [float(np.degrees(Rotation.from_matrix(t[:3, :3]).magnitude())) for t in p.cam0_from_cam]
    trans = [float(np.linalg.norm(t[:3, 3])) for t in p.cam0_from_cam]
    s = dict(samples=len(p.frames), first_frame=int(p.frames[0]), last_frame=int(p.frames[-1]),
             focal_px_video=round(p.f_video(0), 1), fovy_deg=round(fovy_from_f(p.f_video(0), p.H), 2),
             focal_spread=round(float(np.std(p.fx_video) / np.mean(p.fx_video)), 4),
             max_camera_rotation_deg=round(max(rot), 2), max_camera_translation_units=round(max(trans), 3),
             median_depth_frame0=round(float(np.median(p.npz["depths"][0])), 3))
    s["camera_static"] = s["max_camera_rotation_deg"] < 1.0 and s["max_camera_translation_units"] < 0.03 * s["median_depth_frame0"]
    save_json(s, Path(out_dir) / "summary.json")
    return s


def _mask_points(p, i, masks, ids, P, c, cam):
    import cv2
    res = {}
    for oid, m in zip(ids, masks):
        mm = cv2.resize(m.astype(np.uint8), (p.proc_hw[1], p.proc_hw[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        mm = cv2.erode(mm.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        if mm.sum() < 5:
            res[int(oid)] = None
            continue
        mm &= c >= np.percentile(c[mm], 20)                # drop the least confident fifth of the mask's points
        X = P[mm]
        Xw = X @ cam.R.T + cam.pos if cam is not None else X
        res[int(oid)] = dict(pixels=int(mm.sum()), cam=np.median(X, 0), world=np.median(Xw, 0),
                             extent=np.percentile(Xw, 95, 0) - np.percentile(Xw, 5, 0), conf=float(np.median(c[mm])))
    return res


def _camera_and_scale(camera):
    if not Path(camera).exists():
        return None, 1.0
    info = load_json(camera)
    return Camera.from_dict(info), info.get("scale_applied", 1.0)


def objects_from_sam3(pi3x_dir, sam3_dir, frame=0, camera=ANALYSIS / "camera.json"):
    p = Pi3X(pi3x_dir)
    i = p.sample(frame)
    if p.frames[i] != frame:
        print(f"note: nearest Pi3X sample to frame {frame} is frame {p.frames[i]}")
    fr = int(p.frames[i])
    f = Path(sam3_dir) / "frames" / f"frame_{fr:06d}.npz"
    if not f.exists():
        raise SystemExit(f"{sam3_dir}: no SAM3 mask for frame {fr} (not finished? `python -m r2s gpu wait {sam3_dir}`)")
    z = np.load(f)
    cam, s = _camera_and_scale(camera)
    pts = _mask_points(p, i, z["masks"], z["object_ids"], p.points_cam(i) * s, p.conf(i), cam)
    res = []
    for oid, r in pts.items():
        if r is None:
            res.append(dict(object_id=oid, error="mask too small for Pi3X"))
            continue
        res.append(dict(object_id=oid, pixels=r["pixels"], center=r["world"].round(4).tolist(),
                        extent_p5_p95=r["extent"].round(4).tolist(), conf_median=round(r["conf"], 3), frame=fr,
                        image_conf_median=round(float(np.median(p.conf(i))), 3),
                        frame_note="median of the visible-surface points"))
    return res


def track_from_sam3(pi3x_dir, sam3_dir, ids=None, camera=ANALYSIS / "camera.json"):
    """3D path of each SAM3 object over the Pi3X samples (visible-surface median)."""
    p = Pi3X(pi3x_dir)
    cam, s = _camera_and_scale(camera)
    tl = timeline()
    out = {}
    for i, fr in enumerate(p.frames):
        f = Path(sam3_dir) / "frames" / f"frame_{int(fr):06d}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        keep = [j for j, o in enumerate(z["object_ids"]) if ids is None or int(o) in ids]
        if not keep:
            continue
        pts = _mask_points(p, i, z["masks"][keep], z["object_ids"][keep], p.points_cam(i) * s, p.conf(i), cam)
        for oid, r in pts.items():
            if r is None:
                continue
            o = out.setdefault(str(oid), dict(frames=[], t=[], world=[], cam=[], pixels=[], conf=[]))
            o["frames"].append(int(fr)); o["t"].append(float(tl["t"][int(fr)]))
            o["world"].append(r["world"].round(5).tolist()); o["cam"].append(r["cam"].round(5).tolist())
            o["pixels"].append(r["pixels"]); o["conf"].append(round(r["conf"], 3))
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="r2s pi3x", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("camera")
    c.add_argument("--dir", default=str(ANALYSIS / "pi3x"))
    c.add_argument("--region", type=float, nargs=4)
    c.add_argument("--scale", type=float, default=1.0)
    c.add_argument("--out", default=str(ANALYSIS / "camera.json"))
    ct = sub.add_parser("camera-track")
    ct.add_argument("--dir", default=str(ANALYSIS / "pi3x"))
    ct.add_argument("--camera", default=str(ANALYSIS / "camera.json"))
    ct.add_argument("--scale", type=float, help="Pi3X units -> world (default: the camera file's scale_applied, else 1)")
    ct.add_argument("--rotation-only", action="store_true", help="keep the camera centre fixed")
    ct.add_argument("--name", default="video_cam", help="the scene camera it moves")
    ct.add_argument("-o", "--out", default=str(OUTPUT / "camera_track.json"))
    o = sub.add_parser("objects")
    o.add_argument("--dir", default=str(ANALYSIS / "pi3x"))
    o.add_argument("--sam3", required=True)
    o.add_argument("--frame", type=int, default=0)
    o.add_argument("--camera", default=str(ANALYSIS / "camera.json"))
    o.add_argument("-o", "--out", help="save the list (default with > 20 objects: analysis/pi3x_objects_<name>.json)")
    tr = sub.add_parser("track")
    tr.add_argument("--dir", default=str(ANALYSIS / "pi3x"))
    tr.add_argument("--sam3", required=True)
    tr.add_argument("--ids", type=int, nargs="+")
    tr.add_argument("--camera", default=str(ANALYSIS / "camera.json"))
    tr.add_argument("-o", "--out")
    pl = sub.add_parser("plane")
    pl.add_argument("--dir", default=str(ANALYSIS / "pi3x"))
    pl.add_argument("--region", type=float, nargs=4, required=True)
    pl.add_argument("--camera", default=str(ANALYSIS / "camera.json"))
    d = sub.add_parser("depth")
    d.add_argument("--dir", default=str(ANALYSIS / "pi3x"))
    d.add_argument("--uv", type=float, nargs=2, required=True)
    d.add_argument("--frame", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd in ("plane", "camera-track") and not Path(a.camera).exists():
        raise SystemExit(f"no {a.camera}: `python -m r2s pi3x camera` writes one")
    if a.cmd in ("objects", "track") and not Path(a.camera).exists():
        print(f"note: no {a.camera}: positions in Pi3X camera coordinates of each sample")
    if a.cmd == "camera":
        p = Pi3X(a.dir)
        cam, info = p.camera(a.region, a.scale)
        cam.save(a.out, **info)
        print(json.dumps(info, indent=1))
        print(f"fovy={cam.fovy:.2f} deg, camera pos (world) {np.round(cam.pos, 3).tolist()} -> {a.out}")
        print(cam.mjcf())
    elif a.cmd == "camera-track":
        track, st = camera_track(a.dir, Path(a.camera), a.scale, a.rotation_only, a.name)
        save_json(track, a.out)
        print(json.dumps(st, indent=1))
        print(f"{len(track['pos'])} poses of camera {a.name!r} -> {a.out} (frame 0 = {a.camera}; rebuild after changing "
              "that camera)")
    elif a.cmd == "objects":
        res = objects_from_sam3(a.dir, a.sam3, a.frame, Path(a.camera))
        out = a.out or (str(ANALYSIS / f"pi3x_objects_{Path(a.sam3).name.replace('sam3_', '')}.json")
                        if len(res) > 20 else None)
        if out is None:
            print(json.dumps(res, indent=1))
        else:
            save_json(res, out)
            ok = [r for r in res if "center" in r and r["object_id"] != 0]
            if len(res) <= 20:
                print(json.dumps(res, indent=1))
            elif ok:
                c, e = np.array([r["center"] for r in ok]), np.array([r["extent_p5_p95"] for r in ok])
                print(f"{len(ok)} objects with Pi3X points (of {len(res)}; id 0 = a union if listed) on frame "
                      f"{ok[0]['frame']}: centers x {c[:, 0].min():.3f}..{c[:, 0].max():.3f} y {c[:, 1].min():.3f}.."
                      f"{c[:, 1].max():.3f} z {c[:, 2].min():.3f}..{c[:, 2].max():.3f}; median visible extent "
                      f"{np.median(e, 0).round(4).tolist()}; too small for Pi3X: "
                      f"{[r['object_id'] for r in res if 'error' in r] or 'none'}")
            print("->", out)
    elif a.cmd == "track":
        res = track_from_sam3(a.dir, a.sam3, a.ids, Path(a.camera))
        out = a.out or str(ANALYSIS / f"pi3x_track_{Path(a.sam3).name.replace('sam3_', '')}.json")
        save_json(res, out)
        for oid, o in res.items():
            w = np.array(o["world"])
            print(f"id {oid}: {len(o['frames'])} samples, frames {o['frames'][0]}-{o['frames'][-1]}, first {w[0].round(3).tolist()} "
                  f"last {w[-1].round(3).tolist()}")
        print("->", out)
    elif a.cmd == "plane":
        p = Pi3X(a.dir)
        info = load_json(a.camera)
        cam = Camera.from_dict(info)
        s = info.get("scale_applied", 1.0)
        pl = p.fit_plane(0, a.region)
        n_w = cam.R @ pl["normal"]
        if n_w[2] < 0:
            n_w = -n_w
        P = p.points_cam(0)[p.region_mask(a.region)] * s
        c_w = np.median(P, 0) @ cam.R.T + cam.pos
        slope = float(np.degrees(np.arccos(np.clip(n_w[2], -1, 1))))
        downhill = n_w[:2] / (np.linalg.norm(n_w[:2]) + 1e-12)
        print(json.dumps(dict(normal_world=n_w.round(4).tolist(), slope_deg=round(slope, 2),
                              downhill_dir_xy=downhill.round(3).tolist(), region_center_world=c_w.round(4).tolist(),
                              inlier_frac=round(pl["inlier_frac"], 3)), indent=1))
    else:
        p = Pi3X(a.dir)
        print(p.depth_at(p.sample(a.frame), [a.uv]))
