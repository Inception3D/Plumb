"""Visible-motion evaluation v2 (writes BENCH/evaluation/v2/; GT cache BENCH/cache/v2/).

Why v2: in v1, reconstructions that render almost identically could differ ~10x in ADE3D because of how the
agent organised its MJCF, not because of the motion. v2 changes, all on both GT and prediction alike:
1. Visibility: MuJoCo default groups + alpha 0 hidden (v1 scraped geomgroup lines from run.py; 038 hid the moving box).
2. Correspondence on rigid prediction parts (each body, each world geom) instead of root-body groups: one-to-one
   Hungarian on mean IoU over each GT object's first visible frames (identity from the initial layout, not from
   later motion errors) with a floor (MIN_MATCH_IOU), then leftover parts join the GT object holding at
   least MERGE_SHARE of their pixels. A GT object is scored with the union of its parts (v1: nested bodies were one
   group, zero-overlap pairs were allowed, so a pendulum or a rotor on a base was matched to the wrong thing).
3. 3D point: visible-surface centroid from mask + depth, for GT (dataset depth maps, z-depth in metres) and for
   the prediction (rendered depth). v1 compared a PhysInOne actor origin (often the base, 0.2-0.7 m away) or a
   Physion bbox centre with the centre of the prediction's "largest" geom (by size product: a mesh vane, a tube
   wall segment or an invisible reference box).
4. Scale: one positive scale = median GT/pred z-depth ratio of matched objects at their first frame with
   IoU >= SCALE_MIN_IOU (v1: frame 0 of every matched pair, including wrong pairs).
5. Moving objects: translation > MOVE_M or rotation > MOVE_DEG (v1: 5 mm / 1 deg counted a 7 mm nudge).
6. Camera frames: GT and prediction 3D points are compared in the camera frame whose optical axis is the ray
   through the image centre. A principal-point offset plus a compensating camera rotation renders the same image
   but rotates the whole camera-frame scene (040 parent: 205 px, ~15 deg, ADE3D 0.96 -> 0.08 m).
Unchanged: fixed identities for the whole video, no time warping, no per-frame registration, the reconstruction's
focal length is used as delivered: a wrong focal length distorts the 3D scene (lateral vs depth) while the 2D image
can still match, so it stays a real 3D error (017 v7 rep1: fovy 55.8 vs GT 66.8 deg, ADE3D 0.140; 0.040 with GT focal).

The prediction is rendered by `python -m evaluation.render` (trusted passive rollout of output/scene.xml +
initial_state.json, no agent code) in a subprocess with the evaluating Python (sys.executable).
"""
import csv
import io
import math
import os
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from .common import HERE, NO_OUTPUT, egl_env, read_json, save_json, utc_now
from .gt import MaskTruth, motion_class, read_mask_poses, source_rows  # noqa: F401 (motion_class re-exported)

MOVE_M, MOVE_DEG = 0.02, 5.0
MIN_MATCH_IOU = 0.05
EARLY_MIN, EARLY_FRAC = 10, 0.25
MERGE_SHARE = 0.5
SCALE_MIN_IOU = 0.3
GT_CACHE = Path("cache/v2")          # relative to the benchmark directory
OUT = Path("evaluation/v2")
VERSION = "visible_motion_v2"


# ---------------------------------------------------------------- GT side
class GTDepth:
    def __init__(self, task):
        self.task = task
        if task["dataset"] == "physion_v15":
            self.handle = h5py.File(task["source"]["path"], "r")
        else:
            self.handle = zipfile.ZipFile(task["source"]["path"])
            cam = task["source"]["camera"]
            self.members = {int(Path(n).stem): n for n in self.handle.namelist()
                            if f"/{cam}/depth/" in n and n.endswith(".npz")}

    def frame(self, source_frame):
        if self.task["dataset"] == "physion_v15":
            return self.handle["frames"][source_frame]["images"]["_depth_cam0"][()]
        with np.load(io.BytesIO(self.handle.read(self.members[int(source_frame)]))) as p:
            return p["depth"]

    def close(self):
        self.handle.close()


def backproject_grid(P, width, height):
    u = np.arange(width) + .5; v = np.arange(height) + .5
    ax = (2 * u / width - 1 + P[0, 2]) / P[0, 0]; ay = (1 - 2 * v / height + P[1, 2]) / P[1, 1]
    return np.broadcast_to(ax, (height, width)).ravel(), np.broadcast_to(ay[:, None], (height, width)).ravel()


def label_stats(labels, depth, n, ax, ay):
    """area, sum u, sum v, sum X, sum Y, sum Z for labels 1..n (OpenGL camera; z-depth d -> Z = -d)."""
    h, w = labels.shape
    flat = labels.ravel().astype(np.int64)
    d = depth.ravel().astype(np.float64)
    idx = np.arange(flat.size)
    out = np.zeros((n, 6))
    for k, wt in enumerate((None, idx % w, idx // w, ax * d, ay * d, -d)):
        out[:, k] = np.bincount(flat, weights=wt, minlength=n + 1)[1:]
    return out


def gt_poses(task, metadata, gt_objects):
    return read_mask_poses(task, metadata, gt_objects, MOVE_M, MOVE_DEG)


# ---------------------------------------------------------------- scoring helpers
def centroids(st):
    """st (..., 6) -> 2D (..., 2) and 3D (..., 3), NaN where area == 0."""
    a = st[..., 0:1]
    with np.errstate(invalid="ignore", divide="ignore"):
        c2 = np.where(a > 0, st[..., 1:3] / a, np.nan)
        c3 = np.where(a > 0, st[..., 3:6] / a, np.nan)
    return c2, c3


def centred_frame(P):
    """Rotation (rows = new axes) into the camera frame whose optical axis is the ray through the image centre and
    whose x follows the image rows. A principal-point offset (MuJoCo principalpixel) plus a camera rotation that
    compensates it renders the same image as a centred camera, but rotates the whole camera-frame scene
    (040 parent: offset 205 px, ~15 deg, ADE3D 0.96 m; 0.08 m once both frames are centred)."""
    z = -np.array([P[0, 2] / P[0, 0], P[1, 2] / P[1, 1], -1.0]); z /= np.linalg.norm(z)
    x = np.array([1.0, 0.0, 0.0]) - z[0] * z; x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


def mean_or_none(xs):
    xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else None


def score_masked(root, task, directory, pred_dir):
    metadata = root / "meta" / task["job_id"]
    man = read_json(pred_dir / "manifest.json")
    with np.load(pred_dir / "rollout.npz") as c:
        pst = c["part_stats"]
    parts = man["parts"]; npart = len(parts)
    truth = MaskTruth(task, metadata); depth = GTDepth(task)
    rows = source_rows(metadata)
    try:
        ng = len(truth.objects)
        gpos, gproj, classes, defs, _ = gt_poses(task, metadata, truth.objects)
        cache = root / GT_CACHE / f"{task['job_id']}.npz"
        gst = None
        if cache.exists():
            with np.load(cache) as c:
                if c["gt_stats"].shape[:2] == (man["frames"], ng):
                    gst = c["gt_stats"]
        build = gst is None
        if build:
            gst = np.zeros((man["frames"], ng, 6))
        hist = np.zeros((man["frames"], ng + 1, npart + 1), dtype=np.int64)
        grid = None
        for t in range(man["frames"]):
            gl = truth.frame(t)
            pl = np.array(Image.open(pred_dir / "labels" / f"{t:06d}.png"))
            if gl.shape != pl.shape:
                raise ValueError("GT/prediction resolution mismatch; refusing resizing/warping")
            hist[t] = np.bincount((gl.astype(np.int64) * (npart + 1) + pl).ravel(),
                                  minlength=(ng + 1) * (npart + 1)).reshape(ng + 1, npart + 1)
            if build:
                P = gproj[t]
                if grid is None or not np.allclose(P, grid[0]):
                    grid = (P.copy(), *backproject_grid(P, gl.shape[1], gl.shape[0]))
                gst[t] = label_stats(gl, depth.frame(rows[t]["source_frame"]), ng, grid[1], grid[2])
        if build:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp.npz")
            np.savez_compressed(tmp, gt_stats=gst)
            os.replace(tmp, cache)
    finally:
        truth.close(); depth.close()

    W, H = man["width"], man["height"]; diag = math.hypot(W, H)
    # Correspondence. (a) One-to-one Hungarian between all GT objects (static ones too, so a background part
    # cannot be taken by a moving object) and prediction parts on sequence-mean IoU over the GT-visible frames;
    # pairs below MIN_MATCH_IOU are dropped (v1 kept near-zero pairs). (b) Every part left over joins the GT object
    # holding >= MERGE_SHARE of its visible pixels (objects modelled as several bodies: wheels, links, vanes).
    # Identity comes from each GT object's first visible frames (EARLY_MIN frames or EARLY_FRAC of its visible
    # frames, whichever is more): there the reconstruction is anchored by its initial state; later frames carry
    # motion errors, and a small GT object passing through a large wrong part would otherwise "win" it (014).
    ga_t = hist[:, 1:, :].sum(2); pa_t = hist[:, :, 1:].sum(1)
    union = ga_t[:, :, None] + pa_t[:, None, :] - hist[:, 1:, 1:]
    iou_t = np.divide(hist[:, 1:, 1:], union, out=np.zeros(union.shape), where=union > 0)
    early = np.zeros(ga_t.shape, dtype=bool)
    for g in range(ng):
        fr = np.flatnonzero(ga_t[:, g] > 0)
        if len(fr):
            early[fr[:max(EARLY_MIN, math.ceil(EARLY_FRAC * len(fr)))], g] = True
    affinity = (iou_t * early[:, :, None]).sum(0) / np.maximum(early.sum(0)[:, None], 1)
    rr, cc = linear_sum_assignment(-affinity)
    members = {g: [] for g in range(ng)}
    used = set()
    for g, k in zip(rr, cc):
        if affinity[g, k] >= MIN_MATCH_IOU:
            members[int(g)].append(int(k)); used.add(int(k))
    O = hist.sum(0); A = O.sum(0)
    merged = 0
    for k in range(npart):
        if k in used or A[k + 1] == 0:
            continue
        share = O[1:, k + 1] / A[k + 1]
        g = int(np.argmax(share))
        if share[g] >= MERGE_SHARE:
            members[g].append(k); merged += 1
    owner = {k: g for g, ks in members.items() for k in ks}
    pun = np.zeros((man["frames"], ng, 6)); inter = np.zeros((man["frames"], ng))
    for g, ks in members.items():
        if ks:
            pun[:, g] = pst[:, ks].sum(1)
            inter[:, g] = hist[:, g + 1, [k + 1 for k in ks]].sum(1)
    ga, pa = gst[..., 0], pun[..., 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(ga + pa - inter > 0, inter / (ga + pa - inter), 0.0)
    with np.load(pred_dir / "rollout.npz") as c:
        pproj = c["projection"]
    gst = gst.copy(); pun = pun.copy()  # 3D sums into centred camera frames (identity when the offset is zero)
    gst[..., 3:6] = np.einsum("tij,tgj->tgi", np.stack([centred_frame(np.asarray(gproj[t])) for t in range(man["frames"])]), gst[..., 3:6])
    pun[..., 3:6] = pun[..., 3:6] @ centred_frame(pproj[0] if pproj.ndim == 3 else pproj).T
    g2, g3 = centroids(gst); p2, p3 = centroids(pun)
    # scale from initial reliable overlaps
    ratios = []
    for g in range(ng):
        ok = np.flatnonzero((iou[:, g] >= SCALE_MIN_IOU) & np.isfinite(g3[:, g, 2]) & np.isfinite(p3[:, g, 2]) & (p3[:, g, 2] < 0))
        if len(ok):
            ratios.append(g3[ok[0], g, 2] / p3[ok[0], g, 2])
    scale_basis = f"median z-depth ratio at first frame with IoU >= {SCALE_MIN_IOU}"
    if not ratios:
        for g in range(ng):
            ok = np.flatnonzero((ga[:, g] > 0) & (pa[:, g] > 0) & (p3[:, g, 2] < 0))
            if len(ok):
                ratios.append(g3[ok[0], g, 2] / p3[ok[0], g, 2])
        scale_basis = "median z-depth ratio at first jointly visible frame (no overlap reached the IoU floor)"
    scale = float(np.median(ratios)) if ratios else None
    objs, excluded = [], []
    for g, obj in enumerate(truth.objects):
        vis = ga[:, g] > 0
        if classes[g]["class"] != "moving" or not vis.any():
            excluded.append({"gt_object": obj, "reason": "never_visible" if classes[g]["class"] == "moving" else classes[g]["class"]})
            continue
        joint = vis & (pa[:, g] > 0)
        e2 = np.full(vis.sum(), diag)
        e2[joint[vis]] = np.minimum(np.linalg.norm(g2[joint, g] - p2[joint, g], axis=1), diag)
        e3 = np.linalg.norm(scale * p3[joint, g] - g3[joint, g], axis=1) if scale and joint.any() else np.array([])
        ks = members[g]
        objs.append({"gt_index": g, "gt_object": obj, "gt_motion": classes[g],
                     "pred_parts": [parts[k]["name"] for k in ks],
                     "pred_roots": sorted({parts[k]["root_name"] or parts[k]["name"] for k in ks}),
                     "pred_dynamic_parts": sum(parts[k]["dynamic"] for k in ks),
                     "visible_gt_frames": int(vis.sum()), "jointly_visible_frames": int(joint.sum()),
                     "iou": float(iou[vis, g].mean()), "ade2d_pixels": float(e2.mean()),
                     "ade3d_m": float(e3.mean()) if len(e3) else None, "scored3d_frames": int(len(e3))})
    out = {"moving_visible_instance_iou": mean_or_none(o["iou"] for o in objs),
           "moving_visible_ade2d_pixels": mean_or_none(o["ade2d_pixels"] for o in objs),
           "moving_visible_ade3d_m": mean_or_none(o["ade3d_m"] for o in objs),
           "eligible_objects": len(objs), "matched_objects": sum(bool(o["pred_parts"]) for o in objs),
           "objects_scored3d": sum(o["ade3d_m"] is not None for o in objs),
           "scale": scale, "scale_basis": scale_basis, "scale_objects": len(ratios), "merged_parts": merged,
           "correspondence": f"Hungarian on mean IoU over each GT object's first max({EARLY_MIN}, {EARLY_FRAC:.0%}) visible frames (all GT objects x prediction parts, >= {MIN_MATCH_IOU}); leftover parts join the GT object holding >= {MERGE_SHARE:.0%} of their pixels",
           "object_metrics": objs, "excluded_objects": excluded}
    np.savez_compressed(directory / "statistics.npz", gt_stats=gst, pred_union_stats=pun, intersections=inter,
                        part_owner=np.array([owner.get(k, -1) for k in range(npart)]))
    return out


# ---------------------------------------------------------------- driver
def evaluate_job(root, task, egl_device=None):
    """Render (once) and score one task; the row is cached in evaluation/v2/jobs/<job>/metrics.json."""
    directory = root / OUT / "jobs" / task["job_id"]
    work = (root / "tasks" / task["job_id"]).resolve()
    if (directory / "metrics.json").exists():
        row = read_json(directory / "metrics.json")
        if row.get("error") != NO_OUTPUT or not (work / "output/scene.xml").is_file():
            return row
    directory.mkdir(parents=True, exist_ok=True)
    base = {"job_id": task["job_id"], "dataset": task["dataset"], "sample_id": task["sample_id"],
            "version": VERSION, "evaluated_at": utc_now(), "model_calls": False}
    try:
        if not (work / "output/scene.xml").is_file():
            row = {**base, "status": "unavailable", "error": NO_OUTPUT}
            save_json(directory / "metrics.json", row)
            return row
        pred_dir = directory / "prediction"
        if (pred_dir / "manifest.json").exists():  # render complete (manifest is written last): rescore only
            row = {**base, "status": "evaluated", "metrics": score_masked(root, task, directory, pred_dir)}
            save_json(directory / "metrics.json", row)
            return row
        cmd = [sys.executable, "-m", "evaluation.render", "--work", str(work),
               "--metadata", str((root / "meta" / task["job_id"]).resolve()), "--output", str(pred_dir.resolve())]
        env = egl_env(dict(os.environ), egl_device)
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(HERE.parent), env.get("PYTHONPATH")) if p)
        with (directory / "render.stdout.log").open("w") as o, (directory / "render.stderr.log").open("w") as e:
            code = subprocess.call(cmd, cwd=directory, env=env, stdout=o, stderr=e)  # cwd: MUJOCO_LOG.TXT lands here
        if code:
            raise RuntimeError(f"rollout/render failed ({code}); see {directory / 'render.stderr.log'}")
        row = {**base, "status": "evaluated", "metrics": score_masked(root, task, directory, pred_dir)}
    except Exception as exc:  # recorded, the run continues
        row = {**base, "status": "unavailable", "error": f"{type(exc).__name__}: {exc}"}
    save_json(directory / "metrics.json", row)
    return row


def write_csv(root, rows):
    fields = ["job_id", "dataset", "status", "moving_visible_ade3d_m", "moving_visible_ade2d_pixels",
              "moving_visible_instance_iou", "eligible_objects", "matched_objects", "objects_scored3d", "scale", "error"]
    with (root / OUT / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in sorted(rows, key=lambda r: r["job_id"]):
            m = r.get("metrics", {})
            w.writerow({k: r.get(k, m.get(k)) for k in fields})
