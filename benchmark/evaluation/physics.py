"""Physical-quantity metric for real2sim on PhysInOne and Physion v1.5 (writes BENCH/evaluation/physics/).

The same estimators run on the GT trajectory and on the prediction's passive rollout. Both are sampled at the
input-video frame times (meta/<job>/timestamps.csv relative_time_seconds), i.e. the retimed real-time axis of the
benchmark videos, on which g = 9.81 m/s^2 in GT metres. So every quantity is in real-world units: metres, video
seconds, the gravity of each world. Absolute mass is not observable from RGB, so mass enters only as a ratio to a
reference object.

Quantities (an item exists only where the GT shows the event; the prediction is measured the same way):
  speed0       speed over the first 0.1 s of motion, for objects launched at the start (Physion: by the push)
               with >= 0.5 m/s (m/s)
  restitution  first impact after a free fall: upward / downward vertical speed at the impact frame
               (prediction: its impact nearest in time, no free-fall requirement)
  decel        first stopping phase on a level support: speed decay rate / g (sliding friction or rolling
               resistance; prediction: its stopping phase nearest in time, looser length/speed bounds)
  period       swing period of an oscillating object (s)
  spin_rate    mean angular speed of an object turning in place about a fixed axis (rad/s)
  spin_accel   angular acceleration of that spin (rad/s^2)
  mass_rel     mass / geometric mean mass of the scene's scored moving objects that collide with another moving
               object (GT and prediction normalised separately; objects that never exchange momentum are skipped)
GT masses: PhysInOne recorder_stats.json component mass; Physion per-frame objects/masses (static/mass carries
extra template entries and does not align with object_ids). Physion probe_mass is not the simulated mass.
Correspondence: evaluator v2 (evaluation/v2/jobs/<job>/metrics.json, GT object -> prediction parts).
Correct: ratio pred/GT within 1/1.25..1.25 (restitution: |delta| <= 0.1). MRA: mean over thresholds 0.50..0.95.
Diagnostic *_scaled: speed0/decel of the prediction times evaluator v2's depth scale (GT/pred), i.e. with the
scene-size error removed; the primary score keeps the prediction's own metric units.
GT items are cached in BENCH/cache/physics/<job>.json (recomputed when VERSION changes).

v6 (2026-10-15, curated 50-scene benchmark): mass_rel only for movers that collide with another mover (opposite
velocity changes in the same frame) instead of "not all starting together"; GT stops faster than friction allows
(Physion 0.6 g, PhysInOne 1 g) are collisions and give no decel item; Physion push time relative to the first
video frame (clips may start after source frame 0).
"""
import csv
import json
import math
import zipfile
from pathlib import Path

import numpy as np

from .common import NO_OUTPUT, read_json, save_json

VERSION = "physics_v6"
GT_CACHE = Path("cache/physics")     # relative to the benchmark directory
OUT, V2 = Path("evaluation/physics"), Path("evaluation/v2")
G_GT = 9.81            # GT gravity on the retimed axis
MOVE_M, MOVE_RAD = 0.01, math.radians(5)
TOL_RATIO, TOL_E = 1.25, 0.1
THRESH = np.arange(0.50, 0.951, 0.05)
SCALED = {"speed0", "decel"}   # kinds in length units (decel = a / g)
DECEL_MAX = {"physion_v15": 0.6, "physinone": 1.0}   # a / g above which a GT stop is not friction


# ---------------------------------------------------------------- tracks: t (N,), P (N,3) z-up metres, R (N,3,3)

def quat_xyzw_to_mat(q):
    q = np.asarray(q, float); x, y, z, w = (q / np.linalg.norm(q, axis=-1, keepdims=True)).T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def source_rows(metadata):
    with (metadata / "timestamps.csv").open() as f:
        rows = list(csv.DictReader(f))
    return np.array([int(r["source_frame"]) for r in rows]), np.array([float(r["relative_time_seconds"]) for r in rows])


def physinone_gt(task, frames):
    zp = Path(task["source"]["path"]); b = zp.name[:-4]
    with zipfile.ZipFile(zp) as z:
        traj = json.loads(z.read(f"{b}/{b}.json")); stats = json.loads(z.read(f"{b}/recorder_stats.json"))
    mass = {}
    for group in ("solid_object", "interactable_object", "breakable_object"):
        for entry in stats.get(group) or []:
            for name, v in entry.items():
                ms = [c["mass"] for c in v.get("components", []) if c.get("mass") is not None]
                # several simulated components (trolley wheels, chain links): actor mass is not one number
                mass[name] = float(ms[0]) if len(ms) == 1 and ms[0] > 0 else None
    objects = {}
    for name, actor in (traj.get("actors") or {}).items():
        series = [actor.get("transform_data") or []] + [c.get("transform_data") or [] for c in (actor.get("components") or {}).values()]
        best = None
        for td in series:
            if not td:
                continue
            by = {int(x["frame"]): x["transform"] for x in td}
            if not all(int(f) in by for f in frames):
                continue
            P = np.array([[by[int(f)]["location"][k] / 100 for k in "xyz"] for f in frames])
            q = [by[int(f)]["rotation"]["quaternion"] for f in frames]
            R = quat_xyzw_to_mat(np.array([[r["x"], r["y"], r["z"], r["w"]] for r in q]))
            score = np.ptp(P, 0).max() + 0.1 * rot_range(R)
            if best is None or score > best[0]:
                best = (score, P, R)    # actor channel or its most moving component (009 platform = "Wheel")
        if best:
            objects[name] = {"P": best[1], "R": best[2], "mass": mass.get(name), "label": name}
    return objects, {"push_t": 0.0}


def physion_gt(task, frames):
    import h5py
    with h5py.File(task["source"]["path"], "r") as f:
        ids = [int(i) for i in f["static/object_ids"][()]]
        names = [x.decode() for x in f["static/model_names"][()]]
        F = f["frames"]
        masses = F[f"{int(frames[0]):04d}"]["objects/masses"][()]
        C = np.array([F[f"{int(k):04d}"]["objects/center_cam0"][()] for k in frames])
        Q = np.array([F[f"{int(k):04d}"]["objects/rotations_cam0"][()] for k in frames])
        # frame dt 0.01 s; a clip may start after source frame 0 (leading still frames trimmed), so the push time
        # is relative to the first video frame
        push = max(0.0, (float(f["static/push_time"][()]) - int(frames[0])) * 0.01) if "push_time" in f["static"] else 0.0
    G =np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], float)      # Unity y-up -> z-up (swap y, z)
    objects = {}
    for i, oid in enumerate(ids):
        R = np.einsum("ij,njk,kl->nil", G, quat_xyzw_to_mat(Q[:, i]), G)
        objects[oid] = {"P": C[:, i] @ G.T, "R": R, "mass": float(masses[i]) if masses[i] > 0 else None,
                        "label": f"{oid}:{names[i] if i < len(names) else '?'}"}
    return objects, {"push_t": push}


def rot_range(R):
    d = np.einsum("nij,kj->nik", R, R[0])
    return float(np.max(np.arccos(np.clip((np.trace(d, axis1=1, axis2=2) - 1) / 2, -1, 1))))


# ---------------------------------------------------------------- estimators (identical for GT and prediction)

def onset(P, R):
    d = np.linalg.norm(P - P[0], axis=1)
    a = np.arccos(np.clip((np.trace(np.einsum("nij,kj->nik", R, R[0]), axis1=1, axis2=2) - 1) / 2, -1, 1)) if R is not None else 0 * d
    idx = np.nonzero((d > MOVE_M) | (a > MOVE_RAD))[0]
    return int(idx[0]) if len(idx) else None


def speed_window(t, P, t0, width=0.1, at=None):
    """speed of a quadratic fit over [t0, t0 + width], at the window centre (or at t0 + at)"""
    m = (t >= t0 - 1e-9) & (t <= t0 + width + 1e-9)
    if m.sum() < 4:
        return None
    tau = t[m] - t0; A = np.vstack([np.ones_like(tau), tau, tau * tau]).T
    c, *_ = np.linalg.lstsq(A, P[m], rcond=None)
    tc = 0.5 * width if at is None else at
    return float(np.linalg.norm(c[1] + 2 * c[2] * tc))


def ballistic_v(t, z, idx, tk, g):
    """vertical velocity at tk of z(t) = a + v (t-tk) - g/2 (t-tk)^2 fitted on frames idx"""
    if len(idx) < 2:
        return None
    tau = t[idx] - tk; y = z[idx] + 0.5 * g * tau * tau
    A = np.vstack([np.ones_like(tau), tau]).T
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(c[1])


def free_fall(t, z, idx, g):
    """the frames idx follow a free fall: fitted vertical acceleration within 0.6 g .. 1.4 g downward"""
    if len(idx) < 4:
        return False
    tau = t[idx] - t[idx[0]]
    a = 2 * np.polyfit(tau, z[idx], 2)[0]
    return -1.4 * g <= a <= -0.6 * g


def bounces(t, P, g, strict=True, vmin=0.5):
    """impacts while falling: local height minimum after >= vmin downward. strict (GT applicability): the 5 frames
    before follow a free fall. e = upward / downward vertical speed at the impact frame, from ballistic fits."""
    z = P[:, 2]; n = len(z); out = []
    if n < 8:
        return out
    vz = np.gradient(z, t)
    for k in range(5, n - 3):
        if vz[k - 1] < -vmin and vz[k - 2] < -vmin and z[k] <= z[k - 1] and z[k] <= z[k + 1] + 1e-4:
            if strict and not free_fall(t, z, np.arange(k - 5, k), g):
                continue
            vin = ballistic_v(t, z, np.arange(max(0, k - 4), k), t[k], g)
            vout = ballistic_v(t, z, np.arange(k + 1, min(n, k + 5)), t[k], g)
            if vin is None or vout is None or vin > -vmin:
                continue
            out.append({"t": float(t[k]), "v_in": vin, "v_out": vout, "e": float(np.clip(vout / -vin, 0, 1.5))})
    return out


def stopping_phases(t, P, g, strict=True, vstop=0.05):
    """phases that end in a stop (or the video end) while on a level support; speed decay rate / g by a linear fit.
    strict (GT applicability): starts >= 0.3 m/s, >= 6 frames and 0.1 s, near-linear decay. The prediction is
    measured with looser bounds (>= 0.15 m/s, >= 4 frames) so a faster or shorter stop still yields a value."""
    vstart, nmin, dmin, hold = (0.3, 6, 0.1, 0.1) if strict else (0.15, 4, 0.04, 0.05)
    out = []
    if len(t) < 8:
        return out
    v = np.gradient(P, t, axis=0)
    s = np.convolve(np.linalg.norm(v[:, :2], axis=1), np.ones(5) / 5, mode="same")
    vz = np.abs(np.convolve(v[:, 2], np.ones(5) / 5, mode="same"))
    n = len(s)
    ends = [b for b in range(1, n) if s[b] < vstop <= s[b - 1]] + ([n - 1] if s[-1] >= vstop else [])
    prev = 0
    for b in ends:
        # the phase starts at the fastest frame since the last impact / fall (vertical speed >= 0.25) or stop;
        # small bumps on the way (holes, rocking) do not cut it
        lo = max([prev] + [i + 1 for i in range(prev, b) if vz[i] >= 0.25])
        prev = b
        if lo >= b:
            continue
        a = lo + int(np.argmax(s[lo:b + 1]))
        after = (t > t[b]) & (t <= t[b] + hold)
        if b < n - 1 and (t[-1] - t[b] < hold or s[after].max(initial=0) >= 2 * vstop):
            continue      # a stop must hold (turning points and rebounds are not stops; slight rocking is)
        seg = np.arange(a, b + 1); seg = seg[s[seg] > vstop]
        if len(seg) >= 2:
            path = np.linalg.norm(np.diff(P[seg[0]:seg[-1] + 1, :2], axis=0), axis=1).sum()
            if np.linalg.norm(P[seg[-1], :2] - P[seg[0], :2]) < max(0.02, 0.5 * path):
                continue  # not a straight slide or roll (a spinning body's pivot circles)
        if len(seg) < nmin or t[seg[-1]] - t[seg[0]] < dmin or s[seg[0]] < vstart or s[seg[-1]] > 0.5 * s[seg[0]]:
            continue
        if vz[seg].mean() > 0.15 * s[seg].mean():
            continue      # not on a level support (slopes mix in g sin(theta))
        k, c = np.polyfit(t[seg] - t[seg[0]], s[seg], 1)
        rms = float(np.sqrt(np.mean((np.polyval([k, c], t[seg] - t[seg[0]]) - s[seg]) ** 2)))
        if k >= 0 or (strict and rms > 0.2 * s[seg[0]]):
            continue
        out.append({"t0": float(t[seg[0]]), "t1": float(t[seg[-1]]), "v0": float(s[seg[0]]), "decel": float(-k),
                    "value": float(-k / g), "rms": rms})
    return out


def kicks(t, P, g, thr=0.4):
    """frames where the velocity changes by > thr m/s within +-2 frames, more than either free flight or resting on a
    support explains (contacts and impacts; friction changes it by < 0.3 m/s over 4 frames)"""
    v = np.gradient(P, t, axis=0); n = len(t); out = {}
    for k in range(2, n - 2):
        dv = v[k + 2] - v[k - 2]; free = dv + np.array([0, 0, g * (t[k + 2] - t[k - 2])])
        if min(np.linalg.norm(dv), np.linalg.norm(free)) > thr:
            out[k] = dv
    return out


def colliding(t, tracks, g, t_min):
    """keys of moving objects that collide with another moving object: both change velocity at the same frame in
    opposite directions (momentum exchange) after t_min. Parallel lanes that never touch (FrictionStop) and
    simultaneous landings on the floor (same direction) do not count."""
    K = {k: {f: dv for f, dv in kicks(t, P, g).items() if t[f] > t_min} for k, P in tracks.items()}
    out = set(); keys = list(K)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if any(float(K[a][f] @ K[b][f]) < 0 for f in K[a].keys() & K[b].keys()):
                out |= {a, b}
    return out


def nearest(events, t0, key):
    return min(events, key=lambda e: abs(e[key] - t0)) if events else None


def swing_period(t, P, amp_min=0.05):
    xy = P[:, :2] - P[:, :2].mean(0)
    if len(t) < 8 or np.ptp(xy, 0).max() < 2 * amp_min:
        return None
    w, V = np.linalg.eigh(xy.T @ xy); x = xy @ V[:, -1]
    x = x - 0.5 * (x.max() + x.min())
    cross = []
    for i in range(len(x) - 1):
        if x[i] == 0 or x[i] * x[i + 1] < 0:
            cross.append(t[i] + (t[i + 1] - t[i]) * x[i] / (x[i] - x[i + 1]) if x[i] != x[i + 1] else t[i])
    if len(cross) < 2:
        return None
    half = np.diff(cross)
    return {"crossings": len(cross), "value": float(2 * np.mean(half)), "cv": float(np.std(half) / np.mean(half))}


def spin(t, P, R, min_turn=math.pi / 2, max_shift=0.1):
    """an object turning in place about a fixed axis (turntable: vertical; windmill, door panel: any axis).
    The axis is the dominant direction of the rotation vectors of R(t) R(0)^T; the angle is unwrapped about it."""
    if R is None or len(t) < 8 or np.ptp(P, 0).max() > max_shift:
        return None
    d = np.einsum("nij,kj->nik", R, R[0])
    w = np.stack([d[:, 2, 1] - d[:, 1, 2], d[:, 0, 2] - d[:, 2, 0], d[:, 1, 0] - d[:, 0, 1]], 1)   # 2 sin(th) axis
    if np.linalg.norm(w, axis=1).max() < 1e-6:
        return None
    a = np.linalg.svd(w, full_matrices=False)[2][0]
    u = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0]); u /= np.linalg.norm(u)
    v = d @ u
    th = np.unwrap(np.arctan2(v @ np.cross(a, u), v @ u))
    if abs(th[-1] - th[0]) < min_turn:
        return None
    tau = t - t[0]; c = np.polyfit(tau, th, 2)
    return {"rate": float(abs(th[-1] - th[0]) / tau[-1]), "accel": float(2 * c[0] * np.sign(th[-1] - th[0]))}


# ---------------------------------------------------------------- GT items

def gt_items(task, metadata):
    frames, t = source_rows(metadata)
    objects, info = (physion_gt if task["dataset"] == "physion_v15" else physinone_gt)(task, frames)
    pendulum = task.get("category") == "pendulum" or "pendulum" in task["job_id"].lower()
    items, summary = [], {}
    starts = {}
    for key, o in objects.items():
        on = onset(o["P"], o["R"])
        summary[str(key)] = {"label": o["label"], "mass": o["mass"], "onset_t": None if on is None else float(t[on]),
                             "max_disp_m": float(np.linalg.norm(o["P"] - o["P"][0], axis=1).max())}
        if on is None:
            continue
        starts[key] = float(t[on])
        # initial velocity: only objects launched at the start (Physion: by the push) faster than a fall from rest
        # would be at onset; later starts are collision outcomes
        if t[on] <= info["push_t"] + 0.05 and \
                (speed_window(t, o["P"], float(t[on]), at=0.0) or 0) - G_GT * (t[on] - info["push_t"]) >= 0.3:
            v = speed_window(t, o["P"], float(t[on]))
            items.append({"kind": "speed0", "obj": str(key), "value": v, "window": [float(t[on]), float(t[on]) + 0.1]})
        bo = next(iter(bounces(t, o["P"], G_GT)), None)
        if bo and bo["e"] <= 1.0:      # e > 1: deflected by another body, not a bounce
            items.append({"kind": "restitution", "obj": str(key), "value": bo["e"], "detail": bo})
        s = spin(t, o["P"], o["R"])
        sp = next(iter(stopping_phases(t, o["P"], G_GT)), None)
        if sp and sp["value"] > DECEL_MAX[task["dataset"]]:
            sp = None   # faster than friction can stop it: stopped by a collision (Physion mu_dyn = 0.25 everywhere)
        if sp and not pendulum and not s:   # a swing or a spinning body's pivot is not friction
            items.append({"kind": "decel", "obj": str(key), "value": sp["value"], "detail": sp})
        if pendulum:
            pe = swing_period(t, o["P"])
            if pe and (pe["crossings"] == 2 or pe["cv"] < 0.3):
                items.append({"kind": "period", "obj": str(key), "value": pe["value"], "detail": pe})
        if s:
            items.append({"kind": "spin_rate", "obj": str(key), "value": s["rate"]})
            items.append({"kind": "spin_accel", "obj": str(key), "value": s["accel"]})
    # relative mass: moving objects with a usable GT mass that exchange momentum with another moving object;
    # scored after dividing GT and prediction each by the geometric mean over the scored set (absolute mass is
    # not observable). Objects that never touch another mover (FrictionStop lanes) carry no mass information.
    hit = colliding(t, {k: objects[k]["P"] for k in starts}, G_GT, info["push_t"] + 0.05)
    for key in starts:
        if key in hit and objects[key]["mass"]:
            items.append({"kind": "mass_rel", "obj": str(key), "value": objects[key]["mass"]})
    return {"job_id": task["job_id"], "dataset": task["dataset"], "version": VERSION, "items": items,
            "objects": summary, "frames": len(t),
            "time_axis": "timestamps.csv relative_time_seconds (retimed video time, g = 9.81 in GT metres)"}


def gt_cached(root, task):
    path = root / GT_CACHE / f"{task['job_id']}.json"
    if path.exists():
        cached = read_json(path)
        if cached.get("version") == VERSION:
            return cached
    out = gt_items(task, root / "meta" / task["job_id"])
    save_json(path, out)
    return out


# ---------------------------------------------------------------- prediction

def up_basis(gravity):
    up = -np.asarray(gravity, float); g = float(np.linalg.norm(up)); up = up / g
    x = np.cross([0, 1, 0], up) if abs(up[1]) < 0.9 else np.cross([1, 0, 0], up)
    x /= np.linalg.norm(x); y = np.cross(up, x)
    return np.vstack([x, y, up]), g


def pred_tracks(work, metadata, parts_by_gt, manifest):
    import mujoco
    output = work / "output"
    model = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
    if model.nu or model.nmocap:
        raise ValueError("Evaluation requires a passive scene")
    data = mujoco.MjData(model)
    state = read_json(output / "initial_state.json")
    data.qpos[:] = state["qpos"]; data.qvel[:] = state["qvel"]; mujoco.mj_forward(model, data)
    _, t = source_rows(metadata)
    B, g = up_basis(model.opt.gravity)
    def is_dynamic(b):
        while b:
            if model.body_dofnum[b]:
                return True
            b = int(model.body_parentid[b])
        return False
    part_body = {p["name"]: p["body_id"] for p in manifest["parts"] if p["kind"] == "body"}
    sel = {}
    for gt, names in parts_by_gt.items():
        bodies = sorted({part_body[n] for n in names if n in part_body})
        dyn = [b for b in bodies if is_dynamic(b)]
        if bodies:
            use = dyn or bodies       # track the moving bodies; a matched static base (009 foundation) only dilutes
            sel[gt] = {"bodies": use, "mass": float(sum(model.body_mass[b] for b in dyn)) if dyn else None,
                       "main": max(use, key=lambda b: model.body_mass[b])}
    P = {k: [] for k in sel}; R = {k: [] for k in sel}
    for target in t:
        while data.time < target - 1e-10:
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        if not np.isfinite(data.qpos).all():
            raise ValueError("Nonfinite rollout")
        for k, s in sel.items():
            m = np.array([model.body_mass[b] for b in s["bodies"]]); w = m / m.sum() if m.sum() > 0 else np.ones(len(m)) / len(m)
            P[k].append(B @ (w @ data.xipos[s["bodies"]]))
            R[k].append(B @ data.xmat[s["main"]].reshape(3, 3) @ B.T)
    return t, g, {k: {"P": np.array(P[k]), "R": np.array(R[k]), "mass": sel[k]["mass"]} for k in sel}


def measure(kind, item, t, g, o):
    if o is None:
        return None
    if kind == "speed0":
        return speed_window(t, o["P"], item["window"][0])
    if kind == "restitution":      # the prediction's impact closest in time to the GT one
        b = nearest(bounces(t, o["P"], g, strict=False), item["detail"]["t"], "t"); return b["e"] if b else None
    if kind == "decel":
        s = nearest(stopping_phases(t, o["P"], g, strict=False), item["detail"]["t0"], "t0"); return s["value"] if s else None
    if kind == "period":
        p = swing_period(t, o["P"]); return p["value"] if p else None
    if kind in ("spin_rate", "spin_accel"):
        s = spin(t, o["P"], o["R"]); return (s["rate"] if kind == "spin_rate" else s["accel"]) if s else None
    raise ValueError(kind)


def relative_masses(items, tracks):
    """GT and predicted mass, each divided by its geometric mean over the objects the prediction gives a mass"""
    have = [i for i in items if tracks.get(i["obj"]) and tracks[i["obj"]]["mass"]]
    if len(have) < 2:
        return {i["obj"]: (None, None) for i in items}
    lg = np.mean([math.log(i["value"]) for i in have]); lp = np.mean([math.log(tracks[i["obj"]]["mass"]) for i in have])
    out = {}
    for i in items:
        m = tracks.get(i["obj"], {}).get("mass") if tracks.get(i["obj"]) else None
        out[i["obj"]] = (math.exp(math.log(i["value"]) - lg), math.exp(math.log(m) - lp) if m else None)
    return out


def score(kind, gt, pred):
    if pred is None or not math.isfinite(pred):
        return False, 0.0, None
    if kind == "restitution":
        err = abs(pred - gt); rel = err / max(gt, 0.2)
        return err <= TOL_E, float(np.mean(rel < 1 - THRESH)), rel
    if gt == 0 or pred / gt <= 0:
        return False, 0.0, None
    rel = abs(pred - gt) / abs(gt)
    r = pred / gt
    return 1 / TOL_RATIO <= r <= TOL_RATIO, float(np.mean(rel < 1 - THRESH)), rel


def evaluate_job(root, task):
    metadata = root / "meta" / task["job_id"]
    gt = gt_cached(root, task)
    ev = root / V2 / "jobs" / task["job_id"]
    out = {"job_id": task["job_id"], "dataset": task["dataset"], "version": VERSION, "items": []}
    work = root / "tasks" / task["job_id"]
    if not (work / "output/scene.xml").is_file():
        out.update(status="unavailable", reason=NO_OUTPUT); return out
    if not (ev / "metrics.json").exists() or read_json(ev / "metrics.json").get("status") != "evaluated":
        out["status"] = "unscored: no evaluator v2 correspondence"; return out
    m = read_json(ev / "metrics.json")
    parts_by_gt = {str(o["gt_object"]["source_id"]): o.get("pred_parts") or [] for o in m["metrics"].get("object_metrics", [])}
    try:
        t, g, tracks = pred_tracks(work, metadata, parts_by_gt, read_json(ev / "prediction/manifest.json"))
    except Exception as e:  # noqa: BLE001 - recorded, scored as missing
        out["status"] = f"unscored: rollout failed: {e}"; return out
    items = [it for it in gt["items"] if it["obj"] in parts_by_gt]     # evaluator-eligible moving GT objects
    masses = [it for it in items if it["kind"] == "mass_rel"]
    items = [it for it in items if it["kind"] != "mass_rel"] + (masses if len(masses) >= 2 else [])
    rel = relative_masses(masses, tracks) if len(masses) >= 2 else {}
    scale = m["metrics"].get("scale")
    for it in items:
        o = tracks.get(it["obj"])
        if it["kind"] == "mass_rel":
            gt_v, pred = rel[it["obj"]]
        else:
            gt_v, pred = it["value"], measure(it["kind"], it, t, g, o)
        ok, mra, err = score(it["kind"], gt_v, pred)
        # diagnostic: lengths rescaled by evaluator v2's depth scale (GT/pred), separating dynamics from scene size
        pred_s = pred * scale if pred is not None and scale and it["kind"] in SCALED else pred
        ok_s, mra_s, _ = score(it["kind"], gt_v, pred_s)
        out["items"].append({"kind": it["kind"], "obj": it["obj"], "gt": gt_v, "pred": pred, "correct": ok, "mra": mra,
                             "rel_err": err, "matched": bool(o), "gt_raw": it["value"],
                             "pred_scaled": pred_s, "correct_scaled": ok_s, "mra_scaled": mra_s})
    n = len(out["items"])
    out["status"] = "evaluated"
    out["accuracy"] = sum(i["correct"] for i in out["items"]) / n if n else None
    out["mra"] = sum(i["mra"] for i in out["items"]) / n if n else None
    out["accuracy_scaled"] = sum(i["correct_scaled"] for i in out["items"]) / n if n else None
    out["scale"] = scale
    out["gravity_pred"] = g
    return out


def write_csv(root, results):
    """metrics.csv (one row per job) and items.csv (one row per physical-quantity item)"""
    dest = root / OUT
    results = sorted(results, key=lambda r: r["job_id"])
    with (dest / "metrics.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["job_id", "dataset", "status", "items", "accuracy", "mra", "accuracy_scaled", "scale", "reason"])
        for r in results:
            w.writerow([r["job_id"], r["dataset"], r["status"], len(r["items"]), r.get("accuracy"), r.get("mra"),
                        r.get("accuracy_scaled"), r.get("scale"), r.get("reason")])
    with (dest / "items.csv").open("w", newline="") as f:
        cols = ["kind", "obj", "gt", "pred", "correct", "mra", "rel_err", "matched", "gt_raw", "pred_scaled", "correct_scaled", "mra_scaled"]
        w = csv.writer(f); w.writerow(["job_id"] + cols)
        for r in results:
            for i in r["items"]:
                w.writerow([r["job_id"]] + [i[c] for c in cols])
