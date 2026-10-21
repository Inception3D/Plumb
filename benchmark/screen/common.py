"""Shared parts of the scene screening (benchmark/screen/): work files, command line, parallel map, Stage A.

Stage A runs on the retimed time axis of the benchmark videos (g = 9.81 in GT metres): the GT physical-quantity items
of the published metric (evaluation/physics.py gt_items), the moving objects, and physical-plausibility signals
(rise of translational + potential energy, horizontal drag in free flight, the first friction stop of each object)."""
import argparse
import csv
import json
import math
import sys
import tempfile
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))     # benchmark/: evaluation/, build.py
import build  # noqa: E402,F401  (physinone_g)
from evaluation import physics as ep  # noqa: E402

PICKS = HERE.parent / "bench50.json"     # the hand-picked scenes = the published scene list


def load(work, name):
    return json.loads((work / name).read_text(encoding="utf-8"))


def dump(work, name, value, indent=None):
    (work / name).parent.mkdir(parents=True, exist_ok=True)
    (work / name).write_text(json.dumps(value, indent=indent, default=float), encoding="utf-8")


def pmap(fn, jobs, workers):
    with Pool(workers) as p:
        return p.map(fn, jobs, chunksize=1)


def wanted(path, only):
    """--only: keep files whose name contains one of the given strings"""
    return not only or any(s in Path(path).name for s in only)


def cli(doc, stages, argv=None):
    """`script STAGE [--data-root D] --work W [--workers N] [--only NAME ...]` + stage-specific extras"""
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=stages)
    ap.add_argument("subclasses", nargs="?", help="show: comma-separated subclasses")
    ap.add_argument("--data-root", type=Path, help="raw data (physinone/<category>/*.zip, physion_v15/<category>/*/*.hdf5)")
    ap.add_argument("--work", type=Path, required=True, help="directory of the intermediate JSON files and sheets")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", nargs="+", metavar="NAME", help="only files whose name contains one of these (subset runs)")
    ap.add_argument("--picks", type=Path, default=PICKS, help="the hand-picked scene list (default: benchmark/bench50.json)")
    ap.add_argument("--full", action="store_true", help="show: one row per moving object")
    ap.add_argument("--all", action="store_true", help="physion b: every Stage A scene instead of the picks")
    ap.add_argument("--add", nargs="+", default=[], metavar="SAMPLE", help="physion sheets: further scenes")
    a = ap.parse_args(argv)
    if a.stage in ("scan", "a") and not a.data_root:
        ap.error("--data-root is required for this stage")
    a.data_root = a.data_root.absolute() if a.data_root else None
    a.work.mkdir(parents=True, exist_ok=True)
    return a


# ---------------------------------------------------------------- Stage A (both datasets)

def write_meta(d, frames, t):
    d.mkdir(parents=True, exist_ok=True)
    with (d / "timestamps.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["index", "source_frame", "relative_time_seconds"])
        for i, (fr, tt) in enumerate(zip(frames, t)):
            w.writerow([i, fr, f"{tt:.6f}"])


def energy_rise(t, objs, g=9.81, t0=0.03):
    """largest rise of translational KE + PE above its running minimum, relative to the total range"""
    keys = [k for k, o in objs.items() if o["mass"]]
    if not keys:
        return None
    E = 0
    for k in keys:
        P = objs[k]["P"]; v = np.gradient(P, t, axis=0)
        E = E + objs[k]["mass"] * (0.5 * (v ** 2).sum(1) + g * P[:, 2])
    m = t >= t0
    E = E[m]
    if len(E) < 5:
        return None
    E = np.convolve(E, np.ones(3) / 3, mode="valid")
    rise = float(np.max(E - np.minimum.accumulate(E)))
    rng = float(np.ptp(E)) or 1e-9
    floor = 0.02 * sum(objs[k]["mass"] for k in keys) * g
    return {"rise": rise, "range": rng, "ratio": rise / rng, "significant": bool(rise > 0.25 * rng and rise > floor)}


def free_flight_drag(t, P, g=9.81):
    """horizontal speed ratio end/start over the longest ballistic stretch (|a_z + g| < 1.5, >= 0.15 s)"""
    v = np.gradient(P, t, axis=0); a = np.gradient(v, t, axis=0)
    ok = np.abs(a[:, 2] + g) < 1.5
    best = None; s = None
    for i, f in enumerate(list(ok) + [False]):
        if f and s is None:
            s = i
        if not f and s is not None:
            if t[i - 1] - t[s] >= 0.15 and (best is None or i - s > best[1] - best[0]):
                best = (s, i)
            s = None
    if not best:
        return None
    a0, a1 = best[0] + 1, best[1] - 2
    if a1 <= a0:
        return None
    h0, h1 = np.hypot(*v[a0, :2]), np.hypot(*v[a1, :2])
    return {"t": [float(t[a0]), float(t[a1])], "vh0": float(h0), "vh1": float(h1), "ratio": float(h1 / h0) if h0 > 0.3 else None}


def first_stops(task, objs, t):
    """a / g of each object's first stopping phase, where gt_items would make it a decel item but before its cut of
    stops faster than friction allows (physics_v6 drops those as collisions; the screen flags them as implausible)"""
    pendulum = task.get("category") == "pendulum" or "pendulum" in task["job_id"].lower()
    out = {}
    for k, o in objs.items():
        if pendulum or ep.onset(o["P"], o["R"]) is None or ep.spin(t, o["P"], o["R"]):
            continue
        sp = next(iter(ep.stopping_phases(t, o["P"], ep.G_GT)), None)
        if sp:
            out[str(k)] = sp["value"]
    return out


def stage_a(task, frames, t, extra):
    """Stage A record of one scene: GT items, objects, movers (> 2 cm or > 5 deg), energy, first stops"""
    with tempfile.TemporaryDirectory() as d:
        write_meta(Path(d), frames, t)
        gt = ep.gt_items(task, Path(d))
        rows, tr = ep.source_rows(Path(d))
    objs, info = (ep.physion_gt if task["dataset"] == "physion_v15" else ep.physinone_gt)(task, rows)
    movers = {}
    for k, o in objs.items():
        disp = float(np.linalg.norm(o["P"] - o["P"][0], axis=1).max()); rot = math.degrees(ep.rot_range(o["R"]))
        if disp > 0.02 or rot > 5:
            v = np.gradient(o["P"], t, axis=0)
            movers[str(k)] = {"label": o["label"], "mass": o["mass"], "disp": disp, "rot_deg": rot,
                              "vmax": float(np.linalg.norm(v, axis=1).max()),
                              "z0": float(o["P"][0, 2]), "zmin": float(o["P"][:, 2].min()),
                              "drag": free_flight_drag(t, o["P"])}
    er = energy_rise(t, {k: o for k, o in objs.items() if str(k) in movers}, t0=info["push_t"] + 0.03)
    return {"dataset": task["dataset"], "path": task["source"]["path"], "task": task, "frames": len(frames),
            "duration": float(t[-1]), "items": gt["items"], "objects": gt["objects"], "movers": movers, "energy": er,
            "stop_g": first_stops(task, objs, tr), **extra}


def failed(dataset, path, e):
    return {"dataset": dataset, "path": path, "error": repr(e), "tb": traceback.format_exc()[-1500:]}


def item_counts(o):
    """'why' of a scene: its Stage A GT item counts"""
    k = {}
    for i in o["items"]:
        k[i["kind"]] = k.get(i["kind"], 0) + 1
    return "GT items (screen): " + ", ".join(f"{a} {b}" for a, b in sorted(k.items()))
