#!/usr/bin/env python3
"""Evaluate the reconstructions in a benchmark directory (BENCH/benchmark.json, tasks/, meta/).

usage: python benchmark/evaluate.py BENCH [--metrics v2 physics secondview] [--tasks ID_PREFIX ...] [--workers N]
                                    [--egl-devices 0,1] [--videos] [--selftest]

Runs v2 (input view; renders each prediction once in a subprocess, then scores), physics (uses v2's correspondence)
and secondview (uses v2's alignment), in that order. A task is evaluated when BENCH/tasks/<id>/output/scene.xml
exists; otherwise its row is "unavailable". Writes BENCH/evaluation/<metric>/{jobs/<id>/..., metrics.csv} and keeps
GT caches in BENCH/cache/<metric>/. v2 rows are cached per job (delete BENCH/evaluation/v2/jobs/<id> to re-evaluate
a changed reconstruction); physics and secondview are recomputed on every call (their GT side is cached).
"""
import argparse
import concurrent.futures
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluation.common import egl_env, read_json, save_json, utc_now

egl_env()   # before mujoco is imported anywhere (secondview renders in this process)
from evaluation import physics, secondview, v2  # noqa: E402

METRICS = ("v2", "physics", "secondview")


def run(name, fn, jobs, workers, value):
    """fn(*args) for every args tuple in jobs on a process pool (scoring is numpy/GIL-heavy: processes, not threads)"""
    rows = []
    with concurrent.futures.ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(fn, *args) for args in jobs]
        for f in concurrent.futures.as_completed(futs):
            r = f.result(); rows.append(r)
            print(f"[{utc_now()}] {name} {r['job_id'][:48]} {r['status'][:90]} {value(r)} {r.get('error', '')}", flush=True)
    return rows


def all_rows(root, out):
    return [read_json(p) for p in sorted((root / out / "jobs").glob("*/metrics.json"))]


def mean(xs):
    xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
    return f"{np.mean(xs):.4g}" if xs else "-"


def summary(name, rows):
    by = defaultdict(list)
    for r in rows:
        by[r["dataset"]].append(r)
    for ds, rs in sorted(by.items()):
        ok = [r for r in rs if r["status"] == "evaluated"]
        head = f"{name:<11}{ds:<13}{len(ok):>3}/{len(rs):<3} evaluated"
        if name == "v2":
            m = [r["metrics"] for r in ok]
            print(f"{head}  ade3d {mean(x['moving_visible_ade3d_m'] for x in m)} m  ade2d "
                  f"{mean(x['moving_visible_ade2d_pixels'] for x in m)} px  iou {mean(x['moving_visible_instance_iou'] for x in m)}")
        elif name == "physics":
            items = [i for r in ok for i in r["items"]]
            print(f"{head}  {len(items)} items  accuracy {mean(i['correct'] for i in items)}  MRA {mean(i['mra'] for i in items)}"
                  f"  accuracy_scaled {mean(i['correct_scaled'] for i in items)}  (pooled over items)")
        else:
            print(f"{head}  ade2d {mean(r['secondview_ade2d_pixels'] for r in ok)} px  ade2d/diag "
                  f"{mean(r['secondview_ade2d_diag'] for r in ok)}  iou {mean(r['secondview_iou'] for r in ok)}"
                  f"  (input view: {mean(r['input_ade2d_pixels'] for r in ok)} px, iou {mean(r['input_iou'] for r in ok)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench", type=Path, help="benchmark directory")
    ap.add_argument("--metrics", nargs="+", choices=METRICS, default=list(METRICS))
    ap.add_argument("--tasks", nargs="+", default=[], metavar="ID_PREFIX", help="job id prefixes (default: all tasks)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--egl-devices", default="", help="comma-separated EGL device indices, assigned round robin "
                    "(default: MuJoCo's default device)")
    ap.add_argument("--videos", action="store_true", help="secondview: also write jobs/<id>/{gt,pred}.mp4")
    ap.add_argument("--selftest", action="store_true", help="secondview: use the input camera instead (must reproduce "
                    "v2's input-view ADE / IoU); writes evaluation/secondview_selftest/")
    a = ap.parse_args()
    root = a.bench.resolve()
    tasks = [t for t in read_json(root / "benchmark.json")["tasks"]
             if not a.tasks or any(t["job_id"].startswith(p) for p in a.tasks)]
    devices = [int(d) for d in a.egl_devices.split(",") if d.strip()] or [None]
    dev = lambda i: devices[i % len(devices)]
    done = {}
    if "v2" in a.metrics:
        done["v2"] = run("v2", v2.evaluate_job, [(root, t, dev(i)) for i, t in enumerate(tasks)], a.workers,
                         lambda r: r.get("metrics", {}).get("moving_visible_ade3d_m"))
        v2.write_csv(root, all_rows(root, v2.OUT))
    if "physics" in a.metrics:
        for t in tasks:     # GT once, serially (shared cache)
            physics.gt_cached(root, t)
        done["physics"] = run("physics", physics.evaluate_job, [(root, t) for t in tasks], a.workers, lambda r: r.get("accuracy"))
        for r in done["physics"]:
            save_json(root / physics.OUT / "jobs" / r["job_id"] / "metrics.json", r)
        physics.write_csv(root, all_rows(root, physics.OUT))
    if "secondview" in a.metrics:
        out = secondview.OUT_SELFTEST if a.selftest else secondview.OUT
        done["secondview"] = run("secondview", secondview.evaluate_job,
                                 [(root, t, dev(i), a.videos, a.selftest) for i, t in enumerate(tasks)], a.workers,
                                 lambda r: r.get("secondview_ade2d_pixels"))
        for r in done["secondview"]:
            save_json(root / out / "jobs" / r["job_id"] / "metrics.json", r)
        secondview.write_csv(root, all_rows(root, out), a.selftest)
    print(f"\n{root.name}: {len(tasks)} tasks (means over evaluated jobs)")
    for name, rows in done.items():
        summary(name, rows)


if __name__ == "__main__":
    main()
