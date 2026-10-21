#!/usr/bin/env python3
"""GT-side evaluability check of a built benchmark directory (BENCH from benchmark/build.py; no reconstruction needed).

  python benchmark/screen/gt_check.py BENCH [--workers N] [--table FILE]

Per task: evaluator v2's GT objects, their motion class (moving = > 2 cm or > 5 deg, as v2 scores them) and visible
area per frame (>= 200 px at 1024 x 1024) -> BENCH/evaluation/gt_check.json. Also fills v2's GT statistics cache
(BENCH/cache/v2/<job>.npz, the format evaluation/v2.py score_masked reads), so later evaluations skip it.
--table FILE ('-' = stdout): the per-scene Markdown table: GT physical-quantity items (physics_v6) on the objects v2
scores as moving and visible (mass_rel counted when >= 2), moving / visible (>= 50 % of the frames) object counts.
"""
import argparse
import collections
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import physics as ep  # noqa: E402
from evaluation import v2 as ev  # noqa: E402
from evaluation.common import read_json, save_json  # noqa: E402
from evaluation.gt import source_rows  # noqa: E402

DESC = {
    "010": "double pendulum: two lanterns hanging on chains swing", "029": "a wooden box is thrown off a platform into a suitcase (which barely moves)",
    "030": "three cardboard boxes slide on carpet lanes and stop", "038": "a filing box is thrown, lands, slides and stops",
    "101": "two cushions thrown at a sofa", "102": "two cushions thrown at a sofa (other cushions)",
    "103": "an office chair slides into a wall and bounces back", "104": "a volleyball and a wine bottle roll towards each other",
    "105": "a wooden box hits a suitcase and pushes it away", "106": "a stone ball and a flashlight thrown, landing and bouncing",
    "107": "a stone ball and a flashlight thrown (other masses)", "108": "a laptop falls and lands flat",
    "109": "a ball rolls down a ramp and slows down on the floor", "110": "a pen rolls off a tray",
    "111": "a paper ball rolls down a white ramp, bounces and slows down", "112": "a log on a swing frame swings after its release",
    "113": "a hanging chair swings", "114": "two small balls hanging under a table (double pendulum)",
    "115": "three wooden boxes slide on lanes of three materials and stop", "116": "a bowl-shaped turntable spins at a constant rate, a ball inside",
    "117": "a panel spinning about a vertical axis", "118": "a white turntable with holes spins at a constant rate",
    "119": "a wall of woven blocks and a lighter collapses, the blocks bounce", "120": "a volleyball rolls under a table, hits a leg and bounces back",
    "121": "a board and sticks leaning on a panel fall and slide to a stop",
    "001": "a ball knocks over a post with rings", "005": "a torus hits a pyramid", "006": "an octahedron drops into a bowl and tips it over",
    "012": "a ball knocks down a 4-level tower", "016": "a ball drops into a bowl", "019": "a ball hits a stable 2-level tower",
    "025": "a ball knocks over a long post with rings", "028": "a block pushed towards the target zone knocks a tube away",
    "032": "a ball rolls over a step", "033": "a tube rolls", "036": "a cube drops into a cylinder",
    "122": "a small ball knocks over a long block", "123": "a ball hits a torus, which slides into the target zone",
    "124": "a ball knocks over a bowl-and-cylinder tower", "125": "a ball knocks over a stack of bowls and balls",
    "126": "a vase is knocked over, the balls on it scatter", "127": "a chain of 6 dominoes falls", "128": "a domino chain falls",
    "129": "two dominoes and distractors", "130": "a cone drops onto a bowl", "131": "a ball drops onto a cylinder top and bounces off",
    "132": "a ball hits rings on a cone stand", "133": "a ball rolls down a ramp", "134": "a ball pushes a bowl",
    "135": "a ball knocks down a 5-level tower"}


def check(root, task):
    metadata = root / "meta" / task["job_id"]
    truth = ev.MaskTruth(task, metadata); depth = ev.GTDepth(task); rows = source_rows(metadata)
    try:
        ng = len(truth.objects)
        gpos, gproj, classes, defs, _ = ev.gt_poses(task, metadata, truth.objects)
        cache = root / ev.GT_CACHE / f"{task['job_id']}.npz"; gst = None
        if cache.exists():
            with np.load(cache) as c:
                gst = c["gt_stats"] if c["gt_stats"].shape[:2] == (len(rows), ng) else None
        build = gst is None
        gst = np.zeros((len(rows), ng, 6)) if build else gst; grid = None
        for t in range(len(rows) if build else 0):
            gl = truth.frame(t); P = gproj[t]
            if grid is None or not np.allclose(P, grid[0]):
                grid = (P.copy(), *ev.backproject_grid(P, gl.shape[1], gl.shape[0]))
            gst[t] = ev.label_stats(gl, depth.frame(rows[t]["source_frame"]), ng, grid[1], grid[2])
        if build:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_suffix(f".{os.getpid()}.tmp.npz"); np.savez_compressed(tmp, gt_stats=gst); os.replace(tmp, cache)
        gl = truth.frame(0); area = gst[:, :, 0]; px = gl.shape[0] * gl.shape[1]
        objs = []
        for g, o in enumerate(truth.objects):
            vis = area[:, g] >= 200 * px / (1024 * 1024)
            objs.append({"source_id": str(o["source_id"]), "name": o["name"], "class": classes[g]["class"],
                         "visible_frac": float(vis.mean()), "first_visible": int(np.argmax(vis)) if vis.any() else None})
        moving = [o for o in objs if o["class"] == "moving"]
        return {"job_id": task["job_id"], "objects": len(objs), "moving": moving,
                "moving_visible": sum(o["visible_frac"] >= 0.5 for o in moving), "status": "ok"}
    except Exception as e:  # noqa: BLE001
        return {"job_id": task["job_id"], "status": f"error: {e!r}"}
    finally:
        truth.close(); depth.close()


def table(root, tasks, chk):
    rows, kinds, kinds_e = [], collections.Counter(), collections.Counter()
    for t in tasks:
        jid = t["job_id"]; g = ep.gt_cached(root, t); c = chk.get(jid, {})
        moving = {m["source_id"]: m for m in c.get("moving", [])}
        elig = [i for i in g["items"] if i["obj"] in moving and moving[i["obj"]]["visible_frac"] > 0]
        kinds.update(i["kind"] for i in g["items"])
        k = collections.Counter(i["kind"] for i in elig)
        if k["mass_rel"] < 2:
            k.pop("mass_rel", None)
        kinds_e.update(k)
        sub = t["sample_id"].split("__")[0] if t["dataset"] == "physinone" else t["category"]
        rows.append(f"| {jid[:3]} | {t['dataset']} | {sub} | {DESC.get(jid[:3], '')} | {', '.join(f'{a} {b}' for a, b in sorted(k.items()))} | "
                    f"{len(moving)}/{c.get('moving_visible', '?')} | {t.get('first_frame') or ''} |")
    return (f"GT physical-quantity items (physics_v6) on the objects evaluator v2 scores as moving and visible (mass_rel "
            f"only with >= 2 such objects): {sum(kinds_e.values())} items {dict(kinds_e)} ({sum(kinds.values())} before "
            f"this filter). Moving / visible: v2's moving GT objects / those visible (>= 200 px) in >= 50 % of the frames.\n\n"
            "| job | dataset | subclass | scene | GT items | moving / visible | first frame |\n|---|---|---|---|---|---|---|\n"
            + "\n".join(rows) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench", type=Path)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--table", help="write the per-scene Markdown table to this file ('-': stdout)")
    a = ap.parse_args(argv)
    root = a.bench.resolve()
    tasks = read_json(root / "benchmark.json")["tasks"]
    with ProcessPoolExecutor(a.workers) as ex:
        out = list(ex.map(check, [root] * len(tasks), tasks))
    save_json(root / "evaluation/gt_check.json", out)
    for r in out:
        print(r["job_id"][:60], r["status"], r.get("objects"), "moving", len(r.get("moving", [])), "visible", r.get("moving_visible"))
    if a.table:
        text = table(root, tasks, {r["job_id"]: r for r in out})
        sys.stdout.write(text) if a.table == "-" else Path(a.table).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
