#!/usr/bin/env python3
"""Final step: the hand-picked scenes -> a spec in the bench50.json scene format, rebuilt from the screening results.

  python benchmark/screen/spec.py --data-root D --work W [--picks benchmark/bench50.json] [--out W/spec.json]

The picks are data: which screened candidates were chosen by hand after the contact sheets and listings, with their
job number, first frame and trim reason; they are the scene list of --picks (default: the published bench50.json).
Every other field is rebuilt from the work files: category and source (relative to D) from the scene's Stage A
record, why = its Stage A GT item counts. For each pick the automatic screening verdict is printed (PhysInOne: Stage A
survivor, semantic flags, cluster; Physion: no energy rise / restitution <= 0.95, candidate rank, cam0 visibility of
the movers from Stage B), then the rebuilt scenes are compared with --picks field by field.
"""
import argparse
import json
from pathlib import Path

from common import PICKS, item_counts, load
from physion import ok

FIELDS = ("job_id", "dataset", "category", "sample_id", "source", "first_frame", "why", "trim_reason")


def optional(work, name, default):
    return load(work, name) if (work / name).exists() else default


def sample_id(o):
    name = Path(o["path"]).name
    return name[:-len("_trajectory.zip")] if o["dataset"] == "physinone" else name[:-len(".hdf5")]


def screened(W):
    """the screening results a pick is checked against (missing files: not checked)"""
    return {"surv": {p for v in optional(W, "pio_stageA_survivors.json", {}).values() for p in v},
            "sem": optional(W, "pio_semantic.json", {}), "cl": optional(W, "pio_clusters.json", {}),
            "cands": optional(W, "physion_cands.json", {}),
            "vis": {b["path"]: b for b in optional(W, "stage_b_physion_picks.json", []) if "error" not in b}}


def verdict(o, V):
    p = o["path"]
    if o["dataset"] == "physinone":
        sem = [f for f in V["sem"].get(p, {}).get("flags", ["no semantic record"]) if not f.startswith("mass_unchecked")]
        return "; ".join(["Stage A survivor" if p in V["surv"] else "NOT A STAGE A SURVIVOR", f"semantic flags {sem}",
                          "cluster " + next((k for k, v in V["cl"].items() if p in v), "-")])
    out = ["no energy rise, e <= 0.95" if ok(o) else "ENERGY RISE OR e > 0.95"]
    out += [f"candidate {i + 1} of {s}" for s, ps in V["cands"].items() for i, q in enumerate(ps) if q == p] or ["not a candidate"]
    b = V["vis"].get(p)
    if b:
        out.append("cam0 visible frames " + ", ".join(f"{k}:{sum(c >= 200 for c in v) / len(v):.2f}" for k, v in b["vis"].items()))
    return "; ".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True, help="the raw data root the work files were made from")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--picks", type=Path, default=PICKS)
    ap.add_argument("--out", type=Path, help="default: WORK/spec.json")
    a = ap.parse_args(argv)
    W, root = a.work, a.data_root.absolute()
    ref = json.loads(a.picks.read_text(encoding="utf-8"))
    A = {(o["dataset"], sample_id(o)): o for name in ("screen_a_physinone.json", "screen_a_physion.json")
         for o in load(W, name) if "error" not in o}
    V, scenes, same = screened(W), [], 0
    for r in ref["scenes"]:
        o = A[(r["dataset"], r["sample_id"])]
        cat = Path(o["path"]).parent.name if o["dataset"] == "physinone" else Path(o["path"]).parent.parent.name
        s = {"job_id": f"{r['job_id'][:3]}_{o['dataset']}_{cat}_{r['sample_id']}", "dataset": o["dataset"], "category": cat,
             "sample_id": r["sample_id"], "source": str(Path(o["path"]).relative_to(root)),
             "first_frame": int(r.get("first_frame", 0)), "why": item_counts(o),
             **({"trim_reason": r["trim_reason"]} if r.get("trim_reason") else {})}
        scenes.append(s)
        diff = [f for f in FIELDS if s.get(f) != r.get(f)]
        same += not diff
        print(f"{s['job_id'][:3]} {r['sample_id'][:50]:50s} {verdict(o, V)}" + (f"  DIFFERS: {diff}" if diff else ""))
    out = a.out or W / "spec.json"
    out.write_text(json.dumps({"name": ref.get("name"), "criteria": ref.get("criteria"), "scenes": scenes},
                              indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(scenes)} scenes -> {out}; {same}/{len(scenes)} identical to {a.picks.name} in {', '.join(FIELDS)}")


if __name__ == "__main__":
    main()
