#!/usr/bin/env python3
"""Physion v1.5 screening, stages in order (intermediate files in --work):

  a      --data-root D   screen_a_physion.json (Stage A, common.py) for every HDF5, 0.01 s per frame
  cands                  physion_cands.json: per wanted subtype (WANT, 1-4 per category) the 3 best scenes without
                         an energy rise or restitution > 0.95, ranked by GT item kinds (decel > 0.6 g is a collision)
  sheets [--add S ...]   sheets/phy_NN.jpg (+ phy_items.json): the first candidate of each subtype (+ added scenes)
  b      [--all]         stage_b_physion_picks.json: the Physion picks of --picks (or every Stage A scene:
                         stage_b_physion.json): per mover the cam0 visible pixels in every frame (_id_cam0), hull
                         volume from the mesh x scale -> density = mass / volume, push time, room, distractors

usage: python benchmark/screen/physion.py STAGE [--data-root D] --work W [--workers N] [--only NAME ...]
"""
import glob
import io
import json
import re
import traceback
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import ConvexHull

from common import cli, dump, failed, load, pmap, stage_a, wanted

WANT = {   # subtypes per category picked for diversity (beside the 11 Physion scenes kept from the earlier set)
    "collide": ["collision_tiny_ball_box", "collision_yeet_tdw_1_dis_1_occ", "collision_simple_box", "collision_assorted_targets_box"],
    "contain": ["containment-cylinder-bowl", "containment-multi-bowl", "containment-torus", "containment-vase"],
    "dominoes": ["dominoes_1mid_J025R45_boxroom", "dominoes_4midRM1_tdwroom_2", "dominoes_0mid_d3chairs_o1plants_tdwroom", "dominoes_default_boxroom"],
    "drop": ["drop_sizes_box", "drop_sidezone_box", "drop_simple_box", "drop_all_bowls_box"],
    "link": ["linking_nl2-3_mg01_aCone_bCyl_boxroom", "linking_nl1-8_mg000_aNone_bCyl_tdwroom_small_rings", "linking_nl1-8_mg000_aCylcap_bCyl_tdwroom1"],
    "roll": ["rollingSliding_simple_ramp_box_zone_away", "rollingSliding_simple_collision_tdw_1_dis_1_occ", "rollingSliding_simple_ramp_tdw_1_dis_1_occ", "rollingSliding_simple_ledge_box_sphere_small_zone"],
    "support": ["towers_nb5_fr015_SJ030_mono0_dis0_occ0_boxroom_unstable", "towers_nb3_fr015_SJ025_mono1_dis0_occ0_tdwroom_unstable", "towers_nb4_fr015_SJ000_gr01_mono0_dis1_occ1_tdwroom_unstable"]}


def screen(path):
    try:
        with h5py.File(path, "r") as f:
            frames = sorted(f["frames"].keys())
        t = np.arange(len(frames)) * 0.01
        task = {"job_id": "cand_" + Path(path).stem, "dataset": "physion_v15", "category": Path(path).parent.parent.name,
                "source": {"path": path, "camera": "cam0"}}
        return stage_a(task, frames, t, {})
    except Exception as e:
        return failed("physion_v15", path, e)


def subtype(o):
    return re.sub(r"_\d{4}$", "", Path(o["path"]).stem)


def ok(o):
    e = o["energy"]
    return not (e and e["significant"]) and not any(i["kind"] == "restitution" and i["value"] > 0.95 for i in o["items"])


def rank(o):
    k = {i["kind"] for i in o["items"] if not (i["kind"] == "decel" and i["value"] > 0.6)}
    return (len(k - {"mass_rel"}), len(o["items"]))


def stage_b(args):
    path, movers = args
    try:
        f = h5py.File(path, "r"); s = f["static"]
        ids = [int(i) for i in s["object_ids"][()]]; names = [x.decode() for x in s["model_names"][()]]
        cols = s["object_segmentation_colors"][()]; scale = s["scale"][()]
        F = f["frames"]; ks = sorted(F)
        masses = F[ks[0]]["objects/masses"][()]
        size = {}
        for i, oid in enumerate(ids):
            key = f"vertices_{i}"
            if key not in s["mesh"]:
                continue
            V = s["mesh"][key][()] * scale[i]
            try:
                vol = ConvexHull(V).volume
            except Exception:
                vol = None
            size[str(oid)] = {"name": names[i], "extent_m": np.ptp(V, 0).tolist(), "hull_m3": vol, "mass": float(masses[i]),
                              "density": float(masses[i]) / vol if vol else None}
        code = {str(oid): int(c[0]) * 65536 + int(c[1]) * 256 + int(c[2]) for oid, c in zip(ids, cols)}
        vis = {k: [] for k in movers}
        for k in ks:
            raw = np.array(Image.open(io.BytesIO(F[k]["images/_id_cam0"][()].tobytes())))[..., :3].astype(np.int64)
            raw = raw[..., 0] * 65536 + raw[..., 1] * 256 + raw[..., 2]
            u, c = np.unique(raw, return_counts=True); cnt = dict(zip(u.tolist(), c.tolist()))
            for m in movers:
                vis[m].append(int(cnt.get(code.get(m), 0)))
        res = raw.shape
        others = {str(oid): names[i] for i, oid in enumerate(ids)}
        extra = {k: (s[k][()].decode() if isinstance(s[k][()], bytes) else s[k][()].tolist()) for k in
                 ("probe_id", "target_id", "probe_type", "target_type", "push_time", "room", "distractors", "occluders") if k in s}
        return {"path": path, "vis": vis, "size": size, "res": list(res), "names": others,
                "static": {k: (v if not isinstance(v, list) else [x.decode() if isinstance(x, bytes) else x for x in v]) for k, v in extra.items()}}
    except Exception as e:
        return {"path": path, "error": repr(e), "tb": traceback.format_exc()[-800:]}


def tile(p, label, first=0):
    f = h5py.File(p, "r"); ks = sorted(f["frames"], key=int); ks = [k for k in ks if int(k) >= first]
    idx = [round(i * (len(ks) - 1) / 3) for i in range(4)]
    W = Image.new("RGB", (512, 530), "white"); d = ImageDraw.Draw(W)
    for j, i in enumerate(idx):
        im = Image.open(io.BytesIO(f["frames"][ks[i]]["images/_img_cam0"][()].tobytes())).convert("RGB").resize((256, 256))
        W.paste(im, ((j % 2) * 256, 18 + (j // 2) * 256))
    d.text((4, 2), label, fill="black")
    return W


def main(argv=None):
    a = cli(__doc__, ["a", "cands", "sheets", "b"], argv)
    W = a.work
    if a.stage == "a":
        paths = [p for p in sorted(glob.glob(str(a.data_root / "physion_v15/*/*/*.hdf5"))) if wanted(p, a.only)]
        out = pmap(screen, paths, a.workers)
        dump(W, "screen_a_physion.json", out)
        print(len(out), "errors", sum("error" in o for o in out))
        return
    O = [o for o in load(W, "screen_a_physion.json") if "error" not in o]
    if a.stage == "cands":
        cands = {}
        for cat, subs in WANT.items():
            for s in subs:
                L = sorted([o for o in O if subtype(o).endswith(s) and Path(o["path"]).parent.parent.name == cat and ok(o)], key=rank, reverse=True)
                cands[s] = [o["path"] for o in L[:3]]
        dump(W, "physion_cands.json", cands, indent=0)
        for s, ps in cands.items():
            print(s[:50], [Path(p).stem[-4:] for p in ps])
        return
    by_name = {Path(o["path"]).stem: o for o in O}
    if a.stage == "sheets":
        items = [(s, ps[0]) for s, ps in load(W, "physion_cands.json").items() if ps] + [(n, by_name[n]["path"]) for n in a.add]
        for g in range(0, len(items), 4):
            im = Image.new("RGB", (1024, 1060), "white")
            for j, (s, p) in enumerate(items[g:g + 4]):
                im.paste(tile(p, f"{g+j}: {s[:40]} {Path(p).stem[-4:]}"), ((j % 2) * 512, (j // 2) * 530))
            (W / "sheets").mkdir(exist_ok=True)
            im.save(W / f"sheets/phy_{g//4:02d}.jpg", quality=80)
        dump(W, "sheets/phy_items.json", items, indent=0)
        print(len(items), "scenes ->", W / "sheets")
        return
    if a.all:   # b
        todo, name = O, "stage_b_physion.json"
    else:
        picks = [s["sample_id"] for s in json.loads(a.picks.read_text(encoding="utf-8"))["scenes"] if s["dataset"] == "physion_v15"]
        todo, name = [by_name[n] for n in picks if wanted(n + ".hdf5", a.only)], "stage_b_physion_picks.json"
    todo = [o for o in todo if wanted(o["path"], a.only)]
    out = pmap(stage_b, [(o["path"], list(o["movers"])) for o in todo], a.workers)
    dump(W, name, out)
    for o, b in zip(todo, out):
        if "error" in b:
            print(Path(o["path"]).stem, b["error"]); continue
        vis = {k: round(sum(c >= 200 for c in v) / len(v), 2) for k, v in b["vis"].items()}
        dens = {k: round(b["size"][k]["density"], 1) for k in o["movers"] if k in b["size"] and b["size"][k]["density"]}
        dr = max(dens.values()) / min(dens.values()) if len(dens) > 1 else 1
        print(f"{Path(o['path']).stem[-45:]:45s} push={b['static'].get('push_time')} vis={vis} dens={dens} ratio={dr:.1f} "
              f"names={ {k: b['names'][k] for k in o['movers']} }")


if __name__ == "__main__":
    main()
