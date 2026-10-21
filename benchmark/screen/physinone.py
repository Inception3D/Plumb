#!/usr/bin/env python3
"""PhysInOne screening, stages in order (intermediate files in --work):

  scan     --data-root D   pio_meta.json: per zip the actors (moving channel, displacement), assets, recorder
                           components (mass, damping, physical material, stencil), frame counts, clip gravity
  a        --data-root D   screen_a_physinone.json (Stage A, common.py) on the retimed axis; flags (below) ->
                           pio_stageA_survivors.json: scenes without flags, an energy rise alone does not reject
  b                        stage_b_pio.json (survivors): per mover the CineCamera_0 visible pixels in every frame
                           and the size / convex-hull volume from the frame-0 depth + segmentation of 4 GT cameras
  semantic                 pio_semantic.json: GT mass vs real-world priors (density by material / object class x
                           hull volume, fixed real sizes); pio_clusters.json: survivors by subclass x background,
                           motor-driven Accel*Spin left out
  sheets                   sheets/pio_NN.jpg (+ reps.json): one representative per cluster, 4 frames each
  show SUB[,SUB] [--full]  one line (--full: one row per mover) per survivor, for picking scenes by hand

usage: python benchmark/screen/physinone.py STAGE [--data-root D] --work W [--workers N] [--only NAME ...]
"""
import collections
import glob
import io
import json
import math
import re
import traceback
import zipfile
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from common import build, cli, dump, failed, load, pmap, stage_a, wanted

G_PIO = 2.25     # PhysInOne GT gravity (m/s^2) when a clip has no clean free fall
CAMS = ["CineCamera_0", "CineCamera_3", "CineCamera_6", "CineCamera_9"]


def sub(p):
    return Path(p).name.split("__")[0]


# ---------------------------------------------------------------- scan

def scan(zp):
    zp = Path(zp); b = zp.name[:-4]
    try:
        z = zipfile.ZipFile(zp); names = z.namelist()
        d = json.loads(z.read(f"{b}/{b}.json")); st = json.loads(z.read(f"{b}/recorder_stats.json"))
    except Exception as e:
        return {"zip": str(zp), "error": repr(e)}
    rgb = sorted(n for n in names if "/CineCamera_0/rgb/" in n)
    seg = sorted(n for n in names if "/CineCamera_0/seg/" in n)
    actors = {}
    for name, a in (d.get("actors") or {}).items():
        series = [("actor", a.get("transform_data") or [])] + [(cn, c.get("transform_data") or []) for cn, c in (a.get("components") or {}).items()]
        best = None
        for cn, td in series:
            if not td:
                continue
            P = np.array([[x["transform"]["location"][k] / 100 for k in "xyz"] for x in td])
            disp = float(np.linalg.norm(P - P[0], axis=1).max())
            if best is None or disp > best[1]:
                best = (cn, disp, len(td), [float(x["time_seconds"]) for x in td[:2]], td[0]["transform"].get("scale"))
        actors[name] = {"type": a.get("actor_type"), "asset": a.get("asset_path"), "channel": best[0] if best else None,
                        "disp_m": best[1] if best else 0, "n": best[2] if best else 0, "scale": best[4] if best else None,
                        "components": list((a.get("components") or {}).keys())}
    comps = {}
    for group in ("solid_object", "interactable_object", "breakable_object"):
        for entry in st.get(group) or []:
            for name, v in entry.items():
                comps[name] = {"group": group, "tags": v.get("tags"), "components": [
                    {k: c.get(k) for k in ("component_name", "component_type", "mass", "mass_override", "linear_damping", "angular_damping",
                                          "physical_material_path", "custom_stencil_value", "mesh_path")} for c in v.get("components", [])]}
    g, n = build.physinone_g(zp)
    return {"zip": str(zp), "scene": b, "sub": b.split("__")[0], "bg": b.split("__")[1], "seq": d.get("sequence_info"),
            "n_rgb": len(rgb), "n_seg": len(seg), "first_rgb": rgb[0] if rgb else None, "actors": actors, "recorder": comps,
            "scene_location_info": st.get("scene_location_info"), "summary": st.get("summary"), "g_src": g, "g_windows": n}


# ---------------------------------------------------------------- Stage A + flags

def screen(m):
    path = m["zip"]
    try:
        n = m["n_rgb"]
        if n == 0:
            return {"dataset": "physinone", "path": path, "error": "no CineCamera_0 rgb frames"}
        g_src = m["g_src"] if m["g_src"] and abs(m["g_src"] - G_PIO) < 0.225 else G_PIO
        src_dt = 1 / float(m["seq"]["frame_rate"])
        dt = src_dt / math.sqrt(9.81 / g_src)
        frames = list(range(n)); t = np.arange(n) * dt
        task = {"job_id": "cand_" + m["scene"], "dataset": "physinone", "category": Path(path).parent.name,
                "source": {"path": path, "camera": "CineCamera_0"}}
        return stage_a(task, frames, t, {"g_src_measured": m["g_src"], "g_windows": m["g_windows"], "slowdown": src_dt / dt})
    except Exception as e:
        return failed("physinone", path, e)


def flags(o, m):
    """physical-plausibility and evaluability flags of a Stage A record (m: its pio_meta record)"""
    f = []
    if "error" in o:
        return ["unevaluable:" + ("no video" if "rgb" in o["error"] else "no GT trajectories")]
    if o["g_src_measured"] and abs(o["g_src_measured"] - G_PIO) > 0.225: f.append(f"g_inconsistent:{o['g_src_measured']:.2f}")
    if not o["movers"]: f.append("unevaluable:no motion")
    for k, mv in o["movers"].items():
        r = m["recorder"].get(k)
        if r:
            cs = [c for c in r["components"] if c["mass"] is not None]
            ld = max([c["linear_damping"] or 0 for c in cs] or [0]); ad = max([c["angular_damping"] or 0 for c in cs] or [0])
            if ld > 0.05: f.append(f"damping_lin:{k}:{ld:g}")
            if ad > 0.5: f.append(f"damping_ang:{k}:{ad:g}")
            if cs and all((c["mass"] or 0) <= 0 for c in cs): f.append(f"mass0:{k}")
        d = mv.get("drag")
        if d and d["ratio"] is not None and d["ratio"] < 0.85: f.append(f"drag:{k}:{d['ratio']:.2f}")
    for k, v in o["stop_g"].items():
        if v > 1.0: f.append(f"decel>1g:{k}:{v:.2f}")
    for it in o["items"]:
        if it["kind"] == "restitution" and it["value"] > 0.95: f.append(f"e>0.95:{it['obj']}:{it['value']:.2f}")
    if o["energy"] and o["energy"]["significant"]: f.append(f"energy_rise:{o['energy']['ratio']:.2f}")
    nk = sum(it["kind"] != "mass_rel" for it in o["items"]); nm = sum(it["kind"] == "mass_rel" for it in o["items"])
    if nk == 0 and nm < 2: f.append("unevaluable:no physics item")
    return f


def flag_summary(O, M):
    by = collections.defaultdict(list)
    for o in O:
        by[sub(o["path"])].append((o, flags(o, M[o["path"]])))
    for s, L in sorted(by.items()):
        c = collections.Counter(x.split(":")[0] for _, fl in L for x in set(y.split(":")[0] + (":" + y.split(":")[1] if y.startswith("unevaluable") else "") for y in fl))
        clean = [o for o, fl in L if not fl]
        kinds = collections.Counter(it["kind"] for o in clean for it in o["items"])
        print(f"{s:22s} n={len(L):3d} clean={len(clean):3d} {dict(c)}\n      clean items {dict(kinds)}")


# ---------------------------------------------------------------- Stage B

def npz(z, name, key):
    with np.load(io.BytesIO(z.read(name))) as p:
        return p[key]


def stage_b(job):
    """mover visibility in CineCamera_0 over all video frames, and mover size / hull volume from frame-0 depth + seg
    of 4 static cameras (Blender c2w, z-depth metres) -> effective density = mass / hull volume"""
    o, m = job; path = o["path"]
    try:
        b = m["scene"]
        z = zipfile.ZipFile(path); names = set(z.namelist())
        stencil = {}
        for k in o["movers"]:
            r = m["recorder"].get(k)
            if r:
                stencil[k] = sorted({int(c["custom_stencil_value"]) for c in r["components"] if (c.get("custom_stencil_value") or 0) > 0})
        vis = {k: [] for k in stencil}
        for i in range(o["frames"]):
            seg = npz(z, f"{b}/CineCamera_0/seg/{i:04d}.npz", "seg") if f"{b}/CineCamera_0/seg/{i:04d}.npz" in names else None
            if seg is None:
                for k in vis: vis[k].append(-1)
                continue
            u, c = np.unique(seg.astype(np.int64), return_counts=True); cnt = dict(zip(u.tolist(), c.tolist()))
            for k, ids in stencil.items():
                vis[k].append(int(sum(cnt.get(s, 0) for s in ids)))
        pts = {k: [] for k in stencil}
        for cam in CAMS:
            jn = f"{b}/blender_{cam}.json"
            if jn not in names or f"{b}/{cam}/depth/0000.npz" not in names:
                continue
            cj = json.loads(z.read(jn)); W, H = cj["img_w"], cj["img_h"]; f = W / 2 / np.tan(cj["camera_angle_x"] / 2)
            T = np.array(cj["frames"][0]["transform_matrix"])
            d = npz(z, f"{b}/{cam}/depth/0000.npz", "depth"); seg = npz(z, f"{b}/{cam}/seg/0000.npz", "seg").astype(np.int64)
            for k, ids in stencil.items():
                vv, uu = np.nonzero(np.isin(seg, ids))
                if len(uu) < 30:
                    continue
                sel = np.random.default_rng(0).choice(len(uu), min(len(uu), 4000), replace=False)
                uu, vv = uu[sel] + .5, vv[sel] + .5; dd = d[vv.astype(int), uu.astype(int)]
                ok = np.isfinite(dd) & (dd > 0) & (dd < 1e3)
                X = np.stack([(uu - W / 2) / f * dd, -(vv - H / 2) / f * dd, -dd], 1)[ok]
                pts[k].append(X @ T[:3, :3].T + T[:3, 3])
        size = {}
        for k, L in pts.items():
            if not L:
                continue
            P = np.concatenate(L)
            if len(P) < 50:
                continue
            c = P - P.mean(0); _, _, Vt = np.linalg.svd(c, full_matrices=False); ext = np.ptp(c @ Vt.T, 0)
            try:
                vol = ConvexHull(P).volume
            except Exception:
                vol = None
            size[k] = {"extent_m": ext.tolist(), "hull_m3": vol, "views": len(L)}
        return {"path": path, "vis": vis, "size": size}
    except Exception as e:
        return {"path": path, "error": repr(e), "tb": traceback.format_exc()[-800:]}


# ---------------------------------------------------------------- semantic: real-world priors
# plausible effective (hull) density by material / object class, and the typical size of objects whose size is fixed
# in reality; used only to flag implausible GT
MAT = [("gold", (18000, 19500)), ("silver", (9500, 10600)), ("copper", (8000, 9000)), ("iron", (7000, 7900)),
       ("steel", (7000, 8000)), ("alumin", (2500, 2800)), ("glass", (2300, 2700)),
       ("stone", (1500, 2900)), ("granite", (1500, 2900)), ("marble", (1500, 2900)), ("concrete", (1500, 2500)),
       ("brick", (1400, 2200)), ("plaster", (900, 1800)), ("resin", (1000, 1400)), ("rubber", (900, 1300)),
       ("plastic", (800, 1500)), ("leather", (700, 1000)), ("paper", (400, 1200)), ("wood", (350, 900)),
       ("bamboo", (300, 800)), ("fabric", (100, 700)), ("wool", (50, 400)), ("light", (50, 400))]
CLASS = [  # (regex on asset name, (rho_lo, rho_hi) or None, typical max extent m or None)
    (r"golf", (900, 1300), 0.043), (r"baseball", (500, 900), 0.075), (r"basket_?ball|basketballball", (20, 200), 0.24),
    (r"volley", (20, 200), 0.21), (r"foot_?ball|soccer", (20, 200), 0.22), (r"beach_?ball", (5, 150), None),
    (r"toy_?ball|gachapon|^ball$|old_toy_ball", (20, 250), None), (r"paperball", (20, 300), None),
    (r"pill?ow", (15, 200), None), (r"balloon", (0.5, 30), None), (r"suitcase|maleta|handbag", (40, 450), None),
    (r"rubik", (300, 700), 0.057), (r"cola_can|_can$", (40, 1100), 0.12), (r"bottle", (100, 1100), None),
    (r"cup|mug|glass_", (80, 900), None), (r"toaster", (150, 700), None), (r"flashlight", (300, 1500), None),
    (r"chair", (15, 250), None), (r"laptop", (250, 900), None), (r"umbrella", (10, 150), None),
    (r"lantern", (50, 600), None), (r"skateboard", (100, 600), None), (r"chopping_board", (400, 1100), None),
    (r"cannon", (5000, 8000), None), (r"stone|rock|granite|cliff|skull", (1000, 3000), None),
    (r"box|crate|case|cage", (15, 700), None), (r"mirror", (150, 1500), None)]
PRIMITIVE = re.compile(r"^(Cube|Cube_correct|Cylinder_correct|Sphere)$")


def prior(asset, mat):
    a = asset or ""
    if PRIMITIVE.match(a):
        m = (mat or "").lower()
        for k, rng in MAT:
            if k in m:
                return rng, None, "material:" + k
        return None, None, "primitive, unknown material"
    for rx, rng, size in CLASS:
        if re.search(rx, a, re.I):
            return rng, size, "class:" + rx
    return None, None, "no prior"


def factor(dens, rng):
    if dens is None or rng is None:
        return None
    lo, hi = rng
    return dens / lo if dens < lo else dens / hi if dens > hi else 1.0


def mover_rows(o, b, m):
    rows = []
    for k, mv in o["movers"].items():
        v = b.get("vis", {}).get(k, []); sz = b.get("size", {}).get(k)
        r = m["recorder"].get(k, {}); cs = [c for c in r.get("components", []) if c["mass"] is not None]
        mat = ",".join(sorted({(c["physical_material_path"] or "").split("/")[-1].split(".")[0].replace("PM_M_", "").replace("_NEW", "")[:18] for c in cs}))
        asset = (m["actors"][k]["asset"] or "").split("/")[-1].split(".")[0][:22]
        mass = mv["mass"]; vol = sz and sz["hull_m3"]
        rows.append({"obj": k, "asset": asset, "mat": mat, "mass": mass, "override": any(c["mass_override"] for c in cs),
                     "ext": [round(e, 3) for e in sz["extent_m"]] if sz else None, "hull": vol,
                     "dens": (mass / vol) if (mass and vol) else None,
                     "vis_frac": (sum(c >= 200 for c in v) / len(v)) if v else None, "vis0": v[0] if v else None,
                     "disp": mv["disp"], "rot": mv["rot_deg"]})
    return rows


def semantic(o, b, m):
    """flags: size (fixed-size object 0.5-2x off), mass_ratio (colliding movers' GT mass / prior differ > 3x),
    mass_unchecked (informational: movers without a prior)"""
    rows = mover_rows(o, b, m)
    inter = set() if sub(o["path"]) == "FrictionStop" else {r["obj"] for r in rows}
    fl = []; fs = []
    for r in rows:
        rng, size, why = prior(r["asset"], r["mat"])
        f = factor(r["dens"], rng); r["prior"] = why; r["f"] = f
        if size and r["ext"]:
            s = max(r["ext"]) / size; r["size_ratio"] = s
            if not 0.5 <= s <= 2.0: fl.append(f"size:{r['asset']}:{s:.1f}x")
        if r["obj"] in inter and f is not None: fs.append((f, r["asset"]))
    if len(fs) >= 2:
        hi = max(fs); lo = min(fs)
        if hi[0] / lo[0] > 3: fl.append(f"mass_ratio:{hi[1]}/{lo[1]}:{hi[0]/lo[0]:.0f}x")
    unk = [r["asset"] for r in rows if r["obj"] in inter and r["f"] is None]
    if unk and len(inter) >= 2: fl.append(f"mass_unchecked:{','.join(sorted(set(unk)))[:60]}:{len(unk)}/{len(inter)}")
    return {"flags": fl, "rows": rows}


def clean(s):
    return not [f for f in s["flags"] if not f.startswith("mass_unchecked")]


# ---------------------------------------------------------------- contact sheets, listing

def tile(p, n, label):
    from PIL import Image, ImageDraw
    b = Path(p).name[:-4]; z = zipfile.ZipFile(p)
    W = Image.new("RGB", (512, 530), "white"); d = ImageDraw.Draw(W)
    for j, i in enumerate(round(i * (n - 1) / 3) for i in range(4)):
        im = Image.open(io.BytesIO(z.read(f"{b}/CineCamera_0/rgb/{i:04d}.jpg"))).convert("RGB").resize((256, 256))
        W.paste(im, ((j % 2) * 256, 18 + (j // 2) * 256))
    d.text((4, 2), label, fill="black")
    return W


def line(o, s):
    rows = s["rows"]
    items = collections.Counter(i["kind"] for i in o["items"])
    vis = min([r["vis_frac"] for r in rows if r["vis_frac"] is not None] or [0])
    v0 = min([r["vis0"] for r in rows if r["vis0"] is not None] or [0])
    on = min([v["onset_t"] for v in o["objects"].values() if v["onset_t"] is not None] or [9])
    assets = ",".join(sorted({f"{r['asset'][:14]}({r['mass'] and round(r['mass'],1)}kg,{r['ext'] and round(max(r['ext']),2)}m)" for r in rows}))[:150]
    return f"{Path(o['path']).name[:-15]:38s} {o['duration']:.1f}s on={on:.2f} vis={vis:.2f} v0={v0:6d} {dict(items)} {s['flags']} | {assets}"


def main(argv=None):
    a = cli(__doc__, ["scan", "a", "b", "semantic", "sheets", "show"], argv)
    W = a.work
    if a.stage == "scan":
        zs = [z for z in sorted(glob.glob(str(a.data_root / "physinone/*/*.zip"))) if wanted(z, a.only)]
        out = pmap(scan, zs, a.workers)
        dump(W, "pio_meta.json", out)
        print(len(out), "zips,", sum("error" in o for o in out), "unreadable")
        return
    M = {m["zip"]: m for m in load(W, "pio_meta.json")}
    if a.stage == "a":
        out = pmap(screen, [M[p] for p in sorted(M) if wanted(p, a.only)], a.workers)
        dump(W, "screen_a_physinone.json", out)
        print(len(out), "errors", sum("error" in o for o in out))
        flag_summary(out, M)
        surv = collections.defaultdict(list)
        for o in out:
            if not [f for f in flags(o, M[o["path"]]) if not f.startswith("energy_rise")]:
                surv[sub(o["path"])].append(o["path"])
        dump(W, "pio_stageA_survivors.json", surv, indent=0)
        print("survivors", {k: len(v) for k, v in surv.items()}, sum(len(v) for v in surv.values()))
        return
    A = {o["path"]: o for o in load(W, "screen_a_physinone.json")}
    surv = load(W, "pio_stageA_survivors.json")
    paths = [p for p in sorted(p for v in surv.values() for p in v) if wanted(p, a.only)]
    if a.stage == "b":
        out = pmap(stage_b, [(A[p], M[p]) for p in paths], a.workers)
        dump(W, "stage_b_pio.json", out)
        print(len(out), "errors", sum("error" in x for x in out))
        return
    if a.stage == "semantic":
        B = {x["path"]: x for x in load(W, "stage_b_pio.json")}
        out = {p: semantic(A[p], B.get(p, {}), M[p]) for s in surv for p in surv[s] if wanted(p, a.only)}
        dump(W, "pio_semantic.json", out)
        cnt = collections.Counter((sub(p), f.split(":")[0]) for p, v in out.items() for f in v["flags"])
        for s in sorted(surv):
            ps = [p for p in surv[s] if p in out]
            print(f"{s:22s} n={len(ps):3d} semantic-clean={sum(clean(out[p]) for p in ps):3d}", {k[1]: v for k, v in cnt.items() if k[0] == s})
        cl = collections.defaultdict(list)
        for s, ps in surv.items():
            if not s.startswith("Accel"):   # motor-accelerated spins are not passive
                for p in ps:
                    cl[f"{s}__{Path(p).name.split('__')[1]}"].append(p)
        dump(W, "pio_clusters.json", cl, indent=0)
        print(len(cl), "clusters (subclass x background)")
        return
    S = load(W, "pio_semantic.json")
    if a.stage == "sheets":
        from PIL import Image
        reps = []
        for key, ps in sorted(load(W, "pio_clusters.json").items()):
            ok = [p for p in ps if clean(S[p])] or ps
            reps.append((key, max(ok, key=lambda q: len({i["kind"] for i in A[q]["items"]}))))
        dump(W, "sheets/reps.json", reps, indent=0)
        for g in range(0, len(reps), 4):
            im = Image.new("RGB", (1024, 1060), "white")
            for j, (key, p) in enumerate(reps[g:g + 4]):
                im.paste(tile(p, A[p]["frames"], f"{g+j}: {Path(p).name[:-15]}"), ((j % 2) * 512, (j // 2) * 530))
            im.save(W / f"sheets/pio_{g//4:02d}.jpg", quality=80)
        print(len(reps), "representatives ->", W / "sheets")
        return
    for s in (a.subclasses or "").split(","):    # show
        print("=====", s)
        for p in surv.get(s, []):
            if p in S and wanted(p, a.only):
                print(line(A[p], S[p]))
                if a.full:
                    print("     ", {i["kind"] + ":" + i["obj"][:10]: round(i["value"], 2) for i in A[p]["items"] if i["kind"] != "mass_rel"})
                    for r in S[p]["rows"]:
                        print(f"      {r['obj'][:22]:22s} {r['asset'][:20]:20s} {r['mat'][:16]:16s} m={r['mass'] if r['mass'] is None else round(r['mass'],2)} "
                              f"ext={r['ext']} rho={r['dens'] and round(r['dens'])} f={r['f'] and round(r['f'],1)} vis={r['vis_frac'] and round(r['vis_frac'],2)} "
                              f"v0={r['vis0']} disp={r['disp']:.2f}")


if __name__ == "__main__":
    main()
