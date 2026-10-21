"""Roll out a MuJoCo scene on the input video's frame times and compare it with the tracks (measurements only).

The rollout is the loop of the output/run.py template: initial state set once, then mj_step until the first step
at or after each input frame time (decoded PTS, analysis/timeline.json). Any scene.xml works (no spec needed).

  python -m r2s check rollout [--scene output/scene.xml] [--state output/initial_state.json]
        -> analysis/sim_rollout.npz: frame_times, sim_times, body_names, body_xpos (T,nbody,3), body_xquat, qpos
  python -m r2s check compare [--scene S] [--state J] [--camera NAME] [--camera-track FILE|none]
                              [--link TRACK=BODY[,BODY...] ...] [--tracks a b ...] [--every K] [-o analysis/compare.json]
        renders the scene through camera NAME (default: its only camera, else video_cam) on the track frames
        (every K-th) and compares each track with the rendered mask of its bodies. Bodies of a track: the body
        with the track's name unless --link says otherwise; a body contributes its own geoms (not its child
        bodies); MuJoCo default geom groups 0-2 are drawn, geoms with alpha 0 are not. Per track: centroid
        distance in px (mean, first, 1st / 2nd half, last, worst@frame), frames where the bodies are not
        visible, mask IoU with the SAM3 mask (tracks made from SAM3); where analysis/pi3x_track_<sam3 name>.json
        exists (`r2s pi3x track`): on the Pi3X sample frames, the rendered visible-surface centroid vs Pi3X's in
        the camera frame: depth ratio sim / Pi3X and 3D distance after one depth ratio common to all tracks.
  python -m r2s check overlay [--scene S] [--state J] [--camera NAME] [--camera-track FILE|none] [--frames 0 60 ...]
                              [--cols N] [-o analysis/overlay.jpg]
        rendered outlines (one colour per body, legend printed) and track centroids (white) on the video frames.
  python -m r2s check members --sam3 analysis/sam3_dominoes [--scene S] [--state J] [--camera NAME]
                              [--camera-track FILE|none] [--bodies GLOB ...] [-o analysis/members_<name>.json]
        a `gpu members` result vs the scene on its frame: detected members and rendered bodies (bodies matching
        --bodies, default: bodies with >= 50 % of their visible pixels on or next to the members) paired one to
        one by centroid distance (within half the members' spacing): matched (px, IoU) / members with no body /
        visible bodies with no member / bodies not visible; on the other listed frames the IoU of the union
        masks (bodies paired on the members' frame vs the detections). Image
        analysis/members_<name>.jpg: green = matched, red = member without body, blue = body without member.
  compare, overlay and members move the camera along camera_track.json next to the scene when it exists.
  --link accepts body name patterns (fnmatch): --link dominoes_union='domino_*'.

Python (PYTHONPATH contains the task directory):
  from r2s.sim import rollout, compare
  r = rollout("output/scene.xml", state=None, times=None, camera=None, render=None, on_frame=None, camera_track="auto")
      r["times"], r["sim_times"] (T,), r["body_names"], r["xpos"] (T,nbody,3) world, r["xquat"], r["qpos"],
      r["uv"] (T,nbody,2) body origins projected through the camera (no rendering), r["camera"], r["warning"].
      render = frame indices to render; on_frame(k, body_id_image (H,W) int, depth (H,W) float) is called for each.
  c = compare("output/scene.xml", links={"ball": ["ball"]}, every=1)   -> the dict saved by `check compare`
"""
from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path

import numpy as np

from .common import ANALYSIS, OUTPUT, Camera, load_camera_track, load_json, save_json, set_camera_pose, timeline

os.environ.setdefault("MUJOCO_GL", "egl")


def _alpha(model, g):
    rgba = model.geom_rgba[g]
    m = int(model.geom_matid[g])
    if m >= 0 and np.allclose(rgba, [.5, .5, .5, 1]):
        return float(model.mat_rgba[m][3])
    return float(rgba[3])


def scene_camera(model, name=None):
    import mujoco
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
    if name:
        if name not in names:
            raise SystemExit(f"no camera {name!r} in the scene (cameras: {names or 'none'})")
        return name
    if len(names) == 1:
        return names[0]
    if "video_cam" in names:
        return "video_cam"
    raise SystemExit(f"scene has cameras {names or 'none'}: give --camera NAME")


def load_state(scene, state):
    if state is None:
        state = Path(scene).parent / "initial_state.json"
    if isinstance(state, (str, Path)):
        state = json.loads(Path(state).read_text())
    return state


def _camera_track(scene, camera_track, n):
    """None, or (file, pos, quat): camera_track "auto" = camera_track.json next to the scene if it exists."""
    if camera_track in (None, "none"):
        return None
    f = Path(scene).parent / "camera_track.json" if camera_track == "auto" else Path(camera_track)
    if camera_track == "auto" and not f.exists():
        return None
    try:
        _, pos, quat = load_camera_track(f, n)
    except (OSError, ValueError, KeyError) as e:
        raise SystemExit(f"camera track: {e}")
    return f, pos, quat


def rollout(scene=OUTPUT / "scene.xml", state=None, times=None, camera=None, render=None, on_frame=None, model=None,
            camera_track="auto"):
    import mujoco
    m = model or mujoco.MjModel.from_xml_path(str(scene))
    d = mujoco.MjData(m)
    st = load_state(scene, state)
    if len(st["qpos"]) != m.nq or len(st["qvel"]) != m.nv:
        raise SystemExit(f"initial state has {len(st['qpos'])} qpos / {len(st['qvel'])} qvel, model needs {m.nq} / {m.nv}")
    d.qpos[:] = st["qpos"]
    d.qvel[:] = st["qvel"]
    mujoco.mj_forward(m, d)
    tl = timeline()
    W, H = int(tl["width"]), int(tl["height"])
    times = np.array(tl["t"] if times is None else times, float)
    render = set(render or ())
    try:
        cam_name = scene_camera(m, camera) if m.ncam else None
    except SystemExit:
        if camera or render:
            raise
        cam_name = None                                   # several cameras, none chosen: no projection (r["uv"] None)
    trk = None                                            # "auto" applies only on the video timeline with a camera
    if camera_track not in (None, "none") and (camera_track != "auto" or (cam_name is not None and len(times) == tl["n"])):
        trk = _camera_track(scene, camera_track, tl["n"])
    if trk is not None:
        if cam_name is None:
            raise SystemExit("camera track given but no camera chosen (--camera NAME)")
        cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if len(times) != tl["n"] or int(m.cam_bodyid[cam_id]) != 0:
            raise SystemExit("a camera track needs the video timeline and a camera attached to the world body")
    seg = dep = None
    if render:
        if cam_name is None:
            raise SystemExit("the scene has no camera to render")
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), W)
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), H)
        m.vis.quality.offsamples = 0                      # exact segmentation ids (no multisample blending)
        try:
            seg = mujoco.Renderer(m, height=H, width=W)
            dep = mujoco.Renderer(m, height=H, width=W)
        except Exception as e:
            raise RuntimeError(f"offscreen rendering failed ({e}). MUJOCO_GL={os.environ.get('MUJOCO_GL')}: import r2s "
                               "BEFORE mujoco in your own scripts (it sets MUJOCO_GL=egl), or export MUJOCO_GL=egl") from None
        seg.enable_segmentation_rendering()
        dep.enable_depth_rendering()
        opt = mujoco.MjvOption()
        hidden = [g for g in range(m.ngeom) if _alpha(m, g) <= 0]
        geom_body = np.where([_alpha(m, g) > 0 for g in range(m.ngeom)], m.geom_bodyid, -1).astype(np.int32)
        m.geom_group[hidden] = 5                          # not drawn (rendering only; physics unchanged)
    xpos, xquat, qpos, simt, uv, warn = [], [], [], [], [], None
    for k, t in enumerate(times):
        while d.time < t - 1e-10:
            mujoco.mj_step(m, d)
        if not np.isfinite(d.qpos).all():
            warn = f"non-finite state at t={t:.3f}"
            break
        if warn is None and any(int(w.number) for w in d.warning):
            warn = f"MuJoCo warning by t={t:.3f}: " + ", ".join(
                mujoco.mjtWarning(i).name for i, w in enumerate(d.warning) if w.number)
        xpos.append(d.xpos.copy()); xquat.append(d.xquat.copy()); qpos.append(d.qpos.copy()); simt.append(d.time)
        if trk is not None:
            set_camera_pose(d, cam_id, trk[1][k], trk[2][k])
        if cam_name is not None:
            uv.append(Camera.from_mujoco(m, d, cam_name, W, H).project(d.xpos))
        if k in render:
            seg.update_scene(d, camera=cam_name, scene_option=opt)
            s = seg.render()
            ok = (s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (s[..., 0] >= 0) & (s[..., 0] < m.ngeom)
            lab = np.full((H, W), -1, np.int32)
            lab[ok] = geom_body[s[..., 0][ok]]
            dep.update_scene(d, camera=cam_name, scene_option=opt)
            if on_frame is not None:
                on_frame(k, lab, dep.render())
    for r in (seg, dep):
        if r is not None:
            r.close()
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}" for i in range(m.nbody)]
    return dict(times=times[:len(xpos)], sim_times=np.array(simt), body_names=names, xpos=np.array(xpos),
                xquat=np.array(xquat), qpos=np.array(qpos), uv=np.array(uv) if uv else None, camera=cam_name,
                camera_track=str(trk[0]) if trk else None, warning=warn, width=W, height=H)


def _sam3_mask(t, k):
    ms = t.get("mask_source") or {}
    f = Path(ms.get("sam3_dir", "")) / "frames" / f"frame_{k:06d}.npz"
    if not ms or not f.exists():
        return None
    z = np.load(f)
    ids = [int(i) for i in z["object_ids"]]
    return z["masks"][ids.index(int(ms["object_id"]))] if int(ms["object_id"]) in ids else None


def _pi3x_track(t):
    ms = t.get("mask_source") or {}
    if not ms:
        return None
    f = ANALYSIS / f"pi3x_track_{Path(ms['sam3_dir']).name.replace('sam3_', '')}.json"
    if not f.exists():
        return None
    o = load_json(f).get(str(ms["object_id"]))
    return {int(k): np.array(c) for k, c in zip(o["frames"], o["cam"])} if o else None


def compare(scene=OUTPUT / "scene.xml", state=None, camera=None, links=None, tracks=None, every=1, camera_track="auto"):
    import mujoco
    from .track import load_tracks
    tr = load_tracks(tracks)
    if not tr:
        raise SystemExit("no tracks in analysis/tracks/ (r2s track ...)")
    m = mujoco.MjModel.from_xml_path(str(scene))
    names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i): i for i in range(m.nbody)}
    links = dict(links or {})
    bodies = {}
    for tn in tr:
        bl = links.get(tn, [tn] if tn in names else [])
        bl = [n for b in bl for n in (fnmatch.filter(names, b) if any(ch in b for ch in "*?[") else [b])]
        bad = [b for b in bl if b not in names]
        if bad:
            raise SystemExit(f"--link {tn}: no body {bad} in the scene")
        bodies[tn] = [names[b] for b in bl]
    tl = timeline()
    W, H = int(tl["width"]), int(tl["height"])
    cam_name = scene_camera(m, camera)
    p3 = {tn: _pi3x_track(t) for tn, t in tr.items()}
    want = {}
    for tn, t in tr.items():
        for j, k in enumerate(t["frames"]):
            if j % every == 0 or (p3[tn] and k in p3[tn]):
                want.setdefault(int(k), []).append((tn, j))
    acc = {tn: dict(px=[], frames=[], missing=[], iou=[], sim3=[], pi3=[]) for tn in tr}
    f = None

    def on_frame(k, lab, depth):
        nonlocal f
        if f is None:
            f = 0.5 * H / np.tan(np.radians(m.cam_fovy[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)]) / 2)
        for tn, j in want.get(k, []):
            if not bodies[tn]:
                continue
            mk = np.isin(lab, bodies[tn])
            a = acc[tn]
            if not mk.any():
                a["missing"].append(k)
                continue
            vs, us = np.nonzero(mk)
            a["px"].append(float(np.hypot(us.mean() + .5 - tr[tn]["uv"][j][0], vs.mean() + .5 - tr[tn]["uv"][j][1])))
            a["frames"].append(k)
            gm = _sam3_mask(tr[tn], k)
            if gm is not None and gm.shape == mk.shape:
                a["iou"].append(float((gm & mk).sum() / max(1, (gm | mk).sum())))
            if p3[tn] and k in p3[tn]:
                z = depth[mk].astype(float)
                X = np.stack([(us + .5 - W / 2) / f * z, (vs + .5 - H / 2) / f * z, z], 1)
                a["sim3"].append(np.median(X, 0)); a["pi3"].append(p3[tn][k])

    r = rollout(scene, state, camera=cam_name, render=sorted(want), on_frame=on_frame, model=m, camera_track=camera_track)
    ratios = [s[2] / p[2] for tn in tr for s, p in zip(acc[tn]["sim3"], acc[tn]["pi3"]) if p[2] > 0]
    g = float(np.median(ratios)) if ratios else None
    rows = []
    for tn, a in acc.items():
        row = dict(track=tn, bodies=[r["body_names"][b] for b in bodies[tn]], frames_compared=len(a["frames"]),
                   frames_not_visible=len(a["missing"]))
        if not bodies[tn]:
            row["note"] = "no body with this name (use --link TRACK=BODY)"
        if a["px"]:
            e = np.array(a["px"]); w = int(np.argmax(e)); h = len(e) // 2
            row.update(mean_px=float(e.mean()), first_px=float(e[0]), half1_px=float(e[:max(1, h)].mean()),
                       half2_px=float(e[h:].mean()), last_px=float(e[-1]), worst_px=float(e[w]), worst_frame=a["frames"][w])
        if a["missing"]:
            row["not_visible_frames"] = [a["missing"][0], a["missing"][-1]]
        if a["iou"]:
            row["mask_iou"] = float(np.mean(a["iou"]))
        if a["sim3"]:
            S, P = np.array(a["sim3"]), np.array(a["pi3"])
            row.update(pi3x_frames=len(S), depth_ratio=dict(median=float(np.median(S[:, 2] / P[:, 2])),
                                                             first=float(S[0, 2] / P[0, 2]), last=float(S[-1, 2] / P[-1, 2])),
                       dist3d_after_common_ratio=float(np.linalg.norm(S - g * P, axis=1).mean()))
        rows.append(row)
    linked = {b for bl in bodies.values() for b in bl}
    free = [r["body_names"][i] for i in range(1, m.nbody) if m.body_dofnum[i] and i not in linked]
    res = dict(scene=str(scene), camera=cam_name, camera_track=r["camera_track"], frames_rendered=len(want), rows=rows,
               moving_bodies_without_track=free, common_depth_ratio=g, warning=r["warning"])
    return res


def print_compare(c):
    print(f"camera {c['camera']}" + (f" moved by {c['camera_track']}" if c.get("camera_track") else "")
          + f", {c['frames_rendered']} frames rendered")
    print(f"{'track':16s} {'bodies':22s} {'n':>4s} {'mean':>6s} {'first':>6s} {'half1':>6s} {'half2':>6s} {'last':>6s} "
          f"{'worst@f':>9s} {'notvis':>6s} {'iou':>5s} {'z sim/pi3x':>10s} {'3d':>7s}")
    for x in c["rows"]:
        fm = lambda k, n=1: f"{x[k]:.{n}f}" if x.get(k) is not None else "-"
        z = x.get("depth_ratio")
        zr = f"{z['median']:.3f}" if z else "-"
        print(f"{x['track'][:16]:16s} {','.join(x['bodies'])[:22]:22s} {x['frames_compared']:4d} {fm('mean_px'):>6s} "
              f"{fm('first_px'):>6s} {fm('half1_px'):>6s} {fm('half2_px'):>6s} {fm('last_px'):>6s} "
              f"{(fm('worst_px', 0) + '@' + str(x.get('worst_frame'))) if x.get('worst_px') is not None else '-':>9s} "
              f"{x['frames_not_visible']:6d} {fm('mask_iou', 2):>5s} {zr:>10s} "
              f"{fm('dist3d_after_common_ratio', 3):>7s}" + (f"  ({x['note']})" if x.get("note") else ""))
    if c["common_depth_ratio"]:
        print(f"common depth ratio sim/Pi3X {c['common_depth_ratio']:.3f} (3d = mean distance after it, scene units)")
    if c["moving_bodies_without_track"]:
        print("bodies with degrees of freedom and no track:", ", ".join(c["moving_bodies_without_track"]))
    if c["warning"]:
        print("WARNING:", c["warning"])


def overlay(scene=OUTPUT / "scene.xml", state=None, camera=None, frames=None, out=ANALYSIS / "overlay.jpg", cols=None,
            camera_track="auto"):
    import cv2
    from .track import load_tracks
    from .video import Video, _label, grid_images, save_image
    tl = timeline()
    if frames and any(not 0 <= k < tl["n"] for k in frames):
        raise SystemExit(f"--frames must be within 0..{tl['n'] - 1}")
    frames = frames or sorted(set(list(range(0, tl["n"], max(1, tl["n"] // 11))) + [tl["n"] - 1]))
    labs = {}
    r = rollout(scene, state, camera=camera, render=frames, on_frame=lambda k, lab, dep: labs.__setitem__(k, lab),
                camera_track=camera_track)
    v = Video()
    tracks = load_tracks()
    palette = [(255, 60, 60), (60, 220, 60), (60, 120, 255), (255, 200, 0), (220, 0, 220), (0, 220, 220),
               (255, 130, 0), (150, 80, 255)]
    shown = sorted({int(b) for lab in labs.values() for b in np.unique(lab) if b > 0})
    col = {b: palette[i % len(palette)] for i, b in enumerate(shown)}
    tiles = []
    for k in frames:
        img = v[k].copy()
        lab = labs.get(k)
        for b in shown:
            if lab is not None:
                cs, _ = cv2.findContours((lab == b).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img, cs, -1, col[b], 2)
        for t in tracks.values():
            if k in t["frames"]:
                u, vv = t["uv"][t["frames"].index(k)]
                cv2.circle(img, (int(u), int(vv)), 5, (255, 255, 255), 2)
        tiles.append(_label(img, f"f{k} t={tl['t'][k]:.2f}"))
    p = save_image(grid_images(tiles, cols), out)
    print(f"{p} (camera {r['camera']}" + (f" moved by {r['camera_track']}" if r["camera_track"] else "") + ")\n  outlines (RGB): " + ", ".join(f"{r['body_names'][b]}={col[b]}" for b in shown)
          + "; white circles = track centroids")
    if r["warning"]:
        print("WARNING:", r["warning"])


def members(sam3_dir, scene=OUTPUT / "scene.xml", state=None, camera=None, bodies=None, camera_track="auto"):
    """Detected members (r2s gpu members) vs the scene's rendered bodies on the members' frame, one to one."""
    import cv2
    import mujoco
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree
    from .sam3 import load_members
    info = load_members(sam3_dir)
    if info is None:
        raise SystemExit(f"{sam3_dir}: not a `gpu members` result (no members.json)")
    k0 = int(info["anchor_frame"])
    frames = [int(r["frame"]) for r in info["per_frame"]]
    m = mujoco.MjModel.from_xml_path(str(scene))
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or f"body_{i}" for i in range(m.nbody)]
    pattern = None
    if bodies:
        pattern = [i for i in range(1, m.nbody) if any(fnmatch.fnmatch(names[i], g) for g in bodies)]
        if not pattern:
            raise SystemExit(f"--bodies {bodies}: no body matches")
    labs = {}
    r = rollout(scene, state, camera=scene_camera(m, camera), render=frames, model=m, camera_track=camera_track,
                on_frame=lambda k, lab, dep: labs.__setitem__(k, lab))
    z = {k: np.load(Path(sam3_dir) / "frames" / f"frame_{k:06d}.npz") for k in frames}
    union = {k: z[k]["masks"][list(z[k]["object_ids"]).index(0)] for k in frames}
    lab = labs[k0]
    nb = m.nbody
    ids = [int(i) for i in z[k0]["object_ids"] if int(i) > 0]
    masks = {int(i): mk for i, mk in zip(z[k0]["object_ids"], z[k0]["masks"]) if int(i) > 0}
    rows = {mm["id"]: mm for mm in info["members"]}
    if not ids:
        raise SystemExit(f"{sam3_dir}: no members on frame {k0}")
    C = np.array([rows[i]["centroid"] for i in ids], float)
    size = float(np.median([max(b[2] - b[0], b[3] - b[1]) for b in (rows[i]["bbox"] for i in ids)]))
    nn = float(np.median(cKDTree(C).query(C, k=2)[0][:, 1])) if len(ids) > 1 else size
    gate = max(0.5 * nn, 0.25 * size)          # a body this close to a member's centroid is that member's body
    ok = lab >= 0
    area = np.bincount(lab[ok], minlength=nb)
    vs, us = np.nonzero(ok)
    cen = np.stack([np.bincount(lab[ok], us + .0, nb), np.bincount(lab[ok], vs + .0, nb)], 1) / np.maximum(1, area)[:, None]
    near = cv2.dilate(union[k0].astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * int(gate) + 1, 2 * int(gate) + 1))).astype(bool)
    on_union = np.bincount(lab[near & (lab >= 0)], minlength=nb)
    if pattern is None:                         # bodies drawn mostly on (or next to) the members, not the world body
        cand = [i for i in range(1, nb) if area[i] >= 5 and on_union[i] >= 0.5 * area[i]]
    else:
        cand = pattern
    vis = [i for i in cand if area[i] >= 5]
    B = cen[vis].reshape(-1, 2)
    inter = np.zeros((len(ids), len(vis)))
    for a, i in enumerate(ids):
        inter[a] = np.bincount(lab[masks[i] & (lab >= 0)], minlength=nb)[vis]
    marea = np.array([masks[i].sum() for i in ids], float)
    iou = inter / np.maximum(1, marea[:, None] + area[vis][None, :] - inter) if vis else np.zeros((len(ids), 0))
    pairs = []
    if vis:
        D = np.linalg.norm(C[:, None] - B[None], axis=2)
        ra, cb = linear_sum_assignment(np.where(D <= gate, D, 1e6))
        pairs = [(a, b) for a, b in zip(ra, cb) if D[a, b] <= gate]
    used_m, used_b = {a for a, _ in pairs}, {b for _, b in pairs}

    matched = [dict(member=ids[a], body=names[vis[b]], iou=round(float(iou[a, b]), 3),
                    centroid_px=round(float(np.linalg.norm(C[a] - B[b])), 1)) for a, b in pairs]
    missing = [dict(member=ids[a], centroid=rows[ids[a]]["centroid"],
                    nearest_body_px=round(float(np.linalg.norm(B - C[a], axis=1).min()), 1) if vis else None)
               for a in range(len(ids)) if a not in used_m]
    extra = [dict(body=names[vis[b]], visible_px=int(area[vis[b]]), centroid=B[b].round(1).tolist(),
                  nearest_member_px=round(float(np.linalg.norm(C - B[b], axis=1).min()), 1))
             for b in range(len(vis)) if b not in used_b]
    hidden = [names[i] for i in cand if area[i] < 5]
    later = []
    for k in frames[1:]:
        sim_u = np.isin(labs[k], [vis[b] for _, b in pairs] if pairs else vis)
        later.append(dict(frame=k, detections=next(x["members"] for x in info["per_frame"] if x["frame"] == k),
                          union_iou=round(float((sim_u & union[k]).sum() / max(1, (sim_u | union[k]).sum())), 3)))
    res = dict(sam3_dir=str(sam3_dir), scene=str(scene), camera=r["camera"], camera_track=r["camera_track"],
               frame=k0, bodies="pattern " + " ".join(bodies) if bodies else "on the members' union (auto)",
               match_gate_px=round(gate, 1), member_spacing_px=round(nn, 1),
               members=len(ids), bodies_considered=len(cand), bodies_visible=len(vis), matched=len(matched),
               members_without_body=len(missing), bodies_without_member=len(extra), bodies_not_visible=len(hidden),
               median_iou=float(np.median([x["iou"] for x in matched])) if matched else None,
               median_centroid_px=float(np.median([x["centroid_px"] for x in matched])) if matched else None,
               matches=matched, missing=missing, extra=extra, hidden=hidden, later_frames=later, warning=r["warning"])
    # picture: video frame with the outcome
    from .video import Video, _label, save_image
    img = Video()[k0].copy()
    th = max(1, img.shape[1] // 700)
    for a, b in pairs:
        cs, _ = cv2.findContours(masks[ids[a]].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, (0, 230, 0), th)
    for x in missing:
        cs, _ = cv2.findContours(masks[x["member"]].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, (255, 0, 0), th + 1)
        u, v = x["centroid"]
        cv2.putText(img, str(x["member"]), (int(u) - 8, int(v) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    for b in range(len(vis)):
        if b in used_b:
            continue
        cs, _ = cv2.findContours((lab == vis[b]).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, (40, 120, 255), th + 1)
    res["image"] = str(save_image(_label(img, f"f{k0} green {len(matched)} matched | red {len(missing)} no body | "
                                         f"blue {len(extra)} no member"),
                                  ANALYSIS / f"members_{Path(sam3_dir).name.replace('sam3_', '')}.jpg"))
    return res


def print_members(c):
    print(f"frame {c['frame']}, camera {c['camera']}" + (f" moved by {c['camera_track']}" if c.get("camera_track") else "")
          + f"; bodies: {c['bodies']}; pairs: body centroid within {c['match_gate_px']} px of the member's "
            f"(half the member spacing {c['member_spacing_px']} px)")
    print(f"members {c['members']}, bodies {c['bodies_considered']} ({c['bodies_visible']} visible, "
          f"{c['bodies_not_visible']} not visible): matched {c['matched']}"
          + (f" (median centroid {c['median_centroid_px']:.1f} px, IoU {c['median_iou']:.2f})" if c["matched"] else "")
          + f", members without body {c['members_without_body']}, visible bodies without member "
            f"{c['bodies_without_member']}")
    if c["missing"]:
        print("  members without body (id@u,v): " + ", ".join(
            f"{x['member']}@{x['centroid'][0]:.0f},{x['centroid'][1]:.0f}" for x in c["missing"][:40])
              + (" ..." if len(c["missing"]) > 40 else ""))
    if c["extra"]:
        print("  bodies without member: " + ", ".join(
            f"{x['body']}@{x['centroid'][0]:.0f},{x['centroid'][1]:.0f}" for x in c["extra"][:40])
              + (" ..." if len(c["extra"]) > 40 else ""))
    if c["hidden"]:
        print(f"  bodies not visible on frame {c['frame']}: {len(c['hidden'])} ("
              + ", ".join(c["hidden"][:12]) + (" ..." if len(c["hidden"]) > 12 else "") + ")")
    if c["later_frames"]:
        print("  union IoU on other frames: " + ", ".join(f"f{x['frame']}:{x['union_iou']:.2f}" for x in c["later_frames"][:16])
              + (" ..." if len(c["later_frames"]) > 16 else ""))
    print("  image:", c["image"])
    if c["warning"]:
        print("WARNING:", c["warning"])


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="r2s check", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = {c: sub.add_parser(c) for c in ("rollout", "compare", "overlay", "members")}
    for p in ps.values():
        p.add_argument("--scene", default=str(OUTPUT / "scene.xml"))
        p.add_argument("--state", help="default: initial_state.json next to the scene")
    for c in ("compare", "overlay", "members"):
        ps[c].add_argument("--camera", help="default: the scene's only camera, else video_cam")
        ps[c].add_argument("--camera-track", default="auto", metavar="FILE|none",
                           help="per-frame camera pose (default: camera_track.json next to the scene, if any)")
    ps["compare"].add_argument("--link", action="append", default=[], metavar="TRACK=BODY[,BODY...]")
    ps["compare"].add_argument("--tracks", nargs="+", help="only these tracks")
    ps["compare"].add_argument("--every", type=int, default=1, help="every K-th track frame (Pi3X sample frames always)")
    ps["compare"].add_argument("-o", "--out", default=str(ANALYSIS / "compare.json"))
    ps["members"].add_argument("--sam3", required=True, help="a `r2s gpu members` result directory")
    ps["members"].add_argument("--bodies", nargs="+", help="body name patterns (fnmatch), e.g. 'domino_*'")
    ps["members"].add_argument("-o", "--out", help="default analysis/members_<name>.json")
    ps["overlay"].add_argument("--frames", type=int, nargs="+")
    ps["overlay"].add_argument("--cols", type=int)
    ps["overlay"].add_argument("-o", "--out", default=str(ANALYSIS / "overlay.jpg"))
    a = ap.parse_args(argv)
    if not Path(a.scene).exists():
        raise SystemExit(f"no {a.scene}")
    if a.cmd == "rollout":
        r = rollout(a.scene, a.state)
        np.savez(ANALYSIS / "sim_rollout.npz", frame_times=r["times"], sim_times=r["sim_times"],
                 body_names=np.array(r["body_names"]), body_xpos=r["xpos"], body_xquat=r["xquat"], qpos=r["qpos"])
        print("analysis/sim_rollout.npz", r["xpos"].shape, r["warning"] or "")
    elif a.cmd == "compare":
        links = {}
        for s in a.link:
            if "=" not in s:
                raise SystemExit(f"--link {s}: expected TRACK=BODY[,BODY...]")
            k, v = s.split("=", 1)
            links[k] = [b for b in v.split(",") if b]
        c = compare(a.scene, a.state, a.camera, links, a.tracks, max(1, a.every), a.camera_track)
        save_json(c, a.out)
        print_compare(c)
        print("->", a.out)
    elif a.cmd == "members":
        c = members(a.sam3, a.scene, a.state, a.camera, a.bodies, a.camera_track)
        out = a.out or ANALYSIS / f"members_{Path(a.sam3).name.replace('sam3_', '')}.json"
        save_json(c, out)
        print_members(c)
        print("->", out)
    else:
        overlay(a.scene, a.state, a.camera, a.frames, Path(a.out), a.cols, a.camera_track)
