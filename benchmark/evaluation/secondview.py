"""Second-view metric: 2D ADE and IoU of the moving objects seen from a camera the agent never saw.

The input view can be matched by a scene that is wrong in depth (object too far and too big, wrong focal length,
wrong support height); a second view exposes it. The reconstruction is placed in the second camera through evaluator
v2's alignment of the input camera (centred camera frame + depth scale, evaluation/v2/jobs/<job>), so nothing is
fitted to the second view. Only scene.xml + initial_state.json are used (no agent code), as in the v2 render.

Second view (chosen once per scene from GT only, cached in BENCH/cache/secondview/<job>.npz):
  PhysInOne  a real GT camera: of the ~12 static CineCameras, the one with the largest angle to the input camera
             (<= 100 deg) that sees every scored moving object (>= MIN_AREA px, centroid off the border) in >= 80 % of
             12 sampled frames; else the best covered one. GT masks = the dataset's segmentation for that camera.
  Physion    one camera only, so the GT is rebuilt in MuJoCo from the HDF5 (meshes x scale, per-frame poses) and seen
             from a virtual camera turned about the vertical through the moving objects (first of PHIS covering every
             moving object in >= 80 % of 12 sampled frames). Objects whose rebuild does not reproduce the dataset's
             input-view mask (IoU < VALID_IOU, e.g. library models with another scale convention) are hidden unless
             they are scored. The rebuild has no room, so the prediction is rendered with only its parts matched to a
             GT object (its own walls / floor have no GT counterpart).
Scored objects: evaluator v2's eligible objects (GT moving > 2 cm or > 5 deg and visible in the input view), in the
frames where the second-view GT shows them (>= MIN_GT_AREA px at SIZE). Per frame as in evaluator v2: centroid error
of the visible masks (the prediction's own unoccluded centroid if it is hidden; absent from a 2x-wide render of the
scored parts -> the diagonal), IoU of the visible masks inside the frame. Unmatched objects score the diagonal / 0.
Pixels are reported at the evaluation resolution (PhysInOne 1120, Physion 1024); ade2d_diag = ADE / image diagonal.

Options of evaluate_job: videos = also write evaluation/secondview/jobs/<job>/{gt,pred}.mp4 (GT outlines drawn);
selftest = second camera is the input camera with the reconstruction's own focal length: must reproduce evaluator
v2's input-view ADE / IoU (implementation check; writes evaluation/secondview_selftest/).
"""
import copy, csv, io, json, math, os, subprocess, zipfile
from pathlib import Path

from .common import NO_OUTPUT, egl_env, read_json

egl_env()
import h5py  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from . import v2 as ev  # noqa: E402
from .gt import MaskTruth, source_rows  # noqa: E402
from .render import effective_alpha, parts_of  # noqa: E402

VERSION = "secondview_v1"
GT_CACHE = Path("cache/secondview")  # relative to the benchmark directory
OUT, OUT_SELFTEST = Path("evaluation/secondview"), Path("evaluation/secondview_selftest")
SIZE, EXT = 560, 2
MIN_AREA, MARGIN = 300, 0.05          # camera choice: px at full resolution, centroid margin
MIN_GT_AREA = 50                      # px at SIZE for a second-view GT frame to count
PHIS = [90, -90, 75, -75, 105, -105, 60, -60, 120, -120, 45, -45]
VALID_IOU = 0.5
G = np.array([[1.0, 0, 0], [0, 0, 1], [0, 1, 0]])   # Unity (x, y-up, z) <-> MuJoCo z-up (a reflection)
COLORS = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (237, 161, 0), (232, 123, 164), (0, 131, 0), (98, 80, 214), (227, 73, 72)]


# ---------------------------------------------------------------- shared helpers
def inside(mask):
    """seen: >= MIN_AREA px and centroid away from the border (an object at the edge leaves the frame on a small error)"""
    if mask.sum() < MIN_AREA:
        return False
    v, u = np.nonzero(mask); h, w = mask.shape
    return MARGIN * w < u.mean() < (1 - MARGIN) * w and MARGIN * h < v.mean() < (1 - MARGIN) * h


def eligible(root, task, meta):
    """evaluator v2's scored objects: GT moving (> 2 cm / > 5 deg) and visible in the input view"""
    truth = MaskTruth(task, meta)
    try:
        _, _, classes, _, _ = ev.gt_poses(task, meta, truth.objects)
        cache = root / ev.GT_CACHE / f"{task['job_id']}.npz"
        if cache.exists():
            with np.load(cache) as c:
                area = c["gt_stats"][..., 0]
        else:
            area = np.array([[(truth.frame(t) == g + 1).sum() for g in range(len(truth.objects))] for t in range(len(truth.rows))])
        return [g for g in range(len(truth.objects)) if classes[g]["class"] == "moving" and (area[:, g] > 0).any()]
    finally:
        truth.close()


def look_at(c, target):
    f = target - c; f /= np.linalg.norm(f)
    x = np.cross(f, [0, 0, 1.0]); x /= np.linalg.norm(x)
    return np.stack([x, np.cross(x, f), -f], axis=1)   # columns: camera x, y, z (OpenGL, looks along -z)


def draw(rgb, lab, moving):
    out = rgb.copy()
    for i, g in enumerate(moving):
        m = lab == g + 1
        if m.any():
            e = (np.array(Image.fromarray(m.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(5))) > 0) & ~m
            out[e] = COLORS[i % len(COLORS)]
    return out


def ffmpeg(path, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{SIZE}x{SIZE}",
                             "-r", f"{fps:.6f}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "26",
                             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)


# ---------------------------------------------------------------- PhysInOne: a real held-out GT camera
def pio_cameras(task):
    with zipfile.ZipFile(task["source"]["path"]) as z:
        out = {}
        for n in z.namelist():
            if Path(n).name.startswith("blender_") and n.endswith(".json"):
                c = json.loads(z.read(n))
                Ms = [np.array(f["transform_matrix"]) for f in c["frames"]]
                if max(np.abs(M - Ms[0]).max() for M in Ms) < 1e-6:          # static cameras only
                    out[Path(n).stem[8:]] = {"M": Ms[0], "fovx": c["camera_angle_x"], "w": c["img_w"], "h": c["img_h"]}
    return out


def pio_view(task, meta, moving, selftest):
    cams = pio_cameras(task); c0 = task["source"]["camera"]; f0 = -cams[c0]["M"][:3, 2]
    T = len(source_rows(meta)); sample = np.linspace(0, T - 1, 12).astype(int); cand = []
    for name, c in cams.items():
        if (name == c0) != selftest:
            continue
        f = -c["M"][:3, 2]; ang = math.degrees(math.acos(np.clip(f @ f0, -1, 1)))
        t2 = copy.deepcopy(task); t2["source"]["camera"] = name; mt = MaskTruth(t2, meta)
        try:
            vis = np.array([[inside(mt.frame(int(t)) == g + 1) for g in moving] for t in sample])
        finally:
            mt.close()
        cand.append({"camera": name, "angle": ang, "cover": float(vis.mean(0).min()) if moving else 1.0})
    ok = [c for c in cand if c["cover"] >= 0.8 and c["angle"] <= 100]
    best = max(ok, key=lambda c: c["angle"]) if ok else max(cand, key=lambda c: (c["cover"], c["angle"]))
    c, k = cams[c0], cams[best["camera"]]
    half = math.tan(k["fovx"] / 2) * k["h"] / k["w"]
    # camera k in the GT input-camera frame (OpenGL axes)
    view = {**best, "ck": (c["M"][:3, :3].T @ (k["M"][:3, 3] - c["M"][:3, 3])).tolist(), "Rk": (c["M"][:3, :3].T @ k["M"][:3, :3]).tolist(),
            "half": half, "res": k["w"], "candidates": cand, "synthetic": False}
    t2 = copy.deepcopy(task); t2["source"]["camera"] = best["camera"]; mt = MaskTruth(t2, meta)
    try:
        labs = np.stack([np.array(Image.fromarray(mt.frame(t)).resize((SIZE, SIZE), Image.NEAREST)) for t in range(T)]).astype(np.uint8)
    finally:
        mt.close()
    return view, labs


def pio_gt_video(task, meta, view, labs, moving, path, fps):
    rows = source_rows(meta)
    with zipfile.ZipFile(task["source"]["path"]) as z:
        members = {int(Path(n).stem): n for n in z.namelist() if f"/{view['camera']}/rgb/" in n}
        p = ffmpeg(path, fps)
        for t, r in enumerate(rows):
            im = np.array(Image.open(io.BytesIO(z.read(members[int(r["source_frame"])]))).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR))
            p.stdin.write(draw(im, labs[t], moving).tobytes())
        p.stdin.close(); p.wait()


# ---------------------------------------------------------------- Physion: rebuilt GT, virtual camera
class PhysionScene:
    """MuJoCo rebuild of a Physion trial (meshes on mocap bodies, floor); set(t) poses it at clip frame t."""

    def __init__(self, task, meta):
        self.h = h5py.File(task["source"]["path"], "r"); st = self.h["static"]
        self.names = [n.decode() for n in st["model_names"][()]]
        scale, color = st["scale"][()], st["color"][()]
        self.rows = source_rows(meta); self.frames = [self.h["frames"][r["source_frame"]] for r in self.rows]
        f0 = self.frames[0]["camera_matrices"]
        V = f0["camera_matrix_cam0"][()].reshape(4, 4); P = f0["projection_matrix_cam0"][()].reshape(4, 4)
        Rwc = V[:3, :3] @ G
        self.R0 = Rwc.T; self.c0 = -Rwc.T @ V[:3, 3]
        self.fovy = math.degrees(2 * math.atan(1 / float(P[1, 1])))   # float64 as NumPy 1.x did (NumPy 2: float32)
        a = ['<mujoco><compiler inertiafromgeom="false"/><visual><global offwidth="2048" offheight="2048"/>'
             '<quality offsamples="4"/><headlight ambient=".35 .35 .35" diffuse=".45 .45 .45"/></visual><asset>'
             '<texture name="chk" type="2d" builtin="checker" rgb1=".58 .58 .56" rgb2=".5 .5 .48" width="64" height="64"/>'
             '<material name="floor" texture="chk" texrepeat="12 12"/>']
        for i in range(len(self.names)):
            v = (self.h[f"static/mesh/vertices_{i}"][()] * scale[i]) @ G.T
            fa = self.h[f"static/mesh/faces_{i}"][()].reshape(-1, 3)[:, ::-1]
            a.append(f'<mesh name="m{i}" inertia="shell" vertex="{" ".join(f"{x:.6g}" for x in v.ravel())}" '
                     f'face="{" ".join(str(int(x)) for x in fa.ravel())}"/>')
        a.append('</asset><worldbody><light pos="0 0 6" dir="0 0 -1" diffuse=".55 .55 .55" directional="true"/>'
                 '<geom name="floor" type="plane" size="30 30 .1" material="floor" group="1"/><camera name="v" pos="0 0 3"/>')
        for i in range(len(self.names)):
            a.append(f'<body name="o{i}" mocap="true"><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>'
                     f'<geom name="g{i}" type="mesh" mesh="m{i}" rgba="{" ".join(f"{x:.3f}" for x in color[i])} 1" contype="0" conaffinity="0"/></body>')
        a.append("</worldbody></mujoco>")
        self.model = mujoco.MjModel.from_xml_string("".join(a)); self.data = mujoco.MjData(self.model)
        self.geom = np.array([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"g{i}") for i in range(len(self.names))])
        self.label_of_geom = np.zeros(self.model.ngeom, dtype=np.int64); self.label_of_geom[self.geom] = np.arange(len(self.names)) + 1
        self.cam = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "v"); self.model.cam_fovy[self.cam] = self.fovy
        self.opt = mujoco.MjvOption()

    def set(self, t):
        ob = self.frames[t]["objects"]; pos, rot = ob["positions_cam0"][()], ob["rotations_cam0"][()]
        for i in range(len(self.names)):
            self.data.mocap_pos[i] = G @ pos[i]
            q = np.zeros(4); mujoco.mju_mat2Quat(q, (G @ Rotation.from_quat(rot[i]).as_matrix() @ G).ravel()); self.data.mocap_quat[i] = q
        mujoco.mj_forward(self.model, self.data)

    def place(self, pos, R):
        self.model.cam_pos[self.cam] = pos
        q = np.zeros(4); mujoco.mju_mat2Quat(q, np.asarray(R).ravel()); self.model.cam_quat[self.cam] = q

    def labels(self, r):
        r.update_scene(self.data, camera=self.cam, scene_option=self.opt); sg = r.render()
        g = sg[..., 0]; ok = (sg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (g >= 0)
        out = np.zeros(g.shape, dtype=np.uint8); out[ok] = self.label_of_geom[g[ok]]
        return out

    def close(self):
        self.h.close()


def physion_view(task, meta, moving, selftest):
    gs = PhysionScene(task, meta)
    try:
        T = len(gs.rows); sample = np.linspace(0, T - 1, 12).astype(int)
        # input-view check of the rebuild: objects whose mesh does not reproduce the dataset mask are hidden
        mt = MaskTruth(task, meta); gs.place(gs.c0, gs.R0)
        r = mujoco.Renderer(gs.model, 1024, 1024); r.enable_segmentation_rendering(); ious = [[] for _ in gs.names]
        try:
            for t in sample:
                gs.set(t); ren = gs.labels(r); gt = mt.frame(t)
                for i in range(len(gs.names)):
                    g = gt == i + 1
                    if g.sum() >= 300:
                        p = ren == i + 1; ious[i].append((g & p).sum() / (g | p).sum())
        finally:
            r.close(); mt.close()
        check = [float(np.median(x)) if x else None for x in ious]
        bad = [i for i, v in enumerate(check) if v is not None and v < VALID_IOU]
        for i in bad:
            if i not in moving:
                gs.model.geom_group[gs.geom[i]] = 4
        if selftest:
            view = {"phi": 0, "angle": 0.0, "cover": None, "pos": gs.c0.tolist(), "R": gs.R0.tolist()}; cand = []
        else:
            pos = np.array([[G @ fr["objects"]["positions_cam0"][()][i] for i in range(len(gs.names))] for fr in gs.frames])
            pivot = pos[:, moving].reshape(-1, 3).mean(0); f0 = -gs.R0[:, 2]; cand = []
            r = mujoco.Renderer(gs.model, 1024, 1024); r.enable_segmentation_rendering()
            try:
                for phi in PHIS:
                    a = math.radians(phi); Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
                    c = pivot + Rz @ (gs.c0 - pivot); R = look_at(c, pivot); gs.place(c, R); vis = []
                    for t in sample:
                        gs.set(t); lab = gs.labels(r); vis.append([inside(lab == g + 1) for g in moving])
                    cand.append({"phi": phi, "angle": math.degrees(math.acos(np.clip(-R[:, 2] @ f0, -1, 1))),
                                 "cover": float(np.array(vis).mean(0).min()), "pos": c.tolist(), "R": R.tolist()})
                    if cand[-1]["cover"] >= 0.8:
                        break
            finally:
                r.close()
            ok = [c for c in cand if c["cover"] >= 0.8]
            view = ok[0] if ok else max(cand, key=lambda c: c["cover"])
        gs.place(np.array(view["pos"]), np.array(view["R"]))
        sr = mujoco.Renderer(gs.model, SIZE, SIZE); sr.enable_segmentation_rendering(); labs = []
        try:
            for t in range(T):
                gs.set(t); labs.append(gs.labels(sr))
        finally:
            sr.close()
        ck = gs.R0.T @ (np.array(view["pos"]) - gs.c0); Rk = gs.R0.T @ np.array(view["R"])
        out = {**{k: view[k] for k in ("phi", "angle", "cover")}, "camera": f"virtual turn {view['phi']} deg",
               "ck": ck.tolist(), "Rk": Rk.tolist(), "half": math.tan(math.radians(gs.fovy) / 2), "res": 1024,
               "synthetic": True, "rebuild_input_iou": {f"{gs.names[i]}#{i}": check[i] for i in range(len(check))},
               "hidden": [f"{gs.names[i]}#{i}" for i in bad if i not in moving],
               "scored_bad_mesh": [f"{gs.names[i]}#{i}" for i in bad if i in moving],
               "candidates": [{k: c[k] for k in ("phi", "angle", "cover")} for c in cand], "pos": view["pos"], "R": view["R"]}
        return out, np.stack(labs).astype(np.uint8)
    finally:
        gs.close()


def physion_gt_video(task, meta, view, labs, moving, path, fps):
    gs = PhysionScene(task, meta)
    try:
        for name in view["hidden"]:
            gs.model.geom_group[gs.geom[int(name.split("#")[-1])]] = 4
        gs.place(np.array(view["pos"]), np.array(view["R"]))
        rr = mujoco.Renderer(gs.model, SIZE, SIZE); p = ffmpeg(path, fps)
        try:
            for t in range(len(gs.rows)):
                gs.set(t); rr.update_scene(gs.data, camera=gs.cam, scene_option=gs.opt)
                p.stdin.write(draw(rr.render(), labs[t], moving).tobytes())
        finally:
            p.stdin.close(); p.wait(); rr.close()
    finally:
        gs.close()


# ---------------------------------------------------------------- GT side, cached per scene
def gt_view(root, task, meta, selftest=False):
    path = root / GT_CACHE / ("selftest" if selftest else "") / f"{task['job_id']}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as c:
            return json.loads(str(c["view"])), c["labels"]
    moving = eligible(root, task, meta)
    view, labs = (physion_view if task["dataset"] == "physion_v15" else pio_view)(task, meta, moving, selftest)
    view["moving"] = moving
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz"); np.savez_compressed(tmp, view=json.dumps(view), labels=labs); os.replace(tmp, path)
    return view, labs


# ---------------------------------------------------------------- prediction
def render_pred(root, task, meta, view, labs, video=None, selftest=False):
    """per frame and scored object: (visible area in frame, centroid u, v, intersection, area in the 2x render of the
    scored parts alone), centroids in SIZE pixels"""
    jid = task["job_id"]; d = root / ev.OUT / "jobs" / jid
    m = read_json(d / "metrics.json")["metrics"]; man = read_json(d / "prediction/manifest.json")
    owner = np.load(d / "statistics.npz")["part_owner"]; moving = view["moving"]
    with np.load(d / "prediction/rollout.npz") as c:
        Pp = c["projection"]
    Pp = Pp[0] if Pp.ndim == 3 else Pp
    s = m["scale"]
    work = (root / "tasks" / jid).resolve() / "output"
    model = mujoco.MjModel.from_xml_path(str(work / "scene.xml")); data = mujoco.MjData(model)
    st = read_json(work / "initial_state.json"); data.qpos[:] = st["qpos"]; data.qvel[:] = st["qvel"]
    mujoco.mj_forward(model, data)
    ci = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, man["camera"])
    X = data.cam_xmat[ci].reshape(3, 3).copy(); x0 = data.cam_xpos[ci].copy()
    C = ev.centred_frame(Pp).T                                  # centred prediction frame -> prediction camera frame
    pos = x0 + X @ C @ np.array(view["ck"]) / s; R = X @ C @ np.array(view["Rk"])
    q = np.zeros(4); mujoco.mju_mat2Quat(q, R.ravel()); model.cam_pos[ci] = pos; model.cam_quat[ci] = q
    half = 1 / Pp[1, 1] if selftest else view["half"]
    fovy, fovy_ext = math.degrees(2 * math.atan(half)), math.degrees(2 * math.atan(EXT * half))
    model.cam_sensorsize[ci] = 0
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, EXT * SIZE); model.vis.global_.offheight = max(model.vis.global_.offheight, EXT * SIZE)
    model.vis.quality.offsamples = 0
    # as rendered by the evaluator: groups 0-2, transparent geoms hidden. Scored parts -> group 3, parts matched to a
    # static GT object -> group 4, unmatched parts (own floor / walls) stay in 0-2
    model.geom_group[[g for g in range(model.ngeom) if effective_alpha(model, g) <= 0 or model.geom_group[g] > 2]] = 5
    part_of_geom, parts = parts_of(model)
    if len(parts) != len(owner):
        raise ValueError("part list differs from the evaluation render")
    for k, part in enumerate(parts):
        if owner[k] >= 0:
            for g in part["geom_ids"]:
                if model.geom_group[g] <= 2:
                    model.geom_group[g] = 3 if owner[k] in moving else 4
    show = mujoco.MjvOption(); show.geomgroup[3] = show.geomgroup[4] = 1          # everything (real camera: PhysInOne)
    matched = mujoco.MjvOption(); matched.geomgroup[:] = 0; matched.geomgroup[3] = matched.geomgroup[4] = 1
    occl = matched if view["synthetic"] else show    # the Physion rebuild has no room: occluders = matched parts only
    iso = mujoco.MjvOption(); iso.geomgroup[:] = 0; iso.geomgroup[3] = 1
    seg_r = mujoco.Renderer(model, EXT * SIZE, EXT * SIZE); seg_r.enable_segmentation_rendering()
    iso_r = mujoco.Renderer(model, EXT * SIZE, EXT * SIZE); iso_r.enable_segmentation_rendering()
    rgb_r = mujoco.Renderer(model, SIZE, SIZE) if video else None
    off = (EXT - 1) * SIZE / 2
    times = [float(r["relative_time_seconds"]) for r in source_rows(meta)]
    stats = np.zeros((len(times), len(moving), 5)); p = ffmpeg(video, (len(times) - 1) / (times[-1] - times[0])) if video else None

    def owners(sg):
        geoms = sg[..., 0]; valid = (sg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (geoms >= 0) & (geoms < model.ngeom)
        part = np.full(geoms.shape, -1); part[valid] = part_of_geom[geoms[valid]]
        return np.where(part >= 0, owner[np.maximum(part, 0)], -1)
    try:
        for t, target in enumerate(times):
            while data.time < target - 1e-10:
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            model.cam_fovy[ci] = fovy_ext
            seg_r.update_scene(data, camera=ci, scene_option=occl); gl = owners(seg_r.render())
            iso_r.update_scene(data, camera=ci, scene_option=iso); gi = owners(iso_r.render())
            for i, g in enumerate(moving):
                pm = gl == g; pi = gi == g; gm = labs[t] == g + 1
                pin = pm[int(off):int(off) + SIZE, int(off):int(off) + SIZE]
                if pi.any():
                    v, u = np.nonzero(pm if pm.any() else pi)
                    stats[t, i] = (pin.sum(), u.mean() + .5 - off, v.mean() + .5 - off, (pin & gm).sum(), pi.sum())
            if p:
                model.cam_fovy[ci] = fovy; rgb_r.update_scene(data, camera=ci, scene_option=occl)
                p.stdin.write(draw(rgb_r.render(), labs[t], moving).tobytes())
    finally:
        seg_r.close(); iso_r.close()
        if p:
            p.stdin.close(); p.wait(); rgb_r.close()
    return stats, m


def score(stats, labs, moving, matched, res):
    diag = math.hypot(SIZE, SIZE); out = {}
    for i, g in enumerate(moving):
        ga = np.array([(lab == g + 1).sum() for lab in labs]); vis = ga >= MIN_GT_AREA
        if not vis.any():
            out[g] = {"second_view_visible_frames": 0}
            continue
        if g not in matched:
            out[g] = {"second_view_visible_frames": int(vis.sum()), "ade2d_pixels": diag * res / SIZE, "iou": 0.0}
            continue
        gc = np.array([np.argwhere(lab == g + 1).mean(0)[::-1] + .5 if a else (np.nan, np.nan) for lab, a in zip(labs, ga)])
        pa, pc, inter, pext = stats[:, i, 0], stats[:, i, 1:3], stats[:, i, 3], stats[:, i, 4]
        e = np.where(pext > 0, np.minimum(np.linalg.norm(pc - gc, axis=1), diag), diag)[vis]
        out[g] = {"second_view_visible_frames": int(vis.sum()), "ade2d_pixels": float(e.mean()) * res / SIZE,
                  "iou": float((inter / np.maximum(ga + pa - inter, 1))[vis].mean())}
    return out


def evaluate_job(root, task, dev=None, videos=False, selftest=False):
    if dev is not None:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(dev)
    jid = task["job_id"]; meta = root / "meta" / jid
    out = {"job_id": jid, "dataset": task["dataset"], "version": VERSION}
    try:
        if not (root / "tasks" / jid / "output/scene.xml").is_file():
            return {**out, "status": "unavailable", "reason": NO_OUTPUT}
        mp = root / ev.OUT / "jobs" / jid / "metrics.json"
        if not mp.exists() or read_json(mp).get("status") != "evaluated":
            return {**out, "status": "unscored: no evaluator v2 result"}
        view, labs = gt_view(root, task, meta, selftest)
        m2 = read_json(mp)["metrics"]
        if not m2.get("scale"):
            return {**out, "status": "unscored: evaluator v2 has no depth scale (no overlap with any GT object)"}
        om = {o["gt_index"]: o for o in m2["object_metrics"]}
        if set(om) != set(view["moving"]):
            out["eligible_mismatch"] = {"evaluator_v2": sorted(om), "second_view": view["moving"]}
        dest = root / (OUT_SELFTEST if selftest else OUT) / "jobs" / jid
        if videos:
            T = len(labs); times = [float(r["relative_time_seconds"]) for r in source_rows(meta)]; fps = (T - 1) / (times[-1] - times[0])
            (physion_gt_video if view["synthetic"] else pio_gt_video)(task, meta, view, labs, view["moving"], dest / "gt.mp4", fps)
        stats, _ = render_pred(root, task, meta, view, labs, dest / "pred.mp4" if videos else None, selftest)
        matched = {g for g in view["moving"] if g in om and om[g]["pred_parts"]}
        objs = score(stats, labs, view["moving"], matched, view["res"])
        scored = [g for g in view["moving"] if "ade2d_pixels" in objs[g]]
        mean = lambda xs: float(np.mean(xs)) if xs else None
        diag = math.hypot(view["res"], view["res"])
        out.update({"status": "evaluated",
                    "camera": view["camera"], "synthetic_view": view["synthetic"], "angle_deg": view["angle"], "coverage": view["cover"],
                    "secondview_ade2d_pixels": mean([objs[g]["ade2d_pixels"] for g in scored]),
                    "secondview_ade2d_diag": mean([objs[g]["ade2d_pixels"] / diag for g in scored]),
                    "secondview_iou": mean([objs[g]["iou"] for g in scored]),
                    "input_ade2d_pixels": mean([om[g]["ade2d_pixels"] for g in scored if g in om]),
                    "input_iou": mean([om[g]["iou"] for g in scored if g in om]),
                    "scored_objects": len(scored), "eligible_objects": len(view["moving"]), "matched_objects": len(matched),
                    "objects": [{"gt_index": g, "name": om[g]["gt_object"]["name"] if g in om else None, **objs[g],
                                 "input_ade2d_pixels": om[g]["ade2d_pixels"] if g in om else None,
                                 "input_iou": om[g]["iou"] if g in om else None} for g in view["moving"]],
                    "view": {k: v for k, v in view.items() if k not in ("ck", "Rk", "pos", "R")}})
        return out
    except Exception as e:  # noqa: BLE001 - recorded per job
        import traceback
        return {**out, "status": f"error: {type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:]}


def write_csv(root, results, selftest=False):
    cols = ["job_id", "dataset", "status", "camera", "angle_deg", "secondview_ade2d_pixels", "secondview_ade2d_diag", "secondview_iou",
            "input_ade2d_pixels", "input_iou", "scored_objects", "eligible_objects", "matched_objects", "reason"]
    with (root / (OUT_SELFTEST if selftest else OUT) / "metrics.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in sorted(results, key=lambda r: r["job_id"]):
            w.writerow([r.get(c) for c in cols])
