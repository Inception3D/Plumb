#!/usr/bin/env python3
"""Evaluation rollout v2: trusted passive rollout to per-part ID masks + depth. Never executes agent Python.

Run by evaluator v2 in a subprocess: python -m evaluation.render --work TASK_DIR --metadata META_DIR --output DIR
(TASK_DIR holds input/video.mp4 and output/{scene.xml, initial_state.json, run.py}; META_DIR holds timestamps.csv).
Differences to the first (v1) evaluation renderer:
- Visibility is MuJoCo's default (geom groups 0-2) plus "alpha 0 is invisible". Geom-group assignments are no
  longer scraped from the agent's run.py (v1 hid a whole group because run.py hid it for an auxiliary layer).
- Labels are rigid parts, not root-body groups: every non-world body is one part, every world-attached geom is
  one part. The evaluator later assigns parts to GT objects by mask overlap, so nested or split modelling of one
  physical object no longer changes what is scored.
- Depth is rendered too. Per frame and part we store pixel count, pixel-coordinate sums and camera-frame 3D sums
  of the visible surface, so the evaluator can form visible-surface centroids for any union of parts.
- Per body we store xpos and subtree COM.
"""
import argparse
import csv
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from .common import egl_env, read_json, save_json, sha256

egl_env()
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

BACKGROUND_WORDS = ("floor", "ground", "background", "backdrop")


def check_references(path: Path, root: Path, visited: set) -> None:
    """MJCF includes and asset files must stay inside the output directory; no custom plugins."""
    path = path.resolve()
    if not path.is_relative_to(root):
        raise ValueError("XML include escapes the output directory")
    if path in visited:
        raise ValueError("Repeated/cyclic MJCF include")
    visited.add(path)
    tree = ET.parse(path)
    for node in tree.iter():
        if node.tag == "plugin":
            raise ValueError("Custom MuJoCo plugins require manual review and are not auto-accepted")
        for key in ("meshdir", "texturedir", "assetdir"):
            directory = node.get(key)
            if directory and (Path(directory).is_absolute() or ".." in Path(directory).parts):
                raise ValueError("Asset directory escapes output")
        reference = node.get("file")
        if reference:
            if Path(reference).is_absolute() or ".." in Path(reference).parts:
                raise ValueError("Asset/include path escapes output")
            if node.tag == "include":
                check_references(path.parent / reference, root, visited)
    visited.remove(path)


def effective_alpha(model, g):
    rgba = model.geom_rgba[g]
    m = int(model.geom_matid[g])
    if m >= 0 and np.allclose(rgba, [.5, .5, .5, 1]):
        return float(model.mat_rgba[m][3])
    return float(rgba[3])


def parts_of(model):
    """part index per geom (-1 = not labelled) and part descriptors."""
    part_of_geom = np.full(model.ngeom, -1, dtype=np.int64)
    parts, lookup = [], {}
    for g in range(model.ngeom):
        body = int(model.geom_bodyid[g])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom_{g}"
        if model.geom_type[g] in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD):
            continue
        if body == 0 and any(w in name.lower() for w in BACKGROUND_WORDS):
            continue
        key = ("body", body) if body else ("geom", g)
        if key not in lookup:
            root = body
            while root and model.body_parentid[root]:
                root = int(model.body_parentid[root])
            dyn, b = False, body
            while b:
                dyn |= bool(model.body_dofnum[b]); b = int(model.body_parentid[b])
            lookup[key] = len(parts)
            parts.append({"part": len(parts), "kind": key[0], "body_id": body, "root_body_id": root,
                          "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) if body else name,
                          "root_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root) if root else None,
                          "dynamic": dyn, "geom_ids": []})
        part_of_geom[g] = lookup[key]
        parts[lookup[key]]["geom_ids"].append(g)
    return part_of_geom, parts


def camera_projection(model, data, camera, width, height):
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam < 0 or model.cam_bodyid[cam] != 0 or model.cam_mode[cam] != mujoco.mjtCamLight.mjCAMLIGHT_FIXED:
        raise ValueError("Only fixed world cameras are supported")
    if hasattr(model, "cam_orthographic") and model.cam_orthographic[cam]:
        raise ValueError("Orthographic camera unsupported")
    f = height / (2 * np.tan(np.deg2rad(model.cam_fovy[cam]) / 2))
    P = np.zeros((4, 4)); P[0, 0], P[1, 1], P[3, 2] = 2 * f / width, 2 * f / height, -1
    if np.any(model.cam_sensorsize[cam]):
        c = mujoco.MjvCamera(); c.type, c.fixedcamid = mujoco.mjtCamera.mjCAMERA_FIXED, cam
        scene = mujoco.MjvScene(model, maxgeom=max(model.ngeom + 10, 1000))
        mujoco.mjv_updateScene(model, data, mujoco.MjvOption(), None, c, mujoco.mjtCatBit.mjCAT_ALL, scene)
        fr = scene.camera[0]
        half = (fr.frustum_top - fr.frustum_bottom) / 2 * model.cam_sensorsize[cam, 0] / model.cam_sensorsize[cam, 1]
        P[0, 0] = fr.frustum_near / half
        P[1, 1] = 2 * fr.frustum_near / (fr.frustum_top - fr.frustum_bottom)
        P[0, 2] = fr.frustum_center / half
        P[1, 2] = (fr.frustum_top + fr.frustum_bottom) / (fr.frustum_top - fr.frustum_bottom)
    return cam, P


def backproject_grid(P, width, height):
    """Per-pixel factors so that camera X = ax*d, Y = ay*d, Z = -d for z-depth d (OpenGL camera)."""
    u = (np.arange(width) + .5); v = (np.arange(height) + .5)
    nx = 2 * u / width - 1; ny = 1 - 2 * v / height
    ax = (nx + P[0, 2]) / P[0, 0]; ay = (ny + P[1, 2]) / P[1, 1]
    return np.broadcast_to(ax, (height, width)).ravel(), np.broadcast_to(ay[:, None], (height, width)).ravel()


def part_stats(labels, depth, n, ax, ay, width):
    """area, sum u, sum v, sum X, sum Y, sum Z per part (labels 1..n)."""
    flat = labels.ravel().astype(np.int64)
    d = depth.ravel().astype(np.float64)
    idx = np.arange(flat.size)
    u, v = idx % width, idx // width
    out = np.zeros((n, 6))
    for k, w in enumerate((None, u, v, ax * d, ay * d, -d)):
        out[:, k] = np.bincount(flat, weights=w, minlength=n + 1)[1:]
    return out


def render(work, metadata, destination):
    output = work / "output"
    check_references(output / "scene.xml", output.resolve(), set())
    model = mujoco.MjModel.from_xml_path(str(output / "scene.xml"))
    if model.nu or model.nmocap:
        raise ValueError("Evaluation requires a passive scene")
    data = mujoco.MjData(model)
    state = read_json(output / "initial_state.json")
    data.qpos[:] = state["qpos"]
    data.qvel[:] = state["qvel"]
    mujoco.mj_forward(model, data)
    with (metadata / "timestamps.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    times = np.array([float(r["relative_time_seconds"]) for r in rows])
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                      "stream=width,height", "-of", "json", str(work / "input/video.mp4")]))
    width, height = probe["streams"][0]["width"], probe["streams"][0]["height"]
    camera_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
    script = (output / "run.py").read_text()
    used = re.findall(r"update_scene\([^\n]*camera\s*=\s*['\"]([^'\"]+)['\"]", script)
    if used:
        camera = used[0]
    elif len(camera_names) == 1:
        camera = camera_names[0]
    else:
        raise ValueError("Cannot determine preview camera without executing generated code")
    cam, P = camera_projection(model, data, camera, width, height)
    # Visibility: MuJoCo default groups; alpha-0 geoms moved to hidden group 5 (rendering only, physics unchanged).
    invisible = [g for g in range(model.ngeom) if effective_alpha(model, g) <= 0]
    model.geom_group[invisible] = 5
    option = mujoco.MjvOption()
    # Multisampling blends segmentation colours at edges into IDs that do not exist; IDs must be exact.
    model.vis.quality.offsamples = 0
    part_of_geom, parts = parts_of(model)
    visible_geom = np.array([bool(option.geomgroup[min(int(model.geom_group[g]), 5)]) for g in range(model.ngeom)])
    for p in parts:
        p["visible_geom_ids"] = [g for g in p["geom_ids"] if visible_geom[g]]
    destination.mkdir(parents=True, exist_ok=True)
    labels_dir = destination / "labels"
    labels_dir.mkdir(exist_ok=True)
    n = len(parts)
    xpos, com, simulated, stats = [], [], [], []
    # Two renderers instead of toggling one between modes (toggling crashed segmentation on some scenes).
    renderer = mujoco.Renderer(model, height=height, width=width)
    depth_renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.enable_segmentation_rendering()
    depth_renderer.enable_depth_rendering()
    ax, ay = backproject_grid(P, width, height)
    try:
        for index, target in enumerate(times):
            while data.time < target - 1e-10:
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            if not np.isfinite(data.qpos).all() or any(int(w.number) for w in data.warning):
                raise ValueError("Nonfinite or warned evaluation rollout")
            xpos.append(data.xpos.copy()); com.append(data.subtree_com.copy()); simulated.append(float(data.time))
            renderer.update_scene(data, camera=camera, scene_option=option)
            seg = renderer.render()
            depth_renderer.update_scene(data, camera=camera, scene_option=option)
            depth = depth_renderer.render()
            geoms = seg[..., 0]
            valid = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (geoms >= 0) & (geoms < model.ngeom)
            labels = np.zeros((height, width), dtype=np.uint16)
            labels[valid] = (part_of_geom[geoms[valid]] + 1).astype(np.uint16)
            stats.append(part_stats(labels, depth, n, ax, ay, width))
            Image.fromarray(labels).save(labels_dir / f"{index:06d}.png")
    finally:
        renderer.close(); depth_renderer.close()
    np.savez_compressed(destination / "rollout.npz", times=times, simulated_times=np.array(simulated),
                        body_xpos=np.array(xpos), body_subtree_com=np.array(com),
                        part_stats=np.array(stats) if stats else np.zeros((0, n, 6)), projection=P)
    bodies = [{"id": b, "name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b), "parent": int(model.body_parentid[b]),
               "dofnum": int(model.body_dofnum[b])} for b in range(model.nbody)]
    save_json(destination / "manifest.json", {
        "version": 2, "parts": parts, "bodies": bodies, "width": width, "height": height, "frames": len(times),
        "camera": camera, "projection": P.tolist(), "fovy": float(model.cam_fovy[cam]),
        "visibility": "MuJoCo default geom groups 0-2; alpha-0 geoms hidden; run.py not parsed",
        "hidden_alpha0_geoms": len(invisible),
        "part_stats_columns": ["area", "sum_u", "sum_v", "sum_X", "sum_Y", "sum_Z"],
        "camera_frame": "OpenGL camera coordinates (x right, y up, -z forward), metres of the reconstruction",
        "physics": "initial state assigned once; continuous mj_step; fixed model", "gt_access": False,
        "scene_sha256": sha256(output / "scene.xml"), "state_sha256": sha256(output / "initial_state.json"),
        "time_sampling": "first simulation step at or after each input-video PTS",
        "max_time_error_s": float(np.max(np.array(simulated) - times))})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    render(a.work.resolve(), a.metadata.resolve(), a.output.resolve())
