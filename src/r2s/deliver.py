"""Deliverable chores: the run.py template, preview.mp4 on the input timeline, and a format check.

  python -m r2s deliver [--camera NAME] [--template]
        writes output/video_timeline.json (decoded PTS of the input frames); writes output/run.py from the template
        below if there is none (--template: replace yours); renders output/preview.mp4 with that template
        (`run.py --headless --render --duration <video duration> --output output`); then validates.
        With your own run.py, preview.mp4 is yours too (deliver only validates).
  python -m r2s deliver --validate-only

Template output/run.py: loads scene.xml + initial_state.json once, steps MuJoCo continuously, stores qpos and body
positions at the input frame times (simulation.npz), --render writes preview.mp4 from camera NAME (written into
the file literally; default: the scene's only camera, else video_cam) with the input's frame times; with an
output/camera_track.json it moves that camera to the track's pose before drawing each frame (render-only).

Validation (analysis/validation.json): the five files of prompt item 8 exist and are not empty; scene.xml loads, has
no actuators / mocap / plugins and asset paths inside output/; initial_state.json has nq / nv entries and time 0;
a full-length rollout stays finite and raises no MuJoCo warning; preview.mp4 has the input's frame count and PTS;
the preview camera is identifiable from the files (a literal camera="NAME" in run.py's update_scene call, or a
scene with one camera): the scene is re-rendered externally through that camera; run.py does not mention input/,
analysis/, tracks or optimisers (prompt item 8); an output/camera_track.json has one finite pose per input frame
for the preview camera, that camera is attached to the world body, and run.py reads the file.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from .common import OUTPUT, ANALYSIS, load_json, save_json, timeline

TEMPLATE_MARK = "Passive MuJoCo replay of the reconstructed scene"

RUN_PY = r'''#!/usr/bin/env python3
"""Passive MuJoCo replay of the reconstructed scene: fixed model, initial state set once at t=0,
continuous mj_step. No reference video, target trajectory or optimiser is loaded.

usage: python run.py --headless --duration SECONDS --output DIR [--render]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--output", default=str(HERE))
    ap.add_argument("--render", action="store_true", help="also write preview.mp4 from camera @CAMERA@")
    a = ap.parse_args()
    if a.headless or a.render:
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(HERE / "scene.xml"))
    data = mujoco.MjData(model)
    st = json.loads((HERE / "initial_state.json").read_text())
    data.qpos[:] = st["qpos"]
    data.qvel[:] = st["qvel"]
    mujoco.mj_forward(model, data)
    tl = json.loads((HERE / "video_timeline.json").read_text())   # frame times of the input video (seconds)
    times = np.array(tl["t"], float)
    duration = a.duration if a.duration is not None else float(tl["duration"])
    times = times[times <= duration + 1e-9]
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    renderer = None
    frames_dir = None
    cam_track = None
    if a.render and (HERE / "camera_track.json").exists():   # moving camera: one pose of the preview camera per frame
        cam_track = json.loads((HERE / "camera_track.json").read_text())
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "@CAMERA@")
    if a.render:
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(tl["width"]))
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(tl["height"]))
        renderer = mujoco.Renderer(model, height=int(tl["height"]), width=int(tl["width"]))
        frames_dir = out / "_preview_frames"
        frames_dir.mkdir(exist_ok=True)
    qpos, xpos, simt = [], [], []
    for k, t in enumerate(times):
        while data.time < t - 1e-10:
            mujoco.mj_step(model, data)
        qpos.append(data.qpos.copy())
        xpos.append(data.xpos.copy())
        simt.append(data.time)
        if renderer is not None:
            if cam_track is not None:   # render-only: replaces the pose update_scene reads; mj_step recomputes it
                data.cam_xpos[cam_id] = cam_track["pos"][k]
                mujoco.mju_quat2Mat(data.cam_xmat[cam_id], np.array(cam_track["quat"][k], float))
            renderer.update_scene(data, camera="@CAMERA@")
            from PIL import Image
            Image.fromarray(renderer.render()).save(frames_dir / f"{k:06d}.png")
    while data.time < duration - 1e-10:
        mujoco.mj_step(model, data)
    np.savez(out / "simulation.npz", frame_times=times, sim_times=np.array(simt), qpos=np.array(qpos),
             body_xpos=np.array(xpos), body_names=np.array([model.body(i).name for i in range(model.nbody)]))
    if renderer is not None:
        renderer.close()
        enc = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out / "preview.mp4")]
        import subprocess
        if tl.get("uniform") and tl.get("fps_nominal"):
            # same rational frame rate as the input -> identical PTS
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-framerate", str(tl["fps_nominal"]), "-i",
                            str(frames_dir / "%06d.png")] + enc, check=True)
        else:
            lines = []
            for k in range(len(times)):
                lines.append(f"file '{k:06d}.png'")
                dt = times[k + 1] - times[k] if k + 1 < len(times) else (times[1] - times[0] if len(times) > 1 else 0.04)
                lines.append(f"duration {dt:.6f}")
            (frames_dir / "list.txt").write_text("\n".join(lines) + "\n")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(frames_dir / "list.txt"),
                            "-vsync", "0"] + enc, check=True)
        for p in frames_dir.iterdir():
            p.unlink()
        frames_dir.rmdir()
    print(json.dumps({"frames": len(times), "sim_end_time": float(data.time), "output": str(out)}))


if __name__ == "__main__":
    main()
'''


def preview_camera(model, script):
    import mujoco
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
    used = re.findall(r"update_scene\([^\n]*camera\s*=\s*['\"]([^'\"]+)['\"]", script)
    if used:
        return used[0], names
    return (names[0] if len(names) == 1 else None), names


def validate():
    import mujoco
    import xml.etree.ElementTree as ET
    from .sim import rollout
    problems, notes = [], []
    for r in ["scene.xml", "initial_state.json", "parameters.json", "run.py", "preview.mp4"]:
        f = OUTPUT / r
        if not f.exists() or f.stat().st_size == 0:
            problems.append(f"missing/empty output/{r}")
    if not (OUTPUT / "scene.xml").exists():
        return _finish(problems, notes)
    try:
        m = mujoco.MjModel.from_xml_path(str(OUTPUT / "scene.xml"))
    except Exception as e:
        problems.append(f"scene.xml does not load: {e}")
        return _finish(problems, notes)
    if m.nu or m.nmocap:
        problems.append("actuators or mocap bodies present (the final run must be passive)")
    if m.nv == 0:
        problems.append("no degrees of freedom")
    for node in ET.parse(OUTPUT / "scene.xml").iter():
        if node.tag == "plugin":
            problems.append("plugins in scene.xml")
        ref = node.get("file")
        if ref and (Path(ref).is_absolute() or ".." in Path(ref).parts):
            problems.append(f"asset path outside output/: {ref}")
    if (OUTPUT / "initial_state.json").exists():
        st = load_json(OUTPUT / "initial_state.json")
        if len(st.get("qpos", [])) != m.nq or len(st.get("qvel", [])) != m.nv or st.get("time") != 0:
            problems.append(f"initial_state.json: needs {m.nq} qpos, {m.nv} qvel, time 0.0")
        else:
            tl = timeline()
            r = rollout(OUTPUT / "scene.xml", OUTPUT / "initial_state.json", model=m)
            if r["warning"]:
                problems.append(f"rollout: {r['warning']}")
            if len(r["times"]) != tl["n"]:
                problems.append(f"rollout covered {len(r['times'])}/{tl['n']} frames")
    script = (OUTPUT / "run.py").read_text() if (OUTPUT / "run.py").exists() else ""
    cam, names = preview_camera(m, script)
    notes.append(f"cameras in scene.xml: {names}; preview camera: {cam}")
    if cam is None:
        problems.append(f"preview camera not identifiable: scene.xml has cameras {names} and run.py has no literal "
                        "update_scene(..., camera=\"NAME\")")
    elif cam not in names:
        problems.append(f"run.py renders camera {cam!r}, scene.xml has {names}")
    if (OUTPUT / "camera_track.json").exists() and cam in names:
        _check_camera_track(m, cam, script, problems, notes)
    for bad in ("input/video", "analysis/", "tracks", "least_squares", "minimize"):
        if bad in script:
            problems.append(f"run.py mentions '{bad}' (must not load reference data or optimisers)")
    pv = OUTPUT / "preview.mp4"
    if pv.exists():
        tl = timeline()
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_entries",
                              "frame=best_effort_timestamp_time", "-of", "json", str(pv)], capture_output=True, text=True)
        ts = [float(f["best_effort_timestamp_time"]) for f in json.loads(out.stdout or "{}").get("frames", [])
              if f.get("best_effort_timestamp_time") not in (None, "N/A")]
        if len(ts) != tl["n"]:
            problems.append(f"preview.mp4 has {len(ts)} frames, input has {tl['n']}")
        else:
            err = float(np.max(np.abs((np.array(ts) - ts[0]) - np.array(tl["t"]))))
            notes.append(f"preview PTS max deviation from input PTS: {err * 1000:.2f} ms")
            if err > 0.002:
                problems.append(f"preview PTS deviate from input PTS by up to {err * 1000:.1f} ms")
    return _finish(problems, notes)


def _check_camera_track(m, cam, script, problems, notes):
    import mujoco
    from .common import load_camera_track
    try:
        name, pos, quat = load_camera_track(OUTPUT / "camera_track.json", timeline()["n"])
    except (OSError, ValueError, KeyError) as e:
        problems.append(f"camera_track.json: {e}")
        return
    ci = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam)
    if name != cam:
        problems.append(f"camera_track.json is for camera {name!r}, the preview camera is {cam!r}")
    if int(m.cam_bodyid[ci]) != 0 or int(m.cam_mode[ci]) != int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED):
        problems.append("camera_track.json: the preview camera must be attached to the world body (mode fixed)")
    if not (np.isfinite(pos).all() and np.isfinite(quat).all()) or np.abs(np.linalg.norm(quat, axis=1) - 1).max() > 1e-3:
        problems.append("camera_track.json: non-finite poses or quaternions that are not unit")
    if "camera_track.json" not in script:
        problems.append("output/camera_track.json exists but run.py does not read it (the preview would ignore it)")
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    q0 = np.zeros(4)
    mujoco.mju_mat2Quat(q0, d.cam_xmat[ci])
    rot = np.degrees(2 * np.arccos(np.clip(np.abs(quat @ q0), 0, 1)))
    notes.append(f"camera_track.json: {len(pos)} poses of {cam} (render-only); vs the scene's pose up to "
                 f"{rot.max():.2f} deg, {np.linalg.norm(pos - d.cam_xpos[ci], axis=1).max():.4g} units; frame 0 "
                 f"{rot[0]:.2f} deg, {np.linalg.norm(pos[0] - d.cam_xpos[ci]):.4g} units")


def _finish(problems, notes):
    res = dict(ok=not problems, problems=problems, notes=notes)
    save_json(res, ANALYSIS / "validation.json")
    for n in notes:
        print(n)
    print("format ok" if res["ok"] else "problems:\n  - " + "\n  - ".join(problems))
    return res


def deliver(camera=None, template=False, validate_only=False):
    if not validate_only:
        import mujoco
        from .sim import scene_camera
        if not (OUTPUT / "scene.xml").exists():
            raise SystemExit("no output/scene.xml")
        m = mujoco.MjModel.from_xml_path(str(OUTPUT / "scene.xml"))
        tl = timeline()
        save_json({k: tl[k] for k in ("n", "width", "height", "fps", "fps_nominal", "uniform", "duration", "last_t", "t")} |
                  {"note": "decoded PTS of the input video frames (seconds, first frame = 0)"}, OUTPUT / "video_timeline.json")
        own = (OUTPUT / "run.py").exists() and TEMPLATE_MARK not in (OUTPUT / "run.py").read_text()
        if template or not own:
            (OUTPUT / "run.py").write_text(RUN_PY.replace("@CAMERA@", scene_camera(m, camera)))
            own = False
        if own:
            print("output/run.py is your own: preview.mp4 not rendered by deliver")
        else:
            env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"))
            p = subprocess.run([sys.executable, str(OUTPUT / "run.py"), "--headless", "--render", "--duration",
                                str(tl["duration"]), "--output", str(OUTPUT)], capture_output=True, text=True, env=env)
            print(p.stdout.strip()[-500:])
            if p.returncode:
                print(p.stderr[-2000:])
    return validate()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="r2s deliver", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", help="camera the template renders (default: the scene's only camera, else video_cam)")
    ap.add_argument("--template", action="store_true", help="replace an existing output/run.py with the template")
    ap.add_argument("--validate-only", action="store_true")
    a = ap.parse_args(argv)
    deliver(a.camera, a.template, a.validate_only)
