#!/usr/bin/env python3
"""Build the bench50 benchmark directory (benchmark/README.md) from the raw PhysInOne / Physion v1.5 files.

  python benchmark/build.py --data-root DIR --output BENCH [--spec benchmark/bench50.json] [--tasks ID ...]
                            [--no-prompt] [--jobs N]

DIR holds the source files as benchmark/download.py writes them (<dataset>/<category>/...). Every video is on the
real-time axis of the spec ("timing"): one video second is one second of physics with g = 9.81 in GT metres; Physion
0.01 s per frame, PhysInOne per-clip free fall on the GT time_seconds or the dataset value 2.25 m/s^2. Two encoding
routes ("encoding" of each scene): direct = libx264 crf 18 at the retimed rate (scenes 101-135); two_step = the earlier
30 fps encoding of all source frames, then a lossless re-encode with the new timestamps (scenes 001-038). With FFmpeg
4.4.2 / x264 core 163 the videos are byte-identical to the published runs; a video whose sha256 differs from the spec
is kept and reported (other encoder builds). Finished tasks are skipped when the command is run again.
"""
import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import tempfile
import zipfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src/harness"))
from new_task import SKILL, install_skill  # noqa: E402

PAD = "pad=ceil(iw/2)*2:ceil(ih/2)*2"
TDW_TIMING = "https://github.com/neuroailab/tdw_physics/blob/master/tdw_physics/dataset.py"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def run(cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"{cmd[0]} failed: " + r.stderr[-2000:])
    return r.stdout


def ffmpeg(*args):
    run(["ffmpeg", "-nostdin", "-v", "error", "-y", *args])


def probe(video):
    value = json.loads(run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format",
                            "-show_frames", "-show_entries",
                            "stream=width,height,avg_frame_rate,nb_frames,time_base,start_time,duration:"
                            "format=duration:frame=best_effort_timestamp_time,pkt_duration_time", "-of", "json", video]))
    if not value.get("streams") or not value.get("frames"):
        raise ValueError("Video contains no decodable frames")
    return value


def probe_times(video):
    out = run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "frame=best_effort_timestamp_time",
               "-of", "csv=p=0", video])
    return np.array([float(x) for x in out.split() if x.strip()])


def require_monotonic(times):
    if not times or any(not math.isfinite(t) for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Timestamps must be finite and strictly increasing")


def video_duration(media):
    """Duration of the decoded video: last PTS plus the final frame packet duration (else the stream duration)."""
    frames = media["frames"]
    start, last = float(frames[0]["best_effort_timestamp_time"]), float(frames[-1]["best_effort_timestamp_time"])
    interval = frames[-1].get("pkt_duration_time")
    if interval is not None and math.isfinite(float(interval)) and float(interval) > 0:
        return last - start + float(interval), "Last decoded video PTS plus final frame packet duration, relative to first video PTS"
    duration = media["streams"][0].get("duration")
    if duration is not None and math.isfinite(float(duration)) and float(duration) > last - start:
        return float(duration), "Selected video stream duration; container duration excluded"
    raise ValueError("Cannot establish video end time from final frame packet duration or video stream duration")


def physinone_g(zp):
    """per-clip free fall on GT time_seconds (cm -> m); None when the windows disagree"""
    b = zp.name[:-4]
    d = json.loads(zipfile.ZipFile(zp).read(f"{b}/{b}.json"))
    gs = []
    for a in (d.get("actors") or {}).values():
        for td in [a.get("transform_data") or []] + [c.get("transform_data") or [] for c in (a.get("components") or {}).values()]:
            if len(td) < 12:
                continue
            T = np.array([x["time_seconds"] for x in td]); X = np.array([[x["transform"]["location"][k] / 100 for k in "xyz"] for x in td])
            for s in range(len(T) - 10):
                tt = T[s:s + 10] - T[s]; p = X[s:s + 10]
                if np.ptp(p[:, 2]) < 0.1:
                    continue
                A = np.vstack([np.ones(10), tt, tt * tt]).T
                c, *_ = np.linalg.lstsq(A, p, rcond=None); acc = 2 * c[2]
                rms = np.sqrt(((A @ c - p) ** 2).mean())
                if acc[2] < -0.3 and np.hypot(acc[0], acc[1]) < 0.05 * abs(acc[2]) and rms < 0.01 * np.ptp(p[:, 2]) + 3e-4:
                    gs.append(-acc[2])
    if len(gs) >= 5:
        q1, med, q3 = np.percentile(gs, [25, 50, 75])
        if q3 - q1 <= 0.1 * med:
            return float(med), len(gs)
    return None, len(gs)


def physion_dt(path):
    """seconds per frame implied by the GT rotations and angular velocities"""
    with h5py.File(path, "r") as f:
        F = f["frames"]; ks = sorted(F.keys())
        Q = np.array([F[k]["objects/rotations_cam0"][()] for k in ks]); W = np.array([F[k]["objects/angular_velocities"][()] for k in ks])
    ratios = []
    for i in range(Q.shape[1]):
        dth = 2 * np.arccos(np.clip(np.abs(np.sum(Q[1:, i] * Q[:-1, i], -1)), 0, 1))
        wm = 0.5 * (np.linalg.norm(W[1:, i], axis=1) + np.linalg.norm(W[:-1, i], axis=1))
        m = (wm > 1.0) & (dth < 0.5)
        ratios += list(dth[m] / wm[m])
    return float(np.median(ratios)), len(ratios)


def extract(scene, src, frames, first):
    """write the RGB frames from source frame `first` on; return (source frame ids, source fps, suffix, source times)"""
    if scene["dataset"] == "physinone":
        b = src.name[:-4]
        with zipfile.ZipFile(src) as z:
            members = sorted((int(Path(n).stem), n) for n in z.namelist() if "/CineCamera_0/rgb/" in n and n.endswith(".jpg"))
            members = [m for m in members if m[0] >= first]
            cam = json.loads(z.read(f"{b}/blender_CineCamera_0.json"))
            ids = [i for i, _ in members]
            if not ids or ids != list(range(first, first + len(ids))):
                raise ValueError("discontinuous RGB frames")
            for k, (_, n) in enumerate(members):
                (frames / f"{k:06d}.jpg").write_bytes(z.read(n))
        fps = float(cam["fps"])
        recorded = {int(f["frame"]): float(f["time_abs"]) for f in cam.get("frames", []) if "time_abs" in f}
        use = all(i in recorded for i in ids)  # camera time_abs, else frame / fps
        return [str(i) for i in ids], fps, ".jpg", [recorded[i] if use else i / fps for i in ids]
    with h5py.File(src, "r") as f:
        keys = [k for k in sorted(f["frames"], key=int) if int(k) >= first]
        if not keys or [int(k) for k in keys] != list(range(first, first + len(keys))):
            raise ValueError("discontinuous HDF5 frames")
        for k, key in enumerate(keys):
            data = f["frames"][key]["images/_img_cam0"][()].tobytes()
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("expected PNG frames")
            (frames / f"{k:06d}.png").write_bytes(data)
    return keys, 30.0, ".png", [int(k) / 30.0 for k in keys]  # Physion source rate: tdw_physics default 30 fps


def retime(scene, src, src_fps, timing, dt1=None):
    """(new frame dt, ffmpeg rate, calibration status, basis); dt1 = step-1 frame interval (two_step route)"""
    if scene["dataset"] == "physion_v15":
        if dt1 is None:
            return timing["physion_frame_dt"], "100", "measured", \
                "Physion v1.5 frame dt 0.01 s (angular-velocity check in retime_sources.py)"
        dt, n = physion_dt(src)
        if abs(dt - timing["physion_frame_dt"]) > 0.0005:
            raise ValueError(f"angular check gives {dt:.4f} s/frame, not 0.01")
        return timing["physion_frame_dt"], "100", "measured", f"angular-velocity check {dt:.4f} s/frame over {n} samples"
    g, n = physinone_g(src)
    calib = "measured" if g else "assumed"
    basis = f"free fall {g:.3f} m/s^2 over {n} windows" if g else \
        f"assumed dataset value {timing['physinone_default_g']} m/s^2 ({n} candidate free-fall windows, none consistent)"
    s = math.sqrt((g or timing["physinone_default_g"]) / timing["g"])
    if dt1 is None:
        fps = round(src_fps / s, 4); new_dt = 1 / fps
        return new_dt, f"{int(round(10000 / new_dt))}/10000", calib, basis
    fps = round(1 / (dt1 * s), 4)
    return 1 / fps, f"{int(round(fps * 10000))}/10000", calib, basis


def build(scene, data_root, stage, timing):
    """encode the scene's video into stage/video.mp4 and its metadata into stage/meta/; return the TASK record"""
    jid, enc, first = scene["job_id"], scene["encoding"], int(scene["first_frame"])
    src = data_root / scene["source"]; video = stage / "video.mp4"; meta = stage / "meta"
    physion = scene["dataset"] == "physion_v15"
    meta.mkdir(parents=True)
    warnings = []
    with tempfile.TemporaryDirectory(dir=stage) as tmp:
        tmp = Path(tmp); frames = tmp / "frames"; frames.mkdir()
        if enc["route"] == "direct":
            ids, src_fps, suffix, _ = extract(scene, src, frames, first)
            new_dt, rate, calib, basis = retime(scene, src, src_fps, timing)
            ffmpeg("-framerate", rate, "-start_number", "0", "-i", frames / ("%06d" + suffix), "-an", "-c:v", "libx264",
                   "-threads", enc["x264_threads"], "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-vf", PAD,
                   "-r", rate, "-movflags", "+faststart", video)
        elif enc["route"] == "two_step":
            if first:
                raise ValueError(f"{jid}: two_step keeps all source frames")
            ids, src_fps, suffix, times = extract(scene, src, frames, 0)
            if src_fps != enc["step1_fps"] or any(abs(b - a - 1 / src_fps) > 1e-5 for a, b in zip(times, times[1:])):
                raise ValueError(f"{jid}: source timing is not uniform at {enc['step1_fps']} fps")
            step1 = tmp / "step1.mp4"
            ffmpeg("-threads", "1", "-framerate", str(float(enc["step1_fps"])), "-start_number", "0",
                   "-i", frames / ("%06d" + suffix), "-an", "-c:v", "libx264", "-threads", "1", "-preset", "veryfast",
                   "-crf", "18", "-pix_fmt", "yuv420p", "-vf", PAD, "-movflags", "+faststart", step1)
            if sha256(step1) != enc["step1_sha256"]:
                warnings.append("step-1 video sha256 differs from the spec")
            dt1 = float(np.median(np.diff(probe_times(step1))))
            new_dt, rate, calib, basis = retime(scene, src, src_fps, timing, dt1)
            ffmpeg("-i", step1, "-an", "-vf", f"setpts=N/({rate})/TB", "-r", rate, "-c:v", "libx264",
                   "-threads", enc["x264_threads"], "-pix_fmt", "yuv420p", "-qp", "0", "-bf", "0", "-preset", "veryfast", video)
        else:
            raise ValueError(f"{jid}: unknown encoding route {enc['route']}")
    media = probe(video)
    pts = [float(x["best_effort_timestamp_time"]) for x in media["frames"]]
    require_monotonic(pts)
    if len(pts) != len(ids) or np.max(np.abs(np.diff(pts) - new_dt)) > 2e-4:
        raise ValueError(f"{jid}: encoded frames/PTS do not match the source frames")
    duration, dbasis = video_duration(media)
    digest = sha256(video)
    if enc["route"] == "direct":
        rows = [{"index": i, "source_frame": fid, "source_time_seconds": round(int(fid) * new_dt, 6), "frame_file": "",
                 "relative_time_seconds": round(p - pts[0], 6), "original_source_time_seconds": int(fid) / src_fps}
                for i, (fid, p) in enumerate(zip(ids, pts))]
        timeline = {"dataset": scene["dataset"], "frame_count": len(rows), "source_start_seconds": first * new_dt,
                    "first_source_frame": first, "trimmed_leading_frames": first,
                    "last_frame_relative_seconds": rows[-1]["relative_time_seconds"], "video_duration_seconds": duration,
                    "all_source_frames_retained": first == 0, "video_sha256": digest,
                    "camera": "cam0" if physion else "CineCamera_0", "fps": 1 / new_dt, "timing_status": "source_metadata",
                    "timing_basis": f"retimed so g = 9.81 in GT metres ({calib}: {basis}); encoded directly at the retimed rate",
                    "video_duration_basis": dbasis, "container_duration_seconds": float(media["format"]["duration"]),
                    "retiming_calibration_status": calib, "retiming_basis": basis,
                    "trim_reason": scene.get("trim_reason") if first else None}
    else:
        # GT frame identity kept; times are the retimed video PTS, the step-1 source times stay as original_*
        rows = [{"index": i, "source_frame": fid, "source_time_seconds": p, "frame_file": "",
                 "relative_time_seconds": p - pts[0], "original_source_time_seconds": t}
                for i, (fid, p, t) in enumerate(zip(ids, pts, times))]
        timeline = {"dataset": scene["dataset"], "frame_count": len(rows), "source_start_seconds": pts[0],
                    "last_frame_relative_seconds": rows[-1]["relative_time_seconds"], "video_duration_seconds": duration,
                    "all_source_frames_retained": True, "video_sha256": digest,
                    "camera": "cam0" if physion else "CineCamera_0",
                    "fps": float(Fraction(media["streams"][0]["avg_frame_rate"])), "timing_status": "source_metadata",
                    "timing_basis": "Provided retimed MP4 decoded frame PTS; original GT source_frame mapping retained",
                    **({"timing_reference": TDW_TIMING} if physion else {}),
                    "video_duration_basis": dbasis, "container_duration_seconds": float(media["format"]["duration"]),
                    "retiming_calibration_status": calib, "retiming_basis": basis, "original_timeline": None}
    with (meta / "timestamps.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    save_json(meta / "timeline.json", timeline); save_json(meta / "media.json", media)
    expected = scene["video"]
    if digest != expected["sha256"]:
        warnings.append(f"video sha256 {digest} differs from the spec {expected['sha256']} (encoder build?)")
    if len(rows) != expected["frame_count"] or abs(new_dt - expected["frame_dt"]) > 1e-12:
        warnings.append(f"frames/frame dt {len(rows)}/{new_dt} differ from the spec {expected['frame_count']}/{expected['frame_dt']}")
    task = {"job_id": jid, "dataset": scene["dataset"], "category": scene["category"], "sample_id": scene["sample_id"],
            "source": {"kind": "hdf5_rgb" if physion else "zip_rgb", "path": str(src), **({} if physion else {"camera": "CineCamera_0"})},
            "first_frame": first,
            "curation": {"why": scene["why"], **({"trim_reason": scene["trim_reason"]} if scene.get("trim_reason") else {})},
            "video": {"sha256": digest, "frame_count": len(rows), "frame_dt": new_dt}}
    if warnings:
        task["warnings"] = warnings
    return task


def build_job(args):
    scene, data_root, stage, timing = args
    try:
        return scene["job_id"], build(scene, data_root, stage, timing), None
    except Exception as exc:  # reported by the parent; the other tasks go on
        return scene["job_id"], None, f"{type(exc).__name__}: {exc}"


def remove(path):
    if path.exists():
        for p in [path, *path.rglob("*")]:
            if p.is_dir() and not p.is_symlink():
                p.chmod(0o755)
        shutil.rmtree(path)


def install(root, stage, task, prompt):
    """tasks/<id>/ as src/harness/new_task.py makes it, then meta/<id>/ (last: the task is complete)"""
    jid = task["job_id"]; work = root / "tasks" / jid
    (work / "input").mkdir(parents=True); (work / "output").mkdir()
    (stage / "video.mp4").replace(work / "input/video.mp4"); (work / "input/video.mp4").chmod(0o444)
    install_skill(work)
    if prompt:
        shutil.copyfile(SKILL / "prompts/task.md", work / "PROMPT.md")
    (root / "meta").mkdir(exist_ok=True)
    (stage / "meta").replace(root / "meta" / jid)
    remove(stage)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True, help="directory with the downloaded source files")
    ap.add_argument("--output", type=Path, required=True, help="benchmark directory BENCH")
    ap.add_argument("--spec", type=Path, default=HERE / "bench50.json")
    ap.add_argument("--tasks", nargs="+", metavar="ID", help="build only these job ids")
    ap.add_argument("--no-prompt", action="store_true", help="do not write tasks/<id>/PROMPT.md")
    ap.add_argument("--jobs", type=int, default=1, help="scenes built in parallel")
    a = ap.parse_args()
    spec = json.loads(a.spec.read_text(encoding="utf-8")); spec_sha = sha256(a.spec)
    data_root, root = a.data_root.resolve(), a.output.resolve()
    scenes = spec["scenes"]
    if a.tasks:
        unknown = set(a.tasks) - {s["job_id"] for s in scenes}
        if unknown:
            ap.error(f"unknown job ids: {sorted(unknown)}")
        scenes = [s for s in scenes if s["job_id"] in a.tasks]
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            ap.error(f"{tool} not found on PATH")
    index_path = root / "benchmark.json"
    index = {"name": spec["name"], "created_at": datetime.now(timezone.utc).isoformat(), "data_root": str(data_root),
             "spec_sha256": spec_sha, "ffmpeg": run(["ffmpeg", "-version"]).splitlines()[0], "tasks": []}
    if index_path.exists():
        old = json.loads(index_path.read_text(encoding="utf-8"))
        if (old["spec_sha256"], old["data_root"]) != (spec_sha, str(data_root)):
            ap.error(f"{index_path} was built from another spec or data root")
        index.update(created_at=old["created_at"], tasks=old["tasks"])
    done = {t["job_id"]: t for t in index["tasks"]}
    todo = [s for s in scenes if s["job_id"] not in done]
    missing = [s["source"] for s in todo if not (data_root / s["source"]).is_file()]
    if missing:
        ap.error(f"{len(missing)} source files missing under {data_root} (benchmark/download.py), e.g. {missing[0]}")
    for s in todo:  # leftovers of an interrupted build; never remove a task an agent has written to
        work = root / "tasks" / s["job_id"]
        if (work / "output").is_dir() and any((work / "output").iterdir()):
            ap.error(f"{work} is not in {index_path} but has outputs; move it away first")
        remove(work); remove(root / "meta" / s["job_id"]); remove(root / ".build" / s["job_id"])
    root.mkdir(parents=True, exist_ok=True)
    order = {s["job_id"]: i for i, s in enumerate(spec["scenes"])}
    jobs = [(s, data_root, root / ".build" / s["job_id"], spec["timing"]) for s in todo]
    failed = []
    with ProcessPoolExecutor(a.jobs) as pool:
        for n, (jid, task, error) in enumerate(pool.map(build_job, jobs) if a.jobs > 1 else map(build_job, jobs), 1):
            if error:
                failed.append(jid); print(f"[{n}/{len(jobs)}] FAILED {jid}: {error}", flush=True)
                continue
            install(root, root / ".build" / jid, task, not a.no_prompt)
            done[jid] = task
            index["tasks"] = sorted(done.values(), key=lambda t: order[t["job_id"]])
            save_json(index_path, index)
            status = "ok" if "warnings" not in task else "WARNING: " + "; ".join(task["warnings"])
            print(f"[{n}/{len(jobs)}] {jid}: {task['video']['frame_count']} frames, {status}", flush=True)
    if (root / ".build").is_dir() and not any((root / ".build").iterdir()):
        (root / ".build").rmdir()
    if not index_path.exists():
        save_json(index_path, index)
    built = [done[s["job_id"]] for s in scenes if s["job_id"] in done]
    mismatched = [t["job_id"] for t in built if t.get("warnings")]
    print(f"{index_path}: {len(index['tasks'])} tasks; of the {len(scenes)} requested: {len(built) - len(mismatched)} match "
          f"the spec, {len(mismatched)} with warnings, {len(failed)} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
