"""Pi3X / SAM3 on the shared GPU through the host queue: submit without blocking, collect later.

The GPU is outside this sandbox (CUDA does not work in here). Requests go through tools/_gpu_call.py's file
queue (.gpu/requests) to a host service; a call takes 0.5-3 min of GPU time plus the queue wait. These commands
wrap tools/pi3x_recon and tools/sam3_track (same services, same outputs); you can also call those directly.

  python -m r2s gpu pi3x [--max-frames 48]     whole video, evenly sampled, frame 0 included -> analysis/pi3x
  python -m r2s gpu sam3 --name ball --prompt "ball" [--box x0 y0 x1 y1] [--box-only] [--prompt-frame K]
        --box in VIDEO PIXELS on the prompt frame. Without --box-only, the box is an exemplar for the text prompt
        and SAM3 may return several instances (one object id each); with --box-only it tracks the boxed object.
        -> analysis/sam3_<name>
  python -m r2s gpu sam3 --obj NAME PROMPT X0 Y0 X1 Y1 FRAME [--obj ...]   several requests in one command
  python -m r2s gpu members --name dominoes --prompt "domino" [--prompt "red tile" ...] [--frame 0]
        [--frames K ... | --every N] [--region x0 y0 x1 y1] [--threshold 0.4] [--grid auto|0-4]
        many similar objects (a domino run, stacked cups, a pile): every instance on --frame by SAM3's image
        detector, on the whole frame and on finer overlapping tiles while the detector is near its 200-query
        capacity (video tracking loses crowds); prompts are merged. Members 1..N on --frame, id 0 = their union
        (also on --frames / every N-th frame; identities are not followed between frames) -> analysis/sam3_<name>
        (members.json; numbered sheet: `r2s sam3 members DIR`; scene vs members: `r2s check members`)
  python -m r2s gpu wait analysis/pi3x [--timeout 240]   blocks up to --timeout s, prints a summary
  python -m r2s gpu wait --all [--tracks]       every submitted job; --tracks also writes SAM3 tracks
        (r2s track sam3) named <name> (one id) or <name>_<id>
        (give the shell call a timeout >= --timeout + 60 s, or it is killed with exit 130/143)
  python -m r2s gpu status
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .common import ANALYSIS, VIDEO, load_json, save_json, timeline

JOBS = ANALYSIS / "gpu_jobs.json"
QUEUE, RESULTS = Path(".gpu/requests"), Path(".gpu/results")


def _jobs():
    return load_json(JOBS) if JOBS.exists() else {}


def submit(tool, request, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    request = dict(request, output_dir=str(out.resolve()), overwrite=True)   # re-submitting into the same dir is fine
    if not Path("tools/_gpu_call.py").exists():
        raise SystemExit("no tools/_gpu_call.py: GPU tools are not available in this run")
    QUEUE.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(json.dumps(dict(tool=tool, request=request), sort_keys=True).encode()).hexdigest()[:12]
    rid = f"{tool}-{key}.json"
    if not (RESULTS / rid).exists() and not (QUEUE / rid).exists():
        tmp = QUEUE / (rid + ".tmp")
        tmp.write_text(json.dumps(dict(tool=tool, request=request)))
        tmp.rename(QUEUE / rid)
    jobs = _jobs()
    jobs[str(out)] = dict(tool=tool, result=str(RESULTS / rid), queued=str(QUEUE / rid), submitted=time.time(),
                          prompt=request.get("prompt"))
    save_json(jobs, JOBS)
    return str(out)


def done(job):
    return Path(job["result"]).exists() and not Path(job["queued"]).exists()


def wait(out, timeout=240):
    jobs = _jobs()
    job = jobs.get(str(Path(out)))
    if job is None:
        raise SystemExit(f"no job for {out}; known: {list(jobs)}")
    t0 = time.time()
    while not done(job):
        if time.time() - t0 > timeout:
            print(f"{job['tool']} {out}: still queued/running ({time.time() - job['submitted']:.0f}s since submit; "
                  "the GPU is shared, 5-15 min is common)")
            return None
        time.sleep(3)
    res = load_json(job["result"])
    print(f"{job['tool']} {out}: ok={res.get('ok')} service={res.get('service_seconds')}s, "
          f"{time.time() - job['submitted']:.0f}s since submit")
    if not res.get("ok"):
        print("  error:", str(res.get("error", ""))[:600])
        if "out of memory" in json.dumps(res).lower():
            print("  -> GPU OOM: resubmit with fewer frames (pi3x --max-frames 32 / sam3 --max-frames)")
        return res
    if job["tool"] == "pi3x":
        from .pi3x import summarize
        s = summarize(out)
        print("  " + ", ".join(f"{k}={v}" for k, v in s.items()))
    elif res.get("mode") == "members":
        from .sam3 import members_summary
        members_summary(out, verbose=True)
    else:
        from .sam3 import summarize as ssum
        ssum(out, verbose=True)
    return res


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="r2s gpu", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pi3x")
    p.add_argument("--out", default=str(ANALYSIS / "pi3x"))
    p.add_argument("--max-frames", type=int, default=48)
    s = sub.add_parser("sam3")
    s.add_argument("--name", help="short object name; output analysis/sam3_<name>")
    s.add_argument("--prompt", help="text prompt, e.g. 'ball', 'red cube', 'bowl'")
    s.add_argument("--obj", nargs=7, action="append", metavar=("NAME", "PROMPT", "X0", "Y0", "X1", "Y1", "FRAME"),
                   help="submit several objects in ONE command: --obj ball ball 680 315 790 430 0 --obj ring ring ...")
    s.add_argument("--box", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), help="video pixels")
    s.add_argument("--box-only", action="store_true", help="track only the boxed object (box is not just an exemplar)")
    s.add_argument("--prompt-frame", type=int, default=0)
    s.add_argument("--threshold", type=float, default=0.4)
    s.add_argument("--max-frames", type=int)
    s.add_argument("--out", help="output dir (default analysis/sam3_<name>); with several objects a prefix: <out>_<name>")
    mb = sub.add_parser("members")
    mb.add_argument("--name", required=True, help="short class name; output analysis/sam3_<name>")
    mb.add_argument("--prompt", action="append", required=True,
                    help="text prompt; repeat for members of another look (detections of all prompts are merged)")
    mb.add_argument("--frame", type=int, default=0, help="frame whose members are numbered (default 0)")
    mb.add_argument("--frames", type=int, nargs="+", default=[], help="more frames: union of the detections (id 0)")
    mb.add_argument("--every", type=int, help="union (id 0) also on every N-th frame")
    mb.add_argument("--region", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    help="only detect inside this box (video px)")
    mb.add_argument("--threshold", type=float, default=0.4)
    mb.add_argument("--grid", default="auto",
                    help="auto (finer tiles while the detector is crowded and they still add members) or 0-4 rows")
    mb.add_argument("--out", help="output dir (default analysis/sam3_<name>)")
    w = sub.add_parser("wait")
    w.add_argument("out", nargs="?")
    w.add_argument("--all", action="store_true", help="wait for every submitted job")
    w.add_argument("--tracks", action="store_true",
                   help="also convert every finished SAM3 result to tracks (single id -> <name>, else <name>_<id>)")
    w.add_argument("--timeout", type=float, default=240, help="keep < 5 min: longer shell calls get interrupted")
    st = sub.add_parser("status")
    st.add_argument("dir", nargs="?", help="ignored; status lists all jobs")
    a = ap.parse_args(argv)
    if a.cmd == "pi3x":
        tl = timeline()
        dur = max(tl["last_t"], 1e-3)
        fps = tl["fps"]
        step = max(1, int(-(-(tl["n"] - 1) // (a.max_frames - 1))))      # frames between samples
        req = dict(input_path=str(VIDEO.resolve()), sample_fps=fps / step, max_frames=a.max_frames)
        submit("pi3x", req, a.out)
        print(f"submitted pi3x -> {a.out} (every {step} frames, {min(a.max_frames, tl['n'] // step + 1)} samples, "
              f"{dur:.2f}s video); collect with `python -m r2s gpu wait {a.out}`")
    elif a.cmd == "sam3":
        tl = timeline()
        W, H = tl["width"], tl["height"]
        items = []
        for n, pr, *nums in (a.obj or []):
            try:
                items.append((n, pr, [float(v) for v in nums[:4]], int(float(nums[4]))))
            except ValueError:
                raise SystemExit(f"--obj {n} {pr} {' '.join(nums)}: X0 Y0 X1 Y1 FRAME must be numbers "
                                 "(quote multi-word prompts)") from None
        if a.name or a.prompt:
            if not (a.name and a.prompt):
                raise SystemExit("give both --name and --prompt (or use --obj NAME PROMPT X0 Y0 X1 Y1 FRAME)")
            items.append((a.name, a.prompt, a.box, a.prompt_frame))
        if not items:
            raise SystemExit("give --name/--prompt [--box] or one --obj per object")
        for name, prompt, box, frame in items:
            req = dict(mode="video", input_path=str(VIDEO.resolve()), prompt=prompt, prompt_frame=frame,
                       confidence_threshold=a.threshold)
            if a.max_frames:
                req["max_frames"] = a.max_frames
            if box:
                x0, y0, x1, y1 = box
                x0, x1 = sorted((min(max(x0, 0), W - 1), min(max(x1, 0), W - 1)))     # SAM3 rejects boxes outside
                y0, y1 = sorted((min(max(y0, 0), H - 1), min(max(y1, 0), H - 1)))     # the frame
                req.update(prompt_box_xywh=[x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H], prompt_box_only=bool(a.box_only))
            out = (Path(a.out) if len(items) == 1 else Path(f"{a.out}_{name}")) if a.out else ANALYSIS / f"sam3_{name}"
            submit("sam3", req, out)
            print(f"submitted sam3 '{prompt}' -> {out}")
        print("collect with `python -m r2s gpu wait --all [--tracks]`")
    elif a.cmd == "members":
        tl = timeline()
        if not 0 <= a.frame < tl["n"] or any(not 0 <= k < tl["n"] for k in a.frames):
            raise SystemExit(f"frames must be within 0..{tl['n'] - 1}")
        if a.grid not in ("auto", "0", "1", "2", "3", "4"):
            raise SystemExit("--grid: auto or 0-4")
        more = list(a.frames) + (list(range(0, tl["n"], a.every)) if a.every else [])
        frames = [a.frame] + sorted(set(more) - {a.frame})
        req = dict(mode="members", input_path=str(VIDEO.resolve()), prompts=a.prompt, frames=frames,
                   confidence_threshold=a.threshold, grid=a.grid if a.grid == "auto" else int(a.grid))
        if a.region:
            x0, y0, x1, y1 = a.region
            W, H = tl["width"], tl["height"]
            x0, x1 = sorted((min(max(x0, 0), W), min(max(x1, 0), W)))
            y0, y1 = sorted((min(max(y0, 0), H), min(max(y1, 0), H)))
            if x1 - x0 < 8 or y1 - y0 < 8:
                raise SystemExit("--region is empty inside the frame")
            req["region_xyxy"] = [x0, y0, x1, y1]
        out = Path(a.out) if a.out else ANALYSIS / f"sam3_{a.name}"
        submit("sam3", req, out)
        print(f"submitted sam3 members {a.prompt} on frame {a.frame}" + (f" (+ union on {len(frames) - 1} frames)"
              if len(frames) > 1 else "") + f" -> {out}; collect with `python -m r2s gpu wait {out}`")
    elif a.cmd == "wait":
        outs = list(_jobs()) if a.all else [a.out]
        if not outs or outs == [None]:
            raise SystemExit("give a job directory or --all")
        deadline = time.time() + a.timeout            # one budget for all jobs, so the shell call stays short
        for out in outs:
            res = wait(out, max(5.0, deadline - time.time()))
            if a.tracks and res and res.get("ok") and res.get("mode") == "members":
                from .track import from_sam3
                base = Path(out).name.replace("sam3_", "")
                from_sam3(out, [0], [f"{base}_union"])      # members have no tracks (one frame each)
                print(f"  -> track {base}_union (union of the members on each listed frame)")
            elif a.tracks and res and res.get("ok") and _jobs()[str(Path(out))]["tool"] == "sam3":
                from .sam3 import summarize as ssum
                from .track import from_sam3
                ids = sorted(int(k) for k in ssum(out)["objects"])
                base = Path(out).name.replace("sam3_", "")
                names = [base] if len(ids) == 1 else [f"{base}_{i}" for i in ids]
                if ids:
                    from_sam3(out, ids, names)
                    print(f"  -> tracks {names} (masks drawn: `python -m r2s sam3 sheet {out}`; other ids/names: "
                          f"`python -m r2s track sam3 {out} --ids .. --names ..`)")
    else:
        for out, j in _jobs().items():
            print(f"{out}: {j['tool']} {'done' if done(j) else 'queued/running'} ({time.time() - j['submitted']:.0f}s)")
