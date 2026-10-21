"""Video timeline, frame access, contact sheets and frame-difference events.

CLI (run from the task directory):
  python -m r2s probe                         -> analysis/timeline.json (n, fps, size, duration, t = PTS of every
                                                 frame in seconds; one line) (+ frame_times.json = the same t list)
  python -m r2s sheet --every 10 [--frames 0 40 80] [--times 0 1.2] [--crop x0 y0 x1 y1] [--zoom 2]
                      [--grid 100] [--cols 6] [--thumb 480] [-o analysis/sheet.jpg]
        contact sheet, every tile labelled "f<frame> t=<sec>"; --grid draws pixel coordinates so you can read
        positions off the image (use the printed path with your image viewer)
  python -m r2s motion [--thresh 18] [--min-area 30]
        -> analysis/motion.json + analysis/motion.png: per-frame change (mean |frame difference|, median
           filtered), moving blobs per frame vs a median background, and the frames where the change starts
           (onset), peaks and stops (settle).
  python -m r2s dump --frames 0 50 100 [-o analysis/frames]   (PNG files)
"""
from __future__ import annotations

import json
import subprocess
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

from .common import ANALYSIS, VIDEO, save_json


def probe(video=VIDEO, out=ANALYSIS / "timeline.json") -> dict:
    """Decoded per-frame PTS (seconds, relative to frame 0) + stream info."""
    info = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames:format=duration", "-of", "json", str(video)]))
    frames = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames", "-show_entries",
         "frame=best_effort_timestamp_time,pts_time", "-of", "json", str(video)]))["frames"]
    t = np.array([float(f.get("best_effort_timestamp_time") or f.get("pts_time")) for f in frames])
    t = t - t[0]
    st = info["streams"][0]
    dt = np.diff(t)
    tl = dict(video=str(video), n=len(t), width=int(st["width"]), height=int(st["height"]),
              fps_nominal=st.get("avg_frame_rate"), fps=float(1 / np.median(dt)) if len(dt) else None,
              duration=float(info["format"]["duration"]), last_t=float(t[-1]),
              dt_min=float(dt.min()) if len(dt) else None, dt_max=float(dt.max()) if len(dt) else None,
              uniform=bool(len(dt) == 0 or (dt.max() - dt.min()) < 1e-4), t=t.round(6).tolist())
    save_json(tl, out, indent=None)                               # one line; do not print it, use `r2s probe`
    save_json(tl["t"], Path(out).with_name("frame_times.json"), indent=None)
    return tl


class Video:
    """Random-access RGB reader with a small cache. v.n, v.size=(W,H); v[k] -> (H,W,3) uint8 RGB."""

    def __init__(self, path=VIDEO, cache=96):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise IOError(f"cannot open {path}")
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self._pos, self._cache, self._cn = 0, OrderedDict(), cache

    def __len__(self):
        return self.n

    def __getitem__(self, k):
        k = int(k)
        if k in self._cache:
            self._cache.move_to_end(k)
            return self._cache[k]
        if k != self._pos:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, bgr = self.cap.read()
        if not ok:
            raise IndexError(f"frame {k} not readable ({self.path} has {self.n} frames, 0..{self.n - 1})")
        self._pos = k + 1
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._cache[k] = img
        if len(self._cache) > self._cn:
            self._cache.popitem(last=False)
        return img

    def iter(self, start=0, stop=None, step=1):
        stop = self.n if stop is None else min(stop, self.n)
        for k in range(start, stop, step):
            yield k, self[k]


def _label(img, text):
    img = img.copy()
    s = max(0.5, img.shape[1] / 900)
    cv2.putText(img, text, (6, int(24 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (0, 0, 0), int(4 * s), cv2.LINE_AA)
    cv2.putText(img, text, (6, int(24 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, (255, 255, 0), max(1, int(2 * s)), cv2.LINE_AA)
    return img


def draw_grid(img, step=100, origin=(0, 0)):
    """Pixel-coordinate grid in ORIGINAL video pixels (origin = crop offset)."""
    img = img.copy()
    h, w = img.shape[:2]
    x0, y0 = origin
    for gx in range((x0 // step + 1) * step, x0 + w, step):
        cv2.line(img, (gx - x0, 0), (gx - x0, h), (0, 255, 255), 1)
        cv2.putText(img, str(gx), (gx - x0 + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    for gy in range((y0 // step + 1) * step, y0 + h, step):
        cv2.line(img, (0, gy - y0), (w, gy - y0), (0, 255, 255), 1)
        cv2.putText(img, str(gy), (2, gy - y0 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return img


def grid_images(images, cols=None, pad=4, max_width=2400):
    n = len(images)
    if n == 0:
        raise SystemExit("no frames to draw (empty frame list / no masks in range)")
    cols = cols or int(np.ceil(np.sqrt(n * 1.3)))
    rows = int(np.ceil(n / cols))
    h, w = max(im.shape[0] for im in images), max(im.shape[1] for im in images)
    sheet = np.full((rows * (h + pad), cols * (w + pad), 3), 255, np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        sheet[r * (h + pad):r * (h + pad) + im.shape[0], c * (w + pad):c * (w + pad) + im.shape[1]] = im
    if sheet.shape[1] > max_width:
        f = max_width / sheet.shape[1]
        sheet = cv2.resize(sheet, (max_width, int(sheet.shape[0] * f)), interpolation=cv2.INTER_AREA)
    return sheet


def save_image(img, out, quality=90):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return out


def contact_sheet(frames, out, crop=None, zoom=1.0, grid=None, cols=None, video=VIDEO, tl=None, max_width=2400, thumb=None):
    v = Video(video)
    from .common import timeline
    tl = tl or timeline()
    tiles = []
    for k in frames:
        img = v[k]
        x0 = y0 = 0
        if crop:
            x0, y0, x1, y1 = map(int, crop)
            img = img[y0:y1, x0:x1]
        if grid:
            img = draw_grid(img, grid, (x0, y0))
        if zoom != 1:
            img = cv2.resize(img, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST if zoom > 1 else cv2.INTER_AREA)
        if thumb:
            img = cv2.resize(img, (int(thumb), int(img.shape[0] * thumb / img.shape[1])), interpolation=cv2.INTER_AREA)
        tiles.append(_label(img, f"f{k} t={tl['t'][k]:.3f}"))
    return save_image(grid_images(tiles, cols, max_width=max_width), out)


def motion(video=VIDEO, thresh=18, min_area=30, step=1, out=ANALYSIS / "motion.json"):
    """Median-background motion: energy curve, moving blobs, onset/peak/settle events."""
    v = Video(video)
    ks = list(range(0, v.n, max(1, v.n // 60)))
    bg = np.median(np.stack([cv2.cvtColor(v[k], cv2.COLOR_RGB2GRAY) for k in ks]), 0).astype(np.int16)
    prev = None
    energy, diff_prev, blobs = [], [], {}
    for k, img in v.iter(0, None, step):
        g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (5, 5), 0).astype(np.int16)
        m = (np.abs(g - bg) > thresh).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, cen = cv2.connectedComponentsWithStats(m)
        bl = [dict(cx=float(cen[i][0]), cy=float(cen[i][1]), area=int(stats[i, 4]),
                   box=[int(x) for x in stats[i, :4]]) for i in range(1, n) if stats[i, 4] >= min_area]
        blobs[k] = sorted(bl, key=lambda b: -b["area"])[:12]
        energy.append(int(m.sum()))
        diff_prev.append(float(np.abs(g - prev).mean()) if prev is not None else 0.0)
        prev = g
    from scipy.ndimage import median_filter
    from scipy.signal import find_peaks
    raw = np.array(diff_prev)
    d = median_filter(raw, size=5, mode="nearest")      # removes keyframe/compression flicker
    base = np.percentile(d, 20) + 1e-6
    active = d > 2.5 * base + 0.1
    # drop active/inactive runs shorter than 3 frames
    k = 0
    while k < len(active):
        j = k
        while j < len(active) and active[j] == active[k]:
            j += 1
        if j - k < 3 and k > 0:
            active[k:j] = active[k - 1]
        k = j
    events = []
    for k in range(1, len(d)):
        if active[k] and not active[k - 1]:
            events.append(dict(frame=k * step, kind="onset"))
        if not active[k] and active[k - 1]:
            events.append(dict(frame=k * step, kind="settle"))
    pk, _ = find_peaks(d, prominence=max(d.max() * 0.25, 1e-6), distance=5)
    events += [dict(frame=int(p) * step, kind="peak", value=round(float(d[p]), 3)) for p in pk]
    events.sort(key=lambda e: e["frame"])
    res = dict(step=step, thresh=thresh, static_frames_at_start=int(np.argmax(active)) * step if active.any() else None,
               events=events, detail="analysis/motion_detail.json (per-frame change curve and moving blobs; large)")
    save_json(res, out)
    save_json(dict(frame_change=d.round(4).tolist(), frame_change_raw=raw.round(4).tolist(), fg_pixels=energy,
                   blobs={str(k): b for k, b in blobs.items()}), Path(out).with_name("motion_detail.json"), indent=None)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(np.arange(len(d)) * step, d)
        for e in events:
            ax.axvline(e["frame"], color={"onset": "g", "settle": "k", "peak": "r"}[e["kind"]], alpha=.4)
        ax.set_xlabel("frame")
        ax.set_ylabel("mean |frame diff|")
        fig.tight_layout()
        fig.savefig(Path(out).with_suffix(".png"), dpi=90)
    except Exception:
        pass
    return res


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="r2s", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe")
    s = sub.add_parser("sheet")
    s.add_argument("--every", type=int)
    s.add_argument("--frames", type=int, nargs="+")
    s.add_argument("--times", type=float, nargs="+")
    s.add_argument("--start", type=int, default=0)
    s.add_argument("--stop", type=int)
    s.add_argument("--crop", type=int, nargs=4)
    s.add_argument("--zoom", type=float, default=1.0)
    s.add_argument("--grid", type=int)
    s.add_argument("--cols", type=int)
    s.add_argument("--thumb", "--tile", "--width", type=int, dest="thumb", help="tile width in pixels")
    s.add_argument("-o", "--out", default=None)
    s.add_argument("--video", default=str(VIDEO), help="another video on the same timeline, e.g. output/preview.mp4")
    m = sub.add_parser("motion")
    m.add_argument("--thresh", type=float, default=18)
    m.add_argument("--min-area", type=int, default=30)
    d = sub.add_parser("dump")
    d.add_argument("--frames", type=int, nargs="+", required=True)
    d.add_argument("-o", "--out", default=str(ANALYSIS / "frames"))
    a = ap.parse_args(argv)
    from .common import timeline
    if a.cmd == "probe":
        tl = probe()
        print(json.dumps({k: v for k, v in tl.items() if k != "t"}))
        print("first PTS:", tl["t"][:4], "last:", tl["t"][-2:])
    elif a.cmd == "sheet":
        tl = timeline()
        if a.frames:
            fr = a.frames
        elif a.times:
            t = np.array(tl["t"])
            fr = [int(np.argmin(np.abs(t - x))) for x in a.times]
        else:
            stop = a.stop or tl["n"]
            every = a.every or max(1, (stop - a.start) // 24)
            fr = list(range(a.start, stop, every))
            if fr[-1] != stop - 1:
                fr.append(stop - 1)
        bad = [k for k in fr if not 0 <= k < tl["n"]]
        if bad:
            raise SystemExit(f"frames {bad} outside the video (0..{tl['n'] - 1})")
        out = a.out or str(ANALYSIS / f"sheet_{fr[0]}-{fr[-1]}{'_crop' if a.crop else ''}.jpg")
        if not Path(a.video).exists():
            raise SystemExit(f"no video {a.video}")
        print(contact_sheet(fr, out, a.crop, a.zoom, a.grid, a.cols, video=Path(a.video), tl=tl, thumb=a.thumb))
    elif a.cmd == "motion":
        r = motion(thresh=a.thresh, min_area=a.min_area)
        print(f"static frames at start: {r['static_frames_at_start']}")
        print("events:", json.dumps(r["events"]))
        print("plot: analysis/motion.png ; per-frame blobs in analysis/motion_detail.json (large, do not print)")
    elif a.cmd == "dump":
        v = Video()
        Path(a.out).mkdir(parents=True, exist_ok=True)
        for k in a.frames:
            p = Path(a.out) / f"frame_{k:05d}.png"
            cv2.imwrite(str(p), cv2.cvtColor(v[k], cv2.COLOR_RGB2BGR))
            print(p)
