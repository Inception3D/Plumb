"""2D object tracks in ONE format, from SAM3, colour thresholds or manual clicks.

Track file analysis/tracks/<name>.json:
  {"name", "source", "frames": [k...], "uv": [[u,v]...] (mask centroid, video px), "area": [...],
   "bbox": [[x0,y0,x1,y1]...], "mask_source": {"sam3_dir", "object_id"} | null}
`r2s check compare` compares each track with the scene body of the same name (or the bodies given by --link).

  python -m r2s track sam3 analysis/sam3_ball [--ids 0 2] [--names ball cube]   -> one track per SAM3 id
  python -m r2s track color --name red_ball --hsv 0 120 70 10 255 255 [--hsv2 170 120 70 180 255 255]
                            [--roi x0 y0 x1 y1] [--min-area 20] [--start K --stop K]
        HSV ranges in OpenCV units (H 0-180). Pick them from `python -m r2s track hsv --at u v --frame k`.
  python -m r2s track hsv --frame 0 --at 512 400 [--at ...]        print HSV/RGB at pixels (5x5 median)
  python -m r2s track manual --name stick --frames 0 30 60 --uv 100 200 110 205 130 230
  python -m r2s track show [--names a b] [--every 10]  -> analysis/tracks_sheet.jpg (tracks drawn on frames)
  python -m r2s track list
  python -m r2s track lift --name ball --z 0.05 [--frames 0 12] [--camera analysis/camera.json]
        world path of a track's centroids intersected with the plane z = Z of the camera file's world
        -> analysis/lift_<name>.json; prints the least-squares position and velocity over the frame range
        (default the first 11 track frames) and which of those frames have the mask touching the image border.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .common import ANALYSIS, load_json, save_json, timeline

TRACKS = ANALYSIS / "tracks"


def border_cut(t, width, height, margin=2):
    """Per track frame, which image borders (left, top, right, bottom) the mask box touches. A mask cut off by the
    border has its centroid pulled into the image: the object centre lies beyond it, towards that border."""
    n = len(t["frames"])
    try:
        b = np.asarray(t.get("bbox") or [], float)
    except (TypeError, ValueError):                  # boxes missing on some frames
        return np.zeros((n, 4), bool)
    if b.shape != (n, 4):
        return np.zeros((n, 4), bool)
    return np.stack([b[:, 0] <= margin, b[:, 1] <= margin, b[:, 2] >= width - 1 - margin, b[:, 3] >= height - 1 - margin], 1)


def save_track(name, source, frames, uv, area=None, bbox=None, mask_source=None):
    tr = dict(name=name, source=source, frames=[int(k) for k in frames], uv=np.asarray(uv, float).round(2).tolist(),
              area=[int(a) for a in area] if area is not None else None,
              bbox=np.asarray(bbox).round(1).tolist() if bbox is not None else None, mask_source=mask_source)
    save_json(tr, TRACKS / f"{name}.json")
    return tr


def load_tracks(names=None):
    out = {}
    for p in sorted(TRACKS.glob("*.json")):
        t = load_json(p)
        if names and t["name"] not in names:
            continue
        out[t["name"]] = t
    return out


def from_sam3(sam3_dir, ids=None, names=None):
    sam3_dir = Path(sam3_dir)
    frames = sorted(sam3_dir.glob("frames/frame_*.npz"))
    per = {}
    for f in frames:
        k = int(f.stem.split("_")[1])
        z = np.load(f)
        for oid, m in zip(z["object_ids"], z["masks"]):
            oid = int(oid)
            if ids is not None and oid not in ids:
                continue
            ys, xs = np.nonzero(m)
            if len(xs) == 0:
                continue
            per.setdefault(oid, []).append((k, xs.mean(), ys.mean(), len(xs), xs.min(), ys.min(), xs.max(), ys.max()))
    out = []
    order = ids if ids is not None else sorted(per)
    for j, oid in enumerate(order):
        if oid not in per:
            print(f"sam3 id {oid}: no masks")
            continue
        r = np.array(per[oid])
        name = names[j] if names and j < len(names) else f"{sam3_dir.name.replace('sam3_', '')}_{oid}"
        out.append(save_track(name, "sam3", r[:, 0], r[:, 1:3], r[:, 3], r[:, 4:8],
                              dict(sam3_dir=str(sam3_dir), object_id=oid)))
        print(f"track {name}: sam3 id {oid}, {len(r)} frames ({int(r[0,0])}-{int(r[-1,0])}), "
              f"median area {np.median(r[:, 3]):.0f} px")
    return out


def hsv_at(frame, pts):
    from .video import Video
    img = Video()[frame]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    for u, v in pts:
        u, v = int(u), int(v)
        print(f"({u},{v}) RGB={np.median(img[v-2:v+3, u-2:u+3].reshape(-1, 3), 0).astype(int).tolist()} "
              f"HSV={np.median(hsv[v-2:v+3, u-2:u+3].reshape(-1, 3), 0).astype(int).tolist()}")


def color_track(name, ranges, roi=None, min_area=20, start=0, stop=None, largest=True):
    from .video import Video
    v = Video()
    frames, uv, area, bbox = [], [], [], []
    for k, img in v.iter(start, stop):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        m = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        if roi:
            x0, y0, x1, y1 = roi
            mm = np.zeros_like(m)
            mm[y0:y1, x0:x1] = m[y0:y1, x0:x1]
            m = mm
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(m)
        if n <= 1:
            continue
        i = 1 + int(np.argmax(st[1:, 4]))
        if st[i, 4] < min_area:
            continue
        frames.append(k)
        uv.append(cen[i])
        area.append(st[i, 4])
        x, y, w, h = st[i, :4]
        bbox.append([x, y, x + w, y + h])
    if not frames:
        raise SystemExit("no pixels matched: widen the HSV range (check with `r2s track hsv`)")
    tr = save_track(name, "color", frames, uv, area, bbox)
    gaps = np.diff(frames)
    print(f"track {name}: {len(frames)}/{v.n if stop is None else stop - start} frames, area median "
          f"{np.median(area):.0f} px (min {min(area)}, max {max(area)}), largest gap {gaps.max() if len(gaps) else 0}")
    if max(area) > 3 * np.median(area):
        print(f"  note: largest area {max(area)} px > 3x the median")
    return tr


def show(names=None, every=None, out=ANALYSIS / "tracks_sheet.jpg"):
    from .video import Video, grid_images, save_image, _label
    trs = load_tracks(names)
    tl = timeline()
    every = every or max(1, tl["n"] // 16)
    v = Video()
    cols = [(255, 0, 0), (0, 200, 0), (0, 0, 255), (255, 160, 0), (200, 0, 200), (0, 200, 200), (120, 60, 0)]
    tiles = []
    for k in list(range(0, tl["n"], every)) + [tl["n"] - 1]:
        img = v[k].copy()
        for j, (nm, t) in enumerate(trs.items()):
            fr = np.array(t["frames"])
            uv = np.array(t["uv"])
            past = uv[fr <= k]
            c = cols[j % len(cols)]
            if len(past) > 1:
                cv2.polylines(img, [past.astype(np.int32)], False, c, 2)
            if k in t["frames"]:
                u, vv = uv[t["frames"].index(k)]
                cv2.circle(img, (int(u), int(vv)), 6, c, 2)
                cv2.putText(img, nm, (int(u) + 8, int(vv)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        tiles.append(_label(img, f"f{k}"))
    print(save_image(grid_images(tiles), out))


def lift(name, z, frames=None, camera=ANALYSIS / "camera.json"):
    from .common import Camera
    cam = Camera.load(camera)
    t = load_tracks([name]).get(name)
    if t is None:
        raise SystemExit(f"no track {name} (have: {', '.join(load_tracks()) or 'none'})")
    tl = timeline()
    fr = np.array(t["frames"])
    P = cam.lift_to_height(np.array(t["uv"]), z)
    sel = (fr >= frames[0]) & (fr <= frames[1]) if frames else np.arange(len(fr)) < 11
    cut = border_cut(t, tl["width"], tl["height"]).any(1)
    tt = np.array(tl["t"])[fr[sel]]
    res = dict(name=name, z=z, camera=str(camera), frames=fr.tolist(), t=np.array(tl["t"])[fr].round(6).tolist(),
               P=P.round(5).tolist(), border_cut=cut.tolist())
    if sel.sum() >= 2:
        A = np.stack([np.ones_like(tt), tt - tt[0]], 1)
        coef, *_ = np.linalg.lstsq(A, P[sel], rcond=None)
        res.update(fit_frames=[int(fr[sel][0]), int(fr[sel][-1])], fit_position=coef[0].round(5).tolist(),
                   fit_velocity=coef[1].round(5).tolist())
        print(f"{name}: least squares over frames {fr[sel][0]}-{fr[sel][-1]}: position at frame {fr[sel][0]} "
              f"{coef[0].round(4).tolist()}, velocity {coef[1].round(4).tolist()} /s (plane z={z})")
    if (sel & cut).any():
        print(f"{name}: mask touches the image border on frames {fr[sel & cut].tolist()} of that range")
    save_json(res, ANALYSIS / f"lift_{name}.json")
    print(f"world path -> analysis/lift_{name}.json ({len(fr)} frames, extent {np.ptp(P, 0).round(3).tolist()})")
    return res


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="r2s track", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sam3")
    s.add_argument("dir")
    s.add_argument("--ids", type=int, nargs="+")
    s.add_argument("--names", nargs="+")
    c = sub.add_parser("color")
    c.add_argument("--name")
    c.add_argument("--hsv", type=int, nargs=6)
    c.add_argument("--obj", nargs=7, action="append", metavar=("NAME", "HLO", "SLO", "VLO", "HHI", "SHI", "VHI"),
                   help="several colour objects in ONE command")
    c.add_argument("--hsv2", type=int, nargs=6)
    c.add_argument("--roi", type=int, nargs=4)
    c.add_argument("--min-area", type=int, default=20)
    c.add_argument("--start", type=int, default=0)
    c.add_argument("--stop", type=int)
    h = sub.add_parser("hsv")
    h.add_argument("--frame", type=int, default=0)
    h.add_argument("--at", type=float, nargs=2, action="append", required=True)
    m = sub.add_parser("manual")
    m.add_argument("--name", required=True)
    m.add_argument("--frames", type=int, nargs="+", required=True)
    m.add_argument("--uv", type=float, nargs="+", required=True)
    sh = sub.add_parser("show")
    sh.add_argument("--names", nargs="+")
    sh.add_argument("--every", type=int)
    sub.add_parser("list")
    li = sub.add_parser("lift")
    li.add_argument("--name", required=True)
    li.add_argument("--z", type=float, default=0.0)
    li.add_argument("--frames", type=int, nargs=2)
    li.add_argument("--camera", default=str(ANALYSIS / "camera.json"))
    a = ap.parse_args(argv)
    if a.cmd == "sam3":
        if not Path(a.dir).is_dir():
            raise SystemExit(f"no directory {a.dir}")
        if a.names and len(a.names) != len(a.ids or []):
            raise SystemExit("--names needs exactly one name per --ids entry")
        from_sam3(a.dir, a.ids, a.names)
    elif a.cmd == "color":
        jobs = [(o[0], [(list(map(int, o[1:4])), list(map(int, o[4:7])))]) for o in (a.obj or [])]
        if a.name:
            jobs.append((a.name, [(a.hsv[:3], a.hsv[3:])] + ([(a.hsv2[:3], a.hsv2[3:])] if a.hsv2 else [])))
        for name, rng in jobs:
            h = [(lo[0], hi[0]) for lo, hi in rng]
            if len(rng) == 1 and ((h[0][0] <= 3 and h[0][1] <= 15) or (h[0][1] >= 176 and h[0][0] >= 160)):
                other = "170 .. 180" if h[0][0] <= 3 else "0 .. 10"
                print(f"track {name}: hue {h[0][0]}..{h[0][1]} ends at the 0/180 wrap of red; pixels of the same colour at "
                      f"hue {other} are left out (centroid biased) unless you add --hsv2 with that hue range")
            try:
                color_track(name, rng, a.roi, a.min_area, a.start, a.stop)
            except SystemExit as e:
                print(f"track {name}: {e}")
        show()
    elif a.cmd == "hsv":
        hsv_at(a.frame, a.at)
    elif a.cmd == "manual":
        if len(a.uv) != 2 * len(a.frames):
            raise SystemExit(f"--uv needs one U V pair per frame: {len(a.frames)} frames -> {2 * len(a.frames)} numbers, "
                             f"got {len(a.uv)}")
        uv = np.array(a.uv).reshape(-1, 2)
        save_track(a.name, "manual", a.frames, uv)
        print(f"track {a.name}: {len(uv)} manual points")
    elif a.cmd == "show":
        show(a.names, a.every)
    elif a.cmd == "lift":
        lift(a.name, a.z, a.frames, Path(a.camera))
    else:
        for n, t in load_tracks().items():
            print(f"{n}: {t['source']} {len(t['frames'])} frames {t['frames'][0]}-{t['frames'][-1]}")
