"""Read a SAM3 result (python -m r2s gpu sam3 ...).

  python -m r2s sam3 summary analysis/sam3_ball      object ids, frames covered, median area, centroid path
  python -m r2s sam3 sheet analysis/sam3_ball [--every 10] [--ids 0 2] [--out x.jpg]  -> <dir>/sheet.jpg masks
        drawn on frames, each mask labelled with its object id
  python -m r2s sam3 members analysis/sam3_dominoes [--crop x0 y0 x1 y1] [--tile PX] [-o|--out file.jpg]
        a `gpu members` result: numbered member outlines on its frame, in zoomed tiles of ~6 member sizes
        (--tile PX; pages <dir>/members_p<i>.jpg,
        overview <dir>/members.jpg with the tile boxes); candidates below the threshold as yellow '?' boxes
Then convert to tracks: python -m r2s track sam3 analysis/sam3_ball --ids 0 --names ball
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .common import load_json, save_json, timeline


def _frames(d):
    return sorted(Path(d).glob("frames/frame_*.npz"))


def load_members(d):
    """members.json of a `gpu members` result, or None for a video-tracking result."""
    f = Path(d) / "members.json"
    return load_json(f) if f.exists() else None


def members_summary(d, verbose=False):
    info = load_members(d)
    if info is None:
        raise SystemExit(f"{d}: not a `gpu members` result (no members.json)")
    ms, pf = info["members"], info["per_frame"]
    if verbose:
        a = pf[0]
        print(f"  {len(ms)} members on frame {info['anchor_frame']} ("
              + ", ".join(f"{p!r}: {n}" for p, n in a["per_prompt"].items()) + f"); {a['candidates']} candidates at "
              f"score 0.2-{info['threshold']} (yellow '?' on the sheet, listed in members.json)")
        print("  members after each detector pass: " + " -> ".join(
            f"{'whole' if l['rows'] == 0 else str(l['tiles']) + ' tiles'} {l['members']}" for l in a["levels"]))
        if ms:
            area = np.array([m["area"] for m in ms])
            med = float(np.median(area))
            big = [m["id"] for m in ms if m["area"] > 3 * med]
            small = [m["id"] for m in ms if m["area"] < 0.25 * med]
            print(f"  member area median {med:.0f} px" + (f"; > 3x median: ids {big} (two members in one mask, "
                  "or another object?)" if big else "") + (f"; < 1/4 median: ids {small} (mostly hidden, or a "
                  "part of one?)" if small else ""))
        if len(pf) > 1:
            print(f"  union (id 0) on {len(pf) - 1} more frames, detections per frame: "
                  + ", ".join(f"f{r['frame']}:{r['members']}" for r in pf[1:13]) + (" ..." if len(pf) > 13 else ""))
        print(f"  numbered sheet: python -m r2s sam3 members {d}")
    return info


def _palette(n):
    rng = np.random.default_rng(7)
    return rng.integers(70, 256, (max(n, 1) + 1, 3))


def members_sheet(d, crop=None, tile=None, out=None):
    """Numbered member outlines on zoomed tiles of the members' frame; overview with the tile boxes."""
    from .video import Video, _label, grid_images, save_image
    info = members_summary(d)
    k = info["anchor_frame"]
    z = np.load(Path(d) / "frames" / f"frame_{k:06d}.npz")
    img = Video()[k]
    H, W = img.shape[:2]
    ms = {int(i): m for i, m in zip(z["object_ids"], z["masks"]) if int(i) > 0}
    rows = {m["id"]: m for m in info["members"]}
    boxes = [m["bbox"] for m in info["members"]] + [c["bbox"] for c in info["candidates"]]
    if not boxes:
        raise SystemExit(f"{d}: no members or candidates to draw")
    size = float(np.median([max(b[2] - b[0], b[3] - b[1]) for b in boxes]))
    if crop:
        x0, y0, x1, y1 = (int(round(v)) for v in crop)
    else:
        b = np.array(boxes, float)
        x0, y0 = b[:, 0].min() - size, b[:, 1].min() - size
        x1, y1 = b[:, 2].max() + size, b[:, 3].max() + size
    x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))
    T = int(tile or min(max(120, 6 * size), 640))
    nx, ny = max(1, round((x1 - x0) / T)), max(1, round((y1 - y0) / T))
    tw, th = (x1 - x0) / nx, (y1 - y0) / ny
    pad = 0.6 * size                                  # tiles overlap so a member on an edge is whole in one of them
    cores = [(x0 + i * tw, y0 + j * th, x0 + (i + 1) * tw, y0 + (j + 1) * th) for j in range(ny) for i in range(nx)]
    tiles = [(int(max(x0, a - pad)), int(max(y0, b - pad)), int(min(x1, c + pad)), int(min(y1, e + pad)))
             for a, b, c, e in cores]
    col = _palette(max(rows) if rows else 1)
    zoom = min(5.0, 760 / max(tw + 2 * pad, th + 2 * pad))
    pages, listing = [], []
    for n, ((a, b, c, e), core) in enumerate(zip(tiles, cores), 1):
        im = cv2.resize(img[b:e, a:c], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
        here = []
        for i, m in ms.items():
            r = rows.get(i)
            if r is None or r["bbox"][2] <= a or r["bbox"][0] >= c or r["bbox"][3] <= b or r["bbox"][1] >= e:
                continue
            cs, _ = cv2.findContours(m[b:e, a:c].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(im, [(q * zoom).astype(np.int32) for q in cs], -1, tuple(int(v) for v in col[i]), 2)
            u, v = r["centroid"]                      # numbered in the one tile whose core holds the centroid
            if core[0] <= u < core[2] and core[1] <= v < core[3] or (u == x1 and core[2] == x1) or (v == y1 and core[3] == y1):
                here.append(i)
                p = (int((u - a) * zoom) - 9, int((v - b) * zoom) + 5)
                cv2.putText(im, str(i), p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(im, str(i), p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for q in info["candidates"]:
            u0, v0, u1, v1 = q["bbox"]
            if u1 <= a or u0 >= c or v1 <= b or v0 >= e:
                continue
            p0, p1 = (int((u0 - a) * zoom), int((v0 - b) * zoom)), (int((u1 - a) * zoom), int((v1 - b) * zoom))
            cv2.rectangle(im, p0, p1, (255, 230, 0), 1)
            cv2.putText(im, f"?{q['score']:.2f}", (p0[0], max(12, p0[1] - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (255, 230, 0), 1, cv2.LINE_AA)
        pages.append(_label(im, f"tile {n} x{a}-{c} y{b}-{e} f{k}"))
        listing.append((n, (a, b, c, e), sorted(here)))
    out = Path(out) if out else Path(d) / "members.jpg"
    ov = img.copy()
    for i, m in ms.items():
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cs, -1, tuple(int(v) for v in col[i]), max(1, W // 900))
    for n, (a, b, c, e), _ in listing:
        cv2.rectangle(ov, (a, b), (c, e), (255, 255, 0), max(1, W // 600))
        cv2.putText(ov, str(n), (a + 6, b + int(30 * W / 1280)), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * W / 1280,
                    (255, 255, 0), max(1, W // 600), cv2.LINE_AA)
    files = [save_image(_label(ov, f"f{k}: {len(ms)} members, {len(info['candidates'])} candidates"), out)]
    per = 6
    for j in range(0, len(pages), per):
        files.append(save_image(grid_images(pages[j:j + per], cols=min(3, len(pages[j:j + per]))),
                                out.with_name(f"{out.stem}_p{j // per + 1}{out.suffix}")))
    print(f"{len(ms)} members, {len(info['candidates'])} candidates on frame {k}; overview {files[0]} "
          f"(tile boxes), tiles at x{zoom:.1f}: " + " ".join(str(f) for f in files[1:]))
    for n, box, ids in listing:
        print(f"  tile {n} x{box[0]}-{box[2]} y{box[1]}-{box[3]}: {len(ids)} numbered" + (f" (ids {ids[0]}..{ids[-1]})" if ids else ""))
    return files


def summarize(d, verbose=False):
    if load_members(d) is not None:
        return members_summary(d, verbose)
    per = {}
    files = _frames(d)
    for f in files:
        k = int(f.stem.split("_")[1])
        z = np.load(f)
        for oid, m in zip(z["object_ids"], z["masks"]):
            a = int(m.sum())
            if a:
                ys, xs = np.nonzero(m)
                per.setdefault(int(oid), []).append((k, a, xs.mean(), ys.mean()))
    s = dict(dir=str(d), frames_processed=len(files), objects={})
    for oid, r in sorted(per.items()):
        r = np.array(r)
        s["objects"][oid] = dict(frames=len(r), first=int(r[0, 0]), last=int(r[-1, 0]),
                                 median_area=float(np.median(r[:, 1])),
                                 start_uv=r[0, 2:].round(1).tolist(), end_uv=r[-1, 2:].round(1).tolist(),
                                 path_px=float(np.linalg.norm(np.diff(r[:, 2:], axis=0), axis=1).sum()))
    save_json(s, Path(d) / "summary.json")
    if verbose:
        print(f"  {len(s['objects'])} object(s) over {len(files)} frames")
        for oid, o in s["objects"].items():
            print(f"  id {oid}: frames {o['first']}-{o['last']} ({o['frames']}), area~{o['median_area']:.0f}px, "
                  f"start {o['start_uv']} end {o['end_uv']}, path {o['path_px']:.0f}px")
        if not s["objects"]:
            print("  0 objects (gpu sam3 options: --threshold, --box, --prompt-frame)")
        print(f"  masks drawn: python -m r2s sam3 sheet {d}")
    return s


def sheet(d, every=None, ids=None, out=None):
    from .video import Video, grid_images, save_image, _label
    if load_members(d) is not None and ids is None:
        ids = [0]                                     # members: their union per frame (numbered: `sam3 members`)
    tl = timeline()
    files = {int(f.stem.split("_")[1]): f for f in _frames(d)}
    ks = sorted(files)
    if not ks:
        raise SystemExit(f"{d}: no SAM3 frames (not finished or failed: `python -m r2s gpu wait {d}`)")
    every = every or max(1, len(ks) // 16)
    v = Video()
    rng = np.random.default_rng(1)
    col = {}
    tiles = []
    for k in ks[::every]:
        img = v[k].copy()
        z = np.load(files[k])
        for oid, m in zip(z["object_ids"], z["masks"]):
            if ids is not None and int(oid) not in ids:
                continue
            c = col.setdefault(int(oid), rng.integers(60, 255, 3))
            img[m] = (0.5 * img[m] + 0.5 * c).astype(np.uint8)
            ys, xs = np.nonzero(m)
            if len(xs):
                cv2.putText(img, str(int(oid)), (int(xs.mean()), int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 255), 2)
        tiles.append(_label(img, f"f{k} t={tl['t'][k]:.2f}"))
    print(save_image(grid_images(tiles), out or Path(d) / "sheet.jpg"))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="r2s sam3", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary")
    s.add_argument("dir")
    sh = sub.add_parser("sheet")
    sh.add_argument("dir")
    sh.add_argument("--every", type=int)
    sh.add_argument("--ids", type=int, nargs="+", help="only these object ids")
    sh.add_argument("-o", "--out", help="default <dir>/sheet.jpg")
    mb = sub.add_parser("members")
    mb.add_argument("dir")
    mb.add_argument("--crop", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), help="only this box (video px)")
    mb.add_argument("--tile", type=int, help="tile size in video px (default 6 member sizes, 120-640)")
    mb.add_argument("-o", "--out", help="default <dir>/members.jpg (+ _p<i> tile pages)")
    a = ap.parse_args(argv)
    if not Path(a.dir).is_dir():
        raise SystemExit(f"no directory {a.dir}")
    if a.cmd == "summary":
        summarize(a.dir, True)
    elif a.cmd == "members":
        members_sheet(a.dir, a.crop, a.tile, a.out)
    else:
        sheet(a.dir, a.every, a.ids, a.out)
