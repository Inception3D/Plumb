"""SAM3 instance detection of many similar objects on single frames (host side, SAM3 environment).

usage: python sam3_members.py < request.json      (harness/gpu_service.py runs it for sam3 requests with
                                                    mode "members"; the last stdout line is the result JSON)
request: input_path, output_dir, prompts [text, ...], frames [anchor, other frames ...],
         confidence_threshold (0.4), region_xyxy [x0 y0 x1 y1] video px (optional), grid "auto" | rows (0 = whole
         frame only; the other frames use the grids chosen on the anchor), max_grid (4)

Why: SAM3's video predictor holds a crowd badly (069370 spiral: 0 tracks for "domino" while the image detector
finds ~235), and one image pass has 200 detection queries on a 1008 x 1008 resize of the whole frame, so a crowd
saturates it and small members of a 16:9 frame are squeezed to a few pixels. Following the Plumb agent
harness's population detection (src/plumb/perception/sam3_population.py: full frame, then overlapping tiles when the
detector is near capacity), the frame is detected whole and, while some tile still holds >= TRIGGER detections
(score >= 0.1) and the last grid still added members, again in a finer grid of near-square overlapping tiles
(1 row of 2 tiles for 16:9, then 2 x 4, 3 x 5, 4 x 7). A frame with fewer detections is detected whole only.
A detection touching an inner tile edge is
dropped (the overlap exceeds the members' size, so it is whole in a neighbour); finer tiles win over coarser ones
and over other prompts' overlapping detections (IoU > 0.5 or 60 % of the smaller mask), so a coarse mask that
merges two members does not add a third.

Output (output_dir): frames/frame_XXXXXX.npz in the sam3_track format (object_ids, scores, masks, boxes_xywh):
on the anchor frame members 1..N plus id 0 = their union; on the other frames only id 0 (identities are not
followed between frames); members.json (anchor members: id, prompt, score, centroid, bbox, area, mean RGB, tile
level; candidates below the threshold; per-frame counts); manifest.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

COUNT_SCORE = 0.1          # crowding is counted at a low score, independent of the acceptance threshold
CANDIDATE_SCORE = 0.2      # below the threshold but listed (bbox only) so a review can spot misses
TRIGGER = 120              # of SAM3's 200 detection queries: a tile this full is detected again, finer
NMS_IOU = 0.5
CONTAIN = 0.6
MIN_AREA_FRACTION = 0.00002


class Det:
    __slots__ = ("x0", "y0", "m", "score", "prompt", "level", "area")

    def __init__(self, x0, y0, m, score, prompt, level):
        self.x0, self.y0, self.m, self.score, self.prompt, self.level = x0, y0, m, score, prompt, level
        self.area = int(m.sum())

    @property
    def box(self):
        return self.x0, self.y0, self.x0 + self.m.shape[1], self.y0 + self.m.shape[0]


def overlap(a, b):
    """Intersection in pixels of two crop masks placed in the frame."""
    ax0, ay0, ax1, ay1 = a.box
    bx0, by0, bx1, by1 = b.box
    x0, y0, x1, y1 = max(ax0, bx0), max(ay0, by0), min(ax1, bx1), min(ay1, by1)
    if x0 >= x1 or y0 >= y1:
        return 0
    return int((a.m[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0] & b.m[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]).sum())


def suppressed_by(d, kept, grid):
    x0, y0, x1, y1 = d.box
    cands = set()
    for gx in range(x0 // grid, (x1 - 1) // grid + 1):
        for gy in range(y0 // grid, (y1 - 1) // grid + 1):
            cands.update(kept.get((gx, gy), ()))
    for k in cands:
        i = overlap(d, k)
        if i and (i / max(1, d.area + k.area - i) > NMS_IOU or i / max(1, min(d.area, k.area)) > CONTAIN):
            return True
    return False


def merge(dets, cell=64, index=None):
    """Finer level first, then score: greedy suppression of overlapping or contained detections (also by the
    detections already in `index`). Returns the kept ones and the updated index."""
    kept, index = [], {} if index is None else index
    for d in sorted(dets, key=lambda d: (-d.level, -d.score)):
        if suppressed_by(d, index, cell):
            continue
        kept.append(d)
        x0, y0, x1, y1 = d.box
        for gx in range(x0 // cell, (x1 - 1) // cell + 1):
            for gy in range(y0 // cell, (y1 - 1) // cell + 1):
                index.setdefault((gx, gy), []).append(d)
    return kept, index


def grid_tiles(region, rows, size_hint):
    """Near-square tiles covering region (x0, y0, x1, y1) in `rows` rows, overlapping by more than size_hint."""
    x0, y0, x1, y1 = region
    W, H = x1 - x0, y1 - y0
    cols = max(1, round(rows * W / H))
    tw, th = W / cols, H / rows
    ox = min(max(0.1 * tw, 1.2 * size_hint[0]), 0.5 * tw)
    oy = min(max(0.1 * th, 1.2 * size_hint[1]), 0.5 * th)
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append((int(max(x0, x0 + c * tw - ox)), int(max(y0, y0 + r * th - oy)),
                        int(min(x1, x0 + (c + 1) * tw + ox)), int(min(y1, y0 + (r + 1) * th + oy))))
    return out


def cut(a, b, m, tile, region):
    """Whether a tile detection (crop mask m at a, b in the tile) touches an inner edge of its tile."""
    x0, y0, x1, y1 = tile
    return bool((x0 > region[0] and a <= 0) or (x1 < region[2] and a + m.shape[1] >= x1 - x0)
                or (y0 > region[1] and b <= 0) or (y1 < region[3] and b + m.shape[0] >= y1 - y0))


class Detector:
    def __init__(self, checkpoint):
        from perception.sam3_visual import _image_model       # harness/perception (this script's directory)
        self.proc, _ = _image_model(checkpoint)
        self.proc.confidence_threshold = COUNT_SCORE

    def __call__(self, rgb, prompt):
        """[(x0, y0, crop mask, score)] with score >= COUNT_SCORE, in the coordinates of rgb."""
        import torch
        from PIL import Image
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = self.proc.set_image(Image.fromarray(np.ascontiguousarray(rgb)))
            out = self.proc.set_text_prompt(state=state, prompt=prompt)
            masks = out["masks"]
            masks = (masks if masks.dtype == torch.bool else masks > 0.5).reshape(len(masks), *rgb.shape[:2])
            scores = out["scores"].float().cpu().numpy()
            res = []
            for m, s in zip(masks, scores):
                rows, cols = m.any(1).nonzero(), m.any(0).nonzero()
                if not len(rows):
                    continue
                r0, r1, c0, c1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
                res.append((c0, r0, m[r0:r1, c0:c1].cpu().numpy(), float(s)))
        return res


def detect_frame(rgb, prompts, detector, region, thr, grid="auto", max_grid=4):
    """Members (score >= thr) and candidates (CANDIDATE_SCORE <= score < thr, not overlapping a member) of one
    frame, each merged over tile levels and prompts; levels = what each grid added.

    grid "auto": a finer grid while some tile (or the whole region) returned >= TRIGGER detections and the last
    grid still added members (more than 3 % + 1); an int: grids 1..grid (0 = the whole region only)."""
    minimum = MIN_AREA_FRACTION * rgb.shape[0] * rgb.shape[1]
    dets, levels = [], []

    def level(n, tiles):
        worst = 0
        for p in prompts:
            for t in tiles:
                rows = detector(rgb[t[1]:t[3], t[0]:t[2]], p)
                worst = max(worst, len(rows))
                dets.extend(Det(t[0] + a, t[1] + b, m, s, p, n) for a, b, m, s in rows if not cut(a, b, m, t, region))
        members = len(merge([d for d in dets if d.score >= thr and d.area >= minimum])[0])
        levels.append(dict(rows=n, tiles=len(tiles), max_tile_detections=worst, members=members))

    level(0, [tuple(region)])
    n = 1
    while n <= (max_grid if grid == "auto" else int(grid)):
        if grid == "auto" and (levels[-1]["max_tile_detections"] < TRIGGER
                               or (n > 1 and levels[-1]["members"] <= 1.03 * levels[-2]["members"] + 1)):
            break
        prev = [d for d in dets if d.score >= 0.3] or dets
        hint = (np.percentile([d.m.shape[1] for d in prev], 90), np.percentile([d.m.shape[0] for d in prev], 90)) \
            if prev else (0, 0)
        level(n, grid_tiles(region, n, hint))
        n += 1
    dets = [d for d in dets if d.area >= minimum]
    keep, index = merge([d for d in dets if d.score >= thr])
    cand, _ = merge([d for d in dets if CANDIDATE_SCORE <= d.score < thr], index=index)
    return keep, cand, levels


def read_frames(path, wanted):
    import cv2
    wanted, got = sorted(set(wanted)), {}
    cap = cv2.VideoCapture(str(path))
    k = 0
    while wanted and k <= wanted[-1]:
        ok, bgr = cap.read()
        if not ok:
            break
        if k in wanted:
            got[k] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        k += 1
    cap.release()
    missing = sorted(set(wanted) - set(got))
    if missing:
        raise ValueError(f"cannot decode frames {missing[:5]} of {path}")
    return got


def full_mask(d, shape):
    m = np.zeros(shape, bool)
    m[d.y0:d.y0 + d.m.shape[0], d.x0:d.x0 + d.m.shape[1]] = d.m
    return m


def run(req, detector=None):
    out = Path(req["output_dir"])
    prompts = list(dict.fromkeys(str(p) for p in req["prompts"]))
    frames = [int(k) for k in req["frames"]]
    if not prompts or not frames:
        raise ValueError("members mode needs prompts and frames")
    thr = float(req.get("confidence_threshold", 0.4))
    detector = detector or Detector(req.get("checkpoint_path") or os.environ.get("R2S_SAM3_CHECKPOINT"))
    imgs = read_frames(req["input_path"], frames)
    H, W = next(iter(imgs.values())).shape[:2]
    region = req.get("region_xyxy")
    region = (0, 0, W, H) if region is None else tuple(int(round(v)) for v in region)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    anchor = frames[0]
    per_frame, members, candidates = [], [], []
    grid = req.get("grid", "auto")
    for k in dict.fromkeys(frames):
        keep, cand, levels = detect_frame(imgs[k], prompts, detector, region, thr, grid, int(req.get("max_grid", 4)))
        grid = levels[-1]["rows"]                     # the other frames use the anchor frame's grids
        keep.sort(key=lambda d: (d.y0 + d.m.shape[0] / 2, d.x0 + d.m.shape[1] / 2))
        union = np.zeros((H, W), bool)
        for d in keep:
            union[d.y0:d.y0 + d.m.shape[0], d.x0:d.x0 + d.m.shape[1]] |= d.m
        if k == anchor:
            ms = [full_mask(d, (H, W)) for d in keep]
            ids, masks = [0] + list(range(1, len(keep) + 1)), [union] + ms
            scores = [1.0] + [d.score for d in keep]
            rgb = imgs[k]
            for i, (d, m) in enumerate(zip(keep, ms), 1):
                ys, xs = np.nonzero(m)
                members.append(dict(id=i, prompt=d.prompt, score=round(d.score, 4),
                                    centroid=[round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
                                    bbox=[d.x0, d.y0, d.x0 + d.m.shape[1], d.y0 + d.m.shape[0]], area=d.area,
                                    rgb=[int(v) for v in rgb[m].mean(0)], level=d.level))
            candidates = [dict(prompt=d.prompt, score=round(d.score, 4),
                               bbox=[d.x0, d.y0, d.x0 + d.m.shape[1], d.y0 + d.m.shape[0]], area=d.area)
                          for d in cand]
        else:
            ids, masks, scores = [0], [union], [1.0]
        boxes = []
        for m in masks:
            ys, xs = np.nonzero(m)
            boxes.append([xs.min(), ys.min(), xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min()] if len(xs) else [0, 0, 0, 0])
        np.savez_compressed(out / "frames" / f"frame_{k:06d}.npz", object_ids=np.array(ids, np.int64),
                            scores=np.array(scores, np.float32), masks=np.stack(masks).astype(bool),
                            boxes_xywh=np.array(boxes, np.float32))
        per_frame.append(dict(frame=k, members=len(keep), candidates=len(cand),
                              per_prompt={p: sum(d.prompt == p for d in keep) for p in prompts},
                              levels=levels))
    info = dict(mode="members", input_path=req["input_path"], prompts=prompts, anchor_frame=anchor, frames=frames,
                threshold=thr, region_xyxy=list(region), width=W, height=H, trigger=TRIGGER, per_frame=per_frame,
                members=members, candidates=candidates,
                note="ids 1..N = members on the anchor frame (sorted top to bottom); id 0 = their union on every "
                     "listed frame; identities are not followed between frames")
    (out / "members.json").write_text(json.dumps(info, indent=1))
    manifest = dict(schema_version=1, tool="sam3", mode="members", input_path=req["input_path"], prompt=" | ".join(prompts),
                    frame_count=len(per_frame), track_count=len(members) + 1, members_path=str(out / "members.json"))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return dict(ok=True, tool="sam3", mode="members", member_count=len(members), anchor_frame=anchor,
                per_frame=[dict(frame=r["frame"], members=r["members"], candidates=r["candidates"],
                                grid_rows=r["levels"][-1]["rows"]) for r in per_frame],
                members_path=str(out / "members.json"), manifest_path=str(out / "manifest.json"))


if __name__ == "__main__":
    request = json.loads(sys.stdin.read())
    try:
        result = run(request)
    except Exception as e:                               # the service records the error for the session
        import traceback
        traceback.print_exc()
        result = dict(ok=False, error_type=type(e).__name__, error=str(e))
    print(json.dumps(result))
