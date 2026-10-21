"""Noun-free SAM3 prompting for the SAM3 worker: box prompts that survive
propagation, and a point-grid fallback whose chosen masks seed the tracker.

SAM3's video model drops a track that its text detector does not re-find on
8 of the first 15 frames (``hotstart``), including the prompt frame. A box
prompt that finds a body by its pixels can be erased that way.
When no track of a box-assisted request overlaps the box (nothing found, or only
other instances of the noun), the prompt frame is segmented from a
uniform point grid (as SAM1's automatic mask generator does), the masks
overlapping the box most are kept, and the SAM3 tracker follows them. The
candidates join the text tracks in the same review.
"""
from __future__ import annotations

import contextlib
from typing import Any

GRID_POINTS_PER_SIDE = 32
GRID_MIN_PREDICTED_IOU = 0.8
GRID_MIN_STABILITY = 0.92
GRID_MASK_NMS_IOU = 0.7
GRID_MIN_AREA_FRACTION = 0.0005
GRID_MAX_TRACKED_CANDIDATES = 3
GRID_MIN_BOX_OVERLAP = 0.05

_IMAGE_MODELS: dict[str, Any] = {}


@contextlib.contextmanager
def hotstart_disabled(predictor: Any):
    """Keep a visually prompted track that the text detector does not re-find."""
    model = getattr(predictor, "model", None)
    if model is None or not hasattr(model, "hotstart_delay"):
        yield False
        return
    saved = model.hotstart_delay
    model.hotstart_delay = 0
    try:
        yield True
    finally:
        model.hotstart_delay = saved


def box_xyxy(box_xywh, width: int, height: int) -> list[float]:
    x, y, w, h = box_xywh
    return [x * width, y * height, (x + w) * width, (y + h) * height]


def mask_box_xywh(mask) -> list[float] | None:
    import numpy as np
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    height, width = mask.shape
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return [float(x0 / width), float(y0 / height), float((x1 - x0) / width), float((y1 - y0) / height)]


def pick_by_box(candidates: list[dict[str, Any]], box_mask, *, limit: int = GRID_MAX_TRACKED_CANDIDATES,
                minimum: float = GRID_MIN_BOX_OVERLAP) -> list[dict[str, Any]]:
    """The candidates whose mask overlaps the located box most (IoU), best first."""
    import numpy as np
    box_area = float(box_mask.sum())
    scored = []
    for candidate in candidates:
        mask = candidate["mask"]
        inter = float(np.logical_and(mask, box_mask).sum())
        union = float(mask.sum()) + box_area - inter
        overlap = inter / union if union else 0.0
        if overlap >= minimum:
            scored.append({**candidate, "box_overlap": overlap})
    scored.sort(key=lambda row: -row["box_overlap"])
    return scored[:limit]


def mask_nms(masks, scores, threshold: float = GRID_MASK_NMS_IOU) -> list[int]:
    """Greedy mask NMS by score on 4x-strided masks; indices of kept masks."""
    import numpy as np
    if len(masks) == 0:
        return []
    small = np.stack([m[::4, ::4].reshape(-1) for m in masks]).astype(np.float32)
    inter = small @ small.T
    area = small.sum(1)
    ious = inter / np.maximum(area[:, None] + area[None, :] - inter, 1)
    suppressed = np.zeros(len(masks), bool)
    keep = []
    for index in np.argsort(-np.asarray(scores), kind="stable"):
        if suppressed[index]:
            continue
        keep.append(int(index))
        suppressed |= ious[index] > threshold
    return keep


def read_frame(video_path: str, frame_index: int):
    """The RGB frame SAM3 numbers frame_index, decoded in order."""
    import cv2
    capture = cv2.VideoCapture(str(video_path))
    try:
        for index in range(frame_index + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Cannot decode frame {frame_index} of {video_path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def _image_model(checkpoint: str | None):
    key = str(checkpoint)
    if key not in _IMAGE_MODELS:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        model = build_sam3_image_model(checkpoint_path=checkpoint, load_from_HF=checkpoint is None,
                                       enable_inst_interactivity=True)
        _IMAGE_MODELS[key] = (Sam3Processor(model), model.inst_interactive_predictor)
    return _IMAGE_MODELS[key]


def grid_candidates(rgb, checkpoint: str | None, *, points_per_side: int = GRID_POINTS_PER_SIDE,
                    batch: int = 64) -> list[dict[str, Any]]:
    """SAM1-style automatic masks of one frame from SAM3's interactive predictor."""
    import numpy as np
    import torch
    from PIL import Image

    processor, predictor = _image_model(checkpoint)
    height, width = rgb.shape[:2]
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        with autocast:
            state = processor.set_image(Image.fromarray(rgb))
            backbone = state["backbone_out"]["sam2_backbone_out"]
            _, features, _, _ = predictor.model._prepare_backbone_features(backbone)
            features[-1] = features[-1] + predictor.model.no_mem_embed
            maps = [f.permute(1, 2, 0).view(1, -1, *size)
                    for f, size in zip(features[::-1], predictor._bb_feat_sizes[::-1])][::-1]
        predictor._features = {"image_embed": maps[-1], "high_res_feats": maps[:-1]}
        predictor._is_image_set, predictor._is_batch, predictor._orig_hw = True, False, [(height, width)]
        offset = 1.0 / (2 * points_per_side)
        side = np.linspace(offset, 1 - offset, points_per_side)
        xs, ys = np.meshgrid(side, side)
        points = np.stack([xs.ravel() * width, ys.ravel() * height], 1).astype(np.float32)
        minimum_area = GRID_MIN_AREA_FRACTION * height * width
        kept = []
        for start in range(0, len(points), batch):
            chunk = points[start:start + batch]
            _, coords, labels, _ = predictor._prep_prompts(
                chunk[:, None, :], np.ones((len(chunk), 1), np.int32), None, None, True)
            with autocast:
                logits, predicted, _ = predictor._predict(coords, labels, None, None, True, return_logits=True)
            logits = logits.float().flatten(0, 1)
            predicted = predicted.float().flatten()
            binary = logits > 0
            area = binary.sum((1, 2))
            stability = ((logits > 1).sum((1, 2)).float()
                         / (logits > -1).sum((1, 2)).clamp(min=1).float())
            good = ((predicted >= GRID_MIN_PREDICTED_IOU) & (stability >= GRID_MIN_STABILITY)
                    & (area >= minimum_area)).nonzero().flatten().tolist()
            for index in good:
                kept.append({"mask": binary[index].cpu().numpy(), "predicted_iou": float(predicted[index]),
                             "stability": float(stability[index]), "point_xy": chunk[index // 3].tolist()})
    keep = mask_nms([row["mask"] for row in kept], [row["predicted_iou"] for row in kept])
    return sorted((kept[index] for index in keep), key=lambda row: -row["predicted_iou"])


def track_masks(predictor: Any, video_path: str, frame_index: int, masks: list, *,
                max_frames: int | None = None) -> dict[int, dict[str, Any]]:
    """Follow masks with the SAM3 tracker; outputs in the video predictor's per-frame form."""
    import numpy as np
    import torch

    model = predictor.model
    tracker = model.tracker
    tracker.backbone = model.detector.backbone
    frames: dict[int, dict[int, tuple[Any, float]]] = {}
    try:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = tracker.init_state(video_path=str(video_path), offload_video_to_cpu=True)
            for object_id, mask in enumerate(masks, start=1):
                tracker.add_new_mask(state, frame_index, object_id, torch.from_numpy(np.asarray(mask, bool)))
            for reverse in (False, True):
                for index, object_ids, _low, video_masks, scores in tracker.propagate_in_video(
                        state, frame_index, max_frames, reverse=reverse, tqdm_disable=True,
                        propagate_preflight=True):
                    row = frames.setdefault(int(index), {})
                    flat = scores.reshape(-1) if scores is not None else None
                    for position, object_id in enumerate(list(object_ids)):
                        score = float(torch.sigmoid(flat[position].float())) if flat is not None else 1.0
                        row[int(object_id)] = ((video_masks[position, 0] > 0).cpu().numpy(), score)
            del state
    finally:
        tracker.backbone = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    outputs = {}
    for index, row in sorted(frames.items()):
        present = [(object_id, mask, score) for object_id, (mask, score) in sorted(row.items()) if mask.any()]
        outputs[index] = {
            "out_obj_ids": np.array([object_id for object_id, _, _ in present], np.int64),
            "out_probs": np.array([score for _, _, score in present], np.float32),
            "out_binary_masks": (np.stack([mask for _, mask, _ in present]) if present
                                 else np.zeros((0, 1, 1), bool)),
            "out_boxes_xywh": np.array([mask_box_xywh(mask) for _, mask, _ in present], np.float32).reshape(-1, 4),
        }
    return outputs


GRID_TRIGGER_BOX_IOU = 0.1


def _frame_masks(outputs) -> list:
    import numpy as np
    masks = np.asarray(outputs.get("out_binary_masks", [])) if outputs else np.zeros((0,))
    return [np.squeeze(mask).astype(bool) for mask in masks] if masks.size else []


def covers_box(outputs, box_xywh, threshold: float = GRID_TRIGGER_BOX_IOU) -> bool:
    """Whether any track on the prompt frame overlaps the located box (IoU)."""
    import numpy as np
    for mask in _frame_masks(outputs):
        if mask.ndim != 2:
            continue
        height, width = mask.shape
        x0, y0, x1, y1 = (int(round(v)) for v in box_xyxy(box_xywh, width, height))
        box = np.zeros((height, width), bool)
        box[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
        inter = float(np.logical_and(mask, box).sum())
        union = float(mask.sum() + box.sum()) - inter
        if union and inter / union >= threshold:
            return True
    return False


def merge_outputs(text: dict[int, Any], grid: dict[int, Any], *, id_offset: int = 1000) -> dict[int, Any]:
    """Text tracks plus grid tracks (ids offset), in the video predictor's per-frame form."""
    import numpy as np
    merged = {}
    for index in sorted(set(text) | set(grid)):
        ids, probs, masks, boxes = [], [], [], []
        for outputs, offset in ((text.get(index), 0), (grid.get(index), id_offset)):
            if not outputs:
                continue
            frame_masks = _frame_masks(outputs)
            ids += [int(v) + offset for v in np.asarray(outputs.get("out_obj_ids", [])).reshape(-1)]
            probs += [float(v) for v in np.asarray(outputs.get("out_probs", [])).reshape(-1)]
            masks += frame_masks
            boxes += [list(map(float, row)) for row in np.asarray(outputs.get("out_boxes_xywh", [])).reshape(-1, 4)]
        merged[index] = {"out_obj_ids": np.array(ids, np.int64), "out_probs": np.array(probs, np.float32),
                         "out_binary_masks": np.stack(masks) if masks else np.zeros((0, 1, 1), bool),
                         "out_boxes_xywh": np.array(boxes, np.float32).reshape(-1, 4)}
    return merged


def grid_fallback(predictor: Any, request: dict[str, Any], box_xywh, checkpoint: str | None):
    """Grid candidates on the prompt frame that overlap the box, tracked; and an audit."""
    import numpy as np
    frame_index = int(request.get("prompt_frame", 0))
    rgb = read_frame(request["input_path"], frame_index)
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in box_xyxy(box_xywh, width, height))
    box_mask = np.zeros((height, width), bool)
    box_mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
    candidates = grid_candidates(rgb, checkpoint)
    picked = pick_by_box(candidates, box_mask)
    audit = {"method": "point-grid-then-tracker", "points_per_side": GRID_POINTS_PER_SIDE,
             "frame_index": frame_index, "grid_candidate_count": len(candidates),
             "tracked": [{"object_id": n, "box_overlap": round(row["box_overlap"], 4),
                          "predicted_iou": round(row["predicted_iou"], 4), "stability": round(row["stability"], 4),
                          "point_xy": row["point_xy"]} for n, row in enumerate(picked, start=1)],
             "selection": "highest IoU with the located box; identity is decided by the grounding review"}
    if not picked:
        return {}, audit
    max_frames = request.get("max_frames")
    return track_masks(predictor, request["input_path"], frame_index, [row["mask"] for row in picked],
                       max_frames=int(max_frames) if max_frames is not None else None), audit
