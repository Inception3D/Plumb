"""GPU inference worker of the real2sim skill: SAM3 video tracking and Pi3X reconstruction.

usage: <model env python> src/harness/perception/worker.py {sam3|pi3x} < request.json   (one JSON result on stdout)

Run by harness/gpu_service.py with the Python interpreter of the SAM3 or Pi3 environment; model
dependencies are imported inside each backend, so this file needs nothing else installed there.
Adapted from the perception worker of the Plumb agent harness (src/plumb/perception/worker.py; MIT, see
THIRD_PARTY_NOTICES.md): only the SAM3 video and Pi3X backends, unchanged apart from the environment
variable names (R2S_* instead of PLUMB_*) and the import paths.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# The worker is also launched directly by its absolute path in model envs.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}


def _cached_huggingface_file(repo_id: str, filename: str) -> Path | None:
    """Resolve a complete Hugging Face snapshot without a network call."""

    if cache_override := os.environ.get("HF_HUB_CACHE"):
        hub = Path(cache_override).expanduser()
    elif hf_home := os.environ.get("HF_HOME"):
        hub = Path(hf_home).expanduser() / "hub"
    else:
        hub = Path.home() / ".cache" / "huggingface" / "hub"
    repository = hub / f"models--{repo_id.replace('/', '--')}"
    ref = repository / "refs" / "main"
    if ref.is_file():
        commit = ref.read_text(encoding="utf-8").strip()
        candidate = repository / "snapshots" / commit / filename
        if candidate.is_file():
            return candidate.resolve()
    snapshots = repository / "snapshots"
    if snapshots.is_dir():
        for candidate in sorted(snapshots.glob(f"*/{filename}"), reverse=True):
            if candidate.is_file():
                return candidate.resolve()
    return None


def _output_directory(request: dict[str, Any]) -> Path:
    path = Path(request["output_dir"]).expanduser().resolve()
    if path.exists() and any(path.iterdir()) and not request.get("overwrite", False):
        raise FileExistsError(
            f"output_dir is not empty: {path}; choose a new directory or set overwrite=true"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_manifest(output_dir: Path, manifest: dict[str, Any]) -> Path:
    path = output_dir / "manifest.json"
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def _numpy(value: Any, np: Any) -> Any:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    if str(getattr(value, "dtype", "")) == "torch.bfloat16":
        value = value.float()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        value = numpy()
    return np.asarray(value)


def _squeeze_mask(mask: Any, np: Any) -> Any:
    value = _numpy(mask, np)
    while value.ndim > 2 and value.shape[0] == 1:
        value = value[0]
    if value.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {tuple(value.shape)}")
    return value > 0


def _save_binary_mask(mask: Any, path: Path, np: Any, Image: Any) -> None:
    Image.fromarray(_squeeze_mask(mask, np).astype(np.uint8) * 255).save(path)


def _device(request: dict[str, Any], torch: Any) -> str:
    requested = request.get("device", "auto")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return requested


def _sample_video(
    video_path: Path,
    frame_dir: Path,
    *,
    sample_fps: float,
    max_frames: int,
) -> tuple[list[Path], list[dict[str, Any]]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Video input requires OpenCV in this model environment; install opencv-python or provide an image directory"
        ) from exc

    frame_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    interval = max(round((source_fps or sample_fps) / sample_fps), 1)
    frames: list[Path] = []
    records: list[dict[str, Any]] = []
    index = 0
    try:
        while len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if index % interval == 0:
                path = frame_dir / f"{len(frames):06d}.png"
                if not cv2.imwrite(str(path), frame):
                    raise OSError(f"Could not write sampled frame: {path}")
                frames.append(path)
                records.append(
                    {
                        "path": str(path),
                        "source_frame_index": index,
                        "timestamp_seconds": (
                            float(index / source_fps) if source_fps > 0 else None
                        ),
                    }
                )
            index += 1
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"Video contained no readable frames: {video_path}")
    return frames, records


def _input_frames(
    request: dict[str, Any], output_dir: Path
) -> tuple[list[Path], list[dict[str, Any]]]:
    source = Path(request["input_path"]).resolve()
    max_frames = int(request.get("max_frames", 64))
    if source.is_dir():
        frames = sorted(
            path for path in source.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not frames:
            raise ValueError(f"No supported images found in directory: {source}")
        frames = frames[:max_frames]
        return frames, [
            {"path": str(path), "source_frame_index": index, "timestamp_seconds": None}
            for index, path in enumerate(frames)
        ]
    suffix = source.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return [source], [
            {"path": str(source), "source_frame_index": 0, "timestamp_seconds": 0.0}
        ]
    if suffix in VIDEO_SUFFIXES:
        return _sample_video(
            source,
            output_dir / "sampled_frames",
            sample_fps=float(request.get("sample_fps", 1.0)),
            max_frames=max_frames,
        )
    raise ValueError(f"Unsupported image/video input: {source}")


def _sam3_video_checkpoint(request: dict[str, Any]) -> str | None:
    checkpoint = (
        request.get("checkpoint_path")
        or os.environ.get("R2S_SAM3_CHECKPOINT")
        or _cached_huggingface_file("facebook/sam3", "sam3.pt")
    )
    return str(checkpoint) if checkpoint else None


def _build_sam3_video_predictor(request: dict[str, Any]) -> Any:
    from sam3.model_builder import build_sam3_video_predictor
    from perception.sam_video_storage import install_cpu_video_storage

    # The official video predictor downloads from Hugging Face when
    # checkpoint_path is omitted; unlike the image builder it does not expose a
    # load_from_HF keyword.
    return install_cpu_video_storage(
        build_sam3_video_predictor(checkpoint_path=_sam3_video_checkpoint(request))
    )


def _sam3_start_session(predictor, request):
    from perception.sam_video_storage import offload_video_to_cpu

    return predictor.handle_request({
        "type": "start_session", "resource_path": request["input_path"],
        "offload_video_to_cpu": offload_video_to_cpu(request),
    })


def _sam3_propagation_direction(request, predictor, session_id):
    requested = request.get("propagation_direction", "both")
    anchor = int(request.get("prompt_frame", 0))
    direction = requested
    frame_count = None
    if requested == "both" and anchor > 0:
        get_session = getattr(predictor, "_get_session", None)
        session = get_session(session_id) if callable(get_session) else {}
        frame_count = session.get("state", {}).get("num_frames") if isinstance(session, dict) else None
        if type(frame_count) is not int or not 0 <= anchor < frame_count:
            raise ValueError("SAM3 dual propagation requires valid native session frame-count metadata")
        # SAM3's action-history router treats a completed boundary propagation
        # as full coverage. Forward from the last frame computes only that frame;
        # the following backward leg would merely fetch an empty cache. The
        # anchor output is already captured by add_prompt, so backward alone
        # computes every remaining frame with the original session/track IDs.
        if anchor == frame_count - 1:
            direction = "backward"
    return direction, {"requested": requested, "effective": direction,
        "anchor_frame": anchor, "native_frame_count": frame_count,
        "last_frame_boundary_adjusted": direction != requested}


def _sam3_prompt_box(value: Any) -> list[float] | None:
    """A relative [x, y, w, h] box inside the frame, or None."""
    if value is None:
        return None
    if (not isinstance(value, (list, tuple)) or len(value) != 4
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in value)):
        raise ValueError("prompt_box_xywh must be four finite relative numbers")
    x, y, w, h = (float(v) for v in value)
    if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 - x + 1e-9 and 0 < h <= 1 - y + 1e-9):
        raise ValueError("prompt_box_xywh must lie inside the frame in relative coordinates")
    return [x, y, w, h]


def _sam3_video_condition(
    request: dict[str, Any],
    output_dir: Path,
    *,
    predictor: Any,
    session_id: str,
) -> dict[str, Any]:
    import contextlib
    import numpy as np
    from PIL import Image
    from perception.sam3_visual import covers_box, grid_fallback, hotstart_disabled, merge_outputs

    responses: dict[int, dict[str, Any]] = {}
    prompt = {
        "type": "add_prompt",
        "session_id": session_id,
        "frame_index": int(request.get("prompt_frame", 0)),
        "text": request["prompt"],
        "output_prob_thresh": float(request.get("confidence_threshold", 0.5)),
    }
    box = _sam3_prompt_box(request.get("prompt_box_xywh"))
    if box is not None:
        # A positive box exemplar on the prompt frame, in relative xywh. With
        # prompt_box_only the box alone is the (visual) prompt: a text concept
        # that does not match the body would otherwise suppress it.
        prompt.update(bounding_boxes=[box], bounding_box_labels=[1])
        if request.get("prompt_box_only"):
            prompt["text"] = "visual"
    # A box finds the body by its pixels; SAM3's hotstart would erase that track
    # whenever its text detector does not re-find it in the first frames.
    with (hotstart_disabled(predictor) if box is not None else contextlib.nullcontext(False)) as hotstart_off:
        prompted = predictor.handle_request(prompt)
        responses[int(prompted["frame_index"])] = prompted["outputs"]
        direction, propagation_audit = _sam3_propagation_direction(request, predictor, session_id)
        stream_request: dict[str, Any] = {
            "type": "propagate_in_video",
            "session_id": session_id,
            "propagation_direction": direction,
            "start_frame_index": int(request.get("prompt_frame", 0)),
            "output_prob_thresh": float(request.get("confidence_threshold", 0.5)),
        }
        if request.get("max_frames") is not None:
            stream_request["max_frame_num_to_track"] = int(request["max_frames"])
        for response in predictor.handle_stream_request(stream_request):
            responses[int(response["frame_index"])] = response["outputs"]
    visual_fallback = None
    prompt_outputs = responses.get(int(request.get("prompt_frame", 0)))
    if (box is not None and request.get("visual_fallback") == "grid"
            and not covers_box({key: _numpy(value, np) for key, value in (prompt_outputs or {}).items()}, box)):
        tracked, visual_fallback = grid_fallback(predictor, request, box, _sam3_video_checkpoint(request))
        visual_fallback["trigger"] = "no track on the prompt frame overlaps the located box"
        if tracked:
            responses = merge_outputs({index: {key: _numpy(value, np) for key, value in outputs.items()}
                                       for index, outputs in responses.items()}, tracked)

    mask_dir = output_dir / "masks"
    frame_dir = output_dir / "frames"
    mask_dir.mkdir(exist_ok=True)
    frame_dir.mkdir(exist_ok=True)
    frame_records = []
    tracks: dict[int, list[dict[str, Any]]] = {}
    for frame_index, outputs in sorted(responses.items()):
        object_ids = _numpy(outputs.get("out_obj_ids", []), np).reshape(-1)
        scores = _numpy(outputs.get("out_probs", []), np).reshape(-1)
        masks = _numpy(outputs.get("out_binary_masks", []), np)
        boxes = _numpy(outputs.get("out_boxes_xywh", []), np)
        objects = []
        frame_arrays: dict[str, Any] = {
            "object_ids": object_ids.astype(np.int64),
            "scores": scores.astype(np.float32),
            "masks": masks.astype(np.bool_),
            "boxes_xywh": boxes.astype(np.float32),
        }
        npz_path = frame_dir / f"frame_{frame_index:06d}.npz"
        np.savez_compressed(npz_path, **frame_arrays)
        for item_index, object_id_value in enumerate(object_ids):
            object_id = int(object_id_value)
            mask = _squeeze_mask(masks[item_index], np)
            mask_path = mask_dir / f"frame_{frame_index:06d}_object_{object_id:04d}.png"
            _save_binary_mask(mask, mask_path, np, Image)
            ys, xs = np.nonzero(mask)
            centroid = [float(xs.mean()), float(ys.mean())] if len(xs) else [None, None]
            box = boxes[item_index].tolist() if item_index < len(boxes) else None
            score = float(scores[item_index]) if item_index < len(scores) else None
            objects.append(
                {
                    "object_id": object_id,
                    "mask_path": str(mask_path),
                    "box_xywh": box,
                    "score": score,
                    "centroid_xy": centroid,
                }
            )
            tracks.setdefault(object_id, []).append(
                {
                    "frame_index": frame_index,
                    "centroid_xy": centroid,
                    "box_xywh": box,
                    "score": score,
                }
            )
        frame_records.append(
            {
                "frame_index": frame_index,
                "prediction_path": str(npz_path),
                "objects": objects,
            }
        )

    tracks_path = output_dir / "tracks.json"
    tracks_path.write_text(
        json.dumps(
            {
                "tracks": [
                    {"object_id": key, "observations": value}
                    for key, value in sorted(tracks.items())
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "tool": "sam3",
        "mode": "video",
        "input_path": request["input_path"],
        "prompt": request["prompt"],
        **({"prompt_box_xywh": box, "prompt_box_only": bool(request.get("prompt_box_only")),
            "hotstart": "disabled-for-box-prompt" if hotstart_off else "model-default"} if box is not None else {}),
        **({"visual_fallback": visual_fallback} if visual_fallback is not None else {}),
        "propagation": propagation_audit,
        "video_storage": (
            predictor._get_session(session_id).get("state", {}).get("plumb_video_storage")
            if callable(getattr(predictor, "_get_session", None)) else None
        ),
        "frame_count": len(frame_records),
        "track_count": len(tracks),
        "tracks_path": str(tracks_path),
        "frames": frame_records,
    }
    manifest_path = _write_manifest(output_dir, manifest)
    return {
        "ok": True,
        "tool": "sam3",
        "mode": "video",
        "frame_count": len(frame_records),
        "track_count": len(tracks),
        "manifest_path": str(manifest_path),
        "tracks_path": str(tracks_path),
    }


def _sam3_video(request: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    predictor = _build_sam3_video_predictor(request)
    session_id: str | None = None
    try:
        started = _sam3_start_session(predictor, request)
        session_id = str(started["session_id"])
        return _sam3_video_condition(
            request,
            output_dir,
            predictor=predictor,
            session_id=session_id,
        )
    finally:
        if session_id is not None:
            predictor.handle_request(
                {"type": "close_session", "session_id": session_id}
            )


def run_sam3(request: dict[str, Any]) -> dict[str, Any]:
    # Text- or box-prompted video tracking only; mode "members" (single-frame detection with tiles) is
    # harness/sam3_members.py, routed there by gpu_service.py.
    if request.get("mode", "video") != "video":
        raise ValueError(f"SAM3 mode {request.get('mode')!r} is not served by this worker (only 'video')")
    output_dir = _output_directory(request)
    return _sam3_video(request, output_dir)


def _array_summary(arrays: dict[str, Any]) -> dict[str, Any]:
    return {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in sorted(arrays.items())
    }


def _pi3x_repo(request: dict[str, Any]) -> None:
    """Expose an official Pi3 checkout when the package is not installed."""
    repo = request.get("repo_path") or os.environ.get("R2S_PI3X_REPO")
    if repo:
        repo_path = Path(repo).expanduser().resolve()
        if not repo_path.is_dir():
            raise FileNotFoundError(f"R2S_PI3X_REPO does not exist: {repo_path}")
        sys.path.insert(0, str(repo_path))


def _pi3x_checkpoint(request: dict[str, Any]) -> Path:
    checkpoint = request.get("checkpoint_path") or os.environ.get("R2S_PI3X_CHECKPOINT")
    if not checkpoint:
        raise ValueError("Set R2S_PI3X_CHECKPOINT or pass a local checkpoint_path")
    path = Path(checkpoint).expanduser().resolve()
    if path.is_dir():
        path = path / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"Pi3X checkpoint is missing: {path}")
    return path


def run_pi3x(request: dict[str, Any]) -> dict[str, Any]:
    """Export native Pi3X predictions for standalone use and reconstruction."""
    from perception.pi3x_geometry import PI3X_PIXEL_LIMIT, pi3x_model_size

    from perception.pi3x_geometry import PI3X_WINDOW_FRAMES, PI3X_WINDOW_OVERLAP
    integers = {
        "max_frames": (64, 1, None),
        "pixel_limit": (PI3X_PIXEL_LIMIT, 196, None),
        "seed": (0, 0, 4294967295),
        "window_frames": (int(os.environ.get("R2S_PI3X_WINDOW_FRAMES", PI3X_WINDOW_FRAMES)), 16, None),
        "window_overlap": (PI3X_WINDOW_OVERLAP, 2, None),
    }
    options = dict(request)
    for key, (default, minimum, maximum) in integers.items():
        value = options.setdefault(key, default)
        if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
            raise ValueError(f"Invalid Pi3X {key}: {value!r}")
    sample_fps = options.setdefault("sample_fps", 1.0)
    if (type(sample_fps) not in (int, float) or not math.isfinite(sample_fps) or sample_fps <= 0):
        raise ValueError("Pi3X sample_fps must be finite and positive")
    if options["window_overlap"] >= options["window_frames"] // 2:
        raise ValueError("Pi3X window_overlap must be less than half of window_frames")
    if options.get("device", "auto") not in {"auto", "cuda", "cpu"}:
        raise ValueError("Pi3X device must be auto, cuda or cpu")
    precision = options.get("precision", "auto")
    if precision not in {"auto", "no", "fp16", "bf16"}:
        raise ValueError("Pi3X precision must be auto, no, fp16 or bf16")
    checkpoint_path = _pi3x_checkpoint(options)
    _pi3x_repo(options)

    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from pi3.models.pi3x import Pi3X
    from perception.pi3x_geometry import square_pixel_intrinsics

    device = _device(options, torch)
    if precision == "auto":
        precision = "bf16" if device == "cuda" and torch.cuda.is_bf16_supported() else "no"
    if device == "cpu" and precision != "no":
        raise ValueError("Pi3X CPU inference requires precision=no or auto")
    output_dir = _output_directory(options)
    frames, frame_records = _input_frames(options, output_dir)
    images = []
    for path, record in zip(frames, frame_records, strict=True):
        with Image.open(path) as source:
            record["source_size_hw"] = [source.height, source.width]
            images.append(source.convert("RGB"))
    if any(record["source_size_hw"] != frame_records[0]["source_size_hw"] for record in frame_records):
        raise ValueError("Pi3X frames must share one source size")
    # Official loader: one area-limited raster, full-frame Lanczos resize, [0, 1].
    height, width = pi3x_model_size(frame_records[0]["source_size_hw"], options["pixel_limit"])
    # Kept on the host: windows move to the device one at a time.
    batch = torch.from_numpy(np.stack([
        np.asarray(image.resize((width, height), Image.Resampling.LANCZOS)) for image in images
    ])).permute(0, 3, 1, 2).float().div(255)[None]
    for image in images:
        image.close()
    torch.manual_seed(options["seed"])
    # Strictly load every published tensor, then drop the unused conditioning
    # branches: inference is RGB-only.
    model = Pi3X(use_multimodal=True).eval()
    model.load_state_dict(load_file(str(checkpoint_path)), strict=True)
    model.disable_multimodal()
    model = model.to(device)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)

    def infer(start, end):
        """One forward pass; its world is expressed in its first camera."""
        with torch.no_grad(), torch.autocast(device, dtype=dtype, enabled=dtype is not None):
            predictions = model(batch[:, start:end].to(device))
        local_points = predictions["local_points"][0].float()
        intrinsics = square_pixel_intrinsics(local_points.cpu().numpy(), frame_records[0]["source_size_hw"], np)
        # Pi3X has no reference view; express the metric world in the first camera.
        poses = predictions["camera_poses"][0].float()
        to_first = torch.linalg.inv(poses[0])
        camera_to_world = to_first @ poses
        points = predictions["points"][0].float()
        world_points = points @ to_first[:3, :3].T + to_first[:3, 3]
        tensors = {
            "depths": local_points[..., 2],
            "world_points": world_points,
            "camera_to_world": camera_to_world[:, :3],
            "depth_confidence": predictions["conf"][0, ..., 0].float().sigmoid(),
        }
        result = {key: value.detach().cpu().numpy() for key, value in tensors.items()}
        result["intrinsics"] = intrinsics
        metric = float(predictions["metric"][0])
        del predictions, tensors
        if device == "cuda":
            torch.cuda.empty_cache()
        return result, metric

    from perception.pi3x_geometry import merge_pi3x_windows, pi3x_windows
    window, inference = options["window_frames"], None
    if device == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    while True:
        try:
            spans = pi3x_windows(len(frames), window, options["window_overlap"])
            outputs = [(start, *infer(start, end)) for start, end in spans]
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if window <= 16:
                raise
            # Halve the window and restart; a smaller window only adds alignment.
            window = max(16, min(window, len(frames)) // 2)
            if options["window_overlap"] >= window // 2:
                raise
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if len(outputs) == 1:
        arrays, metric_scale = outputs[0][1], outputs[0][2]
    else:
        arrays, alignment = merge_pi3x_windows([(start, result) for start, result, _ in outputs],
                                               options["window_overlap"], np, seed=options["seed"])
        metric_scale = outputs[0][2]
        inference = {"mode": "windowed-sim3-overlap-v1", "window_frames": window,
                     "overlap_frames": options["window_overlap"],
                     "windows": [{**row, "metric_scale": output[2]} for row, output in zip(alignment, outputs)],
                     "world": "first window's first camera; later windows mapped by a similarity fitted on "
                              "confident overlap points"}
    expected_shapes = {"depths": (len(frames), height, width),
                       "world_points": (len(frames), height, width, 3),
                       "camera_to_world": (len(frames), 3, 4), "intrinsics": (len(frames), 3, 3)}
    for key, array in arrays.items():
        if array.shape[0] != len(frames) or not np.isfinite(array).all():
            raise ValueError(f"Invalid Pi3X output: {key}")
        if key in expected_shapes and array.shape != expected_shapes[key]:
            raise ValueError(f"Unexpected Pi3X {key} shape: {array.shape}")
    if not (arrays["depths"] > 0).all() or not (arrays["intrinsics"][:, (0, 1), (0, 1)] > 0).all():
        raise ValueError("Pi3X returned nonpositive depth or focal length")
    if not np.allclose(np.linalg.det(arrays["camera_to_world"][:, :3, :3]), 1, atol=1e-2):
        raise ValueError("Pi3X returned invalid camera rotations")
    predictions_path = output_dir / "predictions.npz"
    np.savez_compressed(predictions_path, **arrays)
    manifest = {
        "schema_version": 1, "tool": "pi3x", "input_path": options["input_path"],
        "source_frames": [str(path) for path in frames], "frames": frame_records,
        "frame_count": len(frames), "pixel_limit": options["pixel_limit"],
        "processed_size_hw": [height, width], "seed": options["seed"], "device": device,
        "precision": precision, "model_source": "yyfz233/Pi3X", "inference_seconds": elapsed,
        "metric_scale": metric_scale,
        "camera_convention": "OpenCV: x right, y down, z forward; camera_to_world is [R|t]",
        "world_convention": "first camera at the origin; approximate metric units",
        "intrinsics_space": "processed pixels after full-frame resize; one square-source-pixel focal length per frame, least-squares fitted to the predicted rays with a centered principal point",
        "geometry_status": "learned-estimate-in-model-world-coordinates",
        "predictions_path": str(predictions_path), "arrays": _array_summary(arrays),
        **({"inference": inference} if inference else {}),
    }
    manifest_path = _write_manifest(output_dir, manifest)
    return {"ok": True, "tool": "pi3x", "frame_count": len(frames),
            "manifest_path": str(manifest_path), "predictions_path": str(predictions_path),
            "arrays": manifest["arrays"]}


def check_dependencies(name: str) -> dict[str, Any]:
    """Import inference APIs and locate local weights without loading models."""
    import importlib

    modules = ["numpy", "torch", "PIL.Image", "cv2"]
    if name == "sam3":
        modules += ["sam3.model.sam3_image_processor", "sam3.model_builder"]
        checkpoint = _sam3_video_checkpoint({})
        if not checkpoint or not Path(checkpoint).is_file():
            raise FileNotFoundError("SAM3 local checkpoint missing; set R2S_SAM3_CHECKPOINT or populate the local HF cache")
    else:
        modules += ["safetensors.torch", "pi3.models.pi3x", "pi3.utils.geometry"]
        checkpoint = str(_pi3x_checkpoint({}))
        _pi3x_repo({})
    for module in modules:
        importlib.import_module(module)
    return {"ok": True, "tool": name, "operation": "check_dependencies",
            "python": sys.executable, "imports": modules, "checkpoint": str(checkpoint),
            "model_loaded": False}


BACKENDS = {
    "sam3": run_sam3,
    "pi3x": run_pi3x,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or argv[0] not in BACKENDS:
        print(
            "usage: python src/harness/perception/worker.py "
            "{sam3|pi3x}",
            file=sys.stderr,
        )
        return 2
    name = argv[0]
    try:
        request = json.load(sys.stdin)
        if not isinstance(request, dict):
            raise TypeError("worker input must be a JSON object")
        if request.get("tool") not in {None, name}:
            raise ValueError(
                f"request tool {request.get('tool')!r} does not match worker {name!r}"
            )
        # Keep model logs out of the protocol stream. The harness forwards stderr
        # to its renderer while reserving stdout for one machine-readable object.
        with contextlib.redirect_stdout(sys.stderr):
            result = (check_dependencies(name) if request.get("operation") == "check_dependencies"
                      else BACKENDS[name](request))
    except Exception as exc:  # noqa: BLE001 - this is the process error boundary.
        if os.environ.get("R2S_GPU_DEBUG") == "1":
            traceback.print_exc(file=sys.stderr)
        result = {
            "ok": False,
            "tool": name,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
