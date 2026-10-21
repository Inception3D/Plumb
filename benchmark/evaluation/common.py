"""Shared helpers of the evaluators: JSON I/O, hashing, timestamps, headless EGL rendering."""
import ctypes
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
NO_OUTPUT = "no reconstruction: tasks/<job_id>/output/scene.xml does not exist"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def nvidia_egl() -> bool:
    try:
        ctypes.CDLL("libEGL_nvidia.so.0")
        return True
    except OSError:
        return False


def egl_env(env=None, device=None):
    """MuJoCo offscreen rendering through EGL (env: os.environ by default; call before mujoco is imported).

    glvnd picks the EGL vendor library from its JSON list. On some hosts that list holds only Mesa (e.g. the NVIDIA
    driver was installed without its glvnd JSON, or only its libraries are mounted into a container), and Mesa cannot
    open the NVIDIA render nodes, so every MuJoCo Renderer fails. When the NVIDIA EGL library is loadable we name it
    explicitly (egl_vendor_nvidia.json next to this file), which is what the vendor list should say anyway on such a
    host; without it (no NVIDIA driver: Mesa / software EGL) glvnd's own choice is kept, so this is safe elsewhere.
    An explicit __EGL_VENDOR_LIBRARY_FILENAMES or MUJOCO_GL (e.g. osmesa) is never overridden. device: the EGL device
    index (MUJOCO_EGL_DEVICE_ID), None = MuJoCo's default."""
    env = os.environ if env is None else env
    env.setdefault("MUJOCO_GL", "egl")
    if env["MUJOCO_GL"] == "egl" and "__EGL_VENDOR_LIBRARY_FILENAMES" not in env and nvidia_egl():
        env["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(HERE / "egl_vendor_nvidia.json")
    if device is not None:
        env["MUJOCO_EGL_DEVICE_ID"] = str(device)
    return env
