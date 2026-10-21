"""GPU tool service of the real2sim skill: runs SAM3 / Pi3X for the tools of (sandboxed) task directories.

usage: python3 src/harness/gpu_service.py ROOT SLOTS [--env-file FILE]   e.g. ... tasks/billiards 0   or ... BATCH 4,4,5
       python3 src/harness/gpu_service.py --check [--env-file FILE]      imports both models' APIs, finds the weights
  ROOT   a task directory (src/harness/new_task.py), a directory of task directories, or a batch directory with
         tasks at jobs/*/attempts/*/work
  SLOTS  GPU indices, one concurrent job per entry (0,0,1: two jobs on GPU 0, one on GPU 1)

The agent's sandbox may not open the GPU, so SAM3 / Pi3X run here, outside it. A tool call inside a task
directory writes <task>/.gpu/requests/<id>.json; this service validates that every path stays inside that task
directory, runs harness/perception/worker.py (harness/sam3_members.py for sam3 requests with mode "members") with
the Python of the model's environment on a free GPU slot and writes <task>/.gpu/results/<id>.json. Slots bound GPU
memory use across parallel tasks.

Configuration: environment variables, or KEY=VALUE lines in --env-file (default: src/harness/gpu.env if it exists;
see src/harness/gpu.env.example; variables already set in the environment win; relative paths are relative to the
repository root). Every entry of the file is passed to
the workers (e.g. CUDA_HOME).
  R2S_SAM3_PYTHON       Python of the SAM3 environment (required)
  R2S_PI3X_PYTHON       Python of the Pi3 environment (required)
  R2S_PI3X_CHECKPOINT   Pi3X model.safetensors or its directory (required)
  R2S_PI3X_REPO         checkout of github.com/yyfz/Pi3 (default: the third_party/Pi3 submodule)
  R2S_SAM3_CHECKPOINT   sam3.pt (default: the local Hugging Face cache of facebook/sam3)
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
PI3_SUBMODULE = HERE.parents[1] / "third_party/Pi3"
REQUIRED = ("R2S_SAM3_PYTHON", "R2S_PI3X_PYTHON", "R2S_PI3X_CHECKPOINT")
PATTERNS = (".gpu/requests/*.json", "*/.gpu/requests/*.json", "jobs/*/attempts/*/work/.gpu/requests/*.json")


def read_env_file(path):
    env = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.removeprefix("export ").partition("=")
        if not sep:
            raise SystemExit(f"{path}: not KEY=VALUE: {line}")
        env[key.strip()] = os.path.expanduser(os.path.expandvars(value.strip().strip("'\"")))
    return env


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("root", nargs="?", type=Path)
ap.add_argument("slots", nargs="?")
ap.add_argument("--env-file", type=Path, default=HERE / "gpu.env")
ap.add_argument("--check", action="store_true", help="check both model environments and exit")
a = ap.parse_args()
if not a.check and not (a.root and a.slots):
    ap.error("ROOT and SLOTS are required (or --check)")
CONFIG = read_env_file(a.env_file) if a.env_file.exists() else {}
CONFIG.update({k: v for k, v in os.environ.items() if k.startswith("R2S_") or k in CONFIG})
for key, value in CONFIG.items():       # relative paths are relative to the repository root
    if key.startswith("R2S_") and "/" in value and not os.path.isabs(value):
        CONFIG[key] = str(HERE.parents[1] / value)
if not CONFIG.get("R2S_PI3X_REPO") and (PI3_SUBMODULE / "pi3").is_dir():
    CONFIG["R2S_PI3X_REPO"] = str(PI3_SUBMODULE)
missing = [k for k in REQUIRED if not CONFIG.get(k)]
if missing:
    raise SystemExit(f"set {', '.join(missing)} (environment or {a.env_file}; see src/harness/gpu.env.example)")
PY = {"sam3": CONFIG["R2S_SAM3_PYTHON"], "pi3x": CONFIG["R2S_PI3X_PYTHON"]}
ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", **CONFIG, "PYTHONPATH": str(HERE)}   # weights are local

if a.check:
    ok = True
    for tool, py in PY.items():
        p = subprocess.run([py, str(HERE / "perception/worker.py"), tool], input=json.dumps(dict(operation="check_dependencies")),
                           capture_output=True, text=True, env=dict(os.environ, **ENV), timeout=600)
        out = p.stdout.strip().splitlines()
        res = json.loads(out[-1]) if out else {"ok": False, "error": p.stderr[-2000:] or "worker printed nothing"}
        ok &= bool(res.get("ok"))
        print(f"{tool}: " + (f"ok ({res['python']}; checkpoint {res['checkpoint']})" if res.get("ok")
                             else f"FAILED {res.get('error_type', '')}: {res.get('error')}"))
    sys.exit(0 if ok else 1)

ROOT = a.root.resolve()
SLOTS = [int(g) for g in a.slots.split(",")]
free = list(SLOTS)
lock = threading.Lock()
seen = set()
log = open(ROOT / "gpu_service.log", "a", buffering=1)


def inside(task, p):
    try:
        return Path(p).resolve().is_relative_to(task.resolve())
    except Exception:
        return False


def run(req_path, gpu):
    task = req_path.parents[2]
    res_path = task / ".gpu/results" / req_path.name
    t0 = time.time()
    try:
        req = json.loads(req_path.read_text())
        tool = req["tool"]
        body = dict(req["request"])
        if tool not in PY:
            raise ValueError(f"unknown tool {tool}")
        for key in ("input_path", "output_dir"):
            if not inside(task, body.get(key, "")):
                raise ValueError(f"{key} must be inside the task directory")
        body["input_path"] = str(Path(body["input_path"]).resolve())
        body["output_dir"] = str(Path(body["output_dir"]).resolve())
        body.setdefault("device", "cuda")
        env = dict(os.environ, **ENV, CUDA_VISIBLE_DEVICES=str(gpu))
        # mode "members" (many similar objects, single-frame detection with tiles) is its own script
        cmd = [PY[tool], str(HERE / "sam3_members.py")] if tool == "sam3" and body.get("mode") == "members" \
            else [PY[tool], str(HERE / "perception/worker.py"), tool]
        p = subprocess.run(cmd, input=json.dumps(body),
                           capture_output=True, text=True, env=env, timeout=4 * 3600)
        out = p.stdout.strip().splitlines()
        result = json.loads(out[-1]) if out else {"ok": False, "error": "worker printed nothing"}
        if not result.get("ok"):
            result["stderr_tail"] = p.stderr[-3000:]
    except Exception as e:
        result = {"ok": False, "error_type": type(e).__name__, "error": str(e)}
    result["service_seconds"] = round(time.time() - t0, 1)
    tmp = res_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    tmp.rename(res_path)
    log.write(f"{time.strftime('%H:%M:%S')} {task.relative_to(ROOT)} {req_path.name} gpu{gpu} ok={result.get('ok')} {result['service_seconds']}s\n")
    done = task / ".gpu/done"            # the request leaves the queue; an identical retry can be queued again
    done.mkdir(exist_ok=True)
    try:
        req_path.rename(done / req_path.name)
    except OSError:
        pass
    seen.discard(req_path)
    with lock:
        free.append(gpu)


def mtime(p):
    try:
        return p.stat().st_mtime
    except OSError:                      # moved to .gpu/done by a finishing worker between glob and stat
        return float("inf")


print(f"serving {ROOT} on GPU slots {SLOTS} (log: {ROOT / 'gpu_service.log'})", flush=True)
while True:
    for req in sorted((p for pattern in PATTERNS for p in ROOT.glob(pattern)), key=mtime):
        if not req.exists():
            continue
        if req in seen:
            continue
        with lock:
            if not free:
                break
            gpu = free.pop(0)
        seen.add(req)
        threading.Thread(target=run, args=(req, gpu), daemon=True).start()
    time.sleep(1)
