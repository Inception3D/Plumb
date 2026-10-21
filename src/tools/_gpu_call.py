"""Submit one GPU tool request and wait for the result (the GPU runs outside this sandbox).

Requests are keyed by their content: running the same command again (for example after the shell
was interrupted) attaches to the request already queued or reuses its finished result instead of
queueing the job a second time. Progress is printed while waiting."""
import hashlib
import json
from pathlib import Path
import sys
import time

TASK = Path(__file__).resolve().parents[1]


def call(tool, request, timeout_s=4 * 3600):
    q, r = TASK / ".gpu/requests", TASK / ".gpu/results"
    q.mkdir(parents=True, exist_ok=True)
    r.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(json.dumps(dict(tool=tool, request=request), sort_keys=True).encode()).hexdigest()[:12]
    rid = f"{tool}-{key}.json"
    if (r / rid).exists() and not (q / rid).exists():
        res = json.loads((r / rid).read_text())
        if res.get("ok"):
            print(f"[gpu] reusing the finished result of an identical request ({rid})", file=sys.stderr, flush=True)
            return res
        (r / rid).unlink()                       # a failed attempt: queue it again
    if (q / rid).exists():
        print(f"[gpu] attaching to the identical request already queued ({rid})", file=sys.stderr, flush=True)
    else:
        tmp = q / (rid + ".tmp")
        tmp.write_text(json.dumps(dict(tool=tool, request=request)))
        tmp.rename(q / rid)
    t0, last = time.time(), 0.0
    while time.time() - t0 < timeout_s:
        if (r / rid).exists() and not (q / rid).exists():
            return json.loads((r / rid).read_text())
        if time.time() - last > 60:
            print(f"[gpu] {tool} waiting for the GPU queue: {int(time.time() - t0)} s", file=sys.stderr, flush=True)
            last = time.time()
        time.sleep(2)
    return {"ok": False, "error": "timed out waiting for the GPU service"}
