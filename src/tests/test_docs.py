"""SKILL.md must list every command and every option of every r2s command (no need to read the source or -h).
usage: PYTHONPATH=src $REAL2SIM_PYTHON src/tests/test_docs.py   (from the repository root)"""
import re, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parents[1]
LEAVES = [["probe"], ["sheet"], ["motion"], ["dump"], ["deliver"]] + \
    [["gpu", c] for c in ("pi3x", "sam3", "members", "wait", "status")] + \
    [["pi3x", c] for c in ("camera", "camera-track", "objects", "track", "plane", "depth")] + \
    [["sam3", c] for c in ("summary", "sheet", "members")] + \
    [["track", c] for c in ("sam3", "color", "hsv", "manual", "show", "list", "lift")] + \
    [["check", c] for c in ("rollout", "compare", "overlay", "members")]
skill = (HERE / "SKILL.md").read_text()
ref = skill[skill.index("## Command reference"):]
missing = []
for c in LEAVES:
    p = subprocess.run([sys.executable, "-m", "r2s", *c, "-h"], capture_output=True, text=True, cwd=HERE)
    if p.returncode:
        missing.append(f"r2s {' '.join(c)} -h failed: {p.stderr[-300:]}")
    name = "r2s " + " ".join(c)
    if f"`{name}" not in ref:
        missing.append(name)
    for opt in sorted(set(re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", p.stdout)) - {"--help"}):
        if opt not in ref:
            missing.append(f"{name} {opt}")
print("\n".join(missing) if missing else "ok: every command and option is in SKILL.md")
sys.exit(1 if missing else 0)
