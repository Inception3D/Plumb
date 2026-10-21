#!/usr/bin/env python3
"""Set up a task directory for one video: the layout the skill's tools and the task prompt expect.

  python3 src/harness/new_task.py VIDEO TASK_DIR [--no-prompt]

TASK_DIR/
  input/video.mp4     the video (copied; the only input)
  skill/SKILL.md      the command reference the prompt points at   } read-only copies of this skill
  r2s/  tools/        helper commands, GPU tools                   }
  PROMPT.md           the task prompt (src/prompts/task.md), unless --no-prompt
  output/             the deliverables go here
Then serve the GPU tools (python3 src/harness/gpu_service.py TASK_DIR 0) and start the agent in TASK_DIR with
REAL2SIM_PYTHON set (README.md: "Run a reconstruction").
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SKILL = Path(__file__).resolve().parents[1]


def install_skill(work: Path, src_root: Path = SKILL, names=("skill", "r2s", "tools")) -> None:
    """Copy the skill into a task directory, read-only: skill/SKILL.md (+ references/ of old versions), r2s/, tools/."""
    for name, src in (("skill", None), ("r2s", src_root / "r2s"), ("tools", src_root / "tools")):
        if name not in names:
            continue
        dst = work / name
        if dst.exists():
            continue
        if name == "skill":
            dst.mkdir()
            shutil.copyfile(src_root / "SKILL.md", dst / "SKILL.md")
            if (src_root / "references").exists():
                shutil.copytree(src_root / "references", dst / "references")
        else:
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in names:
        for p in sorted((work / name).rglob("*"), reverse=True):
            mode = 0o555 if (p.is_dir() or p.suffix == "" and p.parent.name == "tools" and p.name != "_gpu_call.py") else 0o444
            p.chmod(mode)
        (work / name).chmod(0o555)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("task_dir", type=Path)
    ap.add_argument("--no-prompt", action="store_true", help="do not write PROMPT.md")
    a = ap.parse_args()
    if not a.video.is_file():
        ap.error(f"no video at {a.video}")
    work = a.task_dir.resolve()
    if work.exists() and any(work.iterdir()):
        ap.error(f"{work} exists and is not empty")
    (work / "input").mkdir(parents=True)
    (work / "output").mkdir()
    shutil.copyfile(a.video, work / "input/video.mp4")
    (work / "input/video.mp4").chmod(0o444)
    install_skill(work)
    if not a.no_prompt:
        shutil.copyfile(SKILL / "prompts/task.md", work / "PROMPT.md")
    print(f"task directory: {work}\n"
          f"next: python3 {SKILL / 'harness/gpu_service.py'} {work} 0     (GPU tools; another shell)\n"
          f"      cd {work} && export REAL2SIM_PYTHON=/path/to/python PYTHONPATH={work}   then start the agent")


if __name__ == "__main__":
    main()
