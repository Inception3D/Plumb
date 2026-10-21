"""python -m r2s <command> ...   (run from the task directory; every command and option: skill/SKILL.md;
one command: python -m r2s <command> [<sub>] -h)

  probe | sheet | motion | dump                   video timeline, contact sheets, frame-difference events (r2s/video.py)
  gpu pi3x | sam3 | members | wait | status       Pi3X / SAM3 through the host GPU queue               (r2s/gpu.py)
  pi3x camera | objects | track | plane | depth   read a Pi3X result                                  (r2s/pi3x.py)
  sam3 summary | sheet | members                  read a SAM3 result                                  (r2s/sam3.py)
  track sam3 | color | hsv | manual | show | list | lift   2D tracks (mask centroids) and their plane lift (r2s/track.py)
  check rollout | compare | overlay | members     roll out a scene.xml on the video timeline, compare with tracks (r2s/sim.py)
  deliver [--camera NAME] [--template] [--validate-only]   run.py template, preview.mp4, format check (r2s/deliver.py)
"""
import importlib
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")      # before anything imports mujoco

MODULES = {"probe": "video", "sheet": "video", "motion": "video", "dump": "video", "gpu": "gpu", "pi3x": "pi3x",
           "sam3": "sam3", "track": "track", "check": "sim", "deliver": "deliver"}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd not in MODULES:
        print(f"unknown command {cmd}\n{__doc__}")
        sys.exit(2)
    m = importlib.import_module(f"r2s.{MODULES[cmd]}")
    m.main([cmd] + rest if MODULES[cmd] == "video" else rest)


main()
