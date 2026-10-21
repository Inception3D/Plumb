# bench50

50 videos for evaluating reconstructions: 25 from [PhysInOne](https://huggingface.co/datasets/vLAR/PhysInOne) and 25
from [Physion v1.5](https://physion-benchmark.github.io/physion_v15). Each video is retimed so that one video second is
one second of physics with g = 9.81 m/s². The scene list, selection criteria and checksums are in
[bench50.json](bench50.json); [screen/](screen/) has the scripts that selected the scenes.

Run all commands from the repository root, in the agent environment with the benchmark packages installed:

```bash
.venv/bin/pip install -r benchmark/requirements.txt
```

**1. Download the 50 source files** (37.8 GB; PhysInOne from Hugging Face, Physion v1.5 members read by range requests)

```bash
.venv/bin/python benchmark/download.py --data-root data      # --hf-endpoint https://hf-mirror.com if needed
```

**2. Build the task directories** (about 230 MB)

```bash
.venv/bin/python benchmark/build.py --data-root data --output bench50
```

`bench50/tasks/<id>/` are task directories like those of `src/harness/new_task.py`. `bench50/meta/<id>/` holds the
video timestamps the evaluators use, outside the agent's directory. With FFmpeg 4.4.2 the videos are byte-identical to
the published ones. Other FFmpeg builds give the same frames and timestamps, and `build.py` warns when a checksum
differs.

**3. Reconstruct** every task as in the main [README](../README.md#usage):

```bash
python3 src/harness/gpu_service.py bench50/tasks 0,0,1      # in another shell
export REAL2SIM_PYTHON=$PWD/.venv/bin/python
for t in bench50/tasks/*/; do (cd $t && PYTHONPATH=$PWD codex exec --skip-git-repo-check --sandbox workspace-write -C . - < PROMPT.md); done
```

**4. Evaluate**

```bash
.venv/bin/python benchmark/evaluate.py bench50 [--metrics v2 physics secondview] [--tasks 101 112] [--workers 6]
```

Results are written to `bench50/evaluation/{v2,physics,secondview}/metrics.csv`. The evaluator simulates only
`output/scene.xml` and `initial_state.json` (passively, never the agent's code). The published numbers reproduce
exactly with `mujoco==3.6.0`, the version in `requirements.txt`. Other MuJoCo versions simulate the scenes
differently.

- **v2** (input view): 3D ADE (m) of the visible-surface centroids of the moving objects after one depth scale, 2D ADE
  (px) of their mask centroids, and mask IoU.
- **physics**: initial speed, restitution, friction deceleration, swing period, spin rate and acceleration, and
  relative mass, measured the same way on the ground truth and on the reconstruction. An item is correct within a
  factor of 1.25 (restitution: ±0.1).
- **secondview**: 2D ADE and IoU from a camera the agent never saw (PhysInOne: a held-out camera; Physion: the ground
  truth rebuilt in MuJoCo).
