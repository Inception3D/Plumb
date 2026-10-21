"""Many-similar-objects tools: tiled member detection (harness/sam3_members.py, with a stand-in detector) and
`r2s check members` (scene bodies vs members, rendered with MuJoCo; needs EGL).
usage: PYTHONPATH=src $REAL2SIM_PYTHON src/tests/test_members.py   (from the repository root)"""
import json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "harness"))
import sam3_members as S

fails = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        fails.append(what)


def fake_detector(rects, capacity=200, crowd=150):
    """Like SAM3's image detector: every rectangle at least partly inside the crop (cut ones clipped), but a crop
    with more than `crowd` rectangles saturates: only `capacity` returned, most of them at a low score."""
    def det(rgb, prompt, origin=None):
        x0, y0 = det.origin
        h, w = rgb.shape[:2]
        rows = []
        for (a, b, c, e) in rects:
            ca, cb, cc, ce = max(a, x0), max(b, y0), min(c, x0 + w), min(e, y0 + h)
            if ca < cc and cb < ce:
                rows.append((ca - x0, cb - y0, np.ones((ce - cb, cc - ca), bool), 0.8))
        if len(rows) > crowd:
            rng = np.random.default_rng(len(rows))
            rows = [(a, b, m, 0.8 if rng.random() < 0.3 else 0.15) for a, b, m, _ in rows[:capacity]]
        return rows
    det.origin = (0, 0)
    return det


class Cropping:
    """detect_frame passes image crops; the stand-in needs their offset, recovered from the array's memory."""
    def __init__(self, img, det):
        self.img, self.det = img, det

    def __call__(self, rgb, prompt):
        off = (rgb.__array_interface__["data"][0] - self.img.__array_interface__["data"][0]) // 3
        self.det.origin = (off % self.img.shape[1], off // self.img.shape[1])
        return self.det(rgb, prompt)


img = np.zeros((720, 1280, 3), np.uint8)
grid = [(40 + 40 * i, 60 + 40 * j, 40 + 40 * i + 14, 60 + 40 * j + 30) for i in range(30) for j in range(14)]   # 420
keep, cand, levels = S.detect_frame(img, ["domino"], Cropping(img, fake_detector(grid)), (0, 0, 1280, 720), 0.4)
check(len(keep) == len(grid), f"crowd of {len(grid)}: {len(keep)} members after grids "
      + " -> ".join(str(l["members"]) for l in levels))
few = grid[:6]
keep, cand, levels = S.detect_frame(img, ["domino"], Cropping(img, fake_detector(few)), (0, 0, 1280, 720), 0.4)
check(len(keep) == 6 and len(levels) == 1, f"6 objects: whole frame only ({len(levels)} pass), {len(keep)} members")
pair = S.Det(100, 100, np.ones((30, 40), bool), 0.9, "p", 0)              # one coarse mask over two members
left, right = S.Det(100, 100, np.ones((30, 18), bool), 0.6, "p", 1), S.Det(122, 100, np.ones((30, 18), bool), 0.6, "p", 1)
check(len(S.merge([pair, left, right])[0]) == 2, "a coarse mask merging two finer members does not add a third")
check(len(S.merge([S.Det(0, 0, np.ones((20, 20), bool), 0.5, "a", 0), S.Det(2, 1, np.ones((20, 20), bool), 0.7, "b", 0)])[0]) == 1,
      "two prompts' detections of one member count once")

# --- r2s check members: a row of boxes rendered by MuJoCo is the "detection"; scenes with one box less / one more
SCENE = """<mujoco><visual><global offwidth="320" offheight="240"/></visual><worldbody>
<camera name="video_cam" pos="0 -1.2 0.35" xyaxes="1 0 0 0 0.28 0.96" fovy="45"/>
<geom type="plane" size="2 2 .1" rgba=".8 .7 .5 1"/>{bodies}</worldbody></mujoco>"""


def boxes(xs):
    return "".join(f'<body name="dom_{i:02d}" pos="{x:.3f} 0 0.05"><geom type="box" size=".01 .03 .05" rgba=".2 .3 .9 1"/>'
                   f"</body>" for i, x in enumerate(xs))


xs = [-0.35 + 0.1 * i for i in range(8)]
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    (d / "input").mkdir(); (d / "analysis").mkdir(); (d / "output").mkdir()
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=320x240:r=10",
                    "-frames:v", "3", "-pix_fmt", "yuv420p", str(d / "input/video.mp4")], check=True)
    os.chdir(d)
    os.environ.setdefault("MUJOCO_GL", "egl")
    from r2s.video import probe
    from r2s.sim import rollout, members
    probe()
    (d / "output/scene.xml").write_text(SCENE.format(bodies=boxes(xs)))
    ST = {"time": 0, "qpos": [], "qvel": []}
    (d / "output/initial_state.json").write_text(json.dumps(ST))
    labs = {}
    rollout(d / "output/scene.xml", ST, camera="video_cam", render=[0], on_frame=lambda k, lab, dep: labs.__setitem__(k, lab))
    lab = labs[0]
    ids = sorted(int(b) for b in np.unique(lab) if b > 0)
    ms = [lab == b for b in ids]
    sam = d / "analysis/sam3_dom"
    (sam / "frames").mkdir(parents=True)
    np.savez_compressed(sam / "frames/frame_000000.npz", object_ids=np.arange(len(ms) + 1),
                        masks=np.stack([np.any(ms, 0)] + ms), scores=np.ones(len(ms) + 1), boxes_xywh=np.zeros((len(ms) + 1, 4)))
    mem = []
    for i, m in enumerate(ms, 1):
        vs, us = np.nonzero(m)
        mem.append(dict(id=i, centroid=[float(us.mean()), float(vs.mean())],
                        bbox=[int(us.min()), int(vs.min()), int(us.max()) + 1, int(vs.max()) + 1], area=int(m.sum())))
    (sam / "members.json").write_text(json.dumps(dict(mode="members", anchor_frame=0, threshold=0.4, members=mem,
                                                      candidates=[], per_frame=[dict(frame=0, members=len(mem))])))
    check(len(mem) == 8, f"8 boxes rendered as 8 members ({len(mem)})")
    r = members(sam, d / "output/scene.xml", ST)
    check((r["matched"], r["members_without_body"], r["bodies_without_member"]) == (8, 0, 0), f"same scene: {r['matched']} matched")
    (d / "output/scene.xml").write_text(SCENE.format(bodies=boxes(xs[:3] + xs[4:])))
    r = members(sam, d / "output/scene.xml", ST)
    check((r["matched"], r["members_without_body"], r["bodies_without_member"]) == (7, 1, 0),
          f"one box missing: {r['matched']} matched, {r['members_without_body']} without body")
    (d / "output/scene.xml").write_text(SCENE.format(bodies=boxes(xs + [0.47])))
    r = members(sam, d / "output/scene.xml", ST, bodies=["dom_*"])
    check((r["matched"], r["bodies_without_member"]) == (8, 1), f"one box too many: {r['bodies_without_member']} without member")
    dense = [-0.35 + 0.0778 * i for i in range(10)]                   # same run, laid out at a wrong spacing
    (d / "output/scene.xml").write_text(SCENE.format(bodies=boxes(dense)))
    r = members(sam, d / "output/scene.xml", ST, bodies=["dom_*"])
    check((r["members"], r["bodies_visible"], r["bodies_without_member"]) == (8, 10, 2),
          f"10 boxes over the run of 8: {r['matched']} matched, {r['bodies_without_member']} without member")
print("\n".join(["FAILED:"] + fails) if fails else "all ok")
sys.exit(1 if fails else 0)
