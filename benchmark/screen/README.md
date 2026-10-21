# Scene screening: how bench50 was chosen

The 50 scenes of `benchmark/bench50.json` were screened from 500 PhysInOne scenes (SinglePhysics, Train; a seeded random
100 per category folder) and 700 Physion v1.5 scenes (test split, 100 per category) by GT-side checks, contact sheets
and picks by hand, on the retimed time axis of the videos (g = 9.81 m/s^2 in GT metres). `D` = raw data root in the
layout of `benchmark/download.py`, `W` = work directory. All stages take `--workers N` and `--only NAME ...` (subsets).

| # | command (`S=benchmark/screen`) | checks / filters |
|---|---|---|
| 1 | `python $S/physinone.py scan --data-root D --work W` | per zip: actors, recorder components (mass, damping, material), frames, clip gravity |
| 2 | `python $S/physinone.py a --data-root D --work W` | Stage A: GT items (`evaluation/physics.py`), movers, energy; flags (below) -> survivors |
| 3 | `python $S/physinone.py b --work W` | survivors: visibility in camera 0 per frame, size and hull volume from 4 GT depth views |
| 4 | `python $S/physinone.py semantic --work W` | GT mass and size vs real-world priors; clusters subclass x background |
| 5 | `python $S/physinone.py sheets --work W` | one representative per cluster; then `show SUB[,SUB] --full`; pick by hand |
| 6 | `python $S/physion.py a --data-root D --work W` | Stage A |
| 7 | `python $S/physion.py cands --work W` | 3 best per wanted subtype (`WANT`): no energy rise, e <= 0.95, most item kinds |
| 8 | `python $S/physion.py sheets --work W` | first candidate per subtype; pick by hand |
| 9 | `python $S/physion.py b --work W` | the Physion picks: visibility in cam0, density (mass / hull volume), push time |
| 10 | `python $S/spec.py --data-root D --work W` | picks (scene list of bench50.json) + W -> scene records, each pick's verdict |
| 11 | `python $S/gt_check.py BENCH --table -` | after `build.py`: v2's moving / visible GT objects, eligible GT items per scene |

Criteria (bench50.json "criteria"; steps in brackets):
- Plausibility: movers' linear damping <= 0.05, angular <= 0.5, no zero mass; first friction stop <= 1 g; e <= 0.95;
  >= 85 % of the horizontal speed kept in free flight; clip gravity 2.25 m/s^2 +- 10 % when measurable [2]; GT mass /
  real-world prior within 3x across colliding movers, balls, cans, Rubik cube 0.5-2x real size [4]; no energy rise [7];
  by eye: no floating spinning panels, 2-6 m cannonballs, 9.9 kg toy ball next to 10 kg boxes [5, 8].
- Passive reproducibility: all motion from the first frame; still frames before an invisible push or release trimmed
  (`first_frame`: dominoes 15, vase 5, pendulum 18); motor-driven Accel*Spin [4], catapults, aliasing spins excluded.
- Evaluability: video, GT trajectories, depth, segmentation for every frame [2] (drop_sidezone_box has no depth:
  drop_simple_box_0003 instead); main movers >= 200 px in >= 50 % of the frames [3, 9] (towers nb5 0005 instead of 0031,
  whose cubes show in 12-14 %); >= 1 GT physics item [2]; textured spinners; no duplicate GT motion.
- Diversity: PhysInOne 15 of 18 subclasses, <= 2 per subclass except pendulum and collision; Physion all 7 categories,
  1-4 subtypes each; 15 scenes (4 PhysInOne, 11 Physion) kept from an earlier random sample passed the same checks.

The pool (~0.8 TB: PhysInOne 342 GB, Physion 473 GB) must be on local disk; stages read parts of it (JSON, poses, masks,
depth, sheet frames). 32 workers: steps 1-2 take seconds, 6 ~12 min (cold cache), 3 ~1.5 min, 9 ~0.5 min; 11 ~2 min (16).
bench50 was screened with the physics metric of that time (physics_v5: mass ratios of all movers, no 1 g cut, spins
about the vertical only); `why` in bench50.json counts its items. With the published physics_v6 used here a fresh screen
differs where items rank scenes: 3 of 357 PhysInOne survivors lose their last item, 6 of 44 sheet representatives and
7 of 26 Physion candidate lists change (pick 123 drops to rank 2), as do 13 of 50 `why` strings; every pick still passes.
