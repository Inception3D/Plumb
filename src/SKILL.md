---
name: real2sim
description: Work environment for reconstructing a MuJoCo scene from a single video - GPU foundation models (Pi3X 4D reconstruction, SAM3 video segmentation and member detection for many similar objects) through a host queue, and optional helper commands (python -m r2s) for the video timeline, contact sheets, SAM3 / Pi3X results, 2D tracks, rolling out a scene.xml on the video timeline with rendered comparisons (also scene bodies vs detected members), the run.py template, preview.mp4 and a deliverable format check.
---

# real2sim environment

This directory gives you tools, not a method: how to reconstruct the scene is up to you and the task prompt.

- `tools/pi3x_recon`, `tools/sam3_track`: the GPU foundation models (they run outside the sandbox through a file
  queue; their `-h` describes inputs and outputs).
- `python -m r2s ...` (`r2s` below = `"$REAL2SIM_PYTHON" -m r2s`): helper commands for chores that sessions
  otherwise rewrite by hand. All optional; each one is listed with every option below. Python helpers for your
  own scripts are listed at the end (`PYTHONPATH` already contains the task directory; prepend, never replace:
  `PYTHONPATH=.:$PYTHONPATH`, mujoco is found through it).
- Run everything from the task directory. The commands write to `analysis/` (files) and `output/` (deliverables);
  they print short summaries and write JSON / JPG files (open the JPGs with your image viewer).

Conventions
- Frame k of the video is at `analysis/timeline.json` `t[k]` (decoded PTS in seconds, first frame = 0).
- Pixels are video pixels (u right, v down), frames are 0-based.
- Camera files (`analysis/camera.json`): pinhole, square pixels, principal point at the image centre, MuJoCo
  `fovy`, pose `R_world_from_cv` + `pos`. The world of `r2s pi3x camera` is z-up with the fitted support plane at
  z = 0, in Pi3X units (a learned, approximately metric estimate) times `--scale`.
- OpenCV camera frame: x right, y down, z forward. MuJoCo camera frame: x right, y up, looking along -z.
- Tracks (`analysis/tracks/<name>.json`): `frames`, `uv` (mask centroids), `area`, `bbox`, `mask_source`.
- Members (`r2s gpu members` -> `analysis/sam3_<name>/`): `members.json` (`members`: id, prompt, score, centroid,
  bbox x0 y0 x1 y1, area, mean rgb on `anchor_frame`; `candidates`: detections at score 0.2 - threshold, bbox only;
  `per_frame` counts) and `frames/frame_<K>.npz` like a SAM3 result: ids 1..N = members on the anchor frame, id 0 =
  their union on every listed frame (identities are not followed between frames).
- `r2s check` and `r2s deliver` roll out any `scene.xml` + `initial_state.json` the way the `run.py` template
  does: state set once at t = 0, `mj_step` until the first step at or after each input frame time.
- Moving camera (`output/camera_track.json`, optional): `camera` (name of a world-attached scene camera), `t`,
  `pos` (n, 3) and `quat` (n, 4; w x y z of the MuJoCo camera frame in the world), one pose per input frame.
  Render-only: before frame k is drawn, the run.py template and `r2s check compare / overlay` put pose k into that
  camera's `data.cam_xpos` / `cam_xmat` (`mj_step` recomputes them; the simulation does not read them, model
  parameters do not change). Without the file the camera stays at its scene.xml pose.

Many similar objects (domino runs, cup pyramids, piles): SAM3 video tracking does not hold a crowd (on a spiral of
~250 dominoes `gpu sam3 "domino"` returned 0 tracks; `gpu members` found 241), and a count read off the frame by eye
or laid out as path length / spacing drifts (a marble-run reconstruction: 158 visible dominoes for 148 detected,
only 99 of them at a detected domino; the others sat between two real ones). `gpu members` lists them on one frame
(`sam3 members` draws them numbered, `pi3x objects --sam3` gives 3D positions), `check members` pairs the scene's
bodies with them. Candidates below the threshold can be members that are partly hidden, or parts / reflections of
members already found (all 9 in that marble run were the feet of detected translucent dominoes seen through the
rail); the prompt matters (translucent dominoes: "domino" 0, "colored translucent tile" 148).

## Command reference

Video
- `r2s probe` -> analysis/timeline.json + frame_times.json (frame count, size, fps, duration, PTS of every frame).
- `r2s sheet [--every N | --frames K ... | --times S ...] [--start K] [--stop K] [--crop x0 y0 x1 y1] [--zoom Z]
  [--grid PX] [--cols N] [--thumb W (alias --tile, --width)] [--video other.mp4] [-o|--out file.jpg]`
  contact sheet, tiles labelled `f<frame> t=<s>`; `--grid` draws pixel coordinates; `--video` draws another
  video on the same timeline (e.g. output/preview.mp4).
- `r2s motion [--thresh 18] [--min-area 30]` -> analysis/motion.json (frames where the frame-to-frame change
  starts, peaks, stops), motion.png, motion_detail.json (per-frame change curve and moving blobs).
- `r2s dump --frames K ... [-o|--out analysis/frames]` -> PNG files.

GPU (host queue; a call takes 0.5-3 min of GPU time plus the queue wait; submit, then collect)
- `r2s gpu pi3x [--max-frames 48] [--out analysis/pi3x]` whole video, evenly sampled, frame 0 included.
- `r2s gpu sam3 --name NAME --prompt TEXT [--box x0 y0 x1 y1] [--box-only] [--prompt-frame K] [--threshold 0.4]
  [--max-frames N] [--out DIR]` -> analysis/sam3_NAME. `--box` (video px, on the prompt frame) without
  `--box-only` is an exemplar for the text prompt (SAM3 may return several instances, one id each); with
  `--box-only` SAM3 tracks the boxed object.
- `r2s gpu sam3 --obj NAME PROMPT X0 Y0 X1 Y1 FRAME [--obj ...]` several requests in one command.
- `r2s gpu members --name NAME --prompt TEXT [--prompt TEXT ...] [--frame 0] [--frames K ... | --every N]
  [--region x0 y0 x1 y1] [--threshold 0.4] [--grid auto|0-4] [--out DIR]` many similar objects: every instance on
  --frame by SAM3's image detector (prompts merged), on the whole frame and, while the detector is crowded
  (one pass has 200 queries on a 1008 x 1008 resize of the frame) and finer tiles still add members, on 2 / 8 /
  15 / 28 overlapping tiles (`--grid` fixes the number of tile rows); a frame with fewer detections is detected
  whole, like `gpu sam3` on one frame. --frames / --every: the union (id 0) on those frames. 0.5-3 min.
- `r2s gpu wait DIR [--timeout 240]` / `r2s gpu wait --all [--tracks]` blocks up to --timeout s and prints a
  summary; `--tracks` also writes one track per SAM3 id (`<name>` or `<name>_<id>`; a members result: the union
  track `<name>_union`). Give the shell call a timeout >= --timeout + 60 s.
- `r2s gpu status` every submitted job and its state.

Pi3X result (analysis/pi3x)
- `r2s pi3x camera [--dir analysis/pi3x] [--region x0 y0 x1 y1] [--scale S] [--out analysis/camera.json]`
  RANSAC plane through the confident Pi3X points of frame 0 in REGION (default: lower 45 % of the image) ->
  z-up world on that plane; focal = Pi3X's; prints height, pitch, inliers and the MJCF `<camera>` element.
- `r2s pi3x camera-track [--dir analysis/pi3x] [--camera analysis/camera.json] [--scale S] [--rotation-only]
  [--name video_cam] [-o|--out output/camera_track.json]` the camera file's camera moved by Pi3X's camera motion
  between its samples (frame 0 = the camera file's pose; rotation slerp, position linear in between, held outside
  the sampled range; translation x S, default the file's `scale_applied`, else 1) -> a camera_track.json; prints
  the largest rotation / translation from frame 0. The `r2s gpu wait` summary of a Pi3X job has its camera motion.
- `r2s pi3x objects --sam3 DIR [--frame 0] [--dir analysis/pi3x] [--camera analysis/camera.json] [-o|--out FILE]`
  per SAM3 id: median and 5-95 % extent of its Pi3X points on that frame (visible surface), median confidence
  (printed; with more than 20 ids, e.g. members, saved to analysis/pi3x_objects_<name>.json and summarised).
- `r2s pi3x track --sam3 DIR [--ids ...] [--dir analysis/pi3x] [--camera analysis/camera.json] [-o|--out FILE]`
  the same on every Pi3X sample -> analysis/pi3x_track_<sam3 name>.json
  {id: {frames, t, world, cam (OpenCV camera frame), pixels, conf}}.
- `r2s pi3x plane --region x0 y0 x1 y1 [--dir analysis/pi3x] [--camera analysis/camera.json]` plane of a region
  (ramp, table, wall): normal, slope vs world up, region centre in the camera file's world.
- `r2s pi3x depth --uv u v [--frame 0] [--dir analysis/pi3x]` Pi3X depth (its units) at a pixel.

SAM3 result
- `r2s sam3 summary DIR` ids, frames covered, median area, start / end centroid, path length.
- `r2s sam3 sheet DIR [--every N] [--ids ...] [-o|--out file.jpg]` masks drawn on frames, labelled by id (a
  members result: the union, id 0).
- `r2s sam3 members DIR [--crop x0 y0 x1 y1] [--tile PX] [-o|--out file.jpg]` a members result: numbered member
  outlines in zoomed tiles of ~6 member sizes (`<dir>/members_p<i>.jpg`; overview `<dir>/members.jpg` with the
  tile boxes), candidates as yellow `?score` boxes; `summary` / `gpu wait` print the count after each detector pass
  and members much larger (two in one mask, another object) or smaller (mostly hidden, a part) than the median.

Tracks (analysis/tracks/)
- `r2s track sam3 DIR [--ids ...] [--names ...]` one track per SAM3 id (mask centroid, area, bbox per frame).
- `r2s track color --name N --hsv hlo slo vlo hhi shi vhi [--hsv2 ...] [--roi x0 y0 x1 y1] [--min-area 20]
  [--start K] [--stop K]` or `--obj NAME hlo slo vlo hhi shi vhi` (repeatable): largest blob in an OpenCV HSV
  range (H 0-180; red wraps at 0/180: give the second range with --hsv2).
- `r2s track hsv --frame K --at u v [--at ...]` RGB / HSV at pixels (5x5 median).
- `r2s track manual --name N --frames K ... --uv u v ...` points you read off images.
- `r2s track show [--names ...] [--every N]` -> analysis/tracks_sheet.jpg. `r2s track list`.
- `r2s track lift --name N [--z 0] [--frames a b] [--camera analysis/camera.json]` centroids intersected with the
  plane z = Z of the camera file's world -> analysis/lift_N.json; least-squares position and velocity over the
  frame range (default the first 11 track frames); frames where the mask touches the image border.

Scene rollout and comparison (any scene.xml)
- `r2s check rollout [--scene output/scene.xml] [--state output/initial_state.json]` -> analysis/sim_rollout.npz
  (frame_times, sim_times, body_names, body_xpos, body_xquat, qpos).
- `r2s check compare [--scene S] [--state J] [--camera NAME] [--camera-track FILE|none] [--link TRACK=BODY[,BODY...]]
  [--tracks ...] [--every K] [-o|--out analysis/compare.json]` renders the scene through camera NAME (default: its only camera,
  else video_cam) and compares each track with the rendered mask of its bodies (default: the body named like
  the track; a body's own geoms, not its children; geom groups 0-2, alpha-0 geoms not drawn): centroid distance
  in px (mean, first, 1st / 2nd half, last, worst@frame), frames not visible, IoU with the SAM3 mask; with an
  analysis/pi3x_track_<sam3 name>.json: depth ratio sim / Pi3X of the visible-surface centroid (camera frame) and
  the 3D distance after one ratio common to all tracks. Also lists bodies with degrees of freedom and no track.
  `--link` bodies may be name patterns (fnmatch): `--link dominoes_union='domino_*'`.
- `r2s check overlay [--scene S] [--state J] [--camera NAME] [--camera-track FILE|none] [--frames K ...] [--cols N]
  [-o|--out file.jpg]` rendered outlines (one colour per body) and track centroids (white) on video frames.
- `r2s check members --sam3 DIR [--scene S] [--state J] [--camera NAME] [--camera-track FILE|none]
  [--bodies GLOB ...] [-o|--out analysis/members_<name>.json]` a members result vs the scene on its frame: members
  and rendered bodies (`--bodies` name patterns, default: bodies with >= 50 % of their visible pixels on or next to
  the members) paired one to one by centroid distance, within half the members' median spacing: counts of
  matched (px, IoU), members with no body, visible bodies with no member, bodies not visible (off-screen or
  hidden; with --bodies); on the other listed frames the union IoU. analysis/members_<name>.jpg: green matched,
  red member without body, blue body without member.
- compare / overlay / members move the camera along camera_track.json next to the scene when it exists
  (`--camera-track FILE` another one, `none` the fixed scene pose).

Deliverables
- `r2s deliver [--camera NAME] [--template]` writes output/video_timeline.json; writes output/run.py from the
  template if there is none (`--template` replaces yours): loads scene.xml + initial_state.json once, steps
  continuously, stores qpos / body positions at the input frame times, `--render` writes preview.mp4 from camera
  NAME with the input's frame times, moved along output/camera_track.json if present (`run.py --headless
  --duration S --output DIR [--render]`); renders
  preview.mp4 with it; then validates. With your own run.py,
  preview.mp4 is yours too.
- `r2s deliver --validate-only` -> analysis/validation.json: the five files of prompt item 8 exist; scene.xml
  loads with no actuators / mocap / plugins and asset paths inside output/; initial_state.json has nq / nv
  entries and time 0; a full-length rollout stays finite without MuJoCo warnings; preview.mp4 has the input's
  frame count and PTS; the preview camera is identifiable (a literal `camera="NAME"` in run.py's update_scene
  call, or one camera in the scene: the scene is re-rendered externally through it); run.py does not mention
  input/, analysis/, tracks or optimisers; an output/camera_track.json has one finite pose per input frame for the
  preview camera, that camera is attached to the world body, and run.py reads the file.

## Python helpers

```python
from r2s.common import Camera, timeline, load_json, save_json, K_from_fovy, fovy_from_f
from r2s.common import load_camera_track, set_camera_pose   # (camera, pos, quat) ; set_camera_pose(data, cam_id, pos, quat)
cam = Camera.load("analysis/camera.json")          # .fovy .f .K .R (world_from_cv) .pos .width .height
cam.project(X) ; cam.ray(uv) ; cam.lift_to_height(uv, z) ; cam.depth(X) ; cam.world_to_cv(X) ; cam.scaled(s)
cam.mjcf("video_cam")                               # '<camera name=... pos=... xyaxes=... fovy=.../>'
Camera.look_at(fovy, W, H, pos, target) ; Camera.from_mujoco(model, data, "video_cam", W, H) ; cam.save(path)
from r2s.video import Video, contact_sheet          # Video()[k] -> RGB frame k
from r2s.pi3x import Pi3X                           # p = Pi3X("analysis/pi3x"): p.frames, p.f_video(i),
                                                    # p.points_cam(i), p.conf(i), p.depth_at(i, uv), p.fit_plane(i, region)
from r2s.track import load_tracks, save_track
from r2s.sam3 import load_members                   # members.json of a `gpu members` result (None otherwise)
from r2s.sim import rollout, compare, members
r = rollout("output/scene.xml", state=None, times=None, camera=None, render=None, on_frame=None, camera_track="auto")
#   r["times"], r["sim_times"], r["body_names"], r["xpos"] (T,nbody,3), r["xquat"], r["qpos"],
#   r["uv"] (T,nbody,2) body origins projected through the scene camera, r["camera"], r["camera_track"], r["warning"];
#   camera_track: "auto" (camera_track.json next to the scene, on the video timeline), a file, or None
#   render = frame indices; on_frame(k, body_id_image (H,W; -1 = none), depth (H,W)) for each of them
c = compare("output/scene.xml", links={"track": ["body"]}, every=1)   # the dict `r2s check compare` saves
c = members("analysis/sam3_dominoes", "output/scene.xml", bodies=["domino_*"])   # the dict `r2s check members` saves
```
