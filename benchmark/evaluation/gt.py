"""Read genuine local annotations only in the external evaluator: GT masks and GT motion (poses, motion class).

PhysInOne: lossless segmentation NPZ, recorder_stats.json, <sample>_trajectory.json and blender_<camera>.json inside
the sample zip. Physion v1.5: HDF5 object segmentation colours, _id_cam0, per-frame object poses and camera matrices.
"""
import csv
import io
import json
import zipfile
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def source_rows(metadata):
    with (metadata / "timestamps.csv").open() as stream:
        return list(csv.DictReader(stream))


class MaskTruth:
    def __init__(self, task, metadata):
        self.task = task
        self.rows = source_rows(metadata)
        self.objects = []
        self.mapping = {}
        if task["dataset"] == "physion_v15":
            self.handle = h5py.File(task["source"]["path"], "r")
            static = self.handle["static"]
            ids, colors = static["object_ids"][()], static["object_segmentation_colors"][()]
            names = static["model_names"][()]
            for index, (oid, color) in enumerate(zip(ids, colors)):
                self.mapping[int(color[0])*65536+int(color[1])*256+int(color[2])] = index + 1
                self.objects.append({"label": index + 1, "source_id": int(oid), "name": names[index].decode(),
                                     "scope": "all annotated physical objects, including visible supports"})
            self.kind = "HDF5 object_segmentation_colors + lossless _id_cam0"
        elif task["dataset"] == "physinone":
            self.handle = zipfile.ZipFile(task["source"]["path"])
            names = self.handle.namelist()
            stats_name = next(n for n in names if Path(n).name == "recorder_stats.json")
            stats = json.loads(self.handle.read(stats_name))
            for group in ("solid_object", "interactable_object"):
                for entry in stats.get(group, []):
                    for name, obj in entry.items():
                        codes = sorted({int(c["custom_stencil_value"]) for c in obj.get("components", [])
                                        if c.get("custom_stencil_value", 0) > 0})
                        if not codes:
                            continue
                        label = len(self.objects) + 1
                        for code in codes:
                            if code in self.mapping and self.mapping[code] != label:
                                raise ValueError("Ambiguous PhysInOne stencil ID")
                            self.mapping[code] = label
                        self.objects.append({"label": label, "source_id": name, "name": name, "stencil_ids": codes,
                                             "interaction_object": group == "interactable_object"})
            self.mask_members = {int(Path(n).stem): n for n in names
                                 if f"/{task['source']['camera']}/seg/" in n and n.endswith(".npz")}
            self.kind = "Lossless PhysInOne seg NPZ; component stencils merged by recorder actor"
        else:
            raise ValueError(f"No GT masks for dataset {task['dataset']!r}")

    def frame(self, index):
        frame = self.rows[index]["source_frame"]
        if self.task["dataset"] == "physion_v15":
            images = self.handle["frames"][frame]["images"]
            raw = np.array(Image.open(io.BytesIO(images["_id_cam0"][()].tobytes())))
            raw = raw[..., :3].astype(np.uint32)
            raw = raw[..., 0] * 65536 + raw[..., 1] * 256 + raw[..., 2]
        else:
            with np.load(io.BytesIO(self.handle.read(self.mask_members[int(frame)])), allow_pickle=False) as packed:
                raw = packed["seg"]
            if not np.isfinite(raw).all() or not np.equal(raw, np.rint(raw)).all():
                raise ValueError("GT segmentation contains noninteger/nonfinite IDs")
        result = np.zeros(raw.shape, dtype=np.uint16)
        for code, label in self.mapping.items():
            result[raw == code] = label
        return result

    def close(self):
        self.handle.close()


# ---------------------------------------------------------------- GT motion
def motion_class(positions, quaternions, distance=0.005, angle=1.0):
    """XYZ in meters, XYZW quaternions; sign-equivalent rotations are identical."""
    p, q = np.asarray(positions, float), np.asarray(quaternions, float)
    if len(p) < 2 or len(q) != len(p) or not np.isfinite(p).all() or not np.isfinite(q).all():
        return {"class": "unknown", "reason": "Incomplete finite GT pose sequence"}
    norms = np.linalg.norm(q, axis=-1)
    if (norms < 1e-8).any():
        return {"class": "unknown", "reason": "Invalid GT quaternion"}
    q = q / norms[:, None]
    displacement = float(np.linalg.norm(p-p[0], axis=-1).max())
    rotation = float(np.rad2deg(2*np.arccos(np.clip(np.abs(q@q[0]), 0, 1))).max())
    return {"class": "moving" if displacement > distance or rotation > angle else "static",
            "max_displacement_m": displacement, "max_rotation_degrees": rotation,
            "basis": "GT translation OR rotation relative to first input frame"}


def perspective(focal, width, height, frames):
    matrix = np.zeros((4, 4))
    matrix[0, 0], matrix[1, 1], matrix[3, 2] = 2*focal/width, 2*focal/height, -1
    return np.broadcast_to(matrix, (frames, 4, 4))


def read_mask_poses(task, metadata, objects, distance, angle):
    """Per input frame and GT object: camera-frame reference point, GT projection matrix, motion class, point
    definition and camera-frame orientation (OpenGL camera axes)."""
    rows = source_rows(metadata)
    frames, count = len(rows), len(objects)
    centers = np.full((frames, count, 3), np.nan)
    classes, point_definitions = [], []
    orientations = np.full((frames, count, 3, 3), np.nan)
    if task["dataset"] == "physion_v15":
        camera_points, projections, pose_positions, rotations, view_rotations = [], [], [], [], []
        with h5py.File(task["source"]["path"], "r") as handle:
            for row in rows:
                frame = handle["frames"][row["source_frame"]]
                xyz = frame["objects/center_cam0"][()]
                view = frame["camera_matrices/camera_matrix_cam0"][()].reshape(4, 4)
                camera_points.append(xyz @ view[:3, :3].T + view[:3, 3])
                projections.append(frame["camera_matrices/projection_matrix_cam0"][()].reshape(4, 4))
                pose_positions.append(frame["objects/positions_cam0"][()])
                rotations.append(frame["objects/rotations_cam0"][()])
                view_rotations.append(view[:3, :3])
        p, q = np.array(pose_positions), np.array(rotations)
        classes = [motion_class(p[:, i], q[:, i], distance, angle) for i in range(count)]
        point_definitions = ["GT world bounding-box center; prediction representative primitive center"] * count
        norms = np.linalg.norm(q, axis=-1)
        valid = np.isfinite(q).all(axis=-1) & (norms > 1e-8)
        world_rotations = np.full_like(orientations, np.nan)
        world_rotations[valid] = Rotation.from_quat(q[valid]).as_matrix()
        orientations = np.einsum("tij,tnjk->tnik", np.array(view_rotations), world_rotations)
        return np.array(camera_points), np.array(projections), classes, point_definitions, orientations
    with zipfile.ZipFile(task["source"]["path"]) as handle:
        names = handle.namelist()
        trajectory = json.loads(handle.read(next(n for n in names if Path(n).name == task["sample_id"] + "_trajectory.json")))
        camera = json.loads(handle.read(next(n for n in names if Path(n).name == "blender_" + task["source"]["camera"] + ".json")))
    selected_frames = [int(row["source_frame"]) for row in rows]
    def series(records):
        lookup = {int(r["frame"]): r["transform"] for r in records}
        p, q = [], []
        for index in selected_frames:
            value = lookup.get(index)
            if value is None:
                p.append([np.nan]*3); q.append([np.nan]*4)
                continue
            p.append([value["location"][key] for key in ("x", "y", "z")])
            rotation = value["rotation"].get("quaternion", {})
            q.append([rotation.get(key, np.nan) for key in ("x", "y", "z", "w")])
        return np.array(p)*[.01, -.01, .01], np.array(q)
    for i, obj in enumerate(objects):
        actor = trajectory["actors"].get(obj["source_id"])
        if actor is None:
            # Recorder solid actors are the explicitly separated static environment.
            classes.append({"class": "unknown" if obj["interaction_object"] else "static",
                            "basis": "Missing interactable pose" if obj["interaction_object"] else "Recorded solid/environment actor, absent from interactable pose export"})
            point_definitions.append("unavailable: no recorded GT actor/component position")
            continue
        records = actor.get("transform_data", [])
        components = actor.get("components", {})
        channels = [("actor", records)] if records else [(name, c["transform_data"]) for name, c in components.items() if c.get("transform_data")]
        if not channels:
            classes.append({"class": "unknown", "reason": "No GT pose channels"})
            point_definitions.append("unavailable: no pose channels")
            continue
        decoded = {name: series(values) for name, values in channels}
        states = [motion_class(p, q, distance, angle) for p, q in decoded.values()]
        group = "moving" if any(s["class"] == "moving" for s in states) else ("unknown" if any(s["class"] == "unknown" for s in states) else "static")
        classes.append({"class": group, "basis": "Any recorded actor/component translation OR rotation", "channels": dict(zip(decoded, states))})
        primary = next((name for name in ("actor", "Body2", "Body") if name in decoded), None)
        centers[:, i] = decoded[primary][0] if primary else np.mean([p for p, q in decoded.values()], axis=0)
        point_definitions.append("GT " + (primary + " origin" if primary else "mean of component origins") + "; prediction representative primitive center; UE centimeters converted to Blender meters [x,-y,z]/100")
        orientation_channel = primary or next(iter(decoded))
        q = decoded[orientation_channel][1]
        valid = np.isfinite(q).all(axis=-1) & (np.linalg.norm(q, axis=-1) > 1e-8)
        reflection = np.diag([1., -1., 1.])
        orientations[valid, i] = reflection @ Rotation.from_quat(q[valid]).as_matrix() @ reflection
        point_definitions[-1] += "; rotation reference channel: " + orientation_channel
    camera_frames = {int(f["frame"]): np.array(f["transform_matrix"]) for f in camera["frames"]}
    matrices = np.array([camera_frames[index] for index in selected_frames])
    camera_points = np.einsum("tnj,tjk->tnk", centers-matrices[:, None, :3, 3], matrices[:, :3, :3])
    focal = camera["img_w"]/(2*np.tan(camera["camera_angle_x"]/2))
    projection = perspective(focal, camera["img_w"], camera["img_h"], frames)
    orientations = np.einsum("tji,tnjk->tnik", matrices[:, :3, :3], orientations)
    return camera_points, projection, classes, point_definitions, orientations
