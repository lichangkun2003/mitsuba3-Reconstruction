"""Prepare an independent COLMAP model for dense MVS at actual image sizes.

The supplied sparse model remains untouched. Poses, point IDs and world-space
coordinates are preserved; only image selection, intrinsics, 2D observations
and corresponding sparse tracks change in the temporary dense workspace.
"""

from __future__ import annotations

from pathlib import Path
import struct
import tempfile

import numpy as np
from PIL import Image

from reconstruction_dataset import (_BinaryReader, _inside, read_colmap_cameras,
                                    read_colmap_images)


_OBSERVATION = np.dtype([("x", "<f8"), ("y", "<f8"), ("point_id", "<i8")])
_TRACK = np.dtype([("image_id", "<i4"), ("point_index", "<i4")])
_MODEL_FILES = ("cameras.bin", "images.bin", "points3D.bin")


def inspect_dense_inputs(source_model: Path, image_root: Path, split="train") -> dict:
    """Validate inputs and report the selected views and exact camera scaling.

    This function writes nothing. The training split follows reconstruction's
    sorted-filename holdout convention: every eighth registered image is test.
    """
    if split not in ("train", "all"):
        raise ValueError("Dense point-cloud split must be 'train' or 'all'")
    source_model, image_root = Path(source_model).resolve(), Path(image_root).resolve()
    for name in _MODEL_FILES:
        if not (source_model / name).is_file():
            raise FileNotFoundError(source_model / name)
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    cameras = read_colmap_cameras(source_model / "cameras.bin")
    images = read_colmap_images(source_model / "images.bin")
    selected = [image for index, image in enumerate(images) if split == "all" or index % 8 != 0]
    excluded = [] if split == "all" else [image for index, image in enumerate(images) if index % 8 == 0]
    if len(selected) < 2:
        raise ValueError("Dense reconstruction requires at least two selected registered images")
    scaled = {}
    for image in selected:
        camera_id = image["camera_id"]
        if camera_id not in cameras:
            raise ValueError(f"Image {image['name']} refers to missing camera {camera_id}")
        camera = cameras[camera_id]
        if camera["model"] not in ("PINHOLE", "SIMPLE_PINHOLE"):
            raise ValueError(f"Camera {camera_id} uses {camera['model']}; provide undistorted PINHOLE images")
        path = _inside(image_root, image["name"], "COLMAP image filename")
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as reference:
            width, height = reference.size
        if camera_id in scaled:
            previous = scaled[camera_id]
            if (width, height) != (previous["width"], previous["height"]):
                raise ValueError(f"Images sharing camera {camera_id} have inconsistent dimensions: {path}")
            continue
        sx, sy = width / camera["width"], height / camera["height"]
        if camera["model"] == "PINHOLE":
            fx, fy, cx, cy = camera["parameters"]
        else:
            fx, cx, cy = camera["parameters"]
            fy = fx
        if fx <= 0 or fy <= 0:
            raise ValueError(f"Camera {camera_id} has nonpositive focal length")
        # Promotion also handles independently rounded width/height reductions.
        model = "SIMPLE_PINHOLE" if camera["model"] == "SIMPLE_PINHOLE" and sx == sy else "PINHOLE"
        parameters = [fx * sx, cx * sx, cy * sy] if model == "SIMPLE_PINHOLE" else [
            fx * sx, fy * sy, cx * sx, cy * sy]
        scaled[camera_id] = dict(id=camera_id, source_model=camera["model"], model=model,
            source_width=camera["width"], source_height=camera["height"], width=width,
            height=height, scale_x=sx, scale_y=sy, parameters=parameters)
    summary = lambda image: {key: image[key] for key in ("id", "name", "camera_id")}
    return dict(source_model=str(source_model), image_root=str(image_root), split=split,
                registered_images=len(images), selected_image_count=len(selected),
                excluded_image_count=len(excluded), selected_images=[summary(image) for image in selected],
                excluded_images=[summary(image) for image in excluded],
                cameras=[scaled[key] for key in sorted(scaled)])


def _read_observations(path, selected_ids, cameras):
    result = {}
    with _BinaryReader(path) as reader:
        count = reader.count(73)
        for _ in range(count):
            values = reader.unpack("idddddddi")
            name = reader.name().replace("\\", "/")
            count = reader.count(24)
            if values[0] not in selected_ids:
                reader.skip(count * 24)
                continue
            observations = np.frombuffer(reader.stream.read(count * 24), dtype=_OBSERVATION).copy()
            if len(observations) != count:
                raise ValueError(f"Truncated COLMAP observations: {path}")
            if not np.isfinite(observations["x"]).all() or not np.isfinite(observations["y"]).all():
                raise ValueError(f"Nonfinite observations in image {name}")
            camera = cameras[values[-1]]
            observations["x"] *= camera["scale_x"]
            observations["y"] *= camera["scale_y"]
            result[values[0]] = (values, name, observations)
        reader.finish()
    return result


def _read_retained_points(path, images):
    points, point_ids = [], set()
    linked = {image_id: np.zeros(len(image[2]), dtype=np.bool_) for image_id, image in images.items()}
    with _BinaryReader(path) as reader:
        source_count = reader.count(51)
        for _ in range(source_count):
            values = reader.unpack("QdddBBBd")
            point_id = values[0]
            if point_id in point_ids or point_id > np.iinfo(np.int64).max:
                raise ValueError(f"Invalid or duplicate COLMAP point ID {point_id}")
            point_ids.add(point_id)
            if not np.isfinite(values[1:4]).all() or not np.isfinite(values[-1]):
                raise ValueError(f"Nonfinite COLMAP point {point_id}")
            track_count = reader.count(8)
            track = np.frombuffer(reader.stream.read(track_count * 8), dtype=_TRACK)
            kept = []
            seen = set()
            for row in track:
                image_id, index = int(row["image_id"]), int(row["point_index"])
                if image_id not in images:
                    continue
                # Supplied 360 models can associate several distinct features
                # in one image with the same point. Preserve these indices;
                # MVS eligibility still requires two distinct retained views.
                if (image_id, index) in seen:
                    raise ValueError(f"Point {point_id} repeats one sparse track observation")
                seen.add((image_id, index))
                observations = images[image_id][2]
                if not 0 <= index < len(observations) or observations["point_id"][index] != point_id:
                    raise ValueError(f"Inconsistent sparse track for point {point_id}, image {image_id}")
                kept.append((image_id, index))
            if len({image_id for image_id, _ in kept}) >= 2:
                points.append((values, kept))
                for image_id, index in kept:
                    linked[image_id][index] = True
        reader.finish()
    # Preserve observation indices: detached observations stay present with -1.
    # This also handles selected images which only saw removed sparse points.
    for image_id, (_, _, observations) in images.items():
        observations["point_id"][~linked[image_id]] = -1
    if not points:
        raise ValueError("No sparse points retain observations in at least two selected images; "
                         "COLMAP cannot infer MVS source views and depth ranges")
    return points, source_count


def prepare_dense_model(source_model: Path, image_root: Path, output_model: Path, split="train") -> dict:
    """Atomically write a resized, view-filtered model without modifying sources."""
    source_model, image_root, output_model = (Path(value).resolve() for value in
                                             (source_model, image_root, output_model))
    for source in (source_model, image_root):
        if output_model.is_relative_to(source) or source.is_relative_to(output_model):
            raise ValueError("Dense model output must not overlap the original model or image directory")
    if output_model.exists():
        raise FileExistsError(f"Dense model output already exists: {output_model}")
    report = inspect_dense_inputs(source_model, image_root, split)
    cameras = {camera["id"]: camera for camera in report["cameras"]}
    selected_ids = {image["id"] for image in report["selected_images"]}
    images = _read_observations(source_model / "images.bin", selected_ids, cameras)
    points, source_count = _read_retained_points(source_model / "points3D.bin", images)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output_model.name}-", dir=output_model.parent) as temporary:
        staging = Path(temporary)
        with (staging / "cameras.bin").open("wb") as stream:
            stream.write(struct.pack("<Q", len(cameras)))
            for camera in report["cameras"]:
                model_id = 0 if camera["model"] == "SIMPLE_PINHOLE" else 1
                stream.write(struct.pack("<iiQQ", camera["id"], model_id, camera["width"], camera["height"]))
                stream.write(struct.pack("<" + "d" * len(camera["parameters"]), *camera["parameters"]))
        with (staging / "images.bin").open("wb") as stream:
            stream.write(struct.pack("<Q", len(images)))
            for selected in report["selected_images"]:
                values, name, observations = images[selected["id"]]
                stream.write(struct.pack("<idddddddi", *values))
                stream.write(name.encode("utf-8") + b"\0")
                stream.write(struct.pack("<Q", len(observations)))
                stream.write(observations.tobytes())
        with (staging / "points3D.bin").open("wb") as stream:
            stream.write(struct.pack("<Q", len(points)))
            for values, track in points:
                stream.write(struct.pack("<QdddBBBdQ", *values, len(track)))
                for pair in track:
                    stream.write(struct.pack("<ii", *pair))
        staging.rename(output_model)
    report.update(output_model=str(output_model), source_points=source_count,
                  retained_points=len(points), removed_points=source_count - len(points),
                  retained_observations=sum(len(track) for _, track in points))
    return report
