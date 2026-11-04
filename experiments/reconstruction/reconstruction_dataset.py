"""Read COLMAP cameras and prepare explicit bounded/unbounded initializations.

Files and camera poses remain untouched. Coordinates use one camera-derived
similarity transform. Only core points seed the foreground; fixed background
cube shells are allocated separately without coordinate compression.
"""

from __future__ import annotations

import math
from pathlib import Path, PurePosixPath
import struct

import numpy as np


_AXES = np.array(((1., 0., 0.), (0., 0., 1.), (0., -1., 0.)))
_MODELS = {0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4),
           2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5), 4: ("OPENCV", 8),
           5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12),
           7: ("FOV", 5), 8: ("SIMPLE_RADIAL_FISHEYE", 4),
           9: ("RADIAL_FISHEYE", 5), 10: ("THIN_PRISM_FISHEYE", 12)}
_PLY_TYPES = {"char": ("b", 1), "int8": ("b", 1), "uchar": ("B", 1),
              "uint8": ("B", 1), "short": ("h", 2), "int16": ("h", 2),
              "ushort": ("H", 2), "uint16": ("H", 2), "int": ("i", 4),
              "int32": ("i", 4), "uint": ("I", 4), "uint32": ("I", 4),
              "float": ("f", 4), "float32": ("f", 4),
              "double": ("d", 8), "float64": ("d", 8)}


class _BinaryReader:
    def __init__(self, path):
        self.path = Path(path)
        self.size = self.path.stat().st_size
        self.stream = self.path.open("rb")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()

    def unpack(self, fmt):
        size = struct.calcsize("<" + fmt)
        data = self.stream.read(size)
        if len(data) != size:
            raise ValueError(f"Truncated COLMAP file: {self.path}")
        return struct.unpack("<" + fmt, data)

    def count(self, minimum_record_bytes):
        count, = self.unpack("Q")
        if count > (self.size - self.stream.tell()) // minimum_record_bytes:
            raise ValueError(f"Invalid record count in {self.path}")
        return count

    def skip(self, count):
        if count < 0 or count > self.size - self.stream.tell():
            raise ValueError(f"Truncated COLMAP file: {self.path}")
        self.stream.seek(count, 1)

    def name(self):
        value = bytearray()
        for _ in range(4096):
            byte = self.stream.read(1)
            if not byte:
                raise ValueError(f"Unterminated image name in {self.path}")
            if byte == b"\0":
                try:
                    return value.decode("utf-8")
                except UnicodeDecodeError as error:
                    raise ValueError(f"Invalid UTF-8 image name in {self.path}") from error
            value.extend(byte)
        raise ValueError(f"Image name exceeds 4095 bytes in {self.path}")

    def finish(self):
        if self.stream.tell() != self.size:
            raise ValueError(f"Unexpected trailing data in {self.path}")


def read_colmap_cameras(path):
    """Return camera records keyed by COLMAP's (not necessarily contiguous) ID."""
    cameras = {}
    with _BinaryReader(path) as reader:
        count = reader.count(48)
        for _ in range(count):
            camera_id, model_id, width, height = reader.unpack("iiQQ")
            if model_id not in _MODELS:
                raise ValueError(f"Unsupported COLMAP camera model ID {model_id}: {path}")
            model, parameter_count = _MODELS[model_id]
            parameters = reader.unpack("d" * parameter_count)
            if camera_id in cameras or width < 1 or height < 1:
                raise ValueError(f"Invalid or duplicate COLMAP camera {camera_id}: {path}")
            if not np.isfinite(parameters).all():
                raise ValueError(f"Nonfinite parameters in camera {camera_id}: {path}")
            cameras[camera_id] = {"id": camera_id, "model": model, "width": width,
                                  "height": height, "parameters": parameters}
        reader.finish()
    if not cameras:
        raise ValueError(f"No cameras in {path}")
    return cameras


def read_colmap_images(path):
    """Read poses and names, skipping unused 2D feature observations."""
    images = []
    ids, names = set(), set()
    with _BinaryReader(path) as reader:
        count = reader.count(73)
        for _ in range(count):
            values = reader.unpack("idddddddi")
            image_id, camera_id = values[0], values[-1]
            quaternion, translation = values[1:5], values[5:8]
            name = reader.name().replace("\\", "/")
            _safe_relative(name, "COLMAP image name")
            if image_id in ids or name in names:
                raise ValueError(f"Duplicate image ID or filename in {path}: {name}")
            if not np.isfinite(quaternion + translation).all():
                raise ValueError(f"Nonfinite camera pose in {path}: {name}")
            observation_count = reader.count(24)
            reader.skip(observation_count * 24)
            images.append({"id": image_id, "camera_id": camera_id, "name": name,
                           "quaternion": quaternion, "translation": translation})
            ids.add(image_id)
            names.add(name)
        reader.finish()
    if not images:
        raise ValueError(f"No registered images in {path}")
    return sorted(images, key=lambda image: image["name"])


def read_colmap_points(path):
    """Read xyz positions and sRGB colors; skip reprojection errors and tracks."""
    with _BinaryReader(path) as reader:
        count = reader.count(51)
        positions = np.empty((count, 3), dtype=np.float64)
        colors = np.empty((count, 3), dtype=np.float32)
        ids = set()
        for index in range(count):
            values = reader.unpack("QdddBBBd")
            if values[0] in ids:
                raise ValueError(f"Duplicate point ID in {path}: {values[0]}")
            ids.add(values[0])
            positions[index] = values[1:4]
            colors[index] = values[4:7]
            track_count = reader.count(8)
            reader.skip(track_count * 8)
        reader.finish()
    _validate_points(positions, colors, path)
    colors /= 255.
    return positions, colors


def _validate_points(positions, colors, path):
    if not len(positions):
        raise ValueError(f"Point cloud is empty: {path}")
    if not np.isfinite(positions).all() or not np.isfinite(colors).all():
        raise ValueError(f"Point cloud contains nonfinite positions or colors: {path}")
    if np.any(colors < 0) or np.any(colors > 255):
        raise ValueError(f"Point cloud colors must lie in [0, 255]: {path}")


def read_ply_points(path):
    """Read plain xyz/RGB ASCII or binary PLY files, preserving source units."""
    path = Path(path)
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        encoding, count, current_element = None, None, None
        properties, preceding_count = [], 0
        for _ in range(512):
            raw = stream.readline(8193)
            if not raw or len(raw) > 8192:
                raise ValueError(f"Invalid PLY header: {path}")
            fields = raw.decode("ascii").split()
            if not fields or fields[0] in ("comment", "obj_info"):
                continue
            if fields[0] == "end_header":
                break
            if fields[0] == "format":
                if len(fields) != 3 or fields[2] != "1.0":
                    raise ValueError(f"Unsupported PLY format: {path}")
                encoding = fields[1]
            elif fields[0] == "element":
                if len(fields) != 3 or int(fields[2]) < 0:
                    raise ValueError(f"Invalid PLY element: {path}")
                current_element = fields[1]
                if current_element == "vertex":
                    if count is not None or preceding_count:
                        raise ValueError(f"PLY vertex must be the first nonempty element: {path}")
                    count = int(fields[2])
                elif count is None:
                    preceding_count += int(fields[2])
            elif fields[0] == "property" and current_element == "vertex":
                if len(fields) != 3 or fields[1] not in _PLY_TYPES:
                    raise ValueError(f"PLY vertex requires scalar properties: {path}")
                properties.append((fields[2], fields[1]))
        else:
            raise ValueError(f"PLY header exceeds 512 lines: {path}")
        if count is None or encoding not in ("ascii", "binary_little_endian", "binary_big_endian"):
            raise ValueError(f"Missing vertex element or unsupported PLY format: {path}")
        names = [name for name, _ in properties]
        if len(set(names)) != len(names) or not set(("x", "y", "z", "red", "green", "blue")) <= set(names):
            raise ValueError(f"PLY requires unique x/y/z and red/green/blue properties: {path}")
        # Validate the payload before allocating arrays, including hostile counts.
        minimum_size = count * (2 * len(properties) - 1 if encoding == "ascii" else
                                sum(_PLY_TYPES[kind][1] for _, kind in properties))
        if minimum_size > path.stat().st_size - stream.tell():
            raise ValueError(f"Truncated PLY vertex data: {path}")
        if encoding == "ascii":
            values = np.empty((count, len(properties)), dtype=np.float64)
            for index in range(count):
                fields = stream.readline().split()
                if len(fields) != len(properties):
                    raise ValueError(f"Invalid PLY vertex {index}: {path}")
                values[index] = [float(value) for value in fields]
            columns = {name: values[:, index] for index, name in enumerate(names)}
        else:
            order = "<" if encoding == "binary_little_endian" else ">"
            dtype = np.dtype([(name, order + _PLY_TYPES[kind][0]) for name, kind in properties])
            vertices = np.fromfile(stream, dtype=dtype, count=count)
            if len(vertices) != count:
                raise ValueError(f"Truncated PLY vertex data: {path}")
            columns = {name: vertices[name] for name in names}
        positions = np.column_stack([columns[name] for name in ("x", "y", "z")]).astype(np.float64)
        colors = np.column_stack([columns[name] for name in ("red", "green", "blue")]).astype(np.float32)
        _validate_points(positions, colors, path)
        color_types = dict(properties)
        integer_colors = any(_PLY_TYPES[color_types[name]][0] not in ("f", "d")
                             for name in ("red", "green", "blue"))
        if integer_colors or np.max(colors) > 1:
            colors /= 255.
    return positions, colors


def _safe_relative(name, description):
    path = PurePosixPath(str(name).replace("\\", "/"))
    if not path.parts or path.is_absolute() or ".." in path.parts or ":" in str(path):
        raise ValueError(f"{description} must be a relative path inside the dataset: {name}")
    return path


def _inside(root, relative, description):
    name = _safe_relative(relative, description)
    root = Path(root).resolve()
    result = root.joinpath(*name.parts).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"{description} escapes its directory: {relative}")
    return result


def find_colmap_model(dataset):
    dataset = Path(dataset)
    for candidate in (dataset / "sparse" / "0", dataset / "sparse"):
        if (candidate / "cameras.bin").is_file() and (candidate / "images.bin").is_file():
            return candidate
    raise FileNotFoundError(f"Expected cameras.bin and images.bin in {dataset / 'sparse/0'} or {dataset / 'sparse'}")


def detect_dataset(dataset, cameras_file="transforms_train.json"):
    dataset = Path(dataset)
    if (dataset / cameras_file).is_file() or any(dataset.glob("transforms_*.json")):
        return "synthetic"
    find_colmap_model(dataset)
    return "colmap"


def quaternion_rotation(quaternion):
    """COLMAP's scalar-first world-to-camera quaternion as an orthonormal matrix."""
    value = np.asarray(quaternion, dtype=np.float64)
    length = np.linalg.norm(value)
    if value.shape != (4,) or not np.isfinite(length) or length < 1e-12:
        raise ValueError("COLMAP camera quaternion is degenerate")
    w, x, y, z = value / length
    return np.array(((1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)),
                     (2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)),
                     (2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y))))


def rescale_camera(camera, width, height):
    """Return a copy with full pinhole intrinsics scaled to the render image."""
    if int(width) != width or int(height) != height or width < 1 or height < 1:
        raise ValueError("Image dimensions must be positive integers")
    result = dict(camera)
    source_width = camera.get("intrinsic_width", camera["width"])
    source_height = camera.get("intrinsic_height", camera["height"])
    sx, sy = width / source_width, height / source_height
    for name in ("fx", "cx"):
        result[name] = float(camera[name]) * sx
    for name in ("fy", "cy"):
        result[name] = float(camera[name]) * sy
    result.update(width=int(width), height=int(height), intrinsic_width=int(width),
                  intrinsic_height=int(height), camera_angle_x=2. * math.atan(width / (2. * result["fx"])))
    return result


def _image_directory(dataset, images_dir):
    if images_dir is not None:
        path = _inside(dataset, images_dir, "Image directory")
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path
    scene = dataset.name.lower()
    preferred = "images_4" if scene in {"bicycle", "garden", "stump", "flowers", "treehill"} else (
        "images_2" if scene in {"bonsai", "counter", "kitchen", "room"} else "images")
    for name in dict.fromkeys((preferred, "images", "images_2", "images_4", "images_8")):
        path = dataset / name
        if path.is_dir():
            return path
    raise FileNotFoundError(f"No reference image directory in {dataset}")


def prepare_colmap(dataset, split="train", images_dir=None, point_cloud="init_points.ply",
                   initialization=None, start_core_resolution=256, target_core_resolution=512,
                   shell_layers=None, load_points=True, *, scene_mode=None,
                   background_shell_levels=None, background_opacity=None,
                   background_initial_resolution=None, background_refinement=None,
                   background_initial_leaf_budget=200000, background_initial_pixel_threshold=32.0,
                   background_max_resolution=512, background_resolution_mode="per_shell",
                   foreground_refinement="uniform", foreground_max_resolution=None, **removed_options):
    """Prepare normalized cameras and core-only point seeds without contraction.

    New unbounded scenes start with 56 leaves per world-space cube shell and
    uniformly subdivide their background at each foreground refinement.
    Saved projective/radial initializations require a fresh reconstruction.
    """
    from PIL import Image
    from reconstruction_world_shell import (reject_removed_background_fields, validate_world_shell_layout,
                                            world_shell_metadata_layout, background_initial_subdivision,
                                            WORLD_SHELL_VERSION)

    reject_removed_background_fields(removed_options)
    if background_resolution_mode not in ("per_shell", "core_parent", "core"):
        raise ValueError("background_resolution_mode must be per_shell, core_parent or core")
    if removed_options or shell_layers is not None:
        raise ValueError("Legacy shell settings are unsupported; use background_shell_levels")
    dataset = Path(dataset).resolve()
    if split not in ("train", "test", "all"):
        raise ValueError("COLMAP split must be 'train', 'test', or 'all'")
    if scene_mode not in (None, "bounded", "unbounded"):
        raise ValueError("COLMAP scene_mode must be bounded or unbounded")
    saved = None if initialization is None else dict(initialization)
    if saved is not None:
        mode = saved.get("scene_mode")
        mapping = saved.get("scene_mapping")
        if mapping not in ("bounded", "world_shell") or saved.get("version") not in (2, 7, 8):
            raise ValueError("Legacy compressed COLMAP initialization is unsupported; start a new world_shell reconstruction")
        if mapping == "world_shell" and (saved.get("version") not in (7, 8) or mode != "unbounded"):
            raise ValueError("world_shell initialization requires version 7 or 8 and unbounded scene mode")
        if mode not in ("bounded", "unbounded") or (mapping == "bounded" and mode != "bounded"):
            raise ValueError("Saved COLMAP initialization has an invalid scene_mode")
        if scene_mode is not None and scene_mode != mode:
            raise ValueError("Saved COLMAP initialization scene_mode disagrees with the configuration")
    else:
        mode = scene_mode or "unbounded"
    for name, value in (("start_core_resolution", start_core_resolution),
                        ("target_core_resolution", target_core_resolution)):
        if type(value) is not int or value < 1 or value > 1024 or value & (value - 1):
            raise ValueError(f"{name} must be a power of two in [1, 1024]")
    start, target = start_core_resolution, target_core_resolution
    if target < start:
        raise ValueError("Target core resolution must be >= starting core resolution")
    levels = background_shell_levels
    if levels is None:
        levels = saved.get("background_shell_levels", 5) if saved else 5
    initial_resolution = background_initial_resolution
    if initial_resolution is None:
        initial_resolution = saved.get("background_initial_resolution", 2) if saved else 2
    if mode == "unbounded":
        initial_subdivision = background_initial_subdivision(initial_resolution, start)
        refinement, subdivision = world_shell_metadata_layout(saved) if saved else (background_refinement or "uniform", initial_subdivision)
        if saved and background_refinement is not None and background_refinement != refinement:
            raise ValueError("Saved background refinement policy differs from the configuration")
        if refinement == "adaptive" and initial_subdivision != 0:
            raise ValueError("Adaptive initialization starts with the coarsest complete shells")
        if subdivision != initial_subdivision or (saved and saved["version"] == 7 and refinement != "fixed"):
            raise ValueError("Saved COLMAP initialization differs from the initial background resolution")
        validate_world_shell_layout(start, levels, background_subdivision_level=initial_subdivision)
        final_subdivision = initial_subdivision + (target.bit_length() - start.bit_length() if refinement == "uniform" else 0)
        validate_world_shell_layout(target, levels, background_subdivision_level=final_subdivision)
    else:
        levels = 0
    if background_opacity is None:
        background_opacity = saved.get("background_opacity", .5) if saved else .5
    if (isinstance(background_opacity, bool) or not isinstance(background_opacity, (int, float))
            or not math.isfinite(background_opacity) or not 0 < background_opacity < 1):
        raise ValueError("background_opacity must lie strictly between zero and one")

    model = find_colmap_model(dataset)
    cameras = read_colmap_cameras(model / "cameras.bin")
    images = read_colmap_images(model / "images.bin")
    poses = []
    for image in images:
        if image["camera_id"] not in cameras:
            raise ValueError(f"Image {image['name']} refers to missing camera ID {image['camera_id']}")
        rotation = quaternion_rotation(image["quaternion"])
        center = -rotation.T @ np.asarray(image["translation"])
        poses.append((center, rotation.T))
    centers = np.stack([pose[0] for pose in poses])

    preferred_point_path = _inside(dataset, point_cloud, "Point cloud filename") if point_cloud else None
    point_path = preferred_point_path if preferred_point_path and preferred_point_path.is_file() else model / "points3D.bin"
    source_positions, source_colors = None, None
    if load_points or (saved is None and mode == "bounded"):
        if point_path.suffix.lower() == ".ply":
            source_positions, source_colors = read_ply_points(point_path)
        else:
            source_positions, source_colors = read_colmap_points(point_path)

    if saved is None:
        center = centers.mean(axis=0)
        radius = float(np.max(np.linalg.norm(centers - center, axis=1)) * 1.1)
        if mode == "bounded":
            radius = max(radius, float(np.max(np.abs(source_positions - center))) * 1.1)
        if not math.isfinite(radius) or radius <= 1e-12:
            raise ValueError("Cannot estimate scene radius: registered camera centers are coincident")
        image_root = _image_directory(dataset, images_dir)
        initialization = {
            "kind": "colmap", "version": 8, "center": center.tolist(), "radius": radius,
            "start_core_resolution": start, "target_core_resolution": target,
            "image_directory": image_root.relative_to(dataset).as_posix(),
            "source_model": model.relative_to(dataset).as_posix(), "camera_count": len(images),
            "coordinate_system": "(x,y,z)->(x,z,-y)", "split": "sorted_filename_every_8th_test",
            "scene_mode": mode, "scene_mapping": "world_shell" if mode == "unbounded" else "bounded",
            "background_opacity": float(background_opacity),
        }
        if mode == "unbounded":
            initialization.update(background_shell_levels=levels, background_refinement=refinement,
                                  background_subdivision_level=initial_subdivision, world_shell_version=WORLD_SHELL_VERSION,
                                  background_geometry="ellipsoid",
                                  background_voxels=56 * levels, root_half_extent=2 ** levels)
            if foreground_refinement == "adaptive":
                initialization.update(foreground_refinement="adaptive",
                    foreground_max_resolution=foreground_max_resolution or target)
            if refinement == "adaptive":
                initialization.update(background_initial_resolution=2,
                    background_initialization="adaptive_full_shells",
                    background_initial_leaf_budget=background_initial_leaf_budget,
                    background_initial_pixel_threshold=background_initial_pixel_threshold,
                    background_max_resolution=background_max_resolution,
                    background_resolution_mode=background_resolution_mode)
            if initial_subdivision:
                initialization.update(background_initial_resolution=initial_resolution,
                                      background_initialization="point_cloud_sparse", background_voxels=0)
    else:
        initialization = saved
        expected = {"kind": "colmap", "start_core_resolution": start,
                    "target_core_resolution": target,
                    "source_model": model.relative_to(dataset).as_posix(), "camera_count": len(images),
                    "scene_mode": mode, "scene_mapping": "world_shell" if mode == "unbounded" else "bounded",
                    "background_opacity": float(background_opacity)}
        if foreground_refinement == "adaptive":
            expected.update(foreground_refinement="adaptive",
                            foreground_max_resolution=foreground_max_resolution or target)
        if mode == "unbounded":
            reject_removed_background_fields(initialization)
            expected.update(background_shell_levels=levels, root_half_extent=2 ** levels)
            if refinement == "adaptive":
                if initialization.get("background_resolution_mode", "per_shell") != background_resolution_mode:
                    raise ValueError("Saved background_resolution_mode differs from configuration")
                expected.update(background_initial_resolution=2,
                    background_initialization="adaptive_full_shells",
                    background_initial_leaf_budget=background_initial_leaf_budget,
                    background_initial_pixel_threshold=background_initial_pixel_threshold,
                    background_max_resolution=background_max_resolution)
            elif initial_subdivision:
                expected.update(background_initial_resolution=initial_resolution,
                                background_initialization="point_cloud_sparse")
            else:
                expected["background_voxels"] = 56 * levels
        for name, value in expected.items():
            if initialization.get(name) != value:
                raise ValueError(f"Saved COLMAP initialization {name} disagrees with the dataset/configuration")
        center = np.asarray(initialization.get("center"), dtype=np.float64)
        radius = float(initialization.get("radius", 0))
        if center.shape != (3,) or not np.isfinite(center).all() or not math.isfinite(radius) or radius <= 0:
            raise ValueError("Saved COLMAP initialization has an invalid center or radius")
        if images_dir is not None and PurePosixPath(str(images_dir).replace(chr(92), "/")) != PurePosixPath(initialization["image_directory"]):
            raise ValueError("Image directory disagrees with saved COLMAP initialization")
        image_root = _image_directory(dataset, initialization["image_directory"])

    views = []
    first_size = None
    for index, (image, (position, camera_to_world)) in enumerate(zip(images, poses)):
        source = cameras[image["camera_id"]]
        if source["model"] == "SIMPLE_PINHOLE":
            focal, cx, cy = source["parameters"]
            fx = fy = focal
        elif source["model"] == "PINHOLE":
            fx, fy, cx, cy = source["parameters"]
        else:
            raise ValueError(f"Camera {source['id']} uses {source['model']}; provide COLMAP undistorted PINHOLE images and cameras")
        if fx <= 0 or fy <= 0:
            raise ValueError(f"Camera {source['id']} has a nonpositive focal length")
        reference = _inside(image_root, image["name"], "COLMAP image filename")
        if not reference.is_file():
            raise FileNotFoundError(f"Registered COLMAP image has no matching reference: {reference}")
        with Image.open(reference) as reference_image:
            width, height = reference_image.size
        if first_size is None:
            first_size = (width, height)
        elif (width, height) != first_size:
            raise ValueError(f"COLMAP reference images must share one resolution; {reference} is {width}x{height}, expected {first_size[0]}x{first_size[1]}")
        camera = {
            "camera_file": str((model / "images.bin").resolve()), "frame_count": len(images),
            "camera_index": index, "image_id": image["id"], "colmap_camera_id": image["camera_id"],
            "image_name": image["name"], "position": (_AXES @ ((position - center) / radius)).tolist(),
            "right": (_AXES @ camera_to_world[:, 0]).tolist(),
            "up": (-_AXES @ camera_to_world[:, 1]).tolist(),
            "forward": (_AXES @ camera_to_world[:, 2]).tolist(),
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "width": source["width"], "height": source["height"],
            "intrinsic_width": source["width"], "intrinsic_height": source["height"],
            "nerf_to_falcor": "(x, y, z) -> (x, z, -y)", "source": "colmap",
        }
        camera = rescale_camera(camera, width, height)
        is_test = index % 8 == 0
        if split == "all" or (split == "test") == is_test:
            views.append({"index": index, "camera": camera, "reference_path": str(reference)})
    if not views:
        raise ValueError(f"COLMAP {split} split contains no images")

    positions, colors = None, None
    background_colors, background_point_counts = None, None
    background_points = None
    adaptive_background = None
    if load_points:
        normalized = ((source_positions - center) / radius) @ _AXES.T
        if not np.isfinite(normalized).all():
            raise ValueError("Point normalization produced nonfinite coordinates")
        if mode == "unbounded":
            from reconstruction_background import background_point_colors, prepare_sparse_background
            if refinement == "adaptive":
                from reconstruction_adaptive_background import prepare_adaptive_background
                adaptive_background = prepare_adaptive_background(
                    normalized, source_colors, start, [-1.] * 3, [1.] * 3, levels, views, np,
                    initial_leaf_budget=background_initial_leaf_budget,
                    pixel_threshold=background_initial_pixel_threshold,
                    max_depth=background_max_resolution.bit_length() - 2,
                    resolution_mode=background_resolution_mode, max_resolution=background_max_resolution,
                    **(dict(foreground_refinement="adaptive", foreground_max_resolution=foreground_max_resolution or target)
                       if foreground_refinement == "adaptive" else {}))
                background_point_counts = adaptive_background["point_counts"]
            elif initial_subdivision:
                background_points = prepare_sparse_background(normalized, source_colors, start,
                    [-1.] * 3, [1.] * 3, levels, np, background_subdivision_level=initial_subdivision)
                background_point_counts = background_points["point_counts"]
            else:
                background_colors, background_point_counts = background_point_colors(
                    normalized, source_colors, [-1.] * 3, [1.] * 3, levels, np)
            # Keep core occupancy and point/color correspondence unchanged.
            inside = np.all((normalized >= -1.) & (normalized < 1.), axis=1)
            if saved is None:
                initialization["point_filter"] = {
                    "source_count": len(normalized), "core_count": int(inside.sum()),
                    "excluded_count": int((~inside).sum()), "rule": "core_AABB_only",
                }
                initialization["background_color_initialization"] = {
                    "rule": "per_leaf_mean_linear_rgb", "source_color_space": "srgb",
                    "empty_leaf_color": [0., 0., 0.],
                    "colored_points": int(background_point_counts.sum()),
                    "colored_voxels": int(np.count_nonzero(background_point_counts)),
                    "background_voxels": len(background_point_counts),
                }
                if adaptive_background is not None:
                    initialization["background_voxels"] = len(adaptive_background["cell_indices"])
                    initialization["background_color_initialization"]["empty_leaf_rule"] = "allocated"
                    initialization["background_color_initialization"]["empty_leaf_color"] = [.5, .5, .5]
                elif initial_subdivision:
                    initialization["background_voxels"] = len(background_points["cell_indices"])
                    initialization["background_color_initialization"]["empty_leaf_color"] = None
                    initialization["background_color_initialization"]["empty_leaf_rule"] = "unallocated"
            normalized, source_colors = normalized[inside], source_colors[inside]
        positions = np.ascontiguousarray(normalized, dtype=np.float32)
        colors = np.ascontiguousarray(source_colors, dtype=np.float32)
    return {"views": views, "positions": positions, "colors": colors,
            "background_colors": background_colors,
            "background_point_counts": background_point_counts if background_points is None and adaptive_background is None else None,
            "background_points": background_points,
            "adaptive_background": adaptive_background,
            "initialization": initialization, "width": first_size[0], "height": first_size[1],
            "resolution": start, "core_resolution": start,
            "grid_min": [-1.] * 3, "grid_max": [1.] * 3,
            "point_source": str(point_path),
            "source_files": [str(model / "cameras.bin"), str(model / "images.bin"), str(point_path)]}
