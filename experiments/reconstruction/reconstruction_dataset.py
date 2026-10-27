"""Read COLMAP cameras and prepare explicit bounded/unbounded initializations.

Files and camera poses remain untouched. New unbounded scenes use the same
piecewise-projective contraction for point seeds and native ray queries.
Versions 1--4 retain their original point-only radial contraction and layout.
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


def contract_points(positions, center, radius, shell_width):
    """Similarity transform + one-time far-point contraction, in renderer axes."""
    points = (np.asarray(positions, dtype=np.float64) - np.asarray(center)) / radius
    lengths = np.linalg.norm(points, axis=1)
    far = lengths > 1.
    points[far] *= ((1. + shell_width * (1. - 1. / lengths[far])) / lengths[far])[:, None]
    points = (points @ _AXES.T).astype(np.float32)
    # A finite point must remain inside the grid, including float32 rounding at infinity.
    limit = np.nextafter(np.float32(1. + shell_width), np.float32(0.))
    np.clip(points, -limit, limit, out=points)
    if not np.isfinite(points).all():
        raise ValueError("Point normalization produced nonfinite coordinates")
    return np.ascontiguousarray(points)


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


def _grid_layout(start_core_resolution, target_core_resolution, shell_layers, *, max_grid_resolution=1024):
    for name, value in (("start_core_resolution", start_core_resolution),
                        ("target_core_resolution", target_core_resolution), ("shell_layers", shell_layers)):
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    start, target, layers = map(int, (start_core_resolution, target_core_resolution, shell_layers))
    if start & (start - 1) or target & (target - 1) or target < start:
        raise ValueError("Core resolutions must be powers of two and target must be >= start")
    if (start * layers) % target:
        raise ValueError("Final shell layers must give an integer number of layers at the starting core resolution")
    width = layers * 2. / target
    resolution = start + 2 * (start * layers // target)
    if target + 2 * layers > max_grid_resolution:
        raise ValueError(f"Core plus shell storage resolution exceeds the supported maximum of {max_grid_resolution}")
    return resolution, width


def prepare_colmap(dataset, split="train", images_dir=None, point_cloud="init_points.ply",
                   initialization=None, start_core_resolution=256, target_core_resolution=512,
                   shell_layers=16, load_points=True, *, scene_mode=None,
                   background_layers=None, background_opacity=None, background_final_layers=None):
    """Prepare views and point seeds without writing dataset files.

    New explicit ``unbounded`` mode uses ``background_final_layers`` (eight by
    default) at the target resolution, independent of the ray hit capacity.
    Version-5 metadata keeps the physical shell width constant, so a 256 to
    1024 schedule starts with two layers and refines through four to eight.
    Saved version-2 and version-3 initializations retain their original policies.
    ``bounded`` mode fits cameras
    and points into one cube by a similarity transform, without contraction.
    This function does not initialize shell ellipsoids or transform any rays.

    Omitted ``scene_mode`` retains the low-level version-1 initialization for
    existing callers. A supplied checkpoint selects its own saved mode; explicit
    contradictory arguments are rejected. Saved normalization is never refitted.
    """
    from PIL import Image

    dataset = Path(dataset).resolve()
    if split not in ("train", "test", "all"):
        raise ValueError("COLMAP split must be 'train', 'test', or 'all'")
    if scene_mode not in (None, "legacy", "bounded", "unbounded"):
        raise ValueError("COLMAP scene_mode must be bounded, unbounded, or legacy")
    saved = None if initialization is None else dict(initialization)
    version = 1 if saved is None else saved.get("version")
    if saved is not None and version not in (1, 2, 3, 4, 5):
        raise ValueError("Unsupported saved COLMAP initialization version")
    if saved is not None:
        mode = "legacy" if version == 1 else saved.get("scene_mode")
        if mode not in ("legacy", "bounded", "unbounded") or (version >= 2 and mode == "legacy"):
            raise ValueError("Saved COLMAP initialization has an invalid scene_mode")
        if version == 3 and (mode != "unbounded" or saved.get("background_refinement") != "fixed_layers"):
            raise ValueError("Version-3 initialization requires fixed_layers unbounded refinement")
        if version >= 4 and (mode != "unbounded" or saved.get("background_refinement") != "preserve_physical_width"):
            raise ValueError("Version-4/5 initialization requires preserve_physical_width unbounded refinement")
        if version < 4 and any(key in saved for key in ("background_final_layers", "initial_background_layers")):
            raise ValueError("Legacy initialization cannot silently change to final background layers")
        if version < 3 and "background_refinement" in saved:
            raise ValueError("Legacy initialization cannot silently change its background refinement policy")
        if scene_mode is not None and scene_mode != mode:
            raise ValueError("Saved COLMAP initialization scene_mode disagrees with the configuration")
    else:
        mode = scene_mode or "legacy"
        version = 1 if mode == "legacy" else (5 if mode == "unbounded" else 2)

    if version < 4 and background_final_layers is not None:
        raise ValueError("background_final_layers requires a new unbounded reconstruction or saved version-4 initialization")
    if version >= 4 and background_layers is not None:
        raise ValueError("Use background_final_layers for new unbounded reconstructions; background_layers is reserved for legacy saves")

    if mode == "legacy":
        if background_layers is not None or background_opacity is not None:
            raise ValueError("Version-1 COLMAP initialization has no background initialization settings")
        resolution, shell_width = _grid_layout(start_core_resolution, target_core_resolution, shell_layers)
        final_layers = int(shell_layers)
    else:
        if version >= 4:
            if "background_layers" in (saved or {}):
                raise ValueError("Version-4 initialization uses background_final_layers, not background_layers")
            if background_final_layers is None:
                background_final_layers = saved.get("background_final_layers") if saved else 8
            layer_name, layer_value = "background_final_layers", background_final_layers
        else:
            if background_layers is None:
                background_layers = saved.get("background_layers", 8) if saved else 8
            layer_name, layer_value = "background_layers", background_layers
        if background_opacity is None:
            background_opacity = saved.get("background_opacity", .5) if saved else .5
        for name, value in (("start_core_resolution", start_core_resolution),
                            ("target_core_resolution", target_core_resolution),
                            (layer_name, layer_value)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or int(value) != value or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        start, target, requested_layers = map(int, (start_core_resolution, target_core_resolution, layer_value))
        if start & (start - 1) or target & (target - 1) or target < start:
            raise ValueError("Core resolutions must be powers of two and target must be >= start")
        if target > 1024:
            raise ValueError("Core resolution exceeds the supported maximum of 1024")
        if isinstance(background_opacity, bool) or not isinstance(background_opacity, (int, float)) or not math.isfinite(background_opacity) or not 0 < background_opacity < 1:
            raise ValueError("background_opacity must lie strictly between zero and one")
        if mode == "unbounded":
            if version >= 4:
                final_layers = requested_layers
                resolution, shell_width = _grid_layout(start, target, final_layers, max_grid_resolution=1040)
                initial_layers = (resolution - start) // 2
            elif version == 3:
                initial_layers = requested_layers
                final_layers = initial_layers
                if target + 2 * final_layers > 1040:
                    raise ValueError("Core plus shell storage resolution exceeds the supported maximum of 1040")
                resolution, shell_width = start + 2 * initial_layers, 2. * initial_layers / start
            else:
                initial_layers = requested_layers
                final_layers = initial_layers * (target // start)
                resolution, shell_width = _grid_layout(start, target, final_layers)
        else:
            if target > 1024:
                raise ValueError("Core resolution exceeds the supported maximum of 1024")
            resolution, shell_width, final_layers = start, 0., 0

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
    # A fresh bounded fit needs the same point-inclusive scale even for a
    # cameras-only caller. Restoring saved cameras needs no point-cloud access.
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
            "kind": "colmap", "version": version, "center": center.tolist(), "radius": radius,
            "shell_width": shell_width, "shell_layers": final_layers,
            "start_core_resolution": int(start_core_resolution),
            "target_core_resolution": int(target_core_resolution),
            "image_directory": image_root.relative_to(dataset).as_posix(),
            "source_model": model.relative_to(dataset).as_posix(), "camera_count": len(images),
            "coordinate_system": "(x,y,z)->(x,z,-y)", "split": "sorted_filename_every_8th_test",
        }
        if version >= 2:
            initialization.update(scene_mode=mode,
                scene_mapping=("projective_shell" if version == 5 else "radial_seed_shell") if mode == "unbounded" else "bounded",
                background_opacity=float(background_opacity))
            if version >= 4:
                initialization.update(background_final_layers=final_layers,
                                      initial_background_layers=initial_layers,
                                      background_refinement="preserve_physical_width")
            else:
                initialization["background_layers"] = int(background_layers)
        if version == 3:
            initialization["background_refinement"] = "fixed_layers"
    else:
        initialization = saved
        expected = {"kind": "colmap", "version": version, "shell_layers": final_layers,
                    "start_core_resolution": int(start_core_resolution),
                    "target_core_resolution": int(target_core_resolution),
                    "source_model": model.relative_to(dataset).as_posix(), "camera_count": len(images)}
        if version >= 2:
            expected.update(scene_mode=mode,
                scene_mapping=("projective_shell" if version == 5 else "radial_seed_shell") if mode == "unbounded" else "bounded",
                background_opacity=float(background_opacity))
            if version >= 4:
                expected.update(background_final_layers=final_layers,
                                initial_background_layers=initial_layers,
                                background_refinement="preserve_physical_width")
            else:
                expected["background_layers"] = int(background_layers)
        if version == 3:
            expected["background_refinement"] = "fixed_layers"
        for name, value in expected.items():
            if initialization.get(name) != value:
                raise ValueError(f"Saved COLMAP initialization {name} disagrees with the dataset/configuration")
        if not math.isclose(float(initialization.get("shell_width", -1)), shell_width, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Saved COLMAP shell width disagrees with the core schedule")
        center = np.asarray(initialization.get("center"), dtype=np.float64)
        radius = float(initialization.get("radius", 0))
        if center.shape != (3,) or not np.isfinite(center).all() or not math.isfinite(radius) or radius <= 0:
            raise ValueError("Saved COLMAP initialization has an invalid center or radius")
        if images_dir is not None and PurePosixPath(str(images_dir).replace("\\", "/")) != PurePosixPath(initialization["image_directory"]):
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
    if load_points:
        if mode == "bounded":
            positions = np.ascontiguousarray(((source_positions - center) / radius) @ _AXES.T, dtype=np.float32)
            if not np.isfinite(positions).all():
                raise ValueError("Point normalization produced nonfinite coordinates")
        elif version == 5:
            from reconstruction_projective import projective_points
            normalized = ((source_positions - center) / radius) @ _AXES.T
            positions = projective_points(normalized, shell_width, np)
        else:
            positions = contract_points(source_positions, center, radius, shell_width)
        colors = np.ascontiguousarray(source_colors, dtype=np.float32)
    extent = 1. + shell_width
    return {"views": views, "positions": positions, "colors": colors,
            "initialization": initialization, "width": first_size[0], "height": first_size[1],
            "resolution": resolution, "core_resolution": int(start_core_resolution),
            "grid_min": [-extent] * 3, "grid_max": [extent] * 3,
            "point_source": str(point_path),
            "source_files": [str(model / "cameras.bin"), str(model / "images.bin"), str(point_path)]}
