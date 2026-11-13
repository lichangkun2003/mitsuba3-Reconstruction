"""Portable, atomic reconstruction checkpoints with JSON metadata and no pickle.

NumPy is imported lazily: importing this module does not initialize Mitsuba,
Dr.Jit, or CUDA before the caller has selected its runtime environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import zipfile


FORMAT = "mitsuba-voxel-reconstruction"
FORMAT_VERSION = 1
MODEL_FIELDS = ("cell_indices", "centers", "log_scales", "rotations",
                "radiance_sh", "opacity_sh")


def _array_digest(array):
    # memoryview.cast rejects a zero-length dimension in multi-dimensional arrays.
    return hashlib.sha256(memoryview(array).cast("B") if array.size else b"").hexdigest()


def _integer(value, name, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}")
    return value


def _validate_model_metadata(metadata):
    import numpy as np

    if not isinstance(metadata, dict):
        raise ValueError("model_metadata must be a dictionary")
    required = {"resolution", "grid_min", "grid_max", "sh_count", "opacity_sh_count"}
    if not required.issubset(metadata):
        raise ValueError(f"Missing model metadata: {sorted(required - metadata.keys())}")
    from reconstruction_world_shell import (reject_removed_background_fields, validate_world_shell_layout,
                                            world_shell_metadata_layout, adaptive_max_depth)
    resolution = _integer(metadata["resolution"], "resolution", 1, 1024)
    core = _integer(metadata.get("core_resolution", resolution), "core_resolution", 1, 1024)
    mapping = metadata.get("scene_mapping", "bounded")
    if mapping not in ("bounded", "world_shell"):
        raise ValueError("Legacy compressed scene mappings are unsupported; start a new world_shell reconstruction")
    reject_removed_background_fields(metadata)
    if core != resolution:
        raise ValueError("The stored resolution must equal the core resolution; compressed grid padding is unsupported")
    if mapping == "world_shell":
        mode, subdivision = world_shell_metadata_layout(metadata)
        validate_world_shell_layout(core, metadata.get("background_shell_levels"),
                                    metadata["grid_min"], metadata["grid_max"], subdivision)
        if mode == "adaptive":
            maximum = adaptive_max_depth(metadata)
            declared = _integer(metadata.get("background_max_depth", maximum), "background_max_depth", 0, 18)
            if declared != maximum:
                raise ValueError("background_max_depth differs from the per-shell resolution policy")
            if (metadata.get("background_resolution_mode") == "core_parent"
                    and metadata.get("world_shell_version", 1) < 6):
                raise ValueError("core_parent background requires world_shell_version 6")
            if (metadata.get("background_resolution_mode") == "core"
                    and metadata.get("world_shell_version", 1) < 7):
                raise ValueError("core background requires world_shell_version 7")
    elif any(name in metadata for name in ("background_shell_levels", "background_refinement",
                                           "background_subdivision_level", "world_shell_version", "background_geometry")):
        raise ValueError("Bounded models cannot contain background shell metadata")
    from reconstruction_sh_schedule import radiance_model_layout
    representation, _, sh = radiance_model_layout(metadata)
    if representation == "octahedral_scale":
        if "radiance_resolution" not in metadata:
            raise ValueError("Octahedral model metadata must declare radiance_resolution")
        _integer(metadata.get("octahedral_radiance_version", 1), "octahedral_radiance_version", 1, 1)
    if "active_sh_count" in metadata:
        active = _integer(metadata["active_sh_count"], "active_sh_count", 1, sh)
        if representation == "octahedral_scale" and active not in (1, sh):
            raise ValueError("Octahedral active_sh_count must be 1 or the full radiance storage stride")
    opacity_sh = _integer(metadata["opacity_sh_count"], "opacity_sh_count", 1, 16)
    if "active_opacity_sh_count" in metadata:
        _integer(metadata["active_opacity_sh_count"], "active_opacity_sh_count", 1, opacity_sh)
    lo, hi = np.asarray(metadata["grid_min"]), np.asarray(metadata["grid_max"])
    if (lo.shape != (3,) or hi.shape != (3,) or lo.dtype.kind not in "iuf"
            or hi.dtype.kind not in "iuf" or not np.isfinite(lo).all()
            or not np.isfinite(hi).all() or not (lo < hi).all()):
        raise ValueError("grid_min/grid_max must contain three finite, increasing bounds")
    return resolution, core, mapping, sh, opacity_sh


def _validate_mapping_config(metadata, config):
    """Reject a checkpoint/configuration mismatch before restoring device arrays."""
    if not isinstance(config, dict):
        return
    from reconstruction_world_shell import (reject_removed_background_fields, world_shell_metadata_layout,
                                            world_shell_metadata_geometry, background_initial_subdivision)
    data = config.get("data", {})
    model = config.get("model", {})
    if model:
        actual_radiance = metadata.get("radiance_representation", "sh")
        if model.get("radiance_representation", "sh") != actual_radiance:
            raise ValueError("Checkpoint radiance_representation differs between model and configuration")
        if actual_radiance == "octahedral_scale":
            if model.get("radiance_resolution", 5) != metadata["radiance_resolution"]:
                raise ValueError("Checkpoint radiance_resolution differs between model and configuration")
            if "sh_count" in model and model["sh_count"] != metadata["sh_count"]:
                raise ValueError("Checkpoint octahedral storage stride differs from configuration")
    reject_removed_background_fields(data)
    reject_removed_background_fields(model)
    initialization = data.get("initialization") or {}
    actual = metadata.get("scene_mapping", "bounded")
    if initialization:
        expected = initialization.get("scene_mapping", "bounded")
        if expected not in ("bounded", "world_shell") or expected != actual:
            raise ValueError("Checkpoint scene_mapping differs between model and initialization")
    if actual != "world_shell":
        return
    schedule = config.get("schedule", {})
    if schedule and metadata.get("foreground_refinement") == "adaptive":
        if (schedule.get("refinement", "uniform") != "adaptive"
                or schedule.get("target_resolution") != metadata.get("foreground_max_resolution")):
            raise ValueError("Adaptive foreground policy differs from saved training configuration")
    levels = metadata["background_shell_levels"]
    if "background_shell_levels" in data and data["background_shell_levels"] != levels:
        raise ValueError("Checkpoint background_shell_levels differs between model and configuration")
    if metadata.get("background_refinement") == "adaptive":
        policy = metadata.get("background_resolution_mode", "per_shell")
        if "background_resolution_mode" in data and data["background_resolution_mode"] != policy:
            raise ValueError("Checkpoint background_resolution_mode differs from configuration")
        if "background_max_resolution" in data:
            resolution = _integer(data["background_max_resolution"], "background_max_resolution", 2, 1024)
            if (resolution & (resolution - 1)
                    or (policy in ("core_parent", "core") and resolution != metadata.get(
                        "background_max_resolution", 1024 if policy == "core" else 512))
                    or (policy == "per_shell" and resolution.bit_length() - 2 != metadata.get("background_max_depth", 9))):
                raise ValueError("Checkpoint background_max_depth differs from data.background_max_resolution")
        if initialization and initialization.get("background_resolution_mode", "per_shell") != policy:
            raise ValueError("Checkpoint background_resolution_mode differs from initialization")
    if initialization:
        mode, _ = world_shell_metadata_layout(metadata)
        initial_mode, initial_subdivision = world_shell_metadata_layout(initialization)
        initial_resolution = initialization.get("background_initial_resolution", 2)
        expected_initial = (0 if mode == "adaptive" else background_initial_subdivision(initial_resolution,
            initialization.get("start_core_resolution", metadata.get("core_resolution", metadata["resolution"]))))
        if (initialization.get("version") not in (7, 8, 9)
                or initialization.get("background_shell_levels") != levels
                or initial_mode != mode or initial_subdivision != expected_initial
                or (mode != "adaptive" and data.get("background_initial_resolution", 2) != initial_resolution)
                or ("background_refinement" in data and data["background_refinement"] != mode)
                or world_shell_metadata_geometry(initialization) != world_shell_metadata_geometry(metadata)
                or (initialization.get("version") == 7 and initial_mode != "fixed")):
            raise ValueError("Checkpoint world shell layout differs between model and initialization")


def validate_model_state(state: dict, metadata: dict) -> None:
    """Validate the voxel-major AoS layout consumed by the native constructor."""
    import numpy as np

    resolution, core, mapping, sh, opacity_sh = _validate_model_metadata(metadata)
    if not isinstance(state, dict) or set(state) != set(MODEL_FIELDS):
        raise ValueError(f"model_state must contain exactly {MODEL_FIELDS}")
    arrays = {key: np.asarray(value) for key, value in state.items()}
    cells = arrays["cell_indices"]
    if cells.ndim != 1 or cells.dtype.kind not in "ui":
        raise ValueError("cell_indices must be a flat integer array")
    extent = resolution ** 3 + (56 * metadata["background_shell_levels"]
        * 8 ** metadata.get("background_subdivision_level", 0) if mapping == "world_shell" else 0)
    if metadata.get("background_refinement") == "adaptive":
        from reconstruction_world_shell import (ADAPTIVE_TAG, adaptive_decode_numpy,
                                                validate_adaptive_cells_numpy, adaptive_depth_limit)
        validate_adaptive_cells_numpy(cells, core, metadata["background_shell_levels"], np)
        coarse, depth, _ = adaptive_decode_numpy(cells[cells >= ADAPTIVE_TAG], metadata["background_shell_levels"], np)
        if np.any(depth > adaptive_depth_limit(metadata, coarse // 56)):
            raise ValueError("Adaptive leaf depth exceeds background_max_depth")
        _validate_core_depths(cells[cells < ADAPTIVE_TAG], metadata, np)
    else:
        if np.any(cells < 0) or np.any(cells >= extent):
            raise ValueError("cell_indices contains an index outside the foreground/shell address space")
        if np.unique(cells).size != cells.size:
            raise ValueError("cell_indices must be unique")
    count = cells.size
    widths = {"centers": 3, "log_scales": 3, "rotations": 4,
              "radiance_sh": 3 * sh, "opacity_sh": opacity_sh}
    for key, width in widths.items():
        array = arrays[key]
        if array.shape != (count * width,) or array.dtype.kind != "f":
            raise ValueError(f"{key} must be a flat floating array of length {count * width}")
        if not np.isfinite(array).all():
            raise ValueError(f"{key} contains a nonfinite value")
    if np.any(np.linalg.norm(arrays["rotations"].reshape(-1, 4), axis=1) == 0):
        raise ValueError("rotations contains a zero quaternion")


def _validate_core_depths(cells, metadata, np):
    from reconstruction_world_shell import CORE_ADAPTIVE_TAG, core_leaf_depths_numpy
    core = metadata.get("core_resolution", metadata["resolution"])
    tagged = (cells & np.uint64(CORE_ADAPTIVE_TAG)) != 0
    if np.any(tagged) and metadata.get("foreground_refinement") != "adaptive":
        raise ValueError("Tagged foreground requires adaptive foreground metadata")
    depth, morton = core_leaf_depths_numpy(cells, core, np)
    maximum = metadata.get("foreground_max_resolution", core).bit_length() - 1
    if np.any(depth > maximum):
        raise ValueError("Adaptive foreground leaf exceeds foreground_max_resolution")
    return depth, morton


def _is_array_source(value):
    from reconstruction_export import ArraySource
    return isinstance(value, ArraySource)


def _array_description(value):
    import numpy as np

    dtype = np.dtype(value.dtype)
    shape = tuple(value.shape)
    if (dtype.kind not in "biuf" or dtype.hasobject or
            any(type(item) is not int or item < 0 for item in shape)):
        raise ValueError(f"Unsupported checkpoint array dtype or shape: {dtype}, {shape}")
    if math.prod(shape) != value.size:
        raise ValueError("Checkpoint array shape and size disagree")
    return dtype, shape


def _validate_save_layout(model_state, metadata):
    """Check layout without reading a device source or copying a whole table."""
    import numpy as np

    resolution, core, mapping, sh, opacity_sh = _validate_model_metadata(metadata)
    if not isinstance(model_state, dict) or set(model_state) != set(MODEL_FIELDS):
        raise ValueError(f"model_state must contain exactly {MODEL_FIELDS}")
    state = {key: value if _is_array_source(value) else np.asarray(value)
             for key, value in model_state.items()}
    cells = state["cell_indices"]
    dtype, shape = _array_description(cells)
    if len(shape) != 1 or dtype.kind not in "ui":
        raise ValueError("cell_indices must be a flat integer array")
    count = cells.size
    for name, width in (("centers", 3), ("log_scales", 3), ("rotations", 4),
                        ("radiance_sh", 3 * sh), ("opacity_sh", opacity_sh)):
        dtype, shape = _array_description(state[name])
        if shape != (count * width,) or dtype.kind != "f":
            raise ValueError(f"{name} must be a flat floating array of length {count * width}")
    extent = resolution ** 3 + (56 * metadata["background_shell_levels"]
        * 8 ** metadata.get("background_subdivision_level", 0) if mapping == "world_shell" else 0)
    if metadata.get("background_refinement") == "adaptive":
        return state, (resolution, core, mapping, 2**64, metadata["background_shell_levels"],
                       metadata.get("background_max_depth", 9), metadata)
    return state, (resolution, core, mapping, extent)


def _array_chunks(value, *, alignment=1):
    """Yield C-order flat chunks; even strided NumPy inputs stay bounded."""
    import numpy as np
    from reconstruction_export import CHUNK_BYTES

    dtype, _ = _array_description(value)
    if not value.size:
        return
    limit = CHUNK_BYTES // max(dtype.itemsize, 4)
    source = _is_array_source(value)
    if source:
        limit = min(limit, getattr(value, "max_chunk_elements", limit))
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("Checkpoint chunk capacity must be positive")
    if limit >= alignment:
        limit -= limit % alignment
    for start in range(0, value.size, limit):
        count = min(limit, value.size - start)
        if source:
            chunk = value.read_chunk(start, count)
            if not isinstance(chunk, np.ndarray) or chunk.dtype != dtype or chunk.shape != (count,):
                raise ValueError("Checkpoint array source returned an invalid chunk dtype or shape")
        elif value.flags.c_contiguous:
            chunk = value.reshape(-1)[start:start + count]
        else:
            chunk = value.flat[start:start + count]
        yield np.ascontiguousarray(chunk)


class _CellValidation:
    """Exact uniqueness with a disk-backed bitset, including unordered cells.

    Address bits use a disk-backed mapping, including subdivided shells. Sorting
    and all index temporaries cover at most one export chunk, never N rows.
    """

    def __init__(self, layout, directory):
        self.resolution, self.core, self.mapping, self.extent = layout[:4]
        self.adaptive_levels = layout[4] if len(layout) > 4 else None
        self.adaptive_max_depth = layout[5] if len(layout) > 5 else 9
        self.adaptive_metadata = layout[6] if len(layout) > 6 else {"background_max_depth": self.adaptive_max_depth}
        self.directory = directory
        self.stream = self.seen = None
        self.runs = []
        self.interval_runs = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.seen is not None:
            self.seen._mmap.close()
            self.seen = None
        if self.stream is not None:
            self.stream.close()
        for stream, _, _, _ in self.runs:
            stream.close()
        self.runs.clear()
        for stream, _, _, _ in self.interval_runs:
            stream.close()
        self.interval_runs.clear()

    def check_adaptive(self, chunk, np):
        from reconstruction_world_shell import (ADAPTIVE_TAG, adaptive_decode_numpy,
                                                adaptive_intervals_numpy, adaptive_depth_limit)
        tagged = chunk >= ADAPTIVE_TAG
        if np.any(chunk < 0):
            raise ValueError("Adaptive cell addresses must be nonnegative")
        metadata = dict(self.adaptive_metadata, resolution=self.resolution, core_resolution=self.core)
        core_depth, core_morton = _validate_core_depths(chunk[~tagged], metadata, np)
        coarse, depth, _ = adaptive_decode_numpy(chunk[tagged], self.adaptive_levels, np)
        if np.any(depth > adaptive_depth_limit(self.adaptive_metadata, coarse // 56)):
            raise ValueError("Adaptive leaf depth exceeds background_max_depth")
        if np.any(depth > 9 + coarse // 56):
            raise ValueError("Adaptive leaf is smaller than the world-tree base cell")
        start, end = adaptive_intervals_numpy(chunk[tagged], self.adaptive_levels, np)
        # A reserved interval root keeps foreground overlap checks independent
        # of the 560 possible background roots, including across export chunks.
        shift = (3 * (18 - core_depth)).astype(np.uint64)
        core_start = np.uint64(1022 << 54) | np.left_shift(core_morton, shift)
        core_end = core_start + np.left_shift(np.uint64(1), shift)
        start, end = np.r_[start, core_start], np.r_[end, core_end]
        if not len(start):
            return
        order = np.argsort(start)
        intervals = np.column_stack((start[order], end[order]))
        if np.any(intervals[1:, 0] < intervals[:-1, 1]):
            raise ValueError("Adaptive cell_indices must be unique and have no ancestor/descendant overlap")
        lower, upper = int(intervals[0, 0]), int(intervals[-1, 1])
        for stream, count, first, last in self.interval_runs:
            if upper <= first or lower >= last:
                continue
            previous = np.memmap(stream, mode="r", dtype=np.uint64, shape=(count, 2))
            try:
                index = np.searchsorted(previous[:, 0], intervals[:, 0])
                before, after = index > 0, index < count
                if (np.any(previous[index[before] - 1, 1] > intervals[before, 0])
                        or np.any(previous[index[after], 0] < intervals[after, 1])):
                    raise ValueError("Adaptive cell_indices must be unique and have no ancestor/descendant overlap")
            finally:
                previous._mmap.close()
        stream = tempfile.TemporaryFile(mode="w+b", dir=self.directory)
        stream.write(intervals.tobytes())
        stream.flush()
        self.interval_runs.append((stream, len(intervals), lower, upper))

    def check(self, chunk):
        import numpy as np

        extent = self.extent
        if self.adaptive_levels is not None:
            self.check_adaptive(chunk, np)
        elif np.any(chunk < 0) or np.any(chunk >= extent):
            raise ValueError("cell_indices contains an index outside the grid")
        ordered = np.sort(np.asarray(chunk, dtype=np.uint64))
        if np.any(ordered[1:] == ordered[:-1]):
            raise ValueError("cell_indices must be unique")
        if extent > 2 ** 32:
            # Sparse high-address layouts must not allocate a bitset over the
            # logical world. Keep sorted, bounded runs on disk and check only
            # overlapping ranges. This remains exact for unordered exports.
            if not len(ordered):
                return
            lower, upper = int(ordered[0]), int(ordered[-1])
            for stream, count, first_cell, last_cell in self.runs:
                if upper < first_cell or lower > last_cell:
                    continue
                previous = np.memmap(stream, mode="r", dtype=np.uint64, shape=(count,))
                try:
                    indices = np.searchsorted(previous, ordered)
                    inside = indices < count
                    if np.any(previous[indices[inside]] == ordered[inside]):
                        raise ValueError("cell_indices must be unique")
                finally:
                    previous._mmap.close()
            stream = tempfile.TemporaryFile(mode="w+b", dir=self.directory)
            stream.write(ordered.tobytes())
            stream.flush()
            self.runs.append((stream, len(ordered), lower, upper))
            return
        if self.seen is None:
            self.stream = tempfile.TemporaryFile(mode="w+b", dir=self.directory)
            size = (extent + 7) // 8
            self.stream.truncate(size)
            self.seen = np.memmap(self.stream, mode="r+", dtype=np.uint8, shape=(size,))
        byte = ordered >> 3
        first = np.r_[0, np.flatnonzero(byte[1:] != byte[:-1]) + 1]
        bits = np.left_shift(np.uint8(1), (ordered & 7).astype(np.uint8))
        masks = np.bitwise_or.reduceat(bits, first)
        indices = byte[first]
        previous = self.seen[indices]
        if np.any(previous & masks):
            raise ValueError("cell_indices must be unique")
        self.seen[indices] = previous | masks


def _encode(value, arrays, manifest):
    import numpy as np

    if isinstance(value, np.ndarray) or _is_array_source(value):
        dtype, shape = _array_description(value)
        key = f"a{len(arrays):06d}"
        arrays[key] = value
        manifest[key] = {"dtype": dtype.str, "shape": list(shape)}
        return {"type": "array", "key": key}
    if isinstance(value, np.generic):
        return _encode(value.item(), arrays, manifest)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Checkpoint scalar values must be finite")
        return value
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Checkpoint dictionary keys must be strings")
        return {"type": "dict", "items": {key: _encode(item, arrays, manifest)
                                             for key, item in value.items()}}
    if isinstance(value, (list, tuple)):
        return {"type": "tuple" if isinstance(value, tuple) else "list",
                "items": [_encode(item, arrays, manifest) for item in value]}
    raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}")


def _decode(value, arrays):
    if not isinstance(value, dict):
        return value
    kind = value.get("type")
    if kind == "array" and set(value) == {"type", "key"}:
        return arrays[value["key"]].copy()
    if kind == "dict" and set(value) == {"type", "items"} and isinstance(value["items"], dict):
        return {key: _decode(item, arrays) for key, item in value["items"].items()}
    if kind in ("list", "tuple") and set(value) == {"type", "items"} and isinstance(value["items"], list):
        items = [_decode(item, arrays) for item in value["items"]]
        return tuple(items) if kind == "tuple" else items
    raise ValueError("Malformed checkpoint state encoding")


def save_checkpoint(path, model_state, model_metadata, *, training_state=None,
                    optimizer_state=None, config=None, provenance=None) -> Path:
    """Save a complete snapshot; replace the destination only after a full write.

    Arrays may be NumPy arrays or bounded ArraySource exports. The caller must
    keep all sources immutable until the write returns; GPU export factories
    manage their own synchronization. Each array is read only once, validated,
    hashed, and compressed in chunks before writing the final JSON manifest.
    Model arrays are flat AoS; quaternions use wxyz; SH uses coefficient then RGB.
    RNG states and optimizer moments may contain nested numeric array sources.
    """
    import numpy as np

    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Checkpoint filename must end with .npz")
    state, layout = _validate_save_layout(model_state, model_metadata)
    _validate_mapping_config(model_metadata, config)
    payload = {"model_state": state, "model_metadata": model_metadata,
               "training_state": training_state or {}, "optimizer_state": optimizer_state or {},
               "config": config or {}, "provenance": provenance or {}}
    arrays, manifest = {}, {}
    encoded = _encode(payload, arrays, manifest)
    header = {"format": FORMAT, "version": FORMAT_VERSION, "layout":
              "flat voxel-major AoS; rotations wxyz; radiance SH coefficient then RGB",
              "arrays": manifest, "payload": encoded}
    model_keys = {node["key"]: name for name, node in
                  encoded["items"]["model_state"]["items"].items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_DEFLATED,
                                 allowZip64=True) as archive:
                for key, value in arrays.items():
                    name = model_keys.get(key)
                    digest = hashlib.sha256()
                    rotation_tail = np.empty(0, dtype=value.dtype)
                    with _CellValidation(layout, path.parent) as cells:
                        with archive.open(key + ".npy", mode="w", force_zip64=True) as member:
                            np.lib.format.write_array_header_1_0(member, {
                                "descr": np.lib.format.dtype_to_descr(np.dtype(value.dtype)),
                                "fortran_order": False, "shape": tuple(value.shape)})
                            for chunk in _array_chunks(value, alignment=4 if name == "rotations" else 1):
                                if chunk.dtype.kind == "f" and not np.isfinite(chunk).all():
                                    raise ValueError(f"Checkpoint array {name or key} contains a nonfinite value")
                                if name == "cell_indices":
                                    cells.check(chunk)
                                elif name == "rotations":
                                    rotations = (np.concatenate((rotation_tail, chunk))
                                                 if rotation_tail.size else chunk)
                                    complete = rotations.size - rotations.size % 4
                                    if np.any(np.linalg.norm(rotations[:complete].reshape(-1, 4), axis=1) == 0):
                                        raise ValueError("rotations contains a zero quaternion")
                                    rotation_tail = rotations[complete:].copy()
                                raw = memoryview(chunk).cast("B")
                                digest.update(raw)
                                member.write(raw)
                    manifest[key]["sha256"] = digest.hexdigest()
                metadata = np.frombuffer(json.dumps(header, ensure_ascii=False, allow_nan=False,
                    separators=(",", ":")).encode("utf-8"), dtype=np.uint8)
                with archive.open("metadata_json.npy", mode="w", force_zip64=True) as member:
                    np.lib.format.write_array(member, metadata, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return path


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate checkpoint JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError(f"Nonfinite checkpoint JSON value: {value}")


def load_checkpoint(path, expected_metadata=None) -> dict:
    """Load and validate all arrays before returning; never unpickle any object."""
    import numpy as np

    with np.load(Path(path), allow_pickle=False) as archive:
        if "metadata_json" not in archive.files or len(set(archive.files)) != len(archive.files):
            raise ValueError("Not a reconstruction checkpoint, or duplicate archive entries")
        raw = archive["metadata_json"]
        if raw.dtype != np.uint8 or raw.ndim != 1:
            raise ValueError("Checkpoint metadata must be a UTF-8 byte array")
        header = json.loads(raw.tobytes().decode("utf-8"), object_pairs_hook=_unique_json_object,
                            parse_constant=_reject_json_constant)
        if header.get("format") != FORMAT or header.get("version") != FORMAT_VERSION:
            raise ValueError("Unsupported reconstruction checkpoint format/version")
        manifest = header.get("arrays")
        if not isinstance(manifest, dict) or set(archive.files) != set(manifest) | {"metadata_json"}:
            raise ValueError("Checkpoint archive does not match its array manifest")
        arrays = {}
        for key, record in manifest.items():
            array = archive[key]
            if array.dtype.kind not in "biuf" or array.dtype.hasobject:
                raise ValueError(f"Unsupported checkpoint dtype in {key}")
            if array.dtype.str != record["dtype"] or list(array.shape) != record["shape"]:
                raise ValueError(f"Checkpoint array shape/dtype mismatch: {key}")
            contiguous = np.asarray(array, order="C")
            if _array_digest(contiguous) != record["sha256"]:
                raise ValueError(f"Checkpoint array checksum mismatch: {key}")
            if array.dtype.kind == "f" and not np.isfinite(array).all():
                raise ValueError(f"Nonfinite checkpoint array: {key}")
            arrays[key] = array
        payload = _decode(header["payload"], arrays)
    if not isinstance(payload, dict) or set(payload) != {
            "model_state", "model_metadata", "training_state", "optimizer_state", "config", "provenance"}:
        raise ValueError("Checkpoint payload is incomplete")
    validate_model_state(payload["model_state"], payload["model_metadata"])
    _validate_mapping_config(payload["model_metadata"], payload["config"])
    if expected_metadata is not None:
        for key, expected in expected_metadata.items():
            actual = payload["model_metadata"].get(key)
            if not np.array_equal(np.asarray(actual), np.asarray(expected)):
                raise ValueError(f"Checkpoint model metadata differs for {key}: {actual!r} != {expected!r}")
    return payload
