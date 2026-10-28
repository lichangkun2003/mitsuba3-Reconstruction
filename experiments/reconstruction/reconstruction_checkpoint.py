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
    resolution = _integer(metadata["resolution"], "resolution", 1, 1040)
    core = _integer(metadata.get("core_resolution", resolution), "core_resolution", 1, min(resolution, 1024))
    mapping = metadata.get("scene_mapping", "bounded")
    if mapping not in ("bounded", "radial_seed_shell", "projective_shell"):
        raise ValueError("Unknown initialization scene_mapping")
    if mapping in ("radial_seed_shell", "projective_shell"):
        if core >= resolution or (resolution-core) % 2:
            raise ValueError("Shell metadata requires positive symmetric padding")
    policy = metadata.get("background_refinement")
    if policy is not None or "background_layers" in metadata:
        if not ((mapping == "radial_seed_shell" and policy == "fixed_layers") or
                (mapping == "projective_shell" and policy == "preserve_physical_width")):
            raise ValueError("Background refinement metadata disagrees with the scene mapping")
        layers = _integer(metadata.get("background_layers"), "background_layers", 1)
        if resolution != core + 2 * layers:
            raise ValueError("Background layer metadata must match grid padding")
    if mapping == "projective_shell" and policy != "preserve_physical_width":
        raise ValueError("Projective mapping requires preserve_physical_width refinement metadata")
    sh = _integer(metadata["sh_count"], "sh_count", 1, 16)
    if "active_sh_count" in metadata:
        _integer(metadata["active_sh_count"], "active_sh_count", 1, sh)
    opacity_sh = _integer(metadata["opacity_sh_count"], "opacity_sh_count", 1, 16)
    lo, hi = np.asarray(metadata["grid_min"]), np.asarray(metadata["grid_max"])
    if (lo.shape != (3,) or hi.shape != (3,) or lo.dtype.kind not in "iuf"
            or hi.dtype.kind not in "iuf" or not np.isfinite(lo).all()
            or not np.isfinite(hi).all() or not (lo < hi).all()):
        raise ValueError("grid_min/grid_max must contain three finite, increasing bounds")
    if mapping in ("radial_seed_shell", "projective_shell"):
        extent = resolution / metadata["core_resolution"]
        if (not np.allclose(lo, [-extent]*3, rtol=0, atol=1e-7)
                or not np.allclose(hi, [extent]*3, rtol=0, atol=1e-7)):
            raise ValueError("Shell grid bounds must preserve the normalized core")
    return resolution, core, mapping, sh, opacity_sh


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
    if np.any(cells < 0) or np.any(cells >= resolution ** 3):
        raise ValueError("cell_indices contains an index outside the grid")
    if np.unique(cells).size != cells.size:
        raise ValueError("cell_indices must be unique")
    if mapping == "projective_shell":
        padding = (resolution - core) // 2
        outside = np.zeros(cells.size, dtype=np.uint8)
        for stride in (1, resolution, resolution * resolution):
            coordinate = cells // stride % resolution
            outside += ((coordinate < padding) | (coordinate >= padding + core))
        if np.any(outside > 1):
            raise ValueError("Projective voxels must belong to the core or one face slab, not atlas corners")
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
    return state, (resolution, core, mapping)


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

    The largest supported grid needs 134 MiB of mapped address bits. Sorting
    and all index temporaries cover at most one export chunk, never N rows.
    """

    def __init__(self, layout, directory):
        self.resolution, self.core, self.mapping = layout
        self.directory = directory
        self.stream = self.seen = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.seen is not None:
            self.seen._mmap.close()
            self.seen = None
        if self.stream is not None:
            self.stream.close()

    def check(self, chunk):
        import numpy as np

        extent = self.resolution ** 3
        if np.any(chunk < 0) or np.any(chunk >= extent):
            raise ValueError("cell_indices contains an index outside the grid")
        ordered = np.sort(np.asarray(chunk, dtype=np.uint32))
        if np.any(ordered[1:] == ordered[:-1]):
            raise ValueError("cell_indices must be unique")
        if self.mapping == "projective_shell":
            padding = (self.resolution - self.core) // 2
            outside = np.zeros(ordered.size, dtype=np.uint8)
            for stride in (1, self.resolution, self.resolution * self.resolution):
                coordinate = ordered // stride % self.resolution
                outside += (coordinate < padding) | (coordinate >= padding + self.core)
            if np.any(outside > 1):
                raise ValueError("Projective voxels must belong to the core or one face slab, not atlas corners")
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
    if expected_metadata is not None:
        for key, expected in expected_metadata.items():
            actual = payload["model_metadata"].get(key)
            if not np.array_equal(np.asarray(actual), np.asarray(expected)):
                raise ValueError(f"Checkpoint model metadata differs for {key}: {actual!r} != {expected!r}")
    return payload
