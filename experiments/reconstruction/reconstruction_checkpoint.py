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


def validate_model_state(state: dict, metadata: dict) -> None:
    """Validate the voxel-major AoS layout consumed by the native constructor."""
    import numpy as np

    if not isinstance(metadata, dict):
        raise ValueError("model_metadata must be a dictionary")
    required = {"resolution", "grid_min", "grid_max", "sh_count", "opacity_sh_count"}
    if not required.issubset(metadata):
        raise ValueError(f"Missing model metadata: {sorted(required - metadata.keys())}")
    resolution = _integer(metadata["resolution"], "resolution", 1, 1024)
    sh = _integer(metadata["sh_count"], "sh_count", 1, 16)
    opacity_sh = _integer(metadata["opacity_sh_count"], "opacity_sh_count", 1, 16)
    lo, hi = np.asarray(metadata["grid_min"]), np.asarray(metadata["grid_max"])
    if (lo.shape != (3,) or hi.shape != (3,) or lo.dtype.kind not in "iuf"
            or hi.dtype.kind not in "iuf" or not np.isfinite(lo).all()
            or not np.isfinite(hi).all() or not (lo < hi).all()):
        raise ValueError("grid_min/grid_max must contain three finite, increasing bounds")
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


def _encode(value, arrays, manifest):
    import numpy as np

    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf" or value.dtype.hasobject:
            raise ValueError(f"Unsupported checkpoint array dtype: {value.dtype}")
        array = np.asarray(value, order="C")
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise ValueError("Checkpoint arrays must contain only finite values")
        key = f"a{len(arrays):06d}"
        arrays[key] = array
        manifest[key] = {"dtype": array.dtype.str, "shape": list(array.shape),
                         "sha256": _array_digest(array)}
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

    The caller must evaluate/synchronize GPU arrays and convert them to NumPy.
    Model arrays are flat AoS; quaternions use wxyz; SH uses coefficient then RGB.
    RNG states and optimizer moments may contain nested numeric NumPy arrays.
    """
    import numpy as np

    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("Checkpoint filename must end with .npz")
    state = {key: np.asarray(value) for key, value in model_state.items()}
    validate_model_state(state, model_metadata)
    payload = {"model_state": state, "model_metadata": model_metadata,
               "training_state": training_state or {}, "optimizer_state": optimizer_state or {},
               "config": config or {}, "provenance": provenance or {}}
    arrays, manifest = {}, {}
    encoded = _encode(payload, arrays, manifest)
    header = {"format": FORMAT, "version": FORMAT_VERSION, "layout":
              "flat voxel-major AoS; rotations wxyz; radiance SH coefficient then RGB",
              "arrays": manifest, "payload": encoded}
    arrays["metadata_json"] = np.frombuffer(
        json.dumps(header, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8"),
        dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **arrays)
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
