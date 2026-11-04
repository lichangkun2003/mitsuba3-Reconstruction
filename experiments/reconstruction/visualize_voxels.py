"""Export checkpoint leaf boxes to an offline, interactive HTML viewer.

Only cell addresses and model metadata are loaded. Ellipsoid parameters, SH,
Adam state, Mitsuba and CUDA are not needed. Coordinates use the checkpoint's
scene space, including its saved normalization for COLMAP datasets.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reconstruction_checkpoint import (FORMAT, FORMAT_VERSION, _array_digest,
    _decode, _reject_json_constant, _unique_json_object)
from reconstruction_world_shell import (ADAPTIVE_TAG, CORE_ADAPTIVE_TAG, adaptive_decode_numpy,
    core_leaf_depths_numpy,
    leaf_boxes_numpy, validate_world_shell_layout, world_shell_metadata_layout, adaptive_depth_limit)


def _layout(metadata):
    """Validate geometry without imposing current training/appearance options."""
    if not isinstance(metadata, dict):
        raise ValueError("Expected a model metadata dictionary")
    mapping = metadata.get("scene_mapping", "bounded")
    if mapping not in ("bounded", "world_shell"):
        raise ValueError("Legacy compressed/projective scene mappings cannot be displayed as world-space boxes")
    resolution = metadata.get("core_resolution", metadata.get("resolution"))
    if (type(resolution) is not int or not 1 <= resolution <= 1024
            or metadata.get("resolution") != resolution):
        raise ValueError("Expected a core resolution in [1, 1024] without compressed grid padding")
    lower = np.asarray(metadata.get("grid_min"), dtype=np.float64)
    upper = np.asarray(metadata.get("grid_max"), dtype=np.float64)
    if (lower.shape != (3,) or upper.shape != (3,) or not np.isfinite(lower).all()
            or not np.isfinite(upper).all() or not (upper > lower).all()):
        raise ValueError("Expected three finite, increasing grid bounds")
    levels, subdivision = 0, 0
    if mapping == "world_shell":
        _, subdivision = world_shell_metadata_layout(metadata)
        levels = metadata["background_shell_levels"]
        validate_world_shell_layout(resolution, levels, lower, upper, subdivision)
    return mapping, resolution, lower, upper, levels, subdivision


def read_layout(path):
    """Read and verify the address array, without decompressing appearance/Adam."""
    with np.load(path, allow_pickle=False) as archive:
        if "metadata_json" not in archive.files or len(set(archive.files)) != len(archive.files):
            raise ValueError("Expected a reconstruction checkpoint containing metadata_json")
        raw = archive["metadata_json"]
        if raw.dtype != np.uint8 or raw.ndim != 1:
            raise ValueError("Checkpoint metadata must be a UTF-8 byte array")
        header = json.loads(raw.tobytes().decode("utf-8"), object_pairs_hook=_unique_json_object,
                            parse_constant=_reject_json_constant)
        if not isinstance(header, dict) or header.get("format") != FORMAT or header.get("version") != FORMAT_VERSION:
            raise ValueError("Unsupported reconstruction checkpoint format/version")
        manifest = header.get("arrays")
        if not isinstance(manifest, dict) or set(archive.files) != set(manifest) | {"metadata_json"}:
            raise ValueError("Checkpoint archive does not match its array manifest")

        class RequiredArrays:
            def __getitem__(self, key):
                record = manifest[key]
                value = archive[key]
                if (value.dtype.kind not in "biuf" or value.dtype.hasobject
                        or value.dtype.str != record["dtype"] or list(value.shape) != record["shape"]):
                    raise ValueError(f"Checkpoint array shape/dtype mismatch: {key}")
                value = np.ascontiguousarray(value)
                if _array_digest(value) != record["sha256"]:
                    raise ValueError(f"Checkpoint array checksum mismatch: {key}")
                return value

        try:
            payload = header["payload"]
            if payload["type"] != "dict":
                raise ValueError("Malformed checkpoint payload")
            metadata = _decode(payload["items"]["model_metadata"], RequiredArrays())
            state = payload["items"]["model_state"]
            cell_node = state["items"]["cell_indices"]
            if state["type"] != "dict" or cell_node["type"] != "array":
                raise ValueError("Malformed checkpoint cell array")
            cells = RequiredArrays()[cell_node["key"]]
        except (KeyError, TypeError) as error:
            raise ValueError("Checkpoint is missing its voxel layout") from error
    _layout(metadata)
    if cells.ndim != 1 or cells.dtype.kind not in "iu" or np.any(cells < 0):
        raise ValueError("cell_indices must be a flat nonnegative integer array")
    return metadata, cells


def iter_boxes(cells, metadata, chunk_rows=65535):
    """Decode actual cell AABBs, never optimized ellipsoid centers or scales."""
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    mapping, resolution, lower, upper, levels, subdivision = _layout(metadata)
    for start in range(0, len(cells), chunk_rows):
        ids = cells[start:start + chunk_rows]
        if ids.dtype.kind not in "iu" or np.any(ids < 0):
            raise ValueError("Invalid voxel cell address")
        ids = ids.astype(np.uint64, copy=False)
        boxes = np.empty((len(ids), 8), dtype="<f4")
        boxes[:, 6] = 0
        boxes[:, 7] = math.log2(resolution)
        if mapping == "bounded":
            if np.any(ids >= resolution ** 3):
                raise ValueError("Cell address lies outside the bounded grid")
            coords = np.stack((ids % resolution, ids // resolution % resolution,
                               ids // resolution ** 2), axis=1)
            size = (upper - lower) / resolution
            boxes[:, :3] = lower + (coords + .5) * size
            boxes[:, 3:6] = size
        else:
            adaptive = ids >= ADAPTIVE_TAG
            adaptive_core = ~adaptive & ((ids & np.uint64(CORE_ADAPTIVE_TAG)) != 0)
            background = (ids >= resolution ** 3) & ~adaptive_core
            if np.any(adaptive_core) and metadata.get("foreground_refinement") != "adaptive":
                raise ValueError("Tagged foreground cells require adaptive foreground metadata")
            if metadata.get("background_refinement", "fixed") == "adaptive":
                if np.any(background & ~adaptive):
                    raise ValueError("Adaptive model contains legacy background addresses")
            elif np.any(adaptive):
                raise ValueError("Fixed/uniform model contains adaptive background addresses")
            lo, hi = leaf_boxes_numpy(ids, resolution, lower, upper, levels, np, subdivision)
            boxes[:, :3] = (lo + hi) * .5
            boxes[:, 3:6] = hi - lo
            if np.any(adaptive_core):
                depths, _ = core_leaf_depths_numpy(ids[adaptive_core], resolution, np)
                maximum = int(metadata.get("foreground_max_resolution", resolution)).bit_length()-1
                if np.any(depths > maximum):
                    raise ValueError("Adaptive foreground leaf depth exceeds model metadata")
                boxes[adaptive_core, 7] = depths
            legacy = background & ~adaptive
            boxes[legacy, 6] = (ids[legacy] - resolution ** 3) // (56 * 8 ** subdivision) + 1
            boxes[legacy, 7] = subdivision
            if np.any(adaptive):
                coarse, depth, _ = adaptive_decode_numpy(ids[adaptive], levels, np)
                if np.any(depth > adaptive_depth_limit(metadata, coarse // 56)):
                    raise ValueError("Adaptive leaf depth exceeds model metadata")
                boxes[adaptive, 6] = coarse // 56 + 1
                boxes[adaptive, 7] = depth
        if not np.isfinite(boxes).all() or not (boxes[:, 3:6] > 0).all():
            raise ValueError("Voxel boxes cannot be represented by finite positive float32 dimensions")
        yield boxes


def export_viewer(checkpoint, output, *, overwrite=False):
    """Stream every active voxel into one portable HTML file; no export sampling."""
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    if output.suffix.lower() != ".html":
        raise ValueError("Viewer output filename must end in .html")
    if output == checkpoint:
        raise ValueError("Viewer output cannot overwrite its source checkpoint")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}; use --overwrite to replace the viewer")
    metadata, cells = read_layout(checkpoint)
    mapping, resolution, lower, upper, _, _ = _layout(metadata)
    template = Path(__file__).with_name("voxel_viewer.html").read_text(encoding="utf-8")
    if template.count("__VOXEL_DATA_JSON__") != 1:
        raise ValueError("Invalid voxel viewer template")
    prefix, suffix = template.split("__VOXEL_DATA_JSON__")
    summary = dict(source=str(checkpoint), mapping=mapping, resolution=resolution,
                   bounds_min=lower.tolist(), bounds_max=upper.tolist(), total_voxels=len(cells),
                   exported_voxels=len(cells), box_stride=8, sampled=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    groups = Counter()
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            # Escape '<' so even unusual source filenames cannot end the script element.
            encoded = json.dumps(summary, ensure_ascii=True, allow_nan=False).replace("<", "\\u003c")
            stream.write(prefix + encoded[:-1] + ',"boxes_base64":"')
            tail = b""
            for boxes in iter_boxes(cells, metadata):
                pairs, counts = np.unique(boxes[:, 6:8], axis=0, return_counts=True)
                groups.update({(int(pair[0]), float(pair[1])): int(count)
                               for pair, count in zip(pairs, counts)})
                raw = tail + boxes.tobytes()
                complete = len(raw) // 3 * 3
                stream.write(base64.b64encode(raw[:complete]).decode("ascii"))
                tail = raw[complete:]
            stream.write(base64.b64encode(tail).decode("ascii"))
            summary["groups"] = [dict(shell=shell, depth=depth, count=count)
                                 for (shell, depth), count in sorted(groups.items())]
            stream.write('","groups":' + json.dumps(summary["groups"]) + "}" + suffix)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="model.npz or a pre-refinement checkpoint")
    parser.add_argument("--output", type=Path, help="HTML path; defaults to CHECKPOINT_STEM-voxels.html beside the checkpoint")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing exported HTML")
    args = parser.parse_args(argv)
    output = args.output or args.checkpoint.with_name(args.checkpoint.stem + "-voxels.html")
    try:
        summary = export_viewer(args.checkpoint, output, overwrite=args.overwrite)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Voxel viewer export failed: {error}\n")
    print(f"Viewer: {output.resolve()}")
    print(f"Exported all {summary['exported_voxels']:,} active voxels ({summary['mapping']}).")
    print("Open the HTML in a browser. Display limits affect only the view; all cell boxes are embedded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
