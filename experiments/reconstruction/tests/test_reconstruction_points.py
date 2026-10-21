"""Compare renderer-frame point initialization with the existing PLY path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruction_common import initialize_runtime
from preview import RuntimeConfig


def check_points(mi, dr, np, directory):
    positions = np.array([
        [.04, .06, .02], [.12, .19, .10], [.20, .10, .22],
        [-.8, .5, .9], [np.nan, 0., 0.], [4., 0., 0.],
    ], np.float32)
    colors = np.array([
        [1., .5, .2], [.1, .2, .9], [.2, .8, .3],
        [np.nan, .2, .1], [.5, .5, .5], [.5, .5, .5],
    ], np.float32)
    path = directory / "equivalent.ply"
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(positions)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float red\nproperty float green\nproperty float blue\nend_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        for (x, y, z), color in zip(positions, colors):
            # The PLY loader rotates (x, y, z) -> (x, z, -y).
            handle.write(struct.pack("<6f", x, -z, y, *color))
    lo, hi = [-1.25] * 3, [1.25] * 3
    ply_model = mi.VoxelReconstruction(str(path), 10, lo, hi, .05, 9, 1)
    point_model = mi.VoxelReconstruction.from_points(
        positions.ravel(), colors.ravel(), 10, lo, hi, 9, 1, core_resolution=8)
    assert ply_model.core_resolution == ply_model.resolution == 10
    assert point_model.core_resolution == 8 and point_model.resolution == 10
    assert point_model.active_voxel_count == ply_model.active_voxel_count == 2
    for name, value in ply_model.snapshot().items():
        np.testing.assert_array_equal(np.array(value), np.array(point_model.snapshot()[name]),
                                      err_msg=f"{mi.variant()}: {name}")
    assert point_model.statistics() == ply_model.statistics()
    stats = point_model.statistics()
    assert (stats["input_points"], stats["invalid_points"], stats["outside_points"],
            stats["colored_points"]) == (6, 1, 1, 3)
    restored = mi.VoxelReconstruction.from_state(
        10, lo, hi, point_model.snapshot(), 9, 1, core_resolution=8)
    assert restored.core_resolution == 8
    for name, value in restored.snapshot().items():
        np.testing.assert_array_equal(np.array(value), np.array(point_model.snapshot()[name]))
    for make_invalid in (
        lambda: mi.VoxelReconstruction.from_points(
            positions.ravel(), colors.ravel()[:-1], 10, lo, hi),
        lambda: mi.VoxelReconstruction.from_points(
            positions.ravel(), colors.ravel(), 10, lo, hi, core_resolution=11),
        lambda: mi.VoxelReconstruction.from_state(
            10, lo, hi, point_model.snapshot(), 9, 1, core_resolution=11),
        lambda: mi.VoxelReconstruction.from_points(
            positions.ravel().astype(np.float64), colors.ravel(), 10, lo, hi),
    ):
        try:
            make_invalid()
        except (RuntimeError, ValueError, TypeError):
            pass
        else:
            raise AssertionError("Invalid point initialization was accepted")
    return {"variant": mi.variant(), "ply_parity": "exact", "core_restore": "passed",
            "input_validation": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="mitsuba-point-tests-") as temporary:
        directory = Path(temporary)
        mi, dr, np, handles = initialize_runtime(
            RuntimeConfig(args.build_dir, args.gpu), directory, {})
        checks = []
        for variant in ("cuda_ad_rgb", "scalar_rgb"):
            mi.set_variant(variant)
            checks.append(check_points(mi, dr, np, directory))
        print(json.dumps({"status": "passed", "checks": checks}), flush=True)


if __name__ == "__main__":
    main()
