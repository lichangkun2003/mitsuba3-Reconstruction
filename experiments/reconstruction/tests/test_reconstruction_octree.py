"""Generated CUDA regression for octree traversal against the original DDA.

The primary reference is the original leaf-by-leaf traversal on identical model
buffers and rays. Exact face-order and shifted-origin references independently
check long rays where accumulated legacy boundary rounding changes a result.
Checks include per-ray outputs, all parameter gradients, growth/deletion evidence,
and frozen replay across topology changes. No dataset or long training run is used.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preview import RuntimeConfig
from reconstruction_common import initialize_runtime, restore_model
from reconstruction_replay import NativeLossReplay


Y00 = 1. / (2. * math.sqrt(math.pi))


def check_configuration_cpu():
    import preview
    import train

    config = copy.deepcopy(train.CONFIG)
    assert preview.RenderConfig().traversal == config.render.traversal == "octree"
    assert train.settings(config)["use_octree"] is True
    with patch.object(sys, "argv", ["train.py", "--traversal", "dda"]):
        arguments = train.parse_arguments()
    train.apply_arguments(config, arguments, {})
    assert config.render.traversal == "dda" and train.settings(config)["use_octree"] is False
    assert train.final_evaluation_config(config, Path("model.npz")).traversal == "dda"
    old_render = asdict(config.render)
    del old_render["traversal"]
    assert train.normalized_algorithm_group("render", old_render) == \
        train.normalized_algorithm_group("render", old_render | {"traversal": "octree"})
    assert train.normalized_algorithm_group("render", old_render | {"traversal": "dda"}) == \
        train.normalized_algorithm_group("render", old_render | {"traversal": "octree"})
    assert train.normalized_algorithm_group("render", old_render | {"width": old_render["width"]+1}) != \
        train.normalized_algorithm_group("render", old_render)
    legacy = SimpleNamespace(VoxelReconstruction=SimpleNamespace())
    modern = SimpleNamespace(VoxelReconstruction=SimpleNamespace(octree_version=1))
    preview.validate_traversal(config.render, legacy)
    config.render.traversal = "octree"
    preview.validate_traversal(config.render, modern)
    try:
        preview.validate_traversal(config.render, legacy)
    except RuntimeError as error:
        assert "rebuild" in str(error)
    else:
        raise AssertionError("Octree request silently accepted a stale native build")
    calls = []
    model = SimpleNamespace(octree_version=1, trace=lambda *args: calls.append(args))
    preview.trace_model(model, "ray", "background", config.render)
    assert calls[-1][-1] is True and len(calls[-1]) == 6
    config.render.traversal = "dda"
    preview.trace_model(model, "ray", "background", config.render)
    assert calls[-1][-1] is False
    del model.octree_version
    preview.trace_model(model, "ray", "background", config.render)
    assert len(calls[-1]) == 5
    options = {"use_octree": False, "max_hits": 16., "rgb_only": True}
    adapted = preview.native_traversal_settings(model, options)
    assert adapted == {"max_hits": 16., "rgb_only": True} and "use_octree" in options
    model.loss_and_grad = lambda *args: args[-1]
    bypass = NativeLossReplay(SimpleNamespace(), enabled=False)
    assert bypass(model, None, None, None, None, None, options) == adapted
    assert options["use_octree"] is False
    try:
        preview.native_traversal_settings(model, options | {"use_octree": True})
    except RuntimeError as error:
        assert "rebuild" in str(error)
    else:
        raise AssertionError("Legacy loss settings silently accepted octree traversal")
    model.octree_version = 1
    assert preview.native_traversal_settings(model, options) is options
    return {"check": "default_cli_evaluation_stale_build_guard_and_resume_configuration", "status": "passed"}


def state(np, resolution, coordinates, *, opacity=.35):
    count = len(coordinates)
    xyz = np.asarray(coordinates, dtype=np.uint32).reshape(-1, 3)
    colors = np.array([[.2 + .03*(i % 5), .4 + .04*(i % 3), .65 - .02*(i % 7)]
                       for i in range(count)], dtype=np.float32).reshape(-1, 3)
    return {
        "cell_indices": xyz[:, 0] + resolution*(xyz[:, 1] + resolution*xyz[:, 2]),
        "centers": np.tile([.5, .5, .5], count).astype(np.float32),
        "log_scales": np.tile(np.log([.4, .36, .44]), count).astype(np.float32),
        "rotations": np.tile([1., 0., 0., 0.], count).astype(np.float32),
        "radiance_sh": (colors / Y00).ravel(),
        "opacity_sh": np.full(count, math.log(opacity/(1-opacity))/Y00, np.float32),
    }


def payload(np, resolution=8, coordinates=((0, 0, 2), (0, 0, 5), (0, 0, 7))):
    return {
        "model_metadata": {
            "resolution": resolution, "grid_min": [0.] * 3,
            "grid_max": [float(resolution)] * 3, "sh_count": 1, "opacity_sh_count": 1,
        },
        "model_state": state(np, resolution, coordinates),
    }


def device_state(values, mi):
    return {key: (mi.UInt32(value) if key == "cell_indices" else mi.Float(value))
            for key, value in values.items()}


def rays(mi, np, origins, directions, maxt=30.):
    origins = np.asarray(origins, np.float32).reshape(-1, 3)
    directions = np.asarray(directions, np.float32).reshape(-1, 3)
    ray = mi.Ray3f(mi.Point3f(*(mi.Float(origins[:, a].copy()) for a in range(3))),
                   mi.Vector3f(*(mi.Float(directions[:, a].copy()) for a in range(3))))
    ray.maxt = mi.Float(np.asarray(maxt, np.float32).reshape(-1))
    return ray


def axis_rays(mi, np):
    return rays(mi, np, [[.5, .5, -1.], [.55, .5, -1.], [1.5, .5, -1.]], [[0., 0., 1.]])


def host(value, dr, np):
    dr.eval(value)
    dr.sync_thread()

    def flatten(item, prefix=""):
        if isinstance(item, dict):
            for name, child in item.items():
                if name not in ("stage_profile", "traversal_profile"):
                    yield from flatten(child, prefix + name + ".")
        elif isinstance(item, (tuple, list)):
            for index, child in enumerate(item):
                yield from flatten(child, prefix + str(index) + ".")
        else:
            yield prefix.rstrip("."), np.asarray(item).copy()

    return dict(flatten(value))


def compare(expected, actual, np, label):
    assert expected.keys() == actual.keys(), (label, expected.keys(), actual.keys())
    maximum = 0.
    for name, left in expected.items():
        right = actual[name]
        assert left.shape == right.shape, (label, name, left.shape, right.shape)
        assert np.isfinite(left).all() and np.isfinite(right).all(), (label, name)
        if np.issubdtype(left.dtype, np.integer) or np.issubdtype(left.dtype, np.bool_):
            np.testing.assert_array_equal(left, right, err_msg=label + ":" + name)
        else:
            # Small atomic reductions may sum in a different warp order. Float
            # checks stay tight; a missed leaf must never be hidden by counters.
            np.testing.assert_allclose(left, right, rtol=3e-5, atol=3e-6,
                                       err_msg=label + ":" + name)
            if left.size:
                maximum = max(maximum, float(np.max(np.abs(left.astype(float)-right))))
    return maximum


def inputs(mi, ray):
    return ray, mi.Color3f(.1, .25, .35), mi.Float(0.), mi.Color3f(1.), mi.Float(1.)


def settings(**extra):
    return {"rgb_only": 1., "max_hits": 16., "max_candidates": 8.,
            "max_growth_hits": 8., "transmittance_threshold": 1e-5,
            "distortion_loss_weight": .1,
            "collect_rgb_deletion_evidence": 1., **extra}


def compare_loss(model, ray, mi, dr, np, label, **extra):
    options = settings(**extra)
    args = inputs(mi, ray)
    baseline = host(model.loss_and_grad(*args, options | {"use_octree": 0.}), dr, np)
    accelerated = host(model.loss_and_grad(*args, options | {"use_octree": 1.}), dr, np)
    maximum = compare(baseline, accelerated, np, label)
    return accelerated, maximum


def compact_storage(model, unique_cells):
    """Check actual index/hierarchy storage, including hypothetical leaves."""
    statistics = dict(model.statistics())
    required = {"spatial_index_entries", "spatial_index_allocated_bytes",
                "octree_parent_nodes", "octree_allocated_bytes", "octree_levels"}
    assert required.issubset(statistics), \
        "Rebuild this checkout: compact spatial storage statistics are required"
    assert statistics["spatial_index_entries"] == unique_cells, statistics
    # Normal and candidate trees each contribute at most one ancestor per leaf
    # per level. Logical counts exclude replay-compatible empty placeholders.
    assert statistics["octree_parent_nodes"] <= unique_cells * statistics["octree_levels"], statistics
    allocated = statistics["spatial_index_allocated_bytes"] + statistics["octree_allocated_bytes"]
    assert allocated < 1024 * 1024, ("Sparse fixture allocated a dense spatial table", statistics)
    return {key: int(statistics[key]) for key in sorted(required)}


def check_compact_storage_scaling(mi, dr, np):
    """High addresses retain row IDs while sparse allocation stays bounded."""
    maximum, reports = 0., []
    # Check the small model first: an old binary must fail before constructing
    # resolution=1024, whose historical dense leaf index alone used 4 GiB.
    for resolution in (64, 256, 1024, 1040):
        coordinates = [(resolution-1, resolution-1, resolution-1), (0, 0, resolution-1),
                       (resolution//2, resolution//2, resolution//2), (resolution-1, 0, 0)]
        checkpoint = payload(np, resolution, coordinates)
        if resolution > 1024:
            checkpoint["model_metadata"]["core_resolution"] = 1024
        model = restore_model(checkpoint, mi)
        statistics = compact_storage(model, len(coordinates))
        reports.append({"resolution": resolution, **statistics})
        # Deliberately unsorted cell addresses must still select the original
        # parameter rows, not their rank in a sorted/hash spatial structure.
        forward = [[x+.5, y+.5, -1.] for x, y, _ in coordinates]
        backward = [[x+.5, y+.5, resolution+1.] for x, y, _ in coordinates]
        origins = forward + backward + [[.5, resolution-.5, -1.]]
        directions = [[0., 0., 1.]]*4 + [[0., 0., -1.]]*4 + [[0., 0., 1.]]
        ray = rays(mi, np, origins, directions, float(resolution+2))
        result, error = compare_loss(model, ray, mi, dr, np, f"compact_sparse_{resolution}")
        maximum = max(maximum, error)
        np.testing.assert_array_equal(result["hit_count"], [1]*8 + [0])
        np.testing.assert_array_equal(result["appearance_valid"], [2]*4)
        colors = checkpoint["model_state"]["radiance_sh"].reshape(-1, 3) * Y00
        expected = np.vstack((colors*.35 + .65, colors*.35 + .65, np.ones((1, 3))))
        np.testing.assert_allclose(result["raw_rgb"], expected.T, rtol=3e-5, atol=3e-6,
                                   err_msg="Spatial lookup changed the original voxel parameter row")
        maximum = max(maximum, compare(
            host(model.trace(ray, mi.Color3f(1.), 1e-5, 16, 0., False), dr, np),
            host(model.trace(ray, mi.Color3f(1.), 1e-5, 16, 0., True), dr, np),
            np, f"compact_sparse_trace_{resolution}"))
        empty_checkpoint = payload(np, resolution, [])
        if resolution > 1024:
            empty_checkpoint["model_metadata"]["core_resolution"] = 1024
        empty = restore_model(empty_checkpoint, mi)
        compact_storage(empty, 0)
        empty_result, error = compare_loss(empty, ray, mi, dr, np, f"compact_empty_{resolution}")
        maximum = max(maximum, error)
        np.testing.assert_array_equal(empty_result["hit_count"], 0)
        np.testing.assert_array_equal(empty_result["raw_rgb"], 1.)
    first = reports[0]["spatial_index_allocated_bytes"] + reports[0]["octree_allocated_bytes"]
    last = reports[-1]["spatial_index_allocated_bytes"] + reports[-1]["octree_allocated_bytes"]
    assert last <= 16*max(first, 1), ("Storage followed grid volume instead of sparse occupancy", reports)
    return {"check": "compact_sparse_storage_to_1040_preserves_unsorted_rows_and_empty_models",
            "storage": reports, "max_absolute_error": maximum, "status": "passed"}


def check_core_1024_shell_storage(mi, dr, np):
    """Exercise all three coordinate bits above 1023, including Morton bit 32."""
    resolution = 1040
    # All eight combinations are necessary: a truncated Morton sort can alias
    # or misorder upper-z leaves even if one isolated high-address leaf works.
    coordinates = [(5 + 1024*(bits & 1), 7 + 1024*((bits >> 1) & 1),
                    11 + 1024*((bits >> 2) & 1)) for bits in (7, 0, 4, 1, 6, 2, 5, 3)]
    checkpoint = payload(np, resolution, coordinates)
    checkpoint["model_metadata"]["core_resolution"] = 1024
    model = restore_model(checkpoint, mi)
    storage = compact_storage(model, len(coordinates))
    assert storage["octree_levels"] == 11
    assert int(model.core_resolution) == 1024 and int(model.resolution) == 1040
    xy = [(5, 7), (1029, 7), (5, 1031), (1029, 1031)]
    origins = [[x+.5, y+.5, -1.] for x, y in xy] + \
              [[x+.5, y+.5, resolution+1.] for x, y in xy] + [[20.5, 20.5, -1.]]
    directions = [[0., 0., 1.]]*4 + [[0., 0., -1.]]*4 + [[0., 0., 1.]]
    ray = rays(mi, np, origins, directions, float(resolution+2))
    baseline, maximum = compare_loss(model, ray, mi, dr, np, "core1024_high_coordinate_bits")
    np.testing.assert_array_equal(baseline["hit_count"], [2]*8 + [0])
    np.testing.assert_array_equal(baseline["appearance_valid"], [2]*8)
    colors = checkpoint["model_state"]["radiance_sh"].reshape(-1, 3) * Y00
    expected = []
    for backwards in (False, True):
        for x, y in xy:
            rows = sorted((i for i, cell in enumerate(coordinates) if cell[:2] == (x, y)),
                          key=lambda i: coordinates[i][2], reverse=backwards)
            expected.append(.35*colors[rows[0]] + .65*.35*colors[rows[1]] + .65**2)
    expected.append(np.ones(3))
    np.testing.assert_allclose(baseline["raw_rgb"], np.asarray(expected).T,
                               rtol=3e-5, atol=3e-6, err_msg="Independent 33-bit Morton leaf-ID oracle")
    assert np.any(np.abs(baseline["gradients.radiance_sh"]) > 0)
    baseline_trace = host(model.trace(ray, mi.Color3f(1.), 1e-5, 16, 0., True), dr, np)
    maximum = max(maximum, compare(baseline_trace,
        host(model.trace(ray, mi.Color3f(1.), 1e-5, 16, 0., False), dr, np), np, "core1024_trace"))

    proposal_coordinates = [(1029, 1031, 1024), (5, 7, 1024), (1029, 1031, 1024)]
    proposals = state(np, resolution, proposal_coordinates, opacity=.25)
    model.set_growth_candidates(device_state(proposals, mi))
    compact_storage(model, len(coordinates) + 2)
    with_candidates, error = compare_loss(model, ray, mi, dr, np, "core1024_candidate_high_bits",
                                          collect_growth_evidence=1.)
    maximum = max(maximum, error, compare(baseline,
        {key: value for key, value in with_candidates.items() if not key.startswith("growth_")},
        np, "core1024_candidates_leave_real_hits_unchanged"))
    np.testing.assert_array_equal(with_candidates["growth_observed"], [2, 2, 2])
    np.testing.assert_array_equal(with_candidates["growth_unknown"], 0)
    expected_gains = []
    for index in range(len(proposal_coordinates)):
        inserted = copy.deepcopy(checkpoint)
        for key, value in proposals.items():
            width = 1 if key == "cell_indices" else len(value)//len(proposal_coordinates)
            inserted["model_state"][key] = np.concatenate((
                checkpoint["model_state"][key], value[index*width:(index+1)*width]))
        inserted_model = restore_model(inserted, mi)
        inserted_result = host(inserted_model.loss_and_grad(*inputs(mi, ray), settings()), dr, np)
        expected_gains.append(float(np.sum(baseline["rgb_loss"] - inserted_result["rgb_loss"])))
    np.testing.assert_allclose(with_candidates["growth_gain"], expected_gains,
                               rtol=3e-5, atol=3e-6, err_msg="33-bit candidate insertion oracle")
    model.clear_growth_candidates()
    maximum = max(maximum, compare(baseline,
        host(model.loss_and_grad(*inputs(mi, ray), settings()), dr, np), np, "core1024_overlay_clear"))

    # Point initialization and checkpoint construction must enforce the same
    # distinction between core resolution and padded storage resolution.
    positions = np.asarray(coordinates, np.float32).reshape(-1) + np.float32(.5)
    point_model = mi.VoxelReconstruction.from_points(positions, np.full_like(positions, .5),
        resolution, [0.]*3, [float(resolution)]*3, 1, 1, core_resolution=1024)
    assert int(point_model.resolution) == 1040 and int(point_model.core_resolution) == 1024
    np.testing.assert_array_equal(np.sort(np.asarray(point_model.snapshot()["cell_indices"])),
                                  np.sort(checkpoint["model_state"]["cell_indices"]))
    rejected = []
    for actual, core in ((1041, 1024), (1040, 1025), (1040, 0), (512, 1024)):
        bad = payload(np, actual, [(0, 0, 0)])
        if core:
            bad["model_metadata"]["core_resolution"] = core
        for constructor in ("state", "points"):
            try:
                if constructor == "state":
                    restore_model(bad, mi)
                else:
                    mi.VoxelReconstruction.from_points(np.full(3, .5, np.float32),
                        np.full(3, .5, np.float32), actual, [0.]*3, [float(actual)]*3,
                        1, 1, core_resolution=core)
            except (RuntimeError, ValueError):
                rejected.append([constructor, actual, core])
            else:
                raise AssertionError(f"Invalid core/storage resolution accepted: {constructor}, {actual}, {core}")
    return {"check": "core1024_storage1040_33bit_morton_hits_gradients_and_candidates",
            "storage": storage, "rejected_resolutions": rejected,
            "max_absolute_error": maximum, "status": "passed"}


def check_compact_candidate_lifecycle(mi, dr, np):
    """Overlay replacement changes only candidate leaves and preserves siblings."""
    resolution = 64
    checkpoint = payload(np, resolution, [(0, 0, 40), (63, 63, 63)])
    model = restore_model(checkpoint, mi)
    initial_storage = compact_storage(model, 2)
    ray = rays(mi, np, [[.5, .5, -1.], [.55, .5, -1.], [63.5, 63.5, -1.]],
               [[0., 0., 1.]], 66.)
    baseline = host(model.loss_and_grad(*inputs(mi, ray), settings()), dr, np)
    baseline_trace = host(model.trace(ray, mi.Color3f(1.), 1e-5, 16), dr, np)
    proposals = state(np, resolution, [(0, 0, 2), (0, 0, 20), (0, 0, 2)], opacity=.25)
    proposals["radiance_sh"].reshape(-1, 3)[:] = np.asarray(
        [[.02, .08, .12], [.7, .8, .9], [.45, .1, .6]], np.float32) / Y00
    maximum = 0.

    def verify(label, expected_observed):
        nonlocal maximum
        compact_storage(model, 2 + len(np.unique(proposals["cell_indices"])))
        result, error = compare_loss(model, ray, mi, dr, np, label, collect_growth_evidence=1.)
        maximum = max(maximum, error, compare(baseline,
            {key: value for key, value in result.items() if not key.startswith("growth_")}, np, label+"_real"))
        maximum = max(maximum, compare(baseline_trace,
            host(model.trace(ray, mi.Color3f(1.), 1e-5, 16), dr, np), np, label+"_trace"))
        np.testing.assert_array_equal(result["growth_observed"], expected_observed)
        np.testing.assert_array_equal(result["growth_unknown"], 0)
        # An independent actual insertion checks candidate row lookup/order;
        # equal DDA/octree outputs alone could share the same wrong lookup.
        expected_gains = []
        for index in range(len(proposals["cell_indices"])):
            inserted = copy.deepcopy(checkpoint)
            for key, value in proposals.items():
                width = 1 if key == "cell_indices" else len(value)//3
                inserted["model_state"][key] = np.concatenate((
                    checkpoint["model_state"][key], value[index*width:(index+1)*width]))
            inserted_model = restore_model(inserted, mi)
            inserted_result = host(inserted_model.loss_and_grad(*inputs(mi, ray), settings()), dr, np)
            expected_gains.append(float(np.sum(baseline["rgb_loss"] - inserted_result["rgb_loss"])))
        np.testing.assert_allclose(result["growth_gain"], expected_gains, rtol=3e-5, atol=3e-6,
                                   err_msg=label+"_independent_insertion")
        return result

    model.set_growth_candidates(device_state(proposals, mi))
    original = verify("compact_interleaved_siblings", [2, 2, 2])
    assert int(model.growth_candidate_count) == 3
    assert abs(float(original["growth_gain"][0]-original["growth_gain"][2])) > 1e-4
    malformed = copy.deepcopy(proposals)
    malformed["cell_indices"][1] = checkpoint["model_state"]["cell_indices"][1]
    revision = int(model.topology_revision)
    try:
        model.set_growth_candidates(device_state(malformed, mi))
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("Candidate replacement accepted an occupied high-address cell")
    assert int(model.topology_revision) == revision
    maximum = max(maximum, compare(original,
        verify("compact_rejected_replacement_keeps_overlay", [2, 2, 2]), np, "overlay_transaction"))
    # Same allocation sizes, different empty leaf: old occupancy must disappear.
    proposals["cell_indices"][1] = 1 + resolution*(1 + resolution*20)
    model.set_growth_candidates(device_state(proposals, mi))
    replaced = verify("compact_same_size_replacement", [2, 0, 2])
    model.clear_growth_candidates()
    model.clear_growth_candidates()
    assert int(model.growth_candidate_count) == 0
    compact_storage(model, 2)
    maximum = max(maximum, compare(baseline,
        host(model.loss_and_grad(*inputs(mi, ray), settings()), dr, np), np, "compact_cleared"))
    # Both installing an overlay again and rebuilding from saved normal buffers
    # must discard stale candidate links and recover the original row mapping.
    model.set_growth_candidates(device_state(proposals, mi))
    maximum = max(maximum, compare(replaced,
        verify("compact_reinstalled", [2, 0, 2]), np, "same_overlay_reinstalled"))
    model = restore_model({"model_metadata": checkpoint["model_metadata"],
                           "model_state": model.snapshot()}, mi)
    compact_storage(model, 2)
    assert int(model.growth_candidate_count) == 0
    maximum = max(maximum, compare(baseline,
        host(model.loss_and_grad(*inputs(mi, ray), settings()), dr, np), np, "compact_restored"))
    return {"check": "compact_candidate_siblings_replacement_clear_and_reload",
            "initial_storage": initial_storage, "max_absolute_error": maximum, "status": "passed"}


def check_compact_invalid_cells(mi, dr, np):
    """Native sparse construction must reject aliased and illegal leaf keys."""
    rejected = []
    for resolution, coordinates, cells in (
        (1, [(0, 0, 0), (0, 0, 0)], [0, 0]),
        (7, [(6, 6, 6), (0, 0, 0), (3, 2, 1)], [342, 0, 342]),
        (7, [(6, 6, 6), (0, 0, 0), (3, 2, 1)], [342, 343, 66]),
        (7, [(6, 6, 6), (0, 0, 0), (3, 2, 1)], [342, 0, 0xffffffff]),
    ):
        checkpoint = payload(np, resolution, coordinates)
        checkpoint["model_state"]["cell_indices"] = np.asarray(cells, np.uint32)
        try:
            restore_model(checkpoint, mi)
        except (RuntimeError, ValueError):
            rejected.append({"resolution": resolution, "cells": cells})
        else:
            raise AssertionError(f"Invalid sparse constructor cell keys accepted: {resolution}, {cells}")
    # A rejected construction must not poison subsequent valid unsorted keys.
    valid = payload(np, 7, [(6, 6, 6), (0, 0, 0), (3, 2, 1)])
    model = restore_model(valid, mi)
    compact_storage(model, 3)
    np.testing.assert_array_equal(np.asarray(model.snapshot()["cell_indices"]),
                                  valid["model_state"]["cell_indices"])
    return {"check": "native_compact_constructor_rejects_duplicates_and_out_of_range_keys",
            "rejected": rejected, "status": "passed"}


def check_boundaries_and_default(mi, dr, np):
    coordinates = [(0, 0, 2), (0, 0, 5), (0, 0, 7), (3, 3, 3),
                   (4, 3, 3), (4, 4, 3), (4, 4, 4), (3, 4, 4), (3, 3, 4), (7, 7, 7)]
    checkpoint = payload(np, coordinates=coordinates)
    centers = checkpoint["model_state"]["centers"].reshape(-1, 3)
    # Original DDA crosses exact ties x, then y, then z. These four ellipsoids
    # deliberately contain the zero-length intermediate corner segments.
    centers[4], centers[5] = [.1, .9, .9], [.1, .1, .9]
    centers[7], centers[8] = [.9, .1, .1], [.9, .9, .1]
    model = restore_model(checkpoint, mi)
    origins = [[.5, .5, -1.], [.5, .5, 9.], [-1., -1., -1.], [9., 9., 9.],
               [-1., -1.+1e-6, -1.], [-1., -1.-1e-6, -1.],
               [4., 4., -1.], [4., 4., 9.], [.5, .5, 2.1], [.5, .5, -1.],
               [9., .5, -1.], [-1., 4., 4.], [9., 4., 4.], [0., 0., 0.],
               [8., 8., 8.], [3.5, 3.5, -1.], [8., 0., 0.]]
    directions = [[0., 0., 1.], [0., 0., -1.], [1., 1., 1.], [-1., -1., -1.],
                  [1., 1., 1.], [1., 1., 1.], [0., 0., 1.], [0., 0., -1.],
                  [0., 0., 1.], [0., 0., 1.], [0., 0., 1.], [1., 0., 0.],
                  [-1., 0., 0.], [1., 1., 1.], [-1., -1., -1.],
                  [0., 0., 1.], [-1., 0., 0.]]
    maxt = np.full(len(origins), 30., np.float32)
    maxt[8], maxt[9] = .7, 2.5
    ray = rays(mi, np, origins, directions, maxt)
    options = settings()
    baseline = host(model.loss_and_grad(*inputs(mi, ray), options | {"use_octree": 0.}), dr, np)
    default = host(model.loss_and_grad(*inputs(mi, ray), options), dr, np)
    maximum = compare(baseline, default, np, "default_octree_boundary_batch")
    explicit = host(model.loss_and_grad(*inputs(mi, ray), options | {"use_octree": 1.}), dr, np)
    maximum = max(maximum, compare(default, explicit, np, "default_equals_explicit_octree"))
    assert all(baseline["appearance_valid"][i] > 0 for i in (4, 5, 7, 8)), \
        "Fixture must exercise zero-length occupied cells at positive and negative ties"
    assert np.max(baseline["distortion_loss"]) > 1e-4
    assert baseline["hit_count"][9] == 0, "maxt ends before the first occupied cell"
    assert np.sum(baseline["deletion_observed"]) > 0
    for threshold, limit, offset in ((1e-5, 16, 0.), (.8, 16, 0.), (1e-5, 1, 0.),
                                     (1e-5, 16, 1e-4)):
        reference = host(model.trace(ray, mi.Color3f(1.), threshold, limit, offset, False), dr, np)
        current = host(model.trace(ray, mi.Color3f(1.), threshold, limit, offset), dr, np)
        maximum = max(maximum, compare(reference, current, np, "trace_default"))
    return {"check": "default_octree_boundaries_and_real_zero_length_hits",
            "max_absolute_error": maximum, "status": "passed"}


def check_sparse_random_and_non_power_two(mi, dr, np):
    rng = np.random.default_rng(514)
    maximum = 0.
    for resolution in (1, 7, 16):
        total = resolution**3
        cells = rng.choice(total, min(35, total), replace=False)
        coordinates = [(int(i % resolution), int(i // resolution % resolution), int(i // resolution**2))
                       for i in cells]
        checkpoint = payload(np, resolution, coordinates)
        model = restore_model(checkpoint, mi)
        compact_storage(model, len(cells))
        origins = rng.uniform(-.3*resolution, 1.3*resolution, (48, 3))
        directions = rng.uniform(.1*resolution, .9*resolution, (48, 3))-origins
        # Add rays guaranteed to encounter every occupied leaf, in both signs.
        origins = np.concatenate((origins, np.asarray(coordinates)+[.5, .5, -1.],
                                  np.asarray(coordinates)+[.5, .5, 2.]))
        directions = np.concatenate((directions, np.tile([0., 0., 1.], (len(cells), 1)),
                                     np.tile([0., 0., -1.], (len(cells), 1))))
        ray = rays(mi, np, origins, directions, 3.*resolution)
        result, error = compare_loss(model, ray, mi, dr, np, f"resolution_{resolution}")
        maximum = max(maximum, error)
        assert np.all(result["appearance_valid"] > 0)
        empty = restore_model(payload(np, resolution, []), mi)
        compact_storage(empty, 0)
        _, error = compare_loss(empty, ray, mi, dr, np, f"empty_resolution_{resolution}")
        maximum = max(maximum, error)
        if resolution == 1:
            # Zero parent levels must still support the candidate-only tree's
            # leaf lookup, including multiple hypotheses for its only cell.
            proposals = state(np, 1, [(0, 0, 0), (0, 0, 0)], opacity=.25)
            empty.set_growth_candidates(device_state(proposals, mi))
            compact_storage(empty, 1)
            single_ray = rays(mi, np, [[.5, .5, -1.], [.5, .5, 2.]],
                              [[0., 0., 1.], [0., 0., -1.]], 3.)
            candidate_result, error = compare_loss(empty, single_ray, mi, dr, np,
                "single_cell_candidate_only", collect_growth_evidence=1.)
            maximum = max(maximum, error)
            np.testing.assert_array_equal(candidate_result["hit_count"], 0)
            np.testing.assert_array_equal(candidate_result["raw_rgb"], 1.)
            np.testing.assert_array_equal(candidate_result["growth_observed"], [2, 2])
            np.testing.assert_array_equal(candidate_result["growth_unknown"], 0)
            empty.clear_growth_candidates()
            compact_storage(empty, 0)
            cleared_trace = host(empty.trace(single_ray, mi.Color3f(1.), 1e-5, 16), dr, np)
            np.testing.assert_array_equal(cleared_trace["2"], 0)
    return {"check": "sparse_random_single_cell_non_power_two_and_empty_models",
            "max_absolute_error": maximum, "status": "passed"}


def check_long_empty_block_exits(mi, dr, np):
    """A large empty node ends directly at real/candidate face and corner hits.

    The intermediate corner cells have zero ray length but contain the corner
    inside their ellipsoid. Skipping them would change both physical hits and
    growth evidence, even though a point sampled beyond the corner misses them.
    """
    maximum = 0.
    legacy_center_difference = 0.
    for positive in (True, False):
        if positive:
            coordinates = [(32, 31, 31), (32, 32, 31), (32, 32, 32)]
            centers = [[.1, .9, .9], [.1, .1, .9], [.5, .5, .5]]
            origin, direction, terminal = [-1.] * 3, [1.] * 3, (63, 63, 63)
        else:
            coordinates = [(31, 32, 32), (31, 31, 32), (31, 31, 31)]
            centers = [[.9, .1, .1], [.9, .9, .1], [.5, .5, .5]]
            origin, direction, terminal = [65.] * 3, [-1.] * 3, (0, 0, 0)
        checkpoint = payload(np, 64, coordinates)
        checkpoint["model_state"]["centers"][:] = np.asarray(centers).ravel()
        model = restore_model(checkpoint, mi)
        ray = rays(mi, np, [origin], [direction], 100.)
        name = "positive" if positive else "negative"
        reference = host(model.loss_and_grad(*inputs(mi, ray), settings(use_octree=0.)), dr, np)
        result = host(model.loss_and_grad(*inputs(mi, ray), settings(use_octree=1.)), dr, np)
        # Repeated additions in the legacy DDA drift at a distant exact corner.
        # A closer origin on the identical ray, still before every voxel,
        # provides an independent reference without those 32 accumulated steps.
        # Keep strict tolerances; qualify only the known legacy center gradient
        # against that reference instead of widening a global float tolerance.
        close_origin = [31. if positive else 33.] * 3
        close_ray = rays(mi, np, [close_origin], [direction], 100.)
        close_reference = host(model.loss_and_grad(*inputs(mi, close_ray),
                                                  settings(use_octree=0.)), dr, np)
        close_octree = host(model.loss_and_grad(*inputs(mi, close_ray),
                                               settings(use_octree=1.)), dr, np)
        maximum = max(maximum, compare(close_reference, result, np, name + "_close_origin_reference"))
        maximum = max(maximum, compare(close_octree, result, np, name + "_origin_invariant_octree"))
        maximum = max(maximum, compare(
            {k: v for k, v in reference.items() if k != "gradients.centers"},
            {k: v for k, v in result.items() if k != "gradients.centers"}, np,
            name + "_long_empty_corner"))
        legacy_center_difference = max(legacy_center_difference,
            float(np.max(np.abs(reference["gradients.centers"] - result["gradients.centers"]))))
        np.testing.assert_array_equal(result["hit_count"], [3])
        np.testing.assert_array_equal(result["appearance_valid"], 1)
        maximum = max(maximum, compare(
            host(model.trace(ray, mi.Color3f(1.), 1e-5, 16, 0., False), dr, np),
            host(model.trace(ray, mi.Color3f(1.), 1e-5, 16, 0., True), dr, np),
            np, name + "_long_empty_corner_trace"))
        # Only proposals occupy the corner. Their leaf traversal must not
        # insert real alpha or change the distant terminal voxel's gradient.
        candidate_model = restore_model(payload(np, 64, [terminal]), mi)
        before = host(candidate_model.trace(ray, mi.Color3f(1.), 1e-5, 16), dr, np)
        proposals = state(np, 64, coordinates[:2], opacity=.25)
        proposals["centers"][:] = np.asarray(centers[:2]).ravel()
        candidate_model.set_growth_candidates(device_state(proposals, mi))
        result, error = compare_loss(candidate_model, ray, mi, dr, np,
                                    name + "_long_empty_candidate_corner",
                                    collect_growth_evidence=1.)
        maximum = max(maximum, error)
        np.testing.assert_array_equal(result["growth_observed"], [1, 1])
        np.testing.assert_array_equal(result["growth_unknown"], [0, 0])
        np.testing.assert_array_equal(result["hit_count"], [1])
        maximum = max(maximum, compare(before,
            host(candidate_model.trace(ray, mi.Color3f(1.), 1e-5, 16), dr, np),
            np, name + "_candidate_corner_is_invisible"))

    # Padded octree nodes must respect the real extent, and a ray ending in
    # the empty prefix or precisely on the next leaf face must stay empty.
    model = restore_model(payload(np, 47, [(0, 0, 46), (46, 46, 0)]), mi)
    ray = rays(mi, np, [[.5, .5, -1.]] * 4 + [[46.5, 46.5, 48.]] * 4,
               [[0., 0., 1.]] * 4 + [[0., 0., -1.]] * 4,
               [46.75, 47., 47.5, 100.] * 2)
    result, error = compare_loss(model, ray, mi, dr, np, "padded_long_skip_maxt")
    maximum = max(maximum, error)
    np.testing.assert_array_equal(result["hit_count"], [0, 0, 1, 1] * 2)
    maximum = max(maximum, compare(
        host(model.trace(ray, mi.Color3f(1.), 1e-5, 16, 0., False), dr, np),
        host(model.trace(ray, mi.Color3f(1.), 1e-5, 16, 0., True), dr, np),
        np, "padded_long_skip_maxt_trace"))
    return {"check": "long_empty_node_exit_corners_candidates_and_padded_maxt",
            "max_absolute_error": maximum,
            "legacy_accumulated_center_gradient_difference": legacy_center_difference,
            "status": "passed"}


def check_near_miss(mi, dr, np):
    checkpoint = payload(np, coordinates=[(0, 0, 6)])
    checkpoint["model_state"]["log_scales"][:] = math.log(.25)
    model = restore_model(checkpoint, mi)
    ray = rays(mi, np, [[.78, .5, -1.]], [[0., 0., 1.]])
    result, maximum = compare_loss(model, ray, mi, dr, np, "near_miss_only", max_candidates=1.)
    np.testing.assert_array_equal(result["hit_count"], 0)
    np.testing.assert_array_equal(result["appearance_valid"], 0)
    assert result["geometry_valid"][0] > 0
    assert np.linalg.norm(result["gradients.centers"]) > 1e-3
    assert np.linalg.norm(result["gradients.log_scales"]) > 1e-3
    return {"check": "nonhit_occupied_leaf_preserves_near_miss_gradient",
            "max_absolute_error": maximum, "status": "passed"}


def check_growth_and_truncation(mi, dr, np):
    base = payload(np, coordinates=[(0, 0, 2), (0, 0, 6)])
    model = restore_model(base, mi)
    ray = axis_rays(mi, np)
    plain = host(model.loss_and_grad(*inputs(mi, ray), settings()), dr, np)
    plain_trace = host(model.trace(ray, mi.Color3f(1.), 1e-5, 16), dr, np)
    proposals = state(np, 8, [(0, 0, 0), (0, 0, 4), (0, 0, 4)], opacity=.25)
    proposals["radiance_sh"].reshape(-1, 3)[1:] = np.array([[.1, .2, .3], [.8, .7, .6]])/Y00
    model.set_growth_candidates(device_state(proposals, mi))
    maximum = compare(plain_trace, host(model.trace(ray, mi.Color3f(1.), 1e-5, 16), dr, np),
                      np, "candidate_overlay_does_not_change_trace")
    inactive = host(model.loss_and_grad(*inputs(mi, ray), settings()), dr, np)
    maximum = max(maximum, compare(plain, inactive, np, "overlay_without_collection"))
    complete, error = compare_loss(model, ray, mi, dr, np, "multiparent_growth",
                                  collect_growth_evidence=1.)
    maximum = max(maximum, error)
    maximum = max(maximum, compare(plain, {k: v for k, v in complete.items()
                                          if not k.startswith("growth_")}, np,
                                          "candidates_do_not_change_real_hit_or_gradients"))
    np.testing.assert_array_equal(complete["growth_observed"], [2, 2, 2])
    np.testing.assert_array_equal(complete["growth_unknown"], 0)
    assert abs(float(complete["growth_gain"][1]-complete["growth_gain"][2])) > 1e-4
    for name, extra in (("hit_capacity", {"max_hits": 1.}),
                        ("opacity_termination", {"transmittance_threshold": .8}),
                        ("candidate_cell_overflow", {"max_growth_hits": 1.})):
        result, error = compare_loss(model, ray, mi, dr, np, name,
                                    collect_growth_evidence=1., **extra)
        maximum = max(maximum, error)
        assert np.sum(result["growth_unknown"]) > 0, name
        if name != "candidate_cell_overflow":
            assert np.sum(result["deletion_unknown"]) > 0, name
        else:
            np.testing.assert_array_equal(result["growth_observed"], [2, 0, 0])
            np.testing.assert_array_equal(result["growth_unknown"], [0, 2, 2])
    # An empty model must still traverse candidate leaves. No real hit is born
    # merely by recording a proposal's virtual intersection.
    empty = restore_model(payload(np, coordinates=[]), mi)
    empty.set_growth_candidates(device_state(proposals, mi))
    result, error = compare_loss(empty, ray, mi, dr, np, "candidate_only_tree",
                                collect_growth_evidence=1.)
    maximum = max(maximum, error)
    np.testing.assert_array_equal(result["hit_count"], 0)
    np.testing.assert_array_equal(result["raw_rgb"], 1.)
    np.testing.assert_array_equal(result["growth_observed"], [2, 2, 2])
    assert np.any(np.abs(result["growth_gain"]) > 1e-4)
    # Candidate bits must preserve the same zero-length intermediate tie
    # leaves as normal occupancy, despite not counting as physical hits.
    tied = restore_model(payload(np, coordinates=[(7, 7, 7)]), mi)
    corner_proposals = state(np, 8, [(4, 3, 3), (4, 4, 3)], opacity=.25)
    corner_proposals["centers"][:] = [.1, .9, .9, .1, .1, .9]
    tied.set_growth_candidates(device_state(corner_proposals, mi))
    tie_ray = rays(mi, np, [[-1., -1., -1.]], [[1., 1., 1.]])
    result, error = compare_loss(tied, tie_ray, mi, dr, np, "zero_length_candidate_leaves",
                                collect_growth_evidence=1.)
    maximum = max(maximum, error)
    np.testing.assert_array_equal(result["growth_observed"], [1, 1])
    np.testing.assert_array_equal(result["hit_count"], [1])
    return {"check": "multiparent_candidate_only_and_truncated_growth_deletion_evidence",
            "max_absolute_error": maximum, "status": "passed"}


def check_replay_mutations(mi, dr, np):
    checkpoint = payload(np, coordinates=[(0, 0, 2), (0, 0, 6)])
    model = restore_model(checkpoint, mi)
    ray = axis_rays(mi, np)
    args = inputs(mi, ray)
    replay = NativeLossReplay(dr)
    maximum = 0.

    def verify(label, options):
        nonlocal maximum
        baseline = host(model.loss_and_grad(*args, options | {"use_octree": 0.}), dr, np)
        for index in range(2):
            current = host(replay(model, *args, options), dr, np)
            maximum = max(maximum, compare(baseline, current, np, label+str(index)))
        return current

    original = verify("initial_replay", settings())
    before = replay.statistics()["recordings"]
    revision = int(model.topology_revision)
    parameters = {k: v.copy() for k, v in checkpoint["model_state"].items() if k != "cell_indices"}
    parameters["radiance_sh"] *= .55
    parameters["centers"][::3] += .05
    model.set_parameters({k: mi.Float(v) for k, v in parameters.items()})
    assert int(model.topology_revision) == revision
    changed = verify("changed_parameter_buffers", settings())
    assert replay.statistics()["recordings"] == before
    assert not np.array_equal(original["rgb"], changed["rgb"])
    proposals = state(np, 8, [(0, 0, 0), (0, 0, 4)], opacity=.25)
    model.set_growth_candidates(device_state(proposals, mi))
    assert int(model.topology_revision) > revision
    revision = int(model.topology_revision)
    first = verify("candidate_added", settings(collect_growth_evidence=1.))
    # Same number of proposals, different cells; the first is now missed by
    # every ray. The replay wrapper must detect this without caller clear().
    proposals["cell_indices"][0] = 3
    model.set_growth_candidates(device_state(proposals, mi))
    assert int(model.topology_revision) > revision
    revision = int(model.topology_revision)
    replaced = verify("same_count_candidate_cell_replaced", settings(collect_growth_evidence=1.))
    assert first["growth_observed"][0] == 2 and replaced["growth_observed"][0] == 0
    revision = int(model.topology_revision)
    proposals["opacity_sh"] -= 1.5
    model.set_growth_candidates(device_state(proposals, mi))
    assert int(model.topology_revision) > revision
    revision = int(model.topology_revision)
    recolored = verify("same_cells_changed_candidate_parameters", settings(collect_growth_evidence=1.))
    assert abs(float(replaced["growth_gain"][1]-recolored["growth_gain"][1])) > 1e-4
    before = replay.statistics()["recordings"]
    verify("explicit_dda_replay", settings(use_octree=0., collect_growth_evidence=1.))
    assert replay.statistics()["recordings"] > before
    model.clear_growth_candidates()
    assert int(model.topology_revision) > revision
    final = verify("candidate_cleared", settings())
    maximum = max(maximum, compare(changed, final, np, "clear_preserves_normal_model"))
    assert replay.statistics()["topology_changes"] >= 4
    assert replay.statistics()["replays"] >= 6
    return {"check": "frozen_replay_parameter_replacement_candidate_replacement_and_clear",
            "max_absolute_error": maximum, "statistics": replay.statistics(), "status": "passed"}


def check_candidate_partition_invariance(mi, dr, np):
    """Changing empty-node subdivision must not change the real ray solution."""
    resolution = 64
    rng = np.random.default_rng(9024)
    indices = rng.choice(resolution**3, 128, replace=False)
    coordinates = [(int(i % resolution), int(i // resolution % resolution),
                    int(i // resolution**2)) for i in indices]
    coordinates += [(0, 0, 31), (0, 0, 60), (31, 31, 31), (32, 32, 32)]
    coordinates = sorted(set(coordinates))
    checkpoint = payload(np, resolution, coordinates)
    model = restore_model(checkpoint, mi)
    # Guaranteed normal hits, arbitrary oblique rays, and a candidate-rich
    # axis ray that overflows the evidence capacity without ending real tracing.
    origins = np.asarray(coordinates, np.float32)+[.5, .5, -70.]
    directions = np.tile([0., 0., 1.], (len(coordinates), 1))
    arbitrary_origins = rng.uniform(-15., 79., (64, 3))
    arbitrary_directions = rng.uniform(5., 59., (64, 3))-arbitrary_origins
    origins = np.concatenate((origins, arbitrary_origins, [[.5, .5, -1.]]))
    directions = np.concatenate((directions, arbitrary_directions, [[0., 0., 1.]]))
    ray = rays(mi, np, origins, directions, 200.)
    options = settings(use_octree=1.)
    before = host(model.loss_and_grad(*inputs(mi, ray), options), dr, np)
    occupied = set(coordinates)
    proposal_coordinates = [(x, y, z) for z in range(0, resolution, 4)
                            for y in range(0, resolution, 4)
                            for x in range(0, resolution, 4)
                            if (x, y, z) not in occupied]
    proposals = state(np, resolution, proposal_coordinates, opacity=.25)
    model.set_growth_candidates(device_state(proposals, mi))
    after = host(model.loss_and_grad(*inputs(mi, ray),
                                    options | {"collect_growth_evidence": 1.}), dr, np)
    maximum = compare(before, {key: value for key, value in after.items()
                               if not key.startswith("growth_")}, np,
                      "candidate_overlay_changes_empty_node_partition_only")
    assert np.sum(after["growth_observed"]) > 0
    assert np.sum(after["growth_unknown"]) > 0, "Fixture must include candidate record overflow"
    np.testing.assert_array_equal(after["hit_count"][-1], 2)
    model.clear_growth_candidates()
    cleared = host(model.loss_and_grad(*inputs(mi, ray), options), dr, np)
    maximum = max(maximum, compare(before, cleared, np, "empty_node_partition_restored"))
    return {"check": "normal_solution_invariant_to_dense_candidate_tree_partition",
            "rays": len(origins), "candidate_cells": len(proposal_coordinates),
            "max_absolute_error": maximum, "status": "passed"}


def check_traversal_diagnostics(mi, dr, np):
    model = restore_model(payload(np, 64, [(0, 0, 63)]), mi)
    ray = rays(mi, np, [[.5, .5, -1.]], [[0., 0., 1.]], 100.)
    options = settings(profile_traversal=1., max_candidates=0.)
    baseline = host(model.loss_and_grad(*inputs(mi, ray), options | {"use_octree": 0.}), dr, np)
    octree = host(model.loss_and_grad(*inputs(mi, ray), options | {"use_octree": 1.}), dr, np)
    diagnostics = {"traversal_steps", "traversal_skipped_cells", "traversal_leaf_visits",
                   "traversal_tree_queries"}
    assert diagnostics.issubset(baseline) and diagnostics.issubset(octree)
    compare({k: v for k, v in baseline.items() if k not in diagnostics},
            {k: v for k, v in octree.items() if k not in diagnostics}, np, "profiled_sparse_ray")
    assert baseline["traversal_steps"][0] > octree["traversal_steps"][0]
    assert octree["traversal_skipped_cells"][0] > 0
    assert octree["traversal_tree_queries"][0] > 0
    np.testing.assert_array_equal(octree["traversal_leaf_visits"], [1])
    np.testing.assert_array_equal(baseline["traversal_tree_queries"], [0])
    default = host(model.loss_and_grad(*inputs(mi, ray), settings(max_candidates=0.)), dr, np)
    assert not diagnostics.intersection(default), "Diagnostics must remain opt-in"
    return {"check": "opt_in_counters_confirm_empty_space_skips",
            "dda_steps": int(baseline["traversal_steps"][0]),
            "octree_steps": int(octree["traversal_steps"][0]),
            "skipped_cells": int(octree["traversal_skipped_cells"][0]), "status": "passed"}


def rational_leaf_reference(checkpoint, origin, direction, np):
    """Independent exact face ordering and float64 segment/ellipsoid test.

    This small oracle supports this module's identity-rotation, isotropic,
    constant-opacity fixtures. Fraction times prevent the legacy repeated-add
    drift from deciding whether a near-tangent shell voxel is hit.
    """
    resolution = checkpoint["model_metadata"]["resolution"]
    values = checkpoint["model_state"]
    ids = {int(cell): i for i, cell in enumerate(values["cell_indices"])}
    centers = values["centers"].reshape(-1, 3).astype(np.float64)
    scales = np.exp(values["log_scales"].reshape(-1, 3).astype(np.float64))
    np.testing.assert_array_equal(values["rotations"].reshape(-1, 4),
                                  np.tile([1., 0., 0., 0.], (len(ids), 1)))
    colors = values["radiance_sh"].reshape(-1, 3).astype(np.float64) * Y00
    opacities = 1. / (1. + np.exp(-values["opacity_sh"].astype(np.float64) * Y00))
    o = [Fraction(float(v)) for v in origin]
    d = [Fraction(float(v)) for v in direction]
    near, far = [Fraction(0)], [Fraction(400)]
    for a in range(3):
        if d[a]:
            pair = [-o[a]/d[a], (resolution-o[a])/d[a]]
            near.append(min(pair))
            far.append(max(pair))
        else:
            assert 0 <= o[a] <= resolution
    entry, exit_time = max(near), min(far)
    assert entry <= exit_time
    grid_entry = entry
    cell = [min(resolution-1, max(0, math.floor(o[a]+entry*d[a]))) for a in range(3)]
    steps = [1 if v >= 0 else -1 for v in d]
    color, transmittance = np.zeros(3), 1.
    hit_ids, min_gains = [], []
    mass, moment, distortion = 0., 0., 0.
    direction_length = float(np.linalg.norm(direction))
    for _ in range(3*resolution+3):
        next_times = [(cell[a]+int(steps[a] > 0)-o[a])/d[a]
                      if d[a] else exit_time for a in range(3)]
        end = min(*next_times, exit_time)
        index = ids.get(cell[0]+resolution*(cell[1]+resolution*cell[2]))
        if index is not None:
            relative = np.array([float(o[a]+entry*d[a])-cell[a]-centers[index, a]
                                 for a in range(3)])
            segment = np.array([float((end-entry)*d[a]) for a in range(3)])
            u, v = relative/scales[index], segment/scales[index]
            square = float(v@v)
            star = float(np.clip(-(u@v)/square, 0., 1.)) if square > 1e-20 else 0.
            gmin = float((u+star*v)@(u+star*v)-1.)
            if gmin <= 0.:
                hit_ids.append(index)
                min_gains.append(gmin)
                weight = transmittance*opacities[index]
                depth = (float(entry-grid_entry)+star*float(end-entry))*direction_length/(math.sqrt(3.)*resolution)
                distortion += 2.*weight*(depth*mass-moment)
                mass += weight
                moment += weight*depth
                color += weight*colors[index]
                transmittance *= 1.-opacities[index]
        axis = min(range(3), key=lambda a: next_times[a])
        entry = next_times[axis]
        cell[axis] += steps[axis]
        if entry >= exit_time or min(cell) < 0 or max(cell) >= resolution:
            break
    else:
        raise AssertionError("Exact traversal exceeded its face crossing bound")
    raw = color+transmittance
    rgb = np.clip(raw, 0., 1.)
    rgb_loss = float(np.sum((rgb-[.1, .25, .35])**2))
    return {"hit_count": len(hit_ids), "rgb": rgb, "raw_rgb": raw,
            "alpha": 1.-transmittance, "raw_alpha": 1.-transmittance,
            "rgb_loss": rgb_loss, "distortion_loss": distortion,
            "loss": rgb_loss+.1*distortion}, hit_ids, min_gains


def assert_oracle_pixel(actual, index, expected, np):
    for key, value in expected.items():
        observed = actual[key][..., index]
        if key == "hit_count":
            np.testing.assert_array_equal(observed, value)
        else:
            np.testing.assert_allclose(observed, value, rtol=3e-5, atol=3e-6,
                                       err_msg=f"rational_oracle_pixel_{index}:{key}")


def check_far_tangent_reference(mi, dr, np):
    checkpoint = payload(np, 128, [(54, 34, 38), (54, 34, 39),
                                   (52, 28, 77), (52, 28, 78), (52, 28, 79)])
    checkpoint["model_state"]["log_scales"][:] = math.log(.65)
    origin, direction = [64., 64., -140.], [-10.546875, -33.515625, 204.]
    expected, ids, distances = rational_leaf_reference(checkpoint, origin, direction, np)
    assert len(ids) == 5 and -1e-4 < distances[-1] < -1e-5
    model = restore_model(checkpoint, mi)
    ray = rays(mi, np, [origin], [direction], 400.)
    result = host(model.loss_and_grad(*inputs(mi, ray), settings(use_octree=1.)), dr, np)
    assert_oracle_pixel(result, 0, expected, np)
    return {"check": "far_near_tangent_hit_against_exact_face_order_oracle",
            "hit_count": expected["hit_count"], "last_hit_squared_ellipsoid_distance": distances[-1],
            "status": "passed"}


def benchmark_sparse_shell(mi, dr, np, image_size=64):
    """Descriptive warm-replay timings, not a performance pass/fail criterion."""
    resolution = 128
    z, y, x = np.indices((resolution,) * 3, dtype=np.float32)
    radius = np.sqrt((x+.5-64.)**2 + (y+.5-64.)**2 + (z+.5-64.)**2)
    occupied = np.abs(radius-40.) < .6
    zz, yy, xx = np.nonzero(occupied)
    coordinates = list(zip(xx.tolist(), yy.tolist(), zz.tolist()))
    checkpoint = payload(np, resolution, coordinates)
    checkpoint["model_state"]["log_scales"][:] = math.log(.65)
    model = restore_model(checkpoint, mi)
    px, py = np.meshgrid(np.arange(image_size, dtype=np.float32),
                         np.arange(image_size, dtype=np.float32))
    targets = np.column_stack(((px.ravel()+.5)*120./image_size+4.,
                               (py.ravel()+.5)*120./image_size+4.,
                               np.full(image_size**2, 64., np.float32)))
    origin = np.array([64., 64., -140.], np.float32)
    ray = rays(mi, np, [origin], targets-origin, 400.)
    args = inputs(mi, ray)
    options = settings(max_candidates=16., collect_rgb_deletion_evidence=0.)
    replay = NativeLossReplay(dr)
    # Validate this larger fixture as well before timing any mode.
    dda = host(replay(model, *args, options | {"use_octree": 0.}), dr, np)
    octree = host(replay(model, *args, options | {"use_octree": 1.}), dr, np)
    differing_hits = np.flatnonzero(dda["hit_count"] != octree["hit_count"])
    # At long near-tangent segments, the old accumulating DDA can miss a real
    # hit. Such pixels must satisfy an independent exact face-order oracle,
    # not be accepted by a wider image tolerance.
    stable = np.ones(image_size**2, dtype=bool)
    stable[differing_hits] = False
    pixel_keys = ("rgb", "alpha", "raw_rgb", "raw_alpha", "hit_count",
                  "loss", "rgb_loss", "distortion_loss")
    error = compare({key: dda[key][..., stable] for key in pixel_keys},
                    {key: octree[key][..., stable] for key in pixel_keys},
                    np, "benchmark_unchanged_hit_pixels")
    for index in differing_hits:
        expected, _, _ = rational_leaf_reference(checkpoint, origin, targets[index]-origin, np)
        assert_oracle_pixel(octree, index, expected, np)
    # This symmetric shell also puts rays exactly on cell edges. Legacy drift
    # can choose the other zero-length side cell, changing near-miss gradients
    # even when RGB and physical hit counts agree. Report those discrete changes
    # explicitly; strict gradient regressions use the dedicated fixtures above.
    assert all(np.isfinite(value).all() for value in octree.values())
    gradient_differences = {
        key.removeprefix("gradients."): float(np.max(np.abs(dda[key]-value)))
        for key, value in octree.items() if key.startswith("gradients.") and value.size}
    counter_differences = {
        key: int(np.count_nonzero(dda[key] != value)) for key, value in octree.items()
        if key not in pixel_keys and np.issubdtype(value.dtype, np.integer)
        and np.any(dda[key] != value)}
    assert np.count_nonzero(dda["hit_count"]) > image_size**2//5
    timings = {"dda": [], "octree": []}
    # Alternate order across rounds to reduce thermal/order bias. Warm each
    # recorded mode before collecting synchronized completion times.
    for iteration in range(8):
        for mode in (("dda", "octree") if iteration % 2 == 0 else ("octree", "dda")):
            current_options = options | {"use_octree": float(mode == "octree")}
            start = time.perf_counter()
            result = replay(model, *args, current_options)
            dr.eval(result)
            dr.sync_thread()
            elapsed_ms = (time.perf_counter()-start)*1000.
            if iteration >= 3:
                timings[mode].append(elapsed_ms)
    medians = {mode: float(np.median(samples)) for mode, samples in timings.items()}
    return {"check": "generated_sparse_shell_warm_replay_benchmark", "status": "measured",
            "resolution": resolution, "active_voxels": len(coordinates),
            "width": image_size, "height": image_size, "spp": 1,
            "max_candidates": 16, "distortion_weight": .1,
            "wall_ms_samples": timings, "wall_ms_median": medians,
            "dda_over_octree_wall_ratio": medians["dda"]/medians["octree"],
            "legacy_hit_disagreements_checked_by_rational_oracle": differing_hits.tolist(),
            "legacy_gradient_max_absolute_differences": gradient_differences,
            "legacy_counter_differing_entries": counter_differences,
            "max_absolute_error": error, "replay_statistics": replay.statistics(),
            "comparison_scope": "Per-ray RGB, alpha, hit counts and losses; gradient and counter differences are reported separately.",
            "note": "Synthetic scene only; includes forward, gradients and synchronized dispatch, not full training."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true", help="Check configuration without loading CUDA")
    parser.add_argument("--benchmark", action="store_true", help="Also time a generated sparse shell with warm replay")
    parser.add_argument("--benchmark-size", type=int, default=64,
                        help="Square benchmark image size (default: 64)")
    args = parser.parse_args()
    if not 1 <= args.benchmark_size <= 512:
        parser.error("--benchmark-size must be between 1 and 512")
    checks = [check_configuration_cpu()]
    print(json.dumps(checks[-1]), flush=True)
    if args.cpu_only:
        print(json.dumps({"status": "passed", "checks": checks}), flush=True)
        return 0
    with tempfile.TemporaryDirectory(prefix="mitsuba-octree-regression-") as temporary:
        mi, dr, np, handles = initialize_runtime(RuntimeConfig(args.build_dir, args.gpu),
                                                Path(temporary), {})
        assert int(getattr(mi.VoxelReconstruction, "octree_version", 0)) >= 1, \
            "Rebuild this checkout before running octree regression"
        for check in (check_compact_storage_scaling, check_core_1024_shell_storage,
                      check_compact_candidate_lifecycle,
                      check_compact_invalid_cells,
                      check_boundaries_and_default, check_sparse_random_and_non_power_two,
                      check_long_empty_block_exits,
                      check_far_tangent_reference,
                      check_near_miss, check_growth_and_truncation, check_replay_mutations,
                      check_candidate_partition_invariance,
                      check_traversal_diagnostics):
            checks.append(check(mi, dr, np))
            print(json.dumps(checks[-1]), flush=True)
        if args.benchmark:
            checks.append(benchmark_sparse_shell(mi, dr, np, args.benchmark_size))
            print(json.dumps(checks[-1]), flush=True)
    print(json.dumps({"status": "passed", "checks": checks}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
