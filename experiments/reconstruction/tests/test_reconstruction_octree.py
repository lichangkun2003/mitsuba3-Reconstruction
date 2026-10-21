"""Generated CUDA regression for octree traversal against the original DDA.

The reference is the explicitly selected original leaf-by-leaf traversal, on
identical model buffers and rays. Checks include per-ray outputs, all parameter
gradients, growth/deletion evidence, and frozen replay across topology changes.
No dataset, existing experiment, or long training run is used.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
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
        _, error = compare_loss(empty, ray, mi, dr, np, f"empty_resolution_{resolution}")
        maximum = max(maximum, error)
    return {"check": "sparse_random_single_cell_non_power_two_and_empty_models",
            "max_absolute_error": maximum, "status": "passed"}


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
    error = compare(dda, octree, np, "benchmark_shell_equivalence")
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
            "max_absolute_error": error, "replay_statistics": replay.statistics(),
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
        for check in (check_boundaries_and_default, check_sparse_random_and_non_power_two,
                      check_near_miss, check_growth_and_truncation, check_replay_mutations,
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
