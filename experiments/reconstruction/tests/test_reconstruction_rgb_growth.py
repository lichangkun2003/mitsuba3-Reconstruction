"""Generated CUDA checks for online, single-traversal RGB growth evidence.

The separately rendered candidate-inserted models below are test oracles only:
the training evidence implementation must not perform those extra renders.
No dataset, existing output, or long training run is used.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preview import RuntimeConfig
from reconstruction_common import initialize_runtime, restore_model
from reconstruction_growth_rgb import initialize_statistics, observe_numpy, RGBGrowthEvidenceDevice
from reconstruction_replay import NativeLossReplay
from reconstruction_topology import CoarseToFineConfig, TopologyManager, TopologySettings


Y00 = 1. / (2. * math.sqrt(math.pi))


def state(np, cells, colors=None, opacities=None):
    count = len(cells)
    colors = [[.2, .4, .6]] * count if colors is None else colors
    opacities = [.35] * count if opacities is None else opacities
    return {
        "cell_indices": np.asarray(cells, dtype=np.uint32),
        "centers": np.tile([.5, .5, .5], count).astype(np.float32),
        "log_scales": np.full(count * 3, math.log(.4), dtype=np.float32),
        "rotations": np.tile([1., 0., 0., 0.], count).astype(np.float32),
        "radiance_sh": np.asarray(colors, dtype=np.float32).ravel() / Y00,
        "opacity_sh": np.asarray([math.log(a / (1. - a)) / Y00 for a in opacities],
                                  dtype=np.float32),
    }


def payload(np, cells=(16, 48)):
    return {"model_metadata": {"resolution": 4, "grid_min": [0.] * 3,
                               "grid_max": [4.] * 3, "sh_count": 1,
                               "opacity_sh_count": 1},
            "model_state": state(np, cells)}


def device_state(values, mi):
    return {name: (mi.UInt32(value) if name == "cell_indices" else mi.Float(value))
            for name, value in values.items()}


def ray_at(mi, x=.5):
    ray = mi.Ray3f(mi.Point3f(x, .5, -1.), mi.Vector3f(0., 0., 1.))
    ray.maxt = 8.
    return ray


def evaluate(model, ray, target, mi, dr, *, enabled=True, alpha=0., weight=1., **settings):
    options = {"rgb_only": 1., "collect_growth_evidence": float(enabled),
               "max_growth_hits": 8., "max_hits": 16., "transmittance_threshold": .0001}
    options.update(settings)
    result = model.loss_and_grad(ray, mi.Color3f(target), mi.Float(alpha),
                                 mi.Color3f(1.), mi.Float(weight), options)
    dr.eval(result)
    dr.sync_thread()
    return result


def rendered(model, ray, mi, dr, np):
    result = model.trace(ray, mi.Color3f(1.), .0001, 16)
    dr.eval(result)
    dr.sync_thread()
    return np.asarray(result[0], dtype=np.float64).reshape(3, -1)


def inserted(base, proposals, index, np, mi):
    checkpoint = copy.deepcopy(base)
    dimensions = {"cell_indices": 1, "centers": 3, "log_scales": 3,
                  "rotations": 4, "radiance_sh": 3, "opacity_sh": 1}
    for name, dimension in dimensions.items():
        selected = proposals[name][index * dimension:(index + 1) * dimension]
        checkpoint["model_state"][name] = np.concatenate((checkpoint["model_state"][name], selected))
    return restore_model(checkpoint, mi)


def assert_baseline_equal(first, second, np):
    for name in ("rgb", "raw_rgb", "alpha", "raw_alpha", "hit_count", "loss",
                 "appearance_valid", "geometry_valid", "background_carve_valid",
                 "foreground_evidence", "background_evidence"):
        np.testing.assert_array_equal(np.asarray(first[name]), np.asarray(second[name]), err_msg=name)
    for name, values in first["gradients"].items():
        np.testing.assert_array_equal(np.asarray(values), np.asarray(second["gradients"][name]),
                                      err_msg=f"gradients.{name}")


def check_counterfactual_and_invariance(mi, dr, np):
    base = payload(np)
    model = restore_model(base, mi)
    proposals = state(np, [0, 32], [[.1, .8, .2], [.7, .15, .3]], [.45, .55])
    ray = ray_at(mi)
    target = np.asarray([.3, .6, .4], dtype=np.float64)
    baseline = evaluate(model, ray, target.tolist(), mi, dr, enabled=False)
    original_trace = rendered(model, ray, mi, dr, np)
    model.set_growth_candidates(device_state(proposals, mi))
    assert int(model.growth_candidate_count) == 2 and int(model.active_voxel_count) == 2
    result = evaluate(model, ray, target.tolist(), mi, dr)
    assert_baseline_equal(baseline, result, np)
    assert_baseline_equal(baseline, evaluate(model, ray, target.tolist(), mi, dr, enabled=False), np)
    np.testing.assert_array_equal(rendered(model, ray, mi, dr, np), original_trace)
    expected = []
    baseline_loss = float(np.sum((original_trace[:, 0] - target) ** 2))
    for index in range(2):
        rgb = rendered(inserted(base, proposals, index, np, mi), ray, mi, dr, np)[:, 0]
        expected.append(baseline_loss - float(np.sum((rgb - target) ** 2)))
    np.testing.assert_allclose(np.asarray(result["growth_gain"]), expected, atol=3e-7, rtol=3e-5)
    np.testing.assert_array_equal(np.asarray(result["growth_observed"]), [1, 1])
    np.testing.assert_array_equal(np.asarray(result["growth_unknown"]), [0, 0])
    for alpha in (0., .5, 1.):
        alternate = evaluate(model, ray, target.tolist(), mi, dr, alpha=alpha)
        assert_baseline_equal(result, alternate, np)
        for name in ("growth_gain", "growth_observed", "growth_unknown"):
            np.testing.assert_array_equal(np.asarray(result[name]), np.asarray(alternate[name]), err_msg=name)
    model.clear_growth_candidates()
    assert int(model.growth_candidate_count) == 0
    np.testing.assert_array_equal(rendered(model, ray, mi, dr, np), original_trace)
    assert_baseline_equal(baseline, evaluate(model, ray, target.tolist(), mi, dr), np)
    return {"virtual_insertion_matches_separate_render_oracle": "passed",
            "prefix_and_nonwhite_suffix": "passed", "baseline_trace_and_gradients_unchanged": "passed",
            "growth_ignores_reference_alpha": "passed"}


def check_positive_negative_neutral(mi, dr, np):
    base = payload(np, [16])
    model = restore_model(base, mi)
    proposals = state(np, [32], [[.1, .2, .3]], [.4])
    ray = ray_at(mi)
    before = rendered(model, ray, mi, dr, np)[:, 0]
    after = rendered(inserted(base, proposals, 0, np, mi), ray, mi, dr, np)[:, 0]
    model.set_growth_candidates(device_state(proposals, mi))
    positive = evaluate(model, ray, after.tolist(), mi, dr)
    negative = evaluate(model, ray, before.tolist(), mi, dr)
    assert float(np.asarray(positive["growth_gain"])[0]) > 1e-4
    assert float(np.asarray(negative["growth_gain"])[0]) < -1e-4
    neutral = state(np, [32], [[1., 1., 1.]], [.4])
    model.set_growth_candidates(device_state(neutral, mi))
    result = evaluate(model, ray, [0., 0., 0.], mi, dr)
    np.testing.assert_allclose(np.asarray(result["growth_gain"]), 0., atol=2e-7)
    np.testing.assert_array_equal(np.asarray(result["growth_observed"]), 1)
    miss = evaluate(model, ray_at(mi, 1.5), [0., 0., 0.], mi, dr)
    for name in ("growth_gain", "growth_observed", "growth_unknown"):
        np.testing.assert_array_equal(np.asarray(miss[name]), 0, err_msg=name)
    return {"helpful_harmful_and_color_matched_neutral_candidates": "passed",
            "unobserved_ray_is_neutral": "passed"}


def check_empty_baseline(mi, dr, np):
    base = payload(np, [])
    model = restore_model(base, mi)
    proposals = state(np, [0])
    model.set_growth_candidates(device_state(proposals, mi))
    target = [.2, .3, .4]
    result = evaluate(model, ray_at(mi), target, mi, dr)
    rgb = rendered(inserted(base, proposals, 0, np, mi), ray_at(mi), mi, dr, np)[:, 0]
    expected = np.sum((1. - np.asarray(target)) ** 2) - np.sum((rgb - target) ** 2)
    np.testing.assert_allclose(np.asarray(result["growth_gain"]), expected, atol=4e-7, rtol=3e-6)
    np.testing.assert_array_equal(np.asarray(result["growth_observed"]), 1)
    np.testing.assert_array_equal(np.asarray(result["hit_count"]), 0)
    assert int(model.active_voxel_count) == 0
    return {"hypothetical_candidate_in_empty_native_model": "passed"}


def check_same_cell_proposals(mi, dr, np):
    """Each parent shape is a distinct counterfactual, not a pooled voxel."""
    base = payload(np)
    model = restore_model(base, mi)
    # More sibling shapes than the default record budget, plus two proposals in
    # a later cell. Candidate IDs deliberately interleave the two cell groups.
    cells = [0, 32, 0, 0, 32] + [0] * 9
    colors = [[.05 + .01 * i, .12 + .005 * i, .2 + .01 * i]
              for i in range(len(cells))]
    colors[2] = [4., 4., 4.]  # Reliable but harmful sibling in the SAME cell.
    proposals = state(np, cells, colors, [.35] * len(cells))
    proposals["centers"][:3] = [.85, .85, .5]
    proposals["log_scales"][:3] = math.log(.08)  # First sibling misses the ray.
    # A different parent's shape intersects and explains the target exactly.
    proposals["log_scales"][9:12] = np.log([.25, .3, .35])
    ray = ray_at(mi)
    target = rendered(inserted(base, proposals, 3, np, mi), ray, mi, dr, np)[:, 0]
    baseline_rgb = rendered(model, ray, mi, dr, np)[:, 0]
    baseline = evaluate(model, ray, target.tolist(), mi, dr, enabled=False)
    baseline_loss = float(np.sum((baseline_rgb - target) ** 2))
    model.set_growth_candidates(device_state(proposals, mi))
    self_count = len(cells)
    assert int(model.growth_candidate_count) == self_count
    complete = evaluate(model, ray, target.tolist(), mi, dr, max_growth_hits=2.)
    expected = []
    for index in range(self_count):
        inserted_rgb = rendered(inserted(base, proposals, index, np, mi), ray, mi, dr, np)[:, 0]
        expected.append(baseline_loss - float(np.sum((inserted_rgb - target) ** 2)))
    np.testing.assert_allclose(np.asarray(complete["growth_gain"]), expected,
                               atol=1e-6, rtol=3e-5)
    expected_observed = np.ones(self_count, np.uint32)
    expected_observed[0] = 0
    np.testing.assert_array_equal(np.asarray(complete["growth_observed"]), expected_observed)
    np.testing.assert_array_equal(np.asarray(complete["growth_unknown"]), 0)
    assert float(np.asarray(complete["growth_gain"])[2]) < 0.
    assert float(np.asarray(complete["growth_gain"])[3]) > 0.
    assert_baseline_equal(baseline, complete, np)
    np.testing.assert_array_equal(rendered(model, ray, mi, dr, np)[:, 0], baseline_rgb)

    # A cell is recorded once; its many sibling proposals cannot consume slots
    # or silently hide the better parent. Only the farther CELL overflows.
    limited = evaluate(model, ray, target.tolist(), mi, dr, max_growth_hits=1.)
    near = np.asarray(cells) == 0
    far = ~near
    observed = np.asarray(limited["growth_observed"])
    np.testing.assert_array_equal(observed[near], expected_observed[near])
    assert int(observed[near].sum()) > 8
    np.testing.assert_array_equal(observed[far], 0)
    np.testing.assert_array_equal(np.asarray(limited["growth_unknown"])[near], 0)
    np.testing.assert_array_equal(np.asarray(limited["growth_unknown"])[far], 1)
    np.testing.assert_allclose(np.asarray(limited["growth_gain"])[near],
                               np.asarray(complete["growth_gain"])[near],
                               atol=1e-6, rtol=3e-5)
    np.testing.assert_array_equal(np.asarray(limited["growth_gain"])[far], 0.)
    assert_baseline_equal(complete, limited, np)

    # Stats stay proposal-indexed through ray chunks/shards, including a miss.
    xs = [.5, .55, 1.5]
    combined = evaluate(model, ray_at(mi, mi.Float(xs)), target.tolist(), mi, dr,
                        max_growth_hits=1.)
    parts = [evaluate(model, ray_at(mi, x), target.tolist(), mi, dr, max_growth_hits=1.)
             for x in xs]
    for name in ("growth_gain", "growth_observed", "growth_unknown"):
        np.testing.assert_allclose(np.asarray(combined[name]),
                                   sum(np.asarray(part[name]) for part in parts),
                                   atol=2e-6, rtol=3e-5, err_msg=name)

    # Invalid replacement must preserve the complete linked grouping/evidence.
    malformed = copy.deepcopy(proposals)
    malformed["cell_indices"][3] = 16  # Occupied model cell.
    try:
        model.set_growth_candidates(device_state(malformed, mi))
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("Occupied same-cell proposal was accepted")
    after = evaluate(model, ray, target.tolist(), mi, dr, max_growth_hits=1.)
    assert int(model.growth_candidate_count) == self_count
    for name in ("growth_gain", "growth_observed", "growth_unknown"):
        np.testing.assert_array_equal(np.asarray(after[name]), np.asarray(limited[name]))
    model.clear_growth_candidates()
    assert int(model.growth_candidate_count) == 0
    assert_baseline_equal(baseline, evaluate(model, ray, target.tolist(), mi, dr), np)
    return {"same_cell_shapes_match_independent_insertion_oracles": "passed",
            "missed_and_harmful_first_parents_do_not_hide_beneficial_sibling": "passed",
            "more_than_eight_siblings_share_one_cell_record_without_truncation": "passed",
            "interleaved_proposal_ids_and_chunk_reduction_remain_independent": "passed",
            "invalid_duplicate_overlay_replacement_preserves_all_evidence": "passed"}


def check_partial_paths(mi, dr, np):
    model = restore_model(payload(np), mi)
    proposals = state(np, [0, 32])
    model.set_growth_candidates(device_state(proposals, mi))
    for options in ({"max_hits": 1.}, {"transmittance_threshold": .8}):
        result = evaluate(model, ray_at(mi), [.1, .2, .3], mi, dr, **options)
        np.testing.assert_array_equal(np.asarray(result["growth_observed"]), 0)
        np.testing.assert_array_equal(np.asarray(result["growth_gain"]), 0.)
        assert int(np.sum(np.asarray(result["growth_unknown"]))) > 0, options
    complete = evaluate(model, ray_at(mi), [.1, .2, .3], mi, dr, max_growth_hits=2.)
    limited = evaluate(model, ray_at(mi), [.1, .2, .3], mi, dr, max_growth_hits=1.)
    np.testing.assert_array_equal(np.asarray(limited["growth_observed"]), [1, 0])
    np.testing.assert_array_equal(np.asarray(limited["growth_unknown"]), [0, 1])
    np.testing.assert_allclose(np.asarray(limited["growth_gain"])[0],
                               np.asarray(complete["growth_gain"])[0], atol=2e-7, rtol=3e-6)
    np.testing.assert_array_equal(np.asarray(limited["growth_gain"])[1], 0.)
    assert_baseline_equal(complete, limited, np)
    assert_baseline_equal(complete, evaluate(model, ray_at(mi), [.1, .2, .3], mi, dr, enabled=False), np)

    # The second ray starts beyond candidate 0: candidate 32 has a retained
    # record for this ray, despite being dropped from the first ray's record.
    mixed_ray = mi.Ray3f(mi.Point3f(.5, .5, mi.Float([-1., 1.5])), mi.Vector3f(0., 0., 1.))
    mixed_ray.maxt = 8.
    mixed = evaluate(model, mixed_ray, [.1, .2, .3], mi, dr, max_growth_hits=1.)
    known_ray = mi.Ray3f(mi.Point3f(.5, .5, 1.5), mi.Vector3f(0., 0., 1.))
    known_ray.maxt = 8.
    known = evaluate(model, known_ray, [.1, .2, .3], mi, dr, max_growth_hits=1.)
    np.testing.assert_array_equal(np.asarray(mixed["growth_observed"]), [1, 1])
    np.testing.assert_array_equal(np.asarray(mixed["growth_unknown"]), [0, 1])
    np.testing.assert_allclose(np.asarray(mixed["growth_gain"])[1],
                               np.asarray(known["growth_gain"])[1], atol=2e-7, rtol=3e-6)
    return {"hit_cap_and_early_stop_still_abstain": "passed",
            "candidate_overflow_only_marks_dropped_record_unknown": "passed",
            "retained_gain_survives_other_unknown_rays": "passed"}


def check_shards_and_candidate_validation(mi, dr, np):
    model = restore_model(payload(np), mi)
    proposals = state(np, [0, 32])
    model.set_growth_candidates(device_state(proposals, mi))
    xs, weights = [.45, .55, 1.5], [1., 1., 1.]
    combined = evaluate(model, ray_at(mi, mi.Float(xs)), [.2, .3, .4], mi, dr, weight=weights)
    partials = [evaluate(model, ray_at(mi, x), [.2, .3, .4], mi, dr, weight=w)
                for x, w in zip(xs, weights)]
    for name in ("growth_gain", "growth_observed", "growth_unknown"):
        expected = sum(np.asarray(value[name]) for value in partials)
        np.testing.assert_allclose(np.asarray(combined[name]), expected, atol=2e-7, rtol=3e-6)
    assert np.any(np.abs(np.asarray(combined["growth_gain"])) > 1e-6)
    weighted = evaluate(model, ray_at(mi), [.2, .3, .4], mi, dr, weight=2.)
    np.testing.assert_array_equal(np.asarray(weighted["growth_observed"]), 0)
    np.testing.assert_array_equal(np.asarray(weighted["growth_unknown"]), 1)
    original = rendered(model, ray_at(mi), mi, dr, np)
    malformed = []
    for cells in ([16], [64]):
        malformed.append(state(np, cells))
    wrong_size = state(np, [0])
    wrong_size["centers"] = wrong_size["centers"][:2]
    malformed.append(wrong_size)
    nonfinite = state(np, [0])
    nonfinite["opacity_sh"][0] = np.nan
    malformed.append(nonfinite)
    for candidate in malformed:
        try:
            model.set_growth_candidates(device_state(candidate, mi))
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError("An invalid growth overlay was accepted")
        assert int(model.growth_candidate_count) == 2
        np.testing.assert_array_equal(rendered(model, ray_at(mi), mi, dr, np), original)
    model.clear_growth_candidates()
    model.clear_growth_candidates()
    assert int(model.growth_candidate_count) == 0
    return {"pixel_shards_sum_before_view_voting": "passed",
            "nonuniform_sample_weights_abstain": "passed",
            "invalid_candidate_setter_preserves_model_and_overlay": "passed"}


def check_warmup():
    settings = TopologySettings(enable_growth=True, enable_deletion=False,
                                use_growth_evidence=False, use_rgb_growth=True)
    schedule = CoarseToFineConfig(start_resolution=4, target_resolution=8, total_epochs=200)
    manager = TopologyManager(SimpleNamespace(resolution=4, active_voxel_count=2), 4, settings, schedule)
    assert settings.growth_warmup_epochs == settings.growth_fine_warmup_epochs == 20
    assert not any(manager._growth_due(boundary) for boundary in range(1, 30))
    assert manager._growth_due(30) and manager._growth_due(40)
    assert not manager.needs_rendered_alpha(29)
    assert not manager._growth_due(manager.level_budget), "No growth on the refinement boundary"
    manager.resolution = manager.core_resolution = 8
    manager.level_start_epoch = manager.budgets[4]
    for local in range(1, 30):
        assert not manager._growth_due(manager.level_start_epoch + local)
    assert manager._growth_due(manager.level_start_epoch + 30)
    assert not manager._growth_due(manager.level_start_epoch + manager.level_budget)
    manager.settings.enable_growth = False
    assert not manager._growth_due(manager.level_start_epoch + 30)
    return {"coarse_and_fine_20_epoch_warmup_first_event_30": "passed",
            "refine_and_final_boundaries_do_not_grow": "passed"}


def vote_fixture(np):
    pending = {"cells": np.arange(9)}
    initialize_statistics(pending, np)
    # Known positive/negative, mixed known+unknown positive, no hit, invalid,
    # known zero, mixed negative, mixed zero, and unknown-only. Unknown rays
    # never cancel reliable positive, negative or zero observations.
    out = {"growth_gain": np.asarray([4., -3., 20., 0., np.nan, 0., -6., 0., 0.], np.float32),
           "growth_observed": np.asarray([2, 5, 4, 0, 1, 3, 2, 4, 0], np.uint32),
           "growth_unknown": np.asarray([0, 0, 1, 0, 0, 0, 3, 2, 5], np.uint32)}
    return pending, out


def check_view_votes_numpy():
    import numpy as np
    pending, out = vote_fixture(np)
    padded = copy.deepcopy(pending)
    observe_numpy(pending, out, 100, .01, np, normalization="observed")
    observe_numpy(padded, out, 640000, .01, np, normalization="observed")
    np.testing.assert_array_equal(pending["support_views"], [1, 0, 1, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(pending["conflict_views"], [0, 1, 0, 0, 0, 0, 1, 0, 0])
    np.testing.assert_array_equal(pending["observed_views"], [1, 1, 1, 0, 0, 1, 1, 1, 0])
    np.testing.assert_array_equal(pending["unknown_views"], [0, 0, 1, 0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(pending["invalid_views"], [0, 0, 0, 0, 1, 0, 0, 0, 0])
    np.testing.assert_allclose(pending["gain_sum"], [2., -.6, 5., 0., 0., 0., -3., 0., 0.], atol=3e-8)
    for name, values in pending.items():
        np.testing.assert_array_equal(values, padded[name], err_msg=name)
    legacy, _ = vote_fixture(np)
    observe_numpy(legacy, out, 100, .01, np, normalization="full_image")
    np.testing.assert_allclose(legacy["gain_sum"], [.04, -.03, .2, 0., 0., 0., -.06, 0., 0.], atol=1e-8)
    return {"observed_ray_normalization_is_independent_of_unrelated_pixels": "passed",
            "signed_and_zero_gains_keep_all_valid_observations": "passed",
            "mixed_known_unknown_views_keep_positive_negative_and_zero_evidence": "passed",
            "unknown_only_nonfinite_and_unobserved_views_abstain": "passed",
            "legacy_full_image_normalization_retained": "passed"}


def check_unequal_shard_votes_numpy():
    import numpy as np
    pending = {"cells": np.arange(1)}
    initialize_statistics(pending, np)
    # One shard contains two ray gains [4, -1]; the other contains [0, 0, 0].
    # Normalize only AFTER SUM: (4 - 1 + 0 + 0 + 0) / 5 = 0.6, not the
    # unweighted mean of per-shard averages (1.5 + 0) / 2 = 0.75.
    partials = [
        {"growth_gain": np.asarray([3.], np.float32),
         "growth_observed": np.asarray([2], np.uint32),
         "growth_unknown": np.asarray([0], np.uint32)},
        {"growth_gain": np.asarray([0.], np.float32),
         "growth_observed": np.asarray([3], np.uint32),
         "growth_unknown": np.asarray([0], np.uint32)},
    ]
    combined = {name: sum(item[name] for item in partials) for name in partials[0]}
    observe_numpy(pending, combined, 640000, .7, np, normalization="observed")
    np.testing.assert_allclose(pending["gain_sum"], [.6], atol=3e-8)
    np.testing.assert_array_equal(pending["observed_views"], [1])
    np.testing.assert_array_equal(pending["observed_rays"], [5])
    np.testing.assert_array_equal(pending["support_views"], [0])
    # Unknown evidence in either shard must not erase the reliable rays.
    combined["growth_unknown"][0] = 1
    observe_numpy(pending, combined, 640000, .7, np, normalization="observed")
    np.testing.assert_allclose(pending["gain_sum"], [1.2], atol=6e-8)
    np.testing.assert_array_equal(pending["observed_views"], [2])
    np.testing.assert_array_equal(pending["observed_rays"], [10])
    np.testing.assert_array_equal(pending["unknown_views"], [1])
    return {"unequal_shards_sum_raw_gain_and_all_valid_ray_counts_before_normalizing": "passed"}


def check_view_votes_device(mi, dr, np):
    for normalization in ("observed", "full_image"):
        pending, out = vote_fixture(np)
        expected = copy.deepcopy(pending)
        gpu = RGBGrowthEvidenceDevice(pending, mi.Float, mi.UInt32, dr, np)
        device_out = {name: (mi.Float(value) if name == "growth_gain" else mi.UInt32(value))
                      for name, value in out.items()}
        for pixel_count in (100, 640000):
            observe_numpy(expected, out, pixel_count, .01, np, normalization=normalization)
            gpu.observe(device_out, pixel_count, .01, normalization=normalization)
        actual = gpu.materialize()
        for name, values in actual.items():
            np.testing.assert_allclose(values, expected[name], atol=1e-8, rtol=2e-6, err_msg=name)
    pending = {"cells": np.arange(1)}
    initialize_statistics(pending, np)
    gpu = RGBGrowthEvidenceDevice(pending, mi.Float, mi.UInt32, dr, np)
    combined = {"growth_gain": mi.Float(3.) + mi.Float(0.),
                "growth_observed": mi.UInt32(2) + mi.UInt32(3),
                "growth_unknown": mi.UInt32(0) + mi.UInt32(7)}
    gpu.observe(combined, 640000, .7, normalization="observed")
    actual = gpu.materialize()
    np.testing.assert_allclose(actual["gain_sum"], [.6], atol=3e-8)
    np.testing.assert_array_equal(actual["observed_views"], [1])
    np.testing.assert_array_equal(actual["unknown_views"], [1])
    np.testing.assert_array_equal(actual["support_views"], [0])
    return {"GPU_local_and_legacy_vote_accumulation_match_CPU_reference": "passed",
            "GPU_unequal_shard_counts_normalize_once": "passed"}


def check_native_replay(mi, dr, np):
    model = restore_model(payload(np), mi)
    replay = NativeLossReplay(dr)
    options = {"rgb_only": 1., "collect_growth_evidence": 1., "max_growth_hits": 8.,
               "max_hits": 16., "transmittance_threshold": .0001}
    ray, target = ray_at(mi), mi.Color3f(.2, .3, .4)

    def compare_current(*, request_empty=False):
        options["collect_growth_evidence"] = float(request_empty or model.growth_candidate_count > 0)
        arguments = (ray, target, mi.Float(0.), mi.Color3f(1.), mi.Float(1.), options)
        direct = model.loss_and_grad(*arguments)
        dr.eval(direct)
        for _ in range(2):
            recorded = replay(model, *arguments)
            dr.eval(recorded)
            dr.sync_thread()
            assert_baseline_equal(direct, recorded, np)
            for name in ("growth_gain", "growth_observed", "growth_unknown"):
                if name in direct:
                    np.testing.assert_allclose(np.asarray(direct[name]), np.asarray(recorded[name]),
                                               atol=2e-7, rtol=3e-6, err_msg=name)
        return direct

    compare_current()  # Normal training before any growth overlay was installed.
    compare_current(request_empty=True)  # Safe direct bypass for zero-sized evidence outputs.
    proposals = state(np, [0, 32])
    model.set_growth_candidates(device_state(proposals, mi))
    replay.clear()
    before = compare_current()
    proposals["radiance_sh"] *= .25
    model.set_growth_candidates(device_state(proposals, mi))
    replay.clear()  # Driver must invalidate after replacing even the same count.
    after = compare_current()
    assert np.any(np.abs(np.asarray(before["growth_gain"]) - np.asarray(after["growth_gain"])) > 1e-6)
    options["max_growth_hits"] = 1.
    limited = compare_current()
    np.testing.assert_array_equal(np.asarray(limited["growth_observed"]), [1, 0])
    np.testing.assert_array_equal(np.asarray(limited["growth_unknown"]), [0, 1])
    # Same proposal count, different linked cell membership. Clearing the
    # replay must refresh all cell heads/links rather than preserving old IDs.
    proposals = state(np, [0, 0], [[.1, .2, .3], [.6, .4, .2]])
    model.set_growth_candidates(device_state(proposals, mi))
    replay.clear()
    siblings = compare_current()
    np.testing.assert_array_equal(np.asarray(siblings["growth_observed"]), [1, 1])
    np.testing.assert_array_equal(np.asarray(siblings["growth_unknown"]), [0, 0])
    assert abs(float(np.asarray(siblings["growth_gain"])[0]) -
               float(np.asarray(siblings["growth_gain"])[1])) > 1e-5
    proposals["cell_indices"] = np.asarray([32, 0], np.uint32)
    model.set_growth_candidates(device_state(proposals, mi))
    replay.clear()
    regrouped = compare_current()
    np.testing.assert_array_equal(np.asarray(regrouped["growth_observed"]), [0, 1])
    np.testing.assert_array_equal(np.asarray(regrouped["growth_unknown"]), [1, 0])
    model.clear_growth_candidates()
    replay.clear()
    compare_current()
    assert replay.statistics()["replays"] >= 4
    assert replay.statistics()["bypasses"] >= 2
    return {"direct_vs_frozen_native_with_growth_enabled_and_disabled": "passed",
            "empty_requested_growth_outputs_use_safe_direct_path": "passed",
            "same_count_candidate_replace_and_clear_replay_invalidation": "passed",
            "same_count_cell_grouping_and_reordering_replay_invalidation": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true", help="Check scheduling without loading CUDA")
    args = parser.parse_args()
    checks = [check_warmup(), check_view_votes_numpy(), check_unequal_shard_votes_numpy()]
    for item in checks:
        print(json.dumps(item), flush=True)
    if not args.cpu_only:
        with tempfile.TemporaryDirectory(prefix="mitsuba-rgb-growth-regression-") as temporary:
            mi, dr, np, handles = initialize_runtime(RuntimeConfig(args.build_dir, args.gpu),
                                                       Path(temporary), {})
            assert int(getattr(mi.VoxelReconstruction, "rgb_growth_version", 0)) >= 3, \
                "Rebuild this checkout before running RGB growth regression"
            for check in (check_counterfactual_and_invariance, check_positive_negative_neutral,
                          check_empty_baseline, check_same_cell_proposals, check_partial_paths,
                          check_shards_and_candidate_validation,
                          check_view_votes_device, check_native_replay):
                checks.append(check(mi, dr, np))
                print(json.dumps(checks[-1]), flush=True)
    print(json.dumps({"status": "passed", "checks": checks}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
