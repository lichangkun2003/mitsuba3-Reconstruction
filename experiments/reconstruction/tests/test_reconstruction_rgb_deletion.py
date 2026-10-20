"""Generated CUDA checks for RGB deletion evidence from the training hit records.

The remove-and-render comparisons are independent test oracles only. Production
deletion must reuse the normal forward and hit-backward work, without another
ray traversal. No dataset or existing experiment output is used.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preview import RuntimeConfig
from reconstruction_common import initialize_runtime, restore_model
from reconstruction_replay import NativeLossReplay


Y00 = 1. / (2. * math.sqrt(math.pi))
EVIDENCE = ("deletion_gain", "deletion_observed", "deletion_unknown")
DIMENSIONS = {"cell_indices": 1, "centers": 3, "log_scales": 3,
              "rotations": 4, "radiance_sh": 3, "opacity_sh": 1}


def payload(np, cells=(0, 16, 48), colors=None, opacities=None):
    count = len(cells)
    colors = [[.2, .4, .6]] * count if colors is None else colors
    opacities = [.35] * count if opacities is None else opacities
    return {
        "model_metadata": {"resolution": 4, "grid_min": [0.] * 3,
                           "grid_max": [4.] * 3, "sh_count": 1,
                           "opacity_sh_count": 1},
        "model_state": {
            "cell_indices": np.asarray(cells, dtype=np.uint32),
            "centers": np.tile([.5, .5, .5], count).astype(np.float32),
            "log_scales": np.full(count * 3, math.log(.4), dtype=np.float32),
            "rotations": np.tile([1., 0., 0., 0.], count).astype(np.float32),
            "radiance_sh": np.asarray(colors, dtype=np.float32).ravel() / Y00,
            "opacity_sh": np.asarray(
                [math.log(a / (1. - a)) / Y00 for a in opacities], np.float32),
        },
    }


def without_voxel(checkpoint, index, mi, np):
    removed = copy.deepcopy(checkpoint)
    for name, dimension in DIMENSIONS.items():
        values = removed["model_state"][name].reshape(-1, dimension)
        removed["model_state"][name] = np.delete(values, index, axis=0).ravel()
    return restore_model(removed, mi)


def ray_at(mi, x=.5, z=-1., maxt=8.):
    ray = mi.Ray3f(mi.Point3f(x, .5, z), mi.Vector3f(0., 0., 1.))
    ray.maxt = maxt
    return ray


def evaluate(model, ray, target, mi, dr, *, enabled=True, alpha=0., weight=1.,
             replay=None, **settings):
    options = {"rgb_only": 1., "collect_rgb_deletion_evidence": float(enabled),
               "max_hits": 16., "transmittance_threshold": .0001}
    options.update(settings)
    arguments = (ray, mi.Color3f(target), mi.Float(alpha), mi.Color3f(1.),
                 mi.Float(weight), options)
    result = (model.loss_and_grad(*arguments) if replay is None else
              replay(model, *arguments))
    dr.eval(result)
    dr.sync_thread()
    return result


def rendered(model, ray, mi, dr, np, threshold=.0001, max_hits=16):
    result = model.trace(ray, mi.Color3f(1.), threshold, max_hits)
    dr.eval(result)
    dr.sync_thread()
    return np.asarray(result[0], dtype=np.float64).reshape(3, -1)


def assert_baseline_equal(first, second, np):
    for name in ("rgb", "raw_rgb", "alpha", "raw_alpha", "hit_count", "loss",
                 "appearance_valid", "geometry_valid", "background_carve_valid",
                 "foreground_evidence", "background_evidence"):
        np.testing.assert_array_equal(np.asarray(first[name]), np.asarray(second[name]),
                                      err_msg=name)
    for name, values in first["gradients"].items():
        np.testing.assert_array_equal(np.asarray(values),
                                      np.asarray(second["gradients"][name]),
                                      err_msg=f"gradients.{name}")


def assert_evidence_close(first, second, np):
    for name in EVIDENCE:
        if name == "deletion_gain":
            np.testing.assert_allclose(np.asarray(first[name]), np.asarray(second[name]),
                                       atol=3e-7, rtol=3e-5, err_msg=name)
        else:
            np.testing.assert_array_equal(np.asarray(first[name]), np.asarray(second[name]),
                                          err_msg=name)


def check_remove_oracle_and_invariance(mi, dr, np):
    checkpoint = payload(np, colors=[[.1, .8, .3], [.7, .2, .4], [.3, .5, .9]],
                         opacities=[.45, .25, .55])
    model = restore_model(checkpoint, mi)
    ray = ray_at(mi)
    target = np.asarray([.3, .6, .4], dtype=np.float64)
    before = rendered(model, ray, mi, dr, np)[:, 0]
    baseline = evaluate(model, ray, target.tolist(), mi, dr, enabled=False)
    result = evaluate(model, ray, target.tolist(), mi, dr)
    assert_baseline_equal(baseline, result, np)
    np.testing.assert_array_equal(rendered(model, ray, mi, dr, np)[:, 0], before)
    original_loss = np.sum((before - target) ** 2)
    expected = []
    for index in range(3):
        after = rendered(without_voxel(checkpoint, index, mi, np), ray, mi, dr, np)[:, 0]
        expected.append(original_loss - np.sum((after - target) ** 2))
    np.testing.assert_allclose(np.asarray(result["deletion_gain"]), expected,
                               atol=3e-7, rtol=3e-5)
    np.testing.assert_array_equal(np.asarray(result["deletion_observed"]), [1, 1, 1])
    np.testing.assert_array_equal(np.asarray(result["deletion_unknown"]), [0, 0, 0])
    for alpha in (0., .5, 1., float("nan")):
        alternate = evaluate(model, ray, target.tolist(), mi, dr, alpha=alpha)
        assert_baseline_equal(result, alternate, np)
        assert_evidence_close(result, alternate, np)
    return {"virtual_removal_matches_remove_and_trace": "passed",
            "prefix_nonwhite_suffix_and_all_hit_positions": "passed",
            "image_loss_gradients_unchanged": "passed",
            "reference_alpha_including_nan_ignored": "passed"}


def check_positive_negative_neutral_and_miss(mi, dr, np):
    checkpoint = payload(np, [16])
    model = restore_model(checkpoint, mi)
    ray = ray_at(mi)
    before = rendered(model, ray, mi, dr, np)[:, 0]
    helpful = evaluate(model, ray, [1., 1., 1.], mi, dr)
    harmful = evaluate(model, ray, before.tolist(), mi, dr)
    assert float(np.asarray(helpful["deletion_gain"])[0]) > 1e-4
    assert float(np.asarray(harmful["deletion_gain"])[0]) < -1e-4
    neutral_model = restore_model(payload(np, [16], colors=[[1., 1., 1.]]), mi)
    neutral = evaluate(neutral_model, ray, [.1, .2, .3], mi, dr)
    np.testing.assert_allclose(np.asarray(neutral["deletion_gain"]), 0., atol=2e-7)
    np.testing.assert_array_equal(np.asarray(neutral["deletion_observed"]), [1])
    np.testing.assert_array_equal(np.asarray(neutral["deletion_unknown"]), [0])
    miss = evaluate(model, ray_at(mi, 1.5), [.1, .2, .3], mi, dr)
    for name in EVIDENCE:
        np.testing.assert_array_equal(np.asarray(miss[name]), 0, err_msg=name)
    return {"helpful_harmful_neutral_removal": "passed",
            "unhit_voxel_supplies_no_evidence": "passed"}


def check_unknown_rays_and_complete_boundaries(mi, dr, np):
    model = restore_model(payload(np, [16, 48]), mi)
    target = [.1, .2, .3]
    for options in ({"max_hits": 1.}, {"transmittance_threshold": .8}):
        unknown = evaluate(model, ray_at(mi), target, mi, dr, **options)
        np.testing.assert_array_equal(np.asarray(unknown["deletion_gain"]), [0., 0.])
        np.testing.assert_array_equal(np.asarray(unknown["deletion_observed"]), [0, 0])
        np.testing.assert_array_equal(np.asarray(unknown["deletion_unknown"]), [1, 0])
        # One ray stops early in the grid, the other normally reaches maxt in
        # the same occupied cell. The latter's background is its known suffix.
        known_ray = ray_at(mi, z=1.1, maxt=.8)
        known = evaluate(model, known_ray, target, mi, dr, **options)
        np.testing.assert_array_equal(np.asarray(known["deletion_observed"]), [1, 0])
        np.testing.assert_array_equal(np.asarray(known["deletion_unknown"]), [0, 0])
        mixed_ray = ray_at(mi, z=mi.Float([-1., 1.1]), maxt=mi.Float([8., .8]))
        mixed = evaluate(model, mixed_ray, target, mi, dr, **options)
        np.testing.assert_array_equal(np.asarray(mixed["deletion_observed"]), [1, 0])
        np.testing.assert_array_equal(np.asarray(mixed["deletion_unknown"]), [1, 0])
        np.testing.assert_allclose(np.asarray(mixed["deletion_gain"]),
                                   np.asarray(known["deletion_gain"]), atol=3e-7, rtol=3e-5)
    # This hit reaches the grid exit in the same DDA step as the hit limit.
    # Removal needs no spare insertion slot and cannot cause earlier opacity termination.
    last_cell = restore_model(payload(np, [48]), mi)
    at_capacity = evaluate(last_cell, ray_at(mi), target, mi, dr, max_hits=1.)
    np.testing.assert_array_equal(np.asarray(at_capacity["deletion_observed"]), [1])
    np.testing.assert_array_equal(np.asarray(at_capacity["deletion_unknown"]), [0])
    standard = evaluate(last_cell, ray_at(mi), target, mi, dr)
    assert_evidence_close(at_capacity, standard, np)
    return {"early_hit_cap_or_transmittance_is_unknown_per_ray": "passed",
            "known_hits_survive_other_unknown_rays": "passed",
            "maxt_or_grid_exit_with_full_hit_capacity_remains_known": "passed"}


def check_near_opaque_and_invalid_evidence(mi, dr, np):
    checkpoint = payload(np, colors=[[.1, .2, .3], [.8, .2, .1], [.2, .9, .4]],
                         opacities=[.999999, .4, .3])
    model = restore_model(checkpoint, mi)
    ray, target = ray_at(mi), np.asarray([.7, .5, .2], np.float64)
    before = rendered(model, ray, mi, dr, np, threshold=0.)[:, 0]
    result = evaluate(model, ray, target.tolist(), mi, dr, transmittance_threshold=0.)
    expected = []
    for index in range(3):
        after = rendered(without_voxel(checkpoint, index, mi, np), ray, mi, dr, np,
                         threshold=0.)[:, 0]
        expected.append(np.sum((before - target) ** 2) - np.sum((after - target) ** 2))
    np.testing.assert_allclose(np.asarray(result["deletion_gain"]), expected,
                               atol=3e-7, rtol=3e-5)
    np.testing.assert_array_equal(np.asarray(result["deletion_observed"]), [1, 1, 1])
    assert np.all(np.isfinite(np.asarray(result["deletion_gain"])))
    simple = restore_model(payload(np, [16]), mi)
    weighted = evaluate(simple, ray, [.1, .2, .3], mi, dr, weight=2.)
    nonfinite = evaluate(simple, ray, [float("nan"), .2, .3], mi, dr)
    for invalid in (weighted, nonfinite):
        np.testing.assert_array_equal(np.asarray(invalid["deletion_gain"]), [0.])
        np.testing.assert_array_equal(np.asarray(invalid["deletion_observed"]), [0])
        np.testing.assert_array_equal(np.asarray(invalid["deletion_unknown"]), [1])
    return {"near_opaque_removal_uses_stable_backward_suffix": "passed",
            "nonunit_weight_and_nonfinite_reference_abstain": "passed"}


def check_shards_replay_empty_and_config(mi, dr, np):
    model = restore_model(payload(np, [16, 48]), mi)
    xs, zs, maxts = [.45, .5, .55, 1.5], [-1., -1., 1.1, -1.], [8., 8., .8, 8.]
    ray = ray_at(mi, mi.Float(xs), mi.Float(zs), mi.Float(maxts))
    options, target = {"transmittance_threshold": .8}, [.1, .2, .3]
    combined = evaluate(model, ray, target, mi, dr, **options)
    shards = [evaluate(model, ray_at(mi, x, z, maxt), target, mi, dr, **options)
              for x, z, maxt in zip(xs, zs, maxts)]
    for name in EVIDENCE:
        expected = sum(np.asarray(value[name]) for value in shards)
        np.testing.assert_allclose(np.asarray(combined[name]), expected,
                                   atol=3e-7, rtol=3e-5, err_msg=name)
    replay = NativeLossReplay(dr)
    for enabled in (False, True):
        direct = evaluate(model, ray, target, mi, dr, enabled=enabled, **options)
        for _ in range(3):
            frozen = evaluate(model, ray, target, mi, dr, enabled=enabled,
                              replay=replay, **options)
            assert_baseline_equal(direct, frozen, np)
            if enabled:
                assert_evidence_close(direct, frozen, np)
    assert replay.statistics()["replays"] >= 2
    empty = restore_model(payload(np, []), mi)
    empty_result = evaluate(empty, ray, target, mi, dr, replay=replay)
    for name in EVIDENCE:
        assert np.asarray(empty_result[name]).size == 0, name
    assert_baseline_equal(empty_result, evaluate(empty, ray, target, mi, dr, enabled=False), np)
    assert replay.statistics()["empty_model_bypasses"] >= 1
    for invalid_options in ({"rgb_only": 0.}, {"collect_rgb_deletion_evidence": -1.},
                            {"collect_rgb_deletion_evidence": float("nan")}):
        try:
            evaluate(model, ray, target, mi, dr, **invalid_options)
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError(f"Accepted invalid RGB deletion settings: {invalid_options}")
    return {"chunk_or_shard_raw_gain_and_counts_sum": "passed",
            "direct_and_native_replay_parity": "passed",
            "empty_model_and_invalid_configuration": "passed",
            "replay": replay.statistics()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    checks = []
    with tempfile.TemporaryDirectory(prefix="mitsuba-rgb-deletion-regression-") as temporary:
        mi, dr, np, handles = initialize_runtime(
            RuntimeConfig(args.build_dir, args.gpu), Path(temporary), {})
        assert int(getattr(mi.VoxelReconstruction, "rgb_deletion_version", 0)) >= 1, \
            "Rebuild this checkout before running RGB deletion regression"
        for check in (check_remove_oracle_and_invariance, check_positive_negative_neutral_and_miss,
                      check_unknown_rays_and_complete_boundaries,
                      check_near_opaque_and_invalid_evidence, check_shards_replay_empty_and_config):
            checks.append(check(mi, dr, np))
            print(json.dumps(checks[-1]), flush=True)
    print(json.dumps({"status": "passed", "checks": checks}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
