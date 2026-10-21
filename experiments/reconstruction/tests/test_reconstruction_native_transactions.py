"""Device parameter transactions and stable topology installation regressions.

Uses synthetic CUDA/scalar buffers only; no datasets or experiment outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preview import RuntimeConfig
from reconstruction_common import initialize_runtime


def check_transactions(mi, dr, np):
    gpu = mi.variant() == "cuda_ad_rgb"
    Float = mi.Float if gpu else dr.scalar.ArrayXf
    UInt = mi.UInt32 if gpu else dr.scalar.ArrayXu
    Mask = mi.Bool if gpu else bool

    def state(cells):
        n = len(cells)
        return {
            "cell_indices": UInt(cells),
            "centers": Float(np.full(3 * n, .5, np.float32)),
            "log_scales": Float(np.full(3 * n, -2., np.float32)),
            "rotations": Float(np.tile([1., 0., 0., 0.], n).astype(np.float32)),
            "radiance_sh": Float(np.full(3 * n, .2, np.float32)),
            "opacity_sh": Float(np.zeros(n, np.float32)),
        }

    def restore(cells):
        return mi.VoxelReconstruction.from_state(4, [0.] * 3, [4.] * 3, state(cells), 1, 1)

    def snapshot(model):
        return {name: np.array(value, copy=True) for name, value in model.snapshot().items()}

    def assert_snapshot(model, expected):
        for name, value in snapshot(model).items():
            np.testing.assert_array_equal(value, expected[name], err_msg=name)

    # An unordered input must retain row indices; duplicate and invalid indices
    # must fail even when their values alias valid indices after integer wrap.
    cells = [63, 3, 9, 0, 60]
    model = restore(cells)
    np.testing.assert_array_equal(snapshot(model)["cell_indices"], cells)
    invalid_count = 0
    for invalid in ([3, 0, 3], [64], [2**32 - 1], [63, 64, 63]):
        try:
            restore(invalid)
        except RuntimeError as exc:
            assert "unique and in bounds" in str(exc), str(exc)
            invalid_count += 1
        else:
            raise AssertionError(f"Accepted invalid occupancy: {invalid}")

    expected = snapshot(model)
    supplied = {name: Float(values + .25) for name, values in expected.items()
                if name != "cell_indices"}
    accepted = model.set_parameters_guarded(supplied, Mask(False))
    assert not bool(dr.slice(accepted))
    assert_snapshot(model, expected)
    accepted = model.set_parameters_guarded(supplied, Mask(True))
    assert bool(dr.slice(accepted))
    for name in expected:
        if name != "cell_indices":
            expected[name] += np.float32(.25)
    assert_snapshot(model, expected)

    # Rejection must preserve ALL groups, even if only one contains an invalid
    # value. The returned mask is the optimizer's shared transaction gate.
    rejected = 0
    for bad_group in supplied:
        for bad_value in (np.nan, np.inf, -np.inf):
            values = {name: value + np.float32(.5) for name, value in expected.items()
                      if name != "cell_indices"}
            values[bad_group][len(values[bad_group]) // 2] = bad_value
            accepted = model.set_parameters_guarded(
                {name: Float(value) for name, value in values.items()}, Mask(True))
            assert not bool(dr.slice(accepted))
            assert_snapshot(model, expected)
            rejected += 1

    for changes in ({"unknown": Float([1.])}, {"centers": Float([1.])}):
        try:
            model.set_parameters_guarded(changes, Mask(True))
        except RuntimeError as exc:
            assert "unknown key or wrong size" in str(exc), str(exc)
        else:
            raise AssertionError("Accepted invalid parameter metadata")
        assert_snapshot(model, expected)
    if gpu:
        try:
            model.set_parameters_guarded(supplied, mi.Bool([True, False]))
        except RuntimeError as exc:
            assert "scalar mask" in str(exc), str(exc)
        else:
            raise AssertionError("Accepted per-element transaction gate")
        assert_snapshot(model, expected)
    empty = restore([])
    assert bool(dr.slice(empty.set_parameters_guarded({}, Mask(True))))
    empty.set_parameters_guarded({"centers": Float([])}, Mask(True))
    # Exercise both the GPU radix path and scalar stable-sort fallback; the
    # hypothetical overlay must never change the model snapshot.
    model.set_growth_candidates(state([1, 16, 1, 4, 16]))
    assert model.growth_candidate_count == 5
    assert_snapshot(model, expected)
    model.clear_growth_candidates()
    assert model.growth_candidate_count == 0
    assert_snapshot(model, expected)
    return {"variant": mi.variant(), "invalid_occupancies_rejected": invalid_count,
            "nonfinite_transactions_preserve_all_parameters": rejected,
            "false_guard_and_shape_errors_preserve_state": "passed", "empty_model": "passed",
            "overlay_preserves_snapshot": "passed"}


def check_large_interleaved_overlay(mi, dr, np):
    from test_reconstruction_rgb_growth import (
        state, payload, device_state, ray_at, evaluate, restore_model)

    model = restore_model(payload(np), mi)
    rng = np.random.default_rng(41)
    count = 513
    cells = rng.choice(np.array([0, 32], np.uint32), count)
    colors = rng.uniform(.05, .7, (count, 3)).astype(np.float32)
    proposals = state(np, cells, colors, [.25] * count)
    model.set_growth_candidates(device_state(proposals, mi))
    result = evaluate(model, ray_at(mi), [.2, .3, .4], mi, dr, max_growth_hits=2.)
    reference = {key: np.array(result[key], copy=True)
                 for key in ("growth_gain", "growth_observed", "growth_unknown")}
    np.testing.assert_array_equal(reference["growth_observed"], 1)
    np.testing.assert_array_equal(reference["growth_unknown"], 0)
    # Shuffled input retains each row's own evidence, including across repeated
    # replacements and the radix kernel's warp/block boundaries.
    permutation = rng.permutation(count)
    reordered = {name: value.reshape(count, -1)[permutation].ravel()
                 for name, value in proposals.items()}
    model.set_growth_candidates(device_state(reordered, mi))
    actual = evaluate(model, ray_at(mi), [.2, .3, .4], mi, dr, max_growth_hits=2.)
    for key, expected in reference.items():
        np.testing.assert_array_equal(np.asarray(actual[key]), expected[permutation], err_msg=key)
    malformed = {key: value.copy() for key, value in reordered.items()}
    malformed["cell_indices"][count // 2] = 64
    try:
        model.set_growth_candidates(device_state(malformed, mi))
    except RuntimeError as exc:
        assert "in bounds" in str(exc), str(exc)
    else:
        raise AssertionError("Accepted invalid candidate replacement")
    after = evaluate(model, ray_at(mi), [.2, .3, .4], mi, dr, max_growth_hits=2.)
    for key in reference:
        np.testing.assert_array_equal(np.asarray(after[key]), np.asarray(actual[key]), err_msg=key)
    model.clear_growth_candidates()
    model.set_growth_candidates(device_state(state(np, []), mi))
    assert model.growth_candidate_count == 0
    return {"interleaved_candidate_rows": count, "radix_permutation_evidence": "passed",
            "invalid_replacement_transaction": "passed", "empty_overlay": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="mitsuba-native-transactions-") as directory:
        mi, dr, np, handles = initialize_runtime(
            RuntimeConfig(args.build_dir, args.gpu), Path(directory), {})
        checks = [check_transactions(mi, dr, np), check_large_interleaved_overlay(mi, dr, np)]
        mi.set_variant("scalar_rgb")
        checks.append(check_transactions(mi, dr, np))
        print(json.dumps({"status": "passed", "checks": checks}), flush=True)


if __name__ == "__main__":
    main()
