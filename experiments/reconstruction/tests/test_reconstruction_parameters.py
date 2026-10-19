"""Check atomic parameter validation for scalar and CUDA source variants.

Only synthetic buffers are used. Runtime artifacts live in a temporary
directory and are removed automatically; no training outputs are created.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconstruction_common import initialize_runtime
from preview import RuntimeConfig


def check_parameters(mi, dr, np):
    buffer_f = dr.scalar.ArrayXf if mi.variant() == "scalar_rgb" else mi.Float
    buffer_u = dr.scalar.ArrayXu if mi.variant() == "scalar_rgb" else mi.UInt32
    count = 3
    initial = {
        "centers": np.full(count * 3, .5, np.float32),
        "log_scales": np.full(count * 3, -2., np.float32),
        "rotations": np.tile(np.array([1., 0., 0., 0.], np.float32), count),
        "radiance_sh": np.linspace(.1, .9, count * 9 * 3, dtype=np.float32),
        "opacity_sh": np.zeros(count * 9, np.float32),
    }

    def restore(values, cells):
        state = {name: buffer_f(value) for name, value in values.items()}
        state["cell_indices"] = buffer_u(cells)
        return mi.VoxelReconstruction.from_state(
            2, [0., 0., 0.], [1., 1., 1.], state, 9, 9)

    model = restore(initial, np.array([0, 2, 7], np.uint32))

    def snapshot():
        return {name: np.array(value, copy=True) for name, value in model.snapshot().items()}

    def assert_state(expected):
        for name, value in snapshot().items():
            np.testing.assert_array_equal(value, expected[name], err_msg=name)

    def update(values):
        model.set_parameters({name: buffer_f(value) for name, value in values.items()})

    def rejected(values, message):
        before = snapshot()
        try:
            update(values)
        except RuntimeError as exc:
            assert message in str(exc), str(exc)
        else:
            raise AssertionError(f"Invalid update was accepted: {message}")
        assert_state(before)

    # Exercise success with lazy GPU expressions, followed by a partial update.
    model.set_parameters({name: buffer_f(value) + .125 for name, value in initial.items()})
    expected = snapshot()
    for name, value in initial.items():
        np.testing.assert_array_equal(expected[name], value + np.float32(.125))
    update({"opacity_sh": np.full(count * 9, .25, np.float32)})
    expected["opacity_sh"][:] = .25
    assert_state(expected)
    model.set_parameters({})
    assert_state(expected)

    # Every rejected transaction also supplies valid changes to all other
    # groups, so checking only the bad buffer cannot hide a partial commit.
    nonfinite_cases = 0
    for name in initial:
        for bad, index in ((np.nan, 0), (np.inf, len(initial[name]) // 2),
                           (-np.inf, len(initial[name]) - 1)):
            values = {key: value + np.float32(.375) for key, value in initial.items()}
            values[name][index] = bad
            rejected(values, f"non-finite {name}.")
            nonfinite_cases += 1

    for name in initial:
        values = {key: value + np.float32(.5) for key, value in initial.items()}
        values[name] = values[name][:-1]
        rejected(values, f"unknown key or wrong size: {name}.")
    rejected({"centers": initial["centers"] + np.float32(.5),
              "unknown": np.zeros(1, np.float32)}, "unknown key or wrong size: unknown.")

    # The constructor uses the same validation path; an empty model must still
    # accept all zero-length groups and empty/partial updates, but no data.
    model = restore({name: value[:0] for name, value in initial.items()},
                    np.empty(0, np.uint32))
    expected = snapshot()
    model.set_parameters({})
    update({"centers": initial["centers"][:0]})
    assert_state(expected)
    rejected({"centers": np.zeros(1, np.float32)}, "unknown key or wrong size: centers.")
    rejected({"unknown": np.empty(0, np.float32)}, "unknown key or wrong size: unknown.")
    return {"variant": mi.variant(), "nonfinite_transactions": nonfinite_cases,
            "invalid_shape_or_key_transactions": 8,
            "valid_full_partial_empty_updates": "passed", "empty_model": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="mitsuba-parameter-tests-") as directory:
        report = {}
        mi, dr, np, handles = initialize_runtime(
            RuntimeConfig(args.build_dir, args.gpu), Path(directory), report)
        checks = []
        for variant in ("cuda_ad_rgb", "scalar_rgb"):
            mi.set_variant(variant)
            checks.append(check_parameters(mi, dr, np))
        print(json.dumps({"status": "passed", "checks": checks}), flush=True)


if __name__ == "__main__":
    main()
