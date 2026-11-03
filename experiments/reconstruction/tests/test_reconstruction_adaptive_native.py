"""Independent native adaptive-shell geometry and insertion-vote regressions.

--cpu-only validates the fixture geometry without importing Mitsuba/CUDA.
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_reconstruction_world_shell_native import oracle, Y00, ENV

TAG = 1 << 63


def address(coarse, depth, morton=0):
    assert 0 <= depth <= 9 and 0 <= morton < 8 ** depth
    return TAG | (coarse << 32) | (1 << (3 * depth)) | morton


def box(key, core, np):
    """Independent parent-to-child physical decoding, no production helpers."""
    key = int(key)
    if key < core ** 3:
        low = -1 + np.array([key % core, key // core % core, key // core ** 2]) * (2 / core)
        return low, low + 2 / core
    assert key & TAG
    coarse, path = (key & ~TAG) >> 32, key & 0xffffffff
    digits = []
    while path > 1:
        digits.append(path % 8)
        path //= 8
    assert path == 1
    shell, ordinal = divmod(coarse, 56)
    xyz = [(x, y, z) for z in range(4) for y in range(4) for x in range(4)
           if not (1 <= x <= 2 and 1 <= y <= 2 and 1 <= z <= 2)][ordinal]
    edge = float(2 ** shell)
    low = (np.array(xyz, float) - 2) * edge
    for digit in reversed(digits):
        edge *= .5
        low += edge * np.array([digit & 1, (digit >> 1) & 1, (digit >> 2) & 1])
    return low, low + edge


def payload(np, cells, core=4, levels=3):
    count = len(cells)
    sizes = np.array([box(key, core, np)[1] - box(key, core, np)[0] for key in cells]).reshape(-1, 3)
    colors = np.array([[.2 + .07 * (i % 5), .4, .6 - .04 * (i % 7)] for i in range(count)], np.float32)
    return {"model_metadata": {"resolution": core, "core_resolution": core,
        "grid_min": [-1.] * 3, "grid_max": [1.] * 3, "scene_mapping": "world_shell",
        "background_shell_levels": levels, "background_subdivision_level": 0,
        "background_refinement": "adaptive", "world_shell_version": 5,
        "background_geometry": "ellipsoid", "sh_count": 1, "opacity_sh_count": 1},
        "model_state": {"cell_indices": np.asarray(cells, np.uint64),
            "centers": np.full(count * 3, .5, np.float32),
            "log_scales": np.log(sizes * [.37, .28, .41]).astype(np.float32).ravel(),
            "rotations": np.tile(np.array([1., 0., 0., 0.], np.float32), count),
            "radiance_sh": (colors / Y00).ravel(),
            "opacity_sh": np.full(count, math.log(.35 / .65) / Y00, np.float32)}}


def device_state(state, mi):
    return {key: (mi.UInt64(value) if key == "cell_indices" else mi.Float(value))
            for key, value in state.items()}


def restore(data, mi, *, adaptive=None):
    md = data["model_metadata"]
    return mi.VoxelReconstruction.from_state(md["resolution"], list(map(float, md["grid_min"])), list(map(float, md["grid_max"])),
        device_state(data["model_state"], mi), 1, 1, background_shell_levels=md.get("background_shell_levels", 0),
        background_use_ellipsoid=True, background_adaptive=(md.get("scene_mapping") == "world_shell"
                                                           if adaptive is None else adaptive))


def ray_for(key, core, np, *, miss=False, axis=2):
    low, high = box(key, core, np)
    edge = high[0] - low[0]
    origin = (low + high) * .5
    origin[axis] = low[axis] - .25 * edge
    if miss:
        origin[(axis + 1) % 3] = low[(axis + 1) % 3] + .02 * edge
    direction = np.zeros(3)
    direction[axis] = 1.
    return origin, direction, 1.5 * edge


def ray(mi, origin, direction, maxt):
    value = mi.Ray3f(mi.Point3f(*map(float, origin)), mi.Vector3f(*map(float, direction)))
    value.maxt = float(maxt)
    return value


def rendered(model, sample, mi, dr, np):
    result = model.trace(ray(mi, *sample), mi.Color3f(ENV), 0., 128)
    dr.eval(result)
    return np.asarray(result[0]).reshape(3, -1)[:, 0], float(np.asarray(result[1])[0])


def evaluate(model, sample, mi, dr, target=(.3, .4, .5), **settings):
    options = {"rgb_only": 1., "max_hits": 128., "max_candidates": 0.,
               "transmittance_threshold": 0., "collect_adaptive_priority": 1.}
    options.update(settings)
    result = model.loss_and_grad(ray(mi, *sample), mi.Color3f(target), mi.Float(0.),
        mi.Color3f(ENV), mi.Float(1.), options)
    dr.eval(result)
    return result


def check_cpu(np):
    leaves = [0, address(0, 1, 0), address(0, 2, 8), address(0, 4, 9 * 64 + 7),
              address(56, 0), address(112, 9, 54321)]
    boxes = [box(key, 4, np) for key in leaves]
    for i, (low, high) in enumerate(boxes):
        for other_low, other_high in boxes[i + 1:]:
            assert not np.all(np.minimum(high, other_high) > np.maximum(low, other_low))
    assert boxes[-1][1][0] - boxes[-1][0][0] < 2 / 4
    for key in leaves[1:]:
        before = box(key, 4, np)
        after = box(key, 1024, np)
        np.testing.assert_array_equal(before, after)
    return leaves


def foreground_tie_fixture(np, core, axes, negative):
    """The old unit-grid convention visits x, then y, then z at a tie."""
    start = core // 2 - 1
    cells = [start + (i & 1) + core * (start + ((i >> 1) & 1) +
             core * (start + ((i >> 2) & 1))) for i in range(8)]
    data = payload(np, cells, core=core)
    edge = 2 / core
    # Cover the leaf corners so zero-length edge/corner hits are observable.
    data["model_state"]["log_scales"].fill(math.log(edge))
    bits = np.zeros(3, dtype=int)
    if negative:
        bits[:axes] = 1
    origin = -1 + (start + bits + .5) * edge
    direction = np.zeros(3)
    direction[:axes] = -1 if negative else 1
    visited = [int(bits @ np.array([1, 2, 4]))]
    for axis in range(axes):
        bits[axis] = 1 - bits[axis]
        visited.append(int(bits @ np.array([1, 2, 4])))
    colors = data["model_state"]["radiance_sh"].reshape(-1, 3) * Y00
    result, before = np.zeros(3), 1.
    for row in visited:
        result += before * .35 * colors[row]
        before *= .65
    return data, (origin, direction, 1.25 * edge), (result + before * np.array(ENV), 1 - before, visited)


def check_foreground_tie_cpu(np):
    for axes, expected in ((1, [0, 1]), (2, [0, 1, 3]), (3, [0, 1, 3, 7])):
        for core in (4, 256, 1024):
            _, _, positive = foreground_tie_fixture(np, core, axes, False)
            _, _, negative = foreground_tie_fixture(np, core, axes, True)
            assert positive[2] == expected
            assert negative[2] == [((1 << axes) - 1) ^ item for item in expected]
            assert abs(positive[1] - (1 - .65 ** (axes + 1))) < 1e-14


def check_foreground_ties(mi, dr, np):
    for core in (4, 256, 1024):
        for axes in (1, 2, 3):
            for negative in (False, True):
                data, sample, expected = foreground_tie_fixture(np, core, axes, negative)
                legacy, adaptive = restore(data, mi, adaptive=False), restore(data, mi, adaptive=True)
                label = f"foreground core={core}, axes={axes}, negative={negative}"
                reference = evaluate(legacy, sample, mi, dr)
                current = evaluate(adaptive, sample, mi, dr)
                np.testing.assert_array_equal(np.asarray(reference["hit_count"]), [axes + 1], err_msg=label)
                np.testing.assert_allclose(np.asarray(reference["raw_rgb"]).ravel(), expected[0], atol=3e-7, rtol=3e-6, err_msg=label)
                np.testing.assert_array_equal(np.asarray(current["hit_count"]), np.asarray(reference["hit_count"]), err_msg=label)
                for key in ("raw_rgb", "raw_alpha", "loss"):
                    np.testing.assert_allclose(np.asarray(current[key]), np.asarray(reference[key]), atol=3e-7, rtol=3e-6,
                                               err_msg=f"{label}: {key}")
                traced = rendered(adaptive, sample, mi, dr, np)
                np.testing.assert_allclose(traced[0], expected[0], atol=3e-7, rtol=3e-6, err_msg=label)
                np.testing.assert_allclose(traced[1], expected[1], atol=3e-7, rtol=3e-6, err_msg=label)


def check_geometry(mi, dr, np, leaves):
    data = payload(np, leaves)
    model = restore(data, mi)
    lows, highs = zip(*(box(key, 4, np) for key in leaves))
    prepared = np.array(lows), np.array(highs), np.asarray(leaves, np.uint64) < 4 ** 3
    sizes = np.asarray(model.leaf_sizes(mi.UInt32(list(range(len(leaves)))))).T
    np.testing.assert_allclose(sizes, prepared[1] - prepared[0], atol=0, rtol=0)
    for key in leaves:
        for axis in range(3):
            for miss in (False, True):
                sample = ray_for(key, 4, np, miss=miss, axis=axis)
                actual = rendered(model, sample, mi, dr, np)
                expected = oracle(data, *sample[:2], np, maxt=sample[2], threshold=0., prepared=prepared)
                np.testing.assert_allclose(actual[0], expected["rgb"], atol=3e-5, rtol=3e-5)
                np.testing.assert_allclose(actual[1], expected["alpha"], atol=3e-6, rtol=3e-5)

    for key in leaves[1:]:
        data = payload(np, [key])
        model = restore(data, mi)
        low, high = box(key, 4, np)
        bounded = copy.deepcopy(data)
        bounded["model_metadata"] = {"resolution": 1, "grid_min": list(low), "grid_max": list(high)}
        bounded["model_state"]["cell_indices"] = np.array([0], np.uint64)
        reference = restore(bounded, mi)
        sample = ray_for(key, 4, np)
        # An off-center ray provides nonzero center/scale proxy gradients.
        sample[0][0] += .13 * (high[0] - low[0])
        actual, expected = evaluate(model, sample, mi, dr), evaluate(reference, sample, mi, dr)
        for group in actual["gradients"]:
            np.testing.assert_allclose(np.asarray(actual["gradients"][group]),
                np.asarray(expected["gradients"][group]), atol=3e-5, rtol=3e-4, err_msg=group)
        expected_priority = abs(np.asarray(expected["gradients"]["opacity_sh"])[0]) / (Y00 * .65)
        np.testing.assert_allclose(np.asarray(actual["adaptive_priority"]), expected_priority, atol=3e-6, rtol=3e-5)
        assert np.asarray(actual["adaptive_priority"])[0] > 0


def check_growth(mi, dr, np):
    first = address(0, 1, 0)
    candidate = address(0, 1, 1)
    fine = address(0, 3, 135)
    tiny = address(112, 9, 54321)
    base = payload(np, [first])
    proposals = payload(np, [candidate, candidate, fine, tiny])["model_state"]
    # Same target, different shapes: the first must not hide a later sibling hit.
    proposals["centers"].reshape(-1, 3)[0] = [5., 5., 5.]
    model = restore(base, mi)
    target = np.array([.3, .4, .5])
    samples = [ray_for(key, 4, np) for key in (first, candidate, fine, tiny)]
    before = [rendered(model, sample, mi, dr, np)[0] for sample in samples]
    model.set_growth_candidates(device_state(proposals, mi))
    totals = np.zeros(4)
    observed = np.zeros(4)
    for sample, rgb in zip(samples, before):
        result = evaluate(model, sample, mi, dr, collect_growth_evidence=1., max_growth_hits=8.)
        np.testing.assert_allclose(np.asarray(result["raw_rgb"]).ravel(), rgb, atol=1e-7)
        totals += np.asarray(result["growth_gain"])
        observed += np.asarray(result["growth_observed"])
        np.testing.assert_array_equal(np.asarray(result["growth_unknown"]), 0)
    expected = []
    widths = {"cell_indices": 1, "centers": 3, "log_scales": 3, "rotations": 4,
              "radiance_sh": 3, "opacity_sh": 1}
    for row in range(4):
        inserted = copy.deepcopy(base)
        for key, width in widths.items():
            inserted["model_state"][key] = np.concatenate((base["model_state"][key],
                proposals[key][row * width:(row + 1) * width]))
        actual = restore(inserted, mi)
        expected.append(sum(np.sum((rgb - target) ** 2) -
            np.sum((rendered(actual, sample, mi, dr, np)[0] - target) ** 2)
            for sample, rgb in zip(samples, before)))
    np.testing.assert_allclose(totals, expected, atol=3e-6, rtol=3e-5)
    np.testing.assert_array_equal(observed, [0, 1, 1, 1])

    # Coarse candidates may not swallow finer occupied descendants, even when
    # their minimum corner is empty; the opposite containment is also invalid.
    for occupied, proposed in ((address(0, 2, 63), address(0, 0)),
                               (address(0, 0), address(0, 2, 63))):
        current = restore(payload(np, [occupied]), mi)
        try:
            current.set_growth_candidates(device_state(payload(np, [proposed])["model_state"], mi))
        except RuntimeError:
            pass
        else:
            raise AssertionError("Overlapping adaptive growth leaf was accepted")


def check_layout_validation(mi, np):
    for bad in (TAG, TAG | 2, TAG | (1 << 30), TAG | (168 << 32) | 1):
        data = payload(np, [address(0, 0)])
        data["model_state"]["cell_indices"][0] = bad
        try:
            restore(data, mi)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"Invalid adaptive path accepted: {bad}")
    for keys in ([address(0, 0), address(0, 1, 7)],
                 [address(0, 2, 63), address(0, 2, 63)]):
        try:
            restore(payload(np, keys), mi)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Duplicate/overlapping adaptive model leaves accepted")

    # Explicit opt-in is required; tagged keys cannot be decoded as old dense
    # shell IDs when loading a legacy fixed/uniform checkpoint.
    data = payload(np, [address(0, 0)])
    md = data["model_metadata"]
    try:
        mi.VoxelReconstruction.from_state(4, md["grid_min"], md["grid_max"],
            device_state(data["model_state"], mi), 1, 1, background_shell_levels=3)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Adaptive cells accepted without adaptive layout metadata")


def main():
    from preview import RuntimeConfig
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ties-only", action="store_true")
    args = parser.parse_args()
    import numpy as np
    leaves = check_cpu(np)
    check_foreground_tie_cpu(np)
    if args.cpu_only:
        print("adaptive native fixtures: CPU geometry PASS")
        return
    from reconstruction_common import initialize_runtime
    with tempfile.TemporaryDirectory(prefix="adaptive-shell-native-") as directory:
        mi, dr, np, _ = initialize_runtime(RuntimeConfig(args.build_dir, args.gpu), Path(directory), {})
        check_foreground_ties(mi, dr, np)
        if not args.ties_only:
            check_geometry(mi, dr, np, leaves)
            check_growth(mi, dr, np)
            check_layout_validation(mi, np)
    print("adaptive native: mixed-depth geometry, gradients, priority, growth votes PASS")


if __name__ == "__main__":
    main()
