"""Tagged mixed-depth core leaves: independent world-space trace and size oracle."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preview import RuntimeConfig
from reconstruction_common import initialize_runtime

TAG = 1 << 61
Y00 = .5 / math.sqrt(math.pi)


def encode(depth, xyz):
    return TAG | (1 << (3 * depth)) | sum(
        ((value >> bit) & 1) << (3 * bit + axis)
        for axis, value in enumerate(xyz) for bit in range(depth))


def check_cpu():
    assert encode(1, (0, 0, 0)) == TAG | 8
    assert encode(10, (1023, 1023, 1023)) == TAG | ((1 << 31) - 1)
    assert encode(10, (1023, 1023, 1023)) < 1 << 63
    return {"core_tag_and_depth_10_capacity": "passed"}


def check_native(mi, dr, np):
    # Tagged depth 2, 3, 10 plus a disjoint old flat depth-2 foreground cell.
    cells = [encode(2, (0, 0, 0)), encode(3, (4, 0, 0)), encode(10, (768, 64, 64)), 63]
    lower = np.array([[0., 0., 0.], [.5, 0., 0.], [.75, .0625, .0625], [.75] * 3])
    sizes = np.array([.25, .125, 1 / 1024, .25])
    colors = np.array([[.8, .2, .1], [.1, .7, .3], [.3, .2, .9], [.6, .5, .1]], np.float32)
    alphas = np.array([.2, .35, .5, .7])
    state = dict(cell_indices=mi.UInt64(cells), centers=mi.Float([.5] * 12),
        log_scales=mi.Float(np.repeat(np.log(.9 * sizes), 3)),
        rotations=mi.Float(np.tile([1., 0., 0., 0.], 4)),
        radiance_sh=mi.Float((colors / Y00).ravel()), opacity_sh=mi.Float(np.log(alphas / (1 - alphas)) / Y00))
    directions = [(1., 0., 0.), (-1., 0., 0.), (0., 0., 1.), (1., 0., 0.)]
    origins = [(-.2, .0627, .0627), (1.2, .0627, .0627), (.85, .85, -.2), (-.2, .4, .4)]
    for shells in (0, 1):
        model = mi.VoxelReconstruction.from_state(4, [0.] * 3, [1.] * 3, state,
            1, 1, 4, shells, 0, False, bool(shells))
        assert model.native_core_adaptive()
        actual_sizes = model.leaf_sizes(mi.UInt32([0, 1, 2, 3, 99]))
        dr.eval(actual_sizes)
        np.testing.assert_allclose(np.asarray(actual_sizes)[:, :4], np.tile(sizes, (3, 1)), atol=1e-8)
        np.testing.assert_array_equal(np.asarray(actual_sizes)[:, 4], 0.)
        for origin, direction in zip(origins, directions):
            hits = []
            for row, (low, size) in enumerate(zip(lower, sizes)):
                first, last = 0., 5.
                for axis in range(3):
                    if direction[axis] == 0:
                        if not low[axis] <= origin[axis] < low[axis] + size:
                            last = -1.
                    else:
                        interval = sorted(((low[axis] - origin[axis]) / direction[axis],
                                           (low[axis] + size - origin[axis]) / direction[axis]))
                        first, last = max(first, interval[0]), min(last, interval[1])
                if first < last:
                    hits.append((first, row))
            expected, transmission = np.zeros(3), 1.
            for _, row in sorted(hits):
                expected += transmission * alphas[row] * colors[row]
                transmission *= 1 - alphas[row]
            expected += transmission * np.array([.01, .02, .03])
            ray = mi.Ray3f(mi.Point3f(*origin), mi.Vector3f(*direction))
            ray.maxt = 5.
            for octree in (False, True):
                result = model.trace(ray, mi.Color3f(.01, .02, .03), 1e-6, 16, 0., octree)
                dr.eval(result)
                np.testing.assert_allclose(np.asarray(result[0]).ravel(), expected, atol=3e-6, rtol=3e-5)
                np.testing.assert_allclose(np.asarray(result[1]), 1-transmission, atol=3e-6)
                assert int(np.asarray(result[2])[0]) == len(hits)
        # A coarse proposal encompassing the finest leaf must be rejected even
        # when its lower corner is empty; the reverse overlap check is needed.
        proposal = {name: (mi.UInt64([encode(2, (3, 0, 0))]) if name == "cell_indices"
                          else mi.Float(np.asarray(value)[:(4 if name == "rotations" else 1 if name == "opacity_sh" else 3)]))
                    for name, value in state.items()}
        try:
            model.set_growth_candidates(proposal)
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError("coarse core proposal containing existing fine core leaf accepted")
    for bad in (TAG, TAG | 1, TAG | 2, TAG | (1 << 31), TAG | (1 << 32) | 8):
        invalid = {name: (mi.UInt64([bad]) if name == "cell_indices"
                         else mi.Float(np.asarray(value)[:(4 if name == "rotations" else 1 if name == "opacity_sh" else 3)]))
                   for name, value in state.items()}
        try:
            mi.VoxelReconstruction.from_state(4, [0.] * 3, [1.] * 3, invalid, 1, 1)
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError(f"invalid adaptive core address accepted: {bad}")
    return {"bounded_and_world_mixed_core_trace_size_legacy_cell_and_candidate_overlap": "passed",
            "invalid_core_path_reserved_bits_and_depth_rejected": "passed"}


def check_gradient_equivalence(mi, dr, np):
    from reconstruction_common import restore_model
    from test_reconstruction_distortion import fixture, evaluate
    source = fixture(np)
    legacy = restore_model(source, mi)
    state = dict(source["model_state"])
    state["cell_indices"] = np.array([encode(2, (int(k) % 4, int(k) // 4 % 4, int(k) // 16))
                                       for k in state["cell_indices"]], np.uint64)
    state = {name: mi.UInt64(value) if name == "cell_indices" else mi.Float(value)
             for name, value in state.items()}
    tagged = mi.VoxelReconstruction.from_state(4, [0.] * 3, [4.] * 3, state, 1, 1)
    for octree in (False, True):
        options = {"use_octree": float(octree), "max_candidates": 4.,
                   "collect_rgb_deletion_evidence": 1.}
        a = evaluate(legacy, mi, dr, np, coefficient=.1, native_options=options)
        b = evaluate(tagged, mi, dr, np, coefficient=.1, native_options=options)
        for key in ("loss", "rgb", "alpha", "geometry_valid", "appearance_valid", "deletion_gain"):
            if key in a:
                np.testing.assert_allclose(np.asarray(b[key]), np.asarray(a[key]), atol=4e-6, rtol=1e-4,
                                           err_msg="tagged_core." + key)
        for key in a["gradients"]:
            np.testing.assert_allclose(np.asarray(b["gradients"][key]), np.asarray(a["gradients"][key]),
                                       atol=4e-6, rtol=1e-4, err_msg="tagged_core.gradient." + key)
    return {"tagged_core_world_geometry_distortion_and_deletion_gradients_match_flat_reference": "passed"}


def check_oblique_block_exits(mi, dr, np):
    """Non-crossed axes must follow the ray, not retain their block-entry index.

    Ellipsoids cover each complete leaf, so this independent AABB oracle never
    shares native tree traversal, address decoding, or quadratic intersections.
    The first case crosses a large empty half-core before a tiny occupied leaf;
    the second crosses an occupied coarse leaf before two smaller leaves.
    """
    cases = (
        ("large_empty_to_fine", [(4, (8, 5, 2))],
         [(-.1, .05, .13), (1.1, .59, .18)], [(1., .45, .025), (-1., -.45, -.025)]),
        ("occupied_coarse_to_fine", [(2, (0, 0, 0)), (4, (4, 3, 2)), (5, (20, 10, 4))],
         [(-.1, .05, .13), (.8, .41, .1525)], [(1., .4, .025), (-1., -.4, -.025)]),
    )
    rng = np.random.default_rng(7081)
    for name, leaves, explicit_origins, explicit_directions in cases:
        lows = np.array([np.array(xyz) / 2**depth for depth, xyz in leaves])
        sizes = np.array([1 / 2**depth for depth, xyz in leaves])
        n = len(leaves)
        colors = np.array([[.8, .2, .1], [.1, .7, .3], [.3, .2, .9]], np.float32)[:n]
        alphas = np.array([.2, .35, .5])[:n]
        state = dict(cell_indices=mi.UInt64([encode(depth, xyz) for depth, xyz in leaves]),
            centers=mi.Float([.5] * (n * 3)), log_scales=mi.Float(np.repeat(np.log(.9 * sizes), 3)),
            rotations=mi.Float(np.tile([1., 0., 0., 0.], n)),
            radiance_sh=mi.Float((colors / Y00).ravel()),
            opacity_sh=mi.Float(np.log(alphas / (1-alphas)) / Y00))
        origins = np.r_[np.array(explicit_origins),
                        np.column_stack((np.full(128, -.1), rng.uniform(.01, .2, 128), rng.uniform(.10, .16, 128)))]
        directions = np.r_[np.array(explicit_directions),
                           np.column_stack((np.ones(128), rng.uniform(.08, .62, 128), rng.uniform(-.04, .06, 128)))]
        origins, directions = origins.astype(np.float32), directions.astype(np.float32)
        expected = []
        for origin, direction in zip(origins, directions):
            intersections = []
            for row, (low, size) in enumerate(zip(lows, sizes)):
                a, b = (low-origin)/direction, (low+size-origin)/direction
                entry, end = max(0., float(np.minimum(a, b).max())), min(5., float(np.maximum(a, b).min()))
                if entry < end:
                    intersections.append((entry, row))
            rgb, transmittance = np.zeros(3), 1.
            for _, row in sorted(intersections):
                rgb += transmittance * alphas[row] * colors[row]
                transmittance *= 1-alphas[row]
            rgb += transmittance * np.array([.01, .02, .03])
            expected.append((*rgb, 1-transmittance, len(intersections)))
        expected = np.array(expected)
        assert expected[0, 4] >= (1 if n == 1 else 2), name + " must exercise the intended intersections"
        ray = mi.Ray3f(mi.Point3f(*[mi.Float(origins[:, axis]) for axis in range(3)]),
                      mi.Vector3f(*[mi.Float(directions[:, axis]) for axis in range(3)]))
        ray.maxt = 5.
        for shells in (0, 1):
            model = mi.VoxelReconstruction.from_state(4, [0.] * 3, [1.] * 3, state,
                1, 1, 4, shells, 0, False, bool(shells))
            for octree in (False, True):
                rgb, alpha, hits = model.trace(ray, mi.Color3f(.01, .02, .03), 1e-6, 16, 0., octree)
                dr.eval(rgb, alpha, hits)
                actual = np.column_stack((np.asarray(rgb).T, np.asarray(alpha), np.asarray(hits)))
                np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-5,
                    err_msg=f"{name}.shells={shells}.octree={octree}")
    return {"oblique_empty_and_occupied_coarse_block_exits_find_fine_leaves": "passed",
            "fixed_random_oblique_rays_match_independent_AABB_oracle": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    results = check_cpu()
    if not args.cpu_only:
        with tempfile.TemporaryDirectory(prefix="mitsuba-core-adaptive-") as output:
            mi, dr, np, handles = initialize_runtime(RuntimeConfig(args.build_dir, args.gpu), Path(output), {})
            results.update(check_native(mi, dr, np))
            results.update(check_gradient_equivalence(mi, dr, np))
            results.update(check_oblique_block_exits(mi, dr, np))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
