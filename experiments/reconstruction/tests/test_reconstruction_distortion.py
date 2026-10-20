"""Generated CUDA checks for point-contribution ray distortion.

The independent double-precision oracle uses explicit pairwise distances and
finite differences. It does not share the native prefix/suffix recurrence.
Only tiny generated models are used; no dataset or experiment output is read.
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
LAMBDA = .7
ORIGIN = (1.30, 1.43, -1.)
DIRECTION = (.025, .014, 1.)
TARGET = (.2, .45, .7)


def quaternion_product(q, p, np):
    return np.r_[q[0] * p[0] - q[1:] @ p[1:],
                 q[0] * p[1:] + p[0] * q[1:] + np.cross(q[1:], p[1:])]


def rotation_matrix(q, np):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def fixture(np, count=3):
    axis = np.array([.31, -.74, .59])
    axis /= np.linalg.norm(axis)
    rotations = [np.r_[math.cos(angle/2), axis * math.sin(angle/2)]
                 for angle in (.43, -.31, .57)]
    return {
        "model_metadata": {
            "resolution": 4, "grid_min": [0.] * 3, "grid_max": [4.] * 3,
            "sh_count": 1, "opacity_sh_count": 1,
        },
        "model_state": {
            "cell_indices": np.array([5, 21, 37][:count], dtype=np.uint32),
            "centers": np.array([[.51, .49, .42], [.43, .52, .62], [.55, .46, .48]],
                                dtype=np.float64)[:count].ravel(),
            "log_scales": np.log(np.array([[.61, .43, .56], [.58, .40, .66], [.63, .39, .52]],
                                         dtype=np.float64))[:count].ravel(),
            "rotations": np.array(rotations, dtype=np.float64)[:count].ravel(),
            "radiance_sh": (np.array([[.3, .6, .8], [.8, .25, .4], [.45, .7, .2]],
                                    dtype=np.float64)[:count] / Y00).ravel(),
            "opacity_sh": (np.log(np.array([.23, .47, .61]) /
                                 (1 - np.array([.23, .47, .61])))[:count] / Y00),
        },
    }


def ray_interval(origin, direction, lower, upper, np):
    entry, end = 0., 20. * np.linalg.norm(np.asarray(DIRECTION))
    for axis in range(3):
        if direction[axis] == 0:
            if not lower[axis] <= origin[axis] < upper[axis]:
                return None
        else:
            a = (lower[axis] - origin[axis]) / direction[axis]
            b = (upper[axis] - origin[axis]) / direction[axis]
            entry, end = max(entry, min(a, b)), min(end, max(a, b))
    return (entry, end) if entry <= end else None


def oracle(checkpoint, np, origin=ORIGIN, direction=DIRECTION):
    """Independent world-space ellipsoid intersections and O(n²) objective."""
    metadata, state = checkpoint["model_metadata"], checkpoint["model_state"]
    lower, upper = np.array(metadata["grid_min"]), np.array(metadata["grid_max"])
    resolution = metadata["resolution"]
    size = (upper - lower) / resolution
    origin, direction = np.asarray(origin, dtype=float), np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    bounds = ray_interval(origin, direction, lower, upper, np)
    if bounds is None:
        return {"distortion": 0., "weights": np.array([]), "depths": np.array([]), "hits": []}
    grid_entry, grid_end = bounds
    hits = []
    for index, cell in enumerate(state["cell_indices"]):
        xyz = np.array([cell % resolution, cell // resolution % resolution, cell // resolution**2])
        interval = ray_interval(origin, direction, lower + xyz*size, lower + (xyz+1)*size, np)
        if interval is None:
            continue
        entry, end = max(grid_entry, interval[0]), min(grid_end, interval[1])
        if entry > end:
            continue
        center = lower + (xyz + state["centers"].reshape(-1, 3)[index]) * size
        scale = np.exp(state["log_scales"].reshape(-1, 3)[index])
        rotation = rotation_matrix(state["rotations"].reshape(-1, 4)[index], np)
        u = rotation.T @ (origin + entry*direction - center) / scale
        v = rotation.T @ ((end-entry)*direction) / scale
        a, b, c = v @ v, 2*(u @ v), u @ u - 1
        discriminant = b*b - 4*a*c
        if a <= 1e-20:
            intersects = c <= 0
            raw_star = 0.
        else:
            root = math.sqrt(max(discriminant, 0.))
            intersects = discriminant >= 0 and max(0., (-b-root)/(2*a)) <= min(1., (-b+root)/(2*a))
            raw_star = -(u @ v) / a
        if intersects:
            alpha = 1 / (1 + math.exp(-state["opacity_sh"][index]*Y00))
            depth = entry + np.clip(raw_star, 0., 1.)*(end-entry)
            hits.append((entry, index, alpha, (depth-grid_entry)/np.linalg.norm(upper-lower), raw_star))
    hits.sort()
    weights, depths, transmittance = [], [], 1.
    for _, _, alpha, depth, _ in hits:
        weights.append(transmittance*alpha)
        depths.append(depth)
        transmittance *= 1-alpha
    weights, depths = np.array(weights), np.array(depths)
    value = sum(2*weights[i]*weights[j]*abs(depths[i]-depths[j])
                for i in range(len(hits)) for j in range(i+1, len(hits)))
    return {"distortion": float(value), "weights": weights, "depths": depths, "hits": hits}


def perturb(checkpoint, name, index, amount, np):
    result = copy.deepcopy(checkpoint)
    if name == "rotation_tangent":
        voxel, axis = divmod(index, 3)
        delta = np.zeros(4)
        delta[0], delta[axis+1] = math.cos(amount/2), math.sin(amount/2)
        rotations = result["model_state"]["rotations"].reshape(-1, 4)
        rotations[voxel] = quaternion_product(rotations[voxel], delta, np)
    else:
        result["model_state"][name][index] += amount
    return result


def check_numpy_oracle():
    import numpy as np
    assert oracle(fixture(np, 0), np)["distortion"] == 0.
    assert oracle(fixture(np, 1), np)["distortion"] == 0.
    value = oracle(fixture(np), np)
    assert len(value["hits"]) == 3
    assert all(.1 < hit[4] < .9 for hit in value["hits"]), value["hits"]
    assert value["distortion"] > .01
    # For two contributors, the exact normalization/factor of two is explicit.
    two = oracle(fixture(np, 2), np)
    expected = 2*.23*(1-.23)*.47*abs(two["depths"][1]-two["depths"][0])
    np.testing.assert_allclose(two["distortion"], expected, atol=1e-15)
    return {"numpy_pairwise_oracle_and_stable_rotated_hard_hits": "passed"}


def ray(mi, np, origins=None):
    origins = np.asarray([ORIGIN] if origins is None else origins, dtype=np.float32).reshape(-1, 3)
    result = mi.Ray3f(mi.Point3f(*[mi.Float(origins[:, axis]) for axis in range(3)]),
                      mi.Vector3f(*DIRECTION))
    result.maxt = 20.
    return result


def evaluate(model, mi, dr, np, *, coefficient=LAMBDA, alpha=.3, inverse_pdf=1.,
             origins=None, replay=None, omit_weight=False, native_options=None):
    options = {"rgb_only": 1., "max_hits": 8., "max_candidates": 0.,
               "transmittance_threshold": .00001}
    if not omit_weight:
        options["distortion_loss_weight"] = coefficient
    if native_options:
        options.update(native_options)
    arguments = (ray(mi, np, origins), mi.Color3f(*TARGET), mi.Float(alpha),
                 mi.Color3f(1.), mi.Float(inverse_pdf), options)
    result = (model.loss_and_grad(*arguments) if replay is None else replay(model, *arguments))
    dr.eval(result)
    dr.sync_thread()
    return result


def assert_results_equal(a, b, np, *, exact=False):
    compare = np.testing.assert_array_equal if exact else np.testing.assert_allclose
    tolerances = {} if exact else {"atol": 2e-7, "rtol": 3e-5}
    for name in ("rgb", "raw_rgb", "alpha", "raw_alpha", "hit_count", "loss",
                 "rgb_loss", "distortion_loss", "appearance_valid", "geometry_valid"):
        compare(np.asarray(a[name]), np.asarray(b[name]), err_msg=name, **tolerances)
    for name in a["gradients"]:
        compare(np.asarray(a["gradients"][name]), np.asarray(b["gradients"][name]),
                err_msg="gradients."+name, **tolerances)


def check_values_and_baseline(mi, dr, np):
    for count in (0, 1, 2, 3):
        checkpoint = fixture(np, count)
        model = restore_model(checkpoint, mi)
        expected = oracle(checkpoint, np)
        result = evaluate(model, mi, dr, np)
        np.testing.assert_allclose(np.asarray(result["distortion_loss"]), expected["distortion"],
                                   atol=2e-7, rtol=3e-6)
        np.testing.assert_array_equal(np.asarray(result["hit_count"]), count)
        baseline = evaluate(model, mi, dr, np, coefficient=0.)
        default = evaluate(model, mi, dr, np, omit_weight=True)
        assert_results_equal(baseline, default, np, exact=True)
        np.testing.assert_allclose(np.asarray(result["loss"]),
            np.asarray(baseline["loss"]) + LAMBDA*expected["distortion"], atol=3e-7, rtol=3e-6)
        np.testing.assert_array_equal(np.asarray(result["rgb_loss"]), np.asarray(baseline["rgb_loss"]))
        for name in ("rgb", "raw_rgb", "alpha", "raw_alpha", "hit_count"):
            np.testing.assert_array_equal(np.asarray(result[name]), np.asarray(baseline[name]), err_msg=name)
        np.testing.assert_array_equal(np.asarray(result["gradients"]["radiance_sh"]),
                                     np.asarray(baseline["gradients"]["radiance_sh"]))
        if count < 2:
            for name in result["gradients"]:
                np.testing.assert_allclose(np.asarray(result["gradients"][name]),
                    np.asarray(baseline["gradients"][name]), atol=2e-7, rtol=3e-6, err_msg=name)
    model = restore_model(fixture(np), mi)
    first = evaluate(model, mi, dr, np)
    for alpha in (0., 1.):
        assert_results_equal(first, evaluate(model, mi, dr, np, alpha=alpha), np, exact=True)
    for coefficient in (-1., float("nan"), float("inf")):
        try:
            evaluate(model, mi, dr, np, coefficient=coefficient)
        except (RuntimeError, ValueError):
            pass
        else:
            raise AssertionError("Invalid distortion weight was accepted")
    return {"zero_one_multiple_hits_and_numpy_pairwise_value": "passed",
            "zero_weight_baseline_and_alpha_reference_independence": "passed",
            "RGB_plus_weighted_distortion_and_radiance_invariance": "passed"}


def check_single_hit_and_rgb_evidence(mi, dr, np):
    # At the exact closest point the existing RGB geometry proxy has zero
    # radius, so it contributes no geometry update counter. A zero distortion
    # term must not introduce a counter and advance Adam's geometry age.
    checkpoint = fixture(np, 1)
    checkpoint["model_state"]["centers"][:] = [.3375, .451, .5]
    checkpoint["model_state"]["rotations"][:] = [1., 0., 0., 0.]
    checkpoint["model_state"]["log_scales"][:] = math.log(.6)
    model = restore_model(checkpoint, mi)
    off = evaluate(model, mi, dr, np, coefficient=0.)
    on = evaluate(model, mi, dr, np)
    np.testing.assert_array_equal(np.asarray(off["hit_count"]), 1)
    np.testing.assert_array_equal(np.asarray(off["geometry_valid"]), 0)
    assert_results_equal(on, off, np, exact=True)
    for name in ("background_carve_valid", "foreground_evidence", "background_evidence"):
        np.testing.assert_array_equal(np.asarray(on[name]), np.asarray(off[name]))

    # Topology keeps its established RGB-only counterfactual criterion. The
    # added continuous regularizer must not quietly change growth/deletion.
    model = restore_model(fixture(np), mi)
    proposals = fixture(np, 1)["model_state"]
    proposals["cell_indices"][:] = 53
    model.set_growth_candidates({name: (mi.UInt32(value) if name == "cell_indices" else mi.Float(value))
                                 for name, value in proposals.items()})
    options = {"collect_growth_evidence": 1., "collect_rgb_deletion_evidence": 1., "max_growth_hits": 4.}
    off = evaluate(model, mi, dr, np, coefficient=0., native_options=options)
    on = evaluate(model, mi, dr, np, native_options=options)
    assert float(np.asarray(on["distortion_loss"])[0]) > 0
    assert np.asarray(on["growth_observed"]).sum() > 0
    assert np.asarray(on["deletion_observed"]).sum() == 3
    for name in ("growth_gain", "growth_observed", "growth_unknown",
                 "deletion_gain", "deletion_observed", "deletion_unknown"):
        np.testing.assert_allclose(np.asarray(on[name]), np.asarray(off[name]), atol=1e-7, rtol=3e-6, err_msg=name)
    return {"single_center_hit_keeps_zero_geometry_gradient_and_Adam_counter": "passed",
            "growth_and_deletion_evidence_remain_RGB_only": "passed"}


def check_gradients(mi, dr, np):
    checkpoint = fixture(np)
    model = restore_model(checkpoint, mi)
    off, on = evaluate(model, mi, dr, np, coefficient=0.), evaluate(model, mi, dr, np)
    errors = {}
    for name, size in (("centers", 9), ("log_scales", 9), ("rotation_tangent", 9), ("opacity_sh", 3)):
        expected, native_fd = [], []
        # The pre-existing RGB geometry derivative is a proxy. Subtract its
        # identical on/off value before checking the smooth distortion term.
        actual = (np.asarray(on["gradients"][name], dtype=np.float64) -
                  np.asarray(off["gradients"][name], dtype=np.float64)) / LAMBDA
        for index in range(size):
            eps = 2e-5
            positive, negative = perturb(checkpoint, name, index, eps, np), perturb(checkpoint, name, index, -eps, np)
            expected.append((oracle(positive, np)["distortion"] - oracle(negative, np)["distortion"]) / (2*eps))
            eps = .001
            values = []
            for sign in (1, -1):
                changed = perturb(checkpoint, name, index, sign*eps, np)
                candidate = restore_model(changed, mi)
                result = evaluate(candidate, mi, dr, np)
                assert int(np.asarray(result["hit_count"])[0]) == 3
                values.append(float(np.asarray(result["distortion_loss"])[0]))
            native_fd.append((values[0]-values[1])/(2*eps))
        np.testing.assert_allclose(actual, expected, atol=1.5e-5, rtol=8e-4, err_msg=name+" oracle")
        np.testing.assert_allclose(actual, native_fd, atol=4e-5, rtol=8e-3, err_msg=name+" native finite difference")
        errors[name] = float(np.max(np.abs(actual-np.asarray(expected))))
    return {"native_distortion_gradients_vs_independent_and_native_finite_difference": "passed",
            "gradient_max_absolute_errors": errors}


def check_clipped_depth_and_weights(mi, dr, np):
    for center_z, expected_star in ((-.4, 0), (1.4, 1)):
        checkpoint = fixture(np, 2)
        checkpoint["model_state"]["centers"][:3] = [1.34-1, 1.45-1, center_z]
        checkpoint["model_state"]["log_scales"][:3] = np.log([.8, .8, .8])
        checkpoint["model_state"]["rotations"][:4] = [1., 0., 0., 0.]
        expected = oracle(checkpoint, np)
        assert len(expected["hits"]) == 2
        star = expected["hits"][0][4]
        assert (star < 0 if expected_star == 0 else star > 1), star
        model = restore_model(checkpoint, mi)
        off, on = evaluate(model, mi, dr, np, coefficient=0.), evaluate(model, mi, dr, np)
        for name in ("centers", "log_scales", "rotation_tangent"):
            difference = np.asarray(on["gradients"][name])-np.asarray(off["gradients"][name])
            np.testing.assert_allclose(difference[:3], 0., atol=2e-7, err_msg=name)
        np.testing.assert_allclose(np.asarray(on["distortion_loss"]), expected["distortion"], atol=2e-7)
    model = restore_model(fixture(np), mi)
    baseline, weighted = evaluate(model, mi, dr, np), evaluate(model, mi, dr, np, inverse_pdf=2.75)
    baseline_off = evaluate(model, mi, dr, np, coefficient=0.)
    weighted_off = evaluate(model, mi, dr, np, coefficient=0., inverse_pdf=2.75)
    # Legacy RGB weights the color estimator before squaring its residual.
    # Only the newly added distortion objective/gradient is linear in 1/pdf.
    np.testing.assert_allclose(np.asarray(weighted["distortion_loss"]),
                               2.75*np.asarray(baseline["distortion_loss"]), atol=4e-7, rtol=5e-6)
    np.testing.assert_allclose(np.asarray(weighted["loss"])-np.asarray(weighted_off["loss"]),
        2.75*(np.asarray(baseline["loss"])-np.asarray(baseline_off["loss"])), atol=4e-7, rtol=5e-6)
    for name in baseline["gradients"]:
        np.testing.assert_allclose(np.asarray(weighted["gradients"][name])-np.asarray(weighted_off["gradients"][name]),
            2.75*(np.asarray(baseline["gradients"][name])-np.asarray(baseline_off["gradients"][name])),
            atol=3e-6, rtol=3e-5, err_msg=name)
    return {"clipped_hit_depth_has_zero_distortion_geometry_gradient": "passed",
            "inverse_pdf_multiplies_loss_and_gradients_once": "passed"}


def check_chunks_and_replay(mi, dr, np):
    checkpoint = fixture(np)
    model = restore_model(checkpoint, mi)
    origins = np.array([ORIGIN, [1.35, 1.42, -1.], [1.39, 1.46, -1.],
                       [1.28, 1.49, -1.], [1.41, 1.41, -1.]])
    pdf = np.array([.7, 1., 1.5, 2., .9], dtype=np.float32)
    full = evaluate(model, mi, dr, np, origins=origins, inverse_pdf=pdf)
    for partitions in ((slice(0, 2), slice(2, 5)), tuple(slice(i, i+1) for i in range(5))):
        chunks = [evaluate(model, mi, dr, np, origins=origins[s], inverse_pdf=pdf[s]) for s in partitions]
        for name in ("loss", "rgb_loss", "distortion_loss"):
            expected = np.asarray(full[name], dtype=np.float64).sum()/len(origins)
            actual = sum(np.asarray(item[name], dtype=np.float64).sum() for item in chunks)/len(origins)
            np.testing.assert_allclose(actual, expected, atol=3e-7, rtol=3e-6)
        for name in full["gradients"]:
            actual = sum(np.asarray(item["gradients"][name], dtype=np.float64) for item in chunks)/len(origins)
            np.testing.assert_allclose(actual, np.asarray(full["gradients"][name])/len(origins),
                                       atol=3e-7, rtol=3e-5, err_msg=name)
    replay = NativeLossReplay(dr)
    for coefficient in (LAMBDA, 0., LAMBDA):
        direct = evaluate(model, mi, dr, np, coefficient=coefficient, origins=origins, inverse_pdf=pdf)
        for _ in range(2):
            recorded = evaluate(model, mi, dr, np, coefficient=coefficient, origins=origins,
                                inverse_pdf=pdf, replay=replay)
            assert_results_equal(direct, recorded, np)
    checkpoint["model_state"]["opacity_sh"][0] += .3
    model.set_parameters({"opacity_sh": mi.Float(checkpoint["model_state"]["opacity_sh"])})
    changed = evaluate(model, mi, dr, np, origins=origins, inverse_pdf=pdf)
    recorded = evaluate(model, mi, dr, np, origins=origins, inverse_pdf=pdf, replay=replay)
    assert_results_equal(changed, recorded, np)
    assert replay.statistics()["replays"] >= 3, replay.statistics()
    return {"chunk_and_unequal_shard_sum_then_global_normalization": "passed",
            "native_replay_weight_changes_and_live_parameters": "passed"}


def check_training_components(mi, dr, np):
    from reconstruction_common import UniformSampler
    from train import TrainConfig, compute_view_shard, finalize_view, train_view

    class Transfers:
        def __init__(self):
            self.widths = []

        def __getattr__(self, name):
            return getattr(np, name)

        def asarray(self, value, *args, **kwargs):
            if dr.is_array_v(type(value)):
                self.widths.append(dr.width(value))
            return np.asarray(value, *args, **kwargs)

    config = TrainConfig()
    config.training.rgb_only = True
    config.topology.enable_deletion = False
    config.gradient.distortion_loss_weight = LAMBDA
    config.render.width, config.render.height = 5, 3
    config.render.max_hits = 8
    view = {"index": 0, "camera": {"position": [1.4, 1.45, -1.],
            "forward": [0., 0., 1.], "right": [1., 0., 0.],
            "up": [0., 1., 0.], "camera_angle_x": .12}}
    reference = np.tile(np.asarray(TARGET, dtype=np.float32), (3, 5, 1))
    model, results, components = restore_model(fixture(np), mi), [], []
    for chunk in (0, 4):
        config.training.chunk_size = chunk
        transfer, losses = Transfers(), {}
        result = train_view(model, view, reference, config, UniformSampler(np),
            np.random.default_rng(591), mi, dr, transfer, loss_components=losses)
        assert transfer.widths == [4], transfer.widths
        assert result[2] is None
        assert "rgb_loss_sum" not in result[0] and "distortion_loss_sum" not in result[0]
        assert losses["distortion_loss"] > 0
        np.testing.assert_allclose(losses["weighted_distortion_loss"],
                                   LAMBDA*losses["distortion_loss"], atol=1e-12)
        np.testing.assert_allclose(result[1], losses["rgb_loss"]+losses["weighted_distortion_loss"],
                                   atol=3e-7, rtol=3e-6)
        results.append(result)
        components.append(losses)
    for key in components[0]:
        np.testing.assert_allclose(components[0][key], components[1][key], atol=3e-7, rtol=3e-6)
    for key in results[0][0]["gradients"]:
        np.testing.assert_allclose(np.asarray(results[0][0]["gradients"][key]),
            np.asarray(results[1][0]["gradients"][key]), atol=2e-6, rtol=3e-5, err_msg=key)

    # Emulate two unequal pixel shards on the same GPU, combining device sums
    # before finalize_view divides by the full image's fifteen samples.
    sampler = UniformSampler(np)
    offsets, inverse_pdf, token = sampler.sample(0, reference, np.random.default_rng(591))
    partials = [compute_view_shard(model, view, reference, config, offsets, inverse_pdf,
                token, mi, dr, pixel_begin=begin, pixel_end=end)
                for begin, end in ((0, 7), (7, 15))]
    a, b = partials[0][0], partials[1][0]
    accumulator = {"gradients": {name: a["gradients"][name]+b["gradients"][name]
                                for name in a["gradients"]}}
    for name in a:
        if name != "gradients":
            accumulator[name] = (dr.maximum(a[name], b[name]) if name in
                ("foreground_evidence", "background_evidence") else a[name]+b[name])
    combined = (accumulator, partials[0][1]+partials[1][1], partials[0][2] & partials[1][2], None, None)
    transfer, losses = Transfers(), {}
    result = finalize_view(combined, view, config, sampler, token, mi, dr, transfer, loss_components=losses)
    assert transfer.widths == [4], transfer.widths
    for key in losses:
        np.testing.assert_allclose(losses[key], components[0][key], atol=3e-7, rtol=3e-6)
    np.testing.assert_allclose(result[1], results[0][1], atol=3e-7, rtol=3e-6)
    for name in result[0]["gradients"]:
        np.testing.assert_allclose(np.asarray(result[0]["gradients"][name]),
            np.asarray(results[0][0]["gradients"][name]), atol=2e-6, rtol=3e-5, err_msg=name)

    config.gradient.distortion_loss_weight = 0.
    transfer, losses = Transfers(), {}
    result = train_view(model, view, reference, config, UniformSampler(np),
        np.random.default_rng(591), mi, dr, transfer, loss_components=losses)
    assert transfer.widths == [2], transfer.widths
    assert losses == {"rgb_loss": result[1], "distortion_loss": 0., "weighted_distortion_loss": 0.}
    return {"train_view_components_chunk_and_unequal_shard_normalization": "passed",
            "enabled_four_scalar_disabled_two_scalar_single_readback": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    checks = [check_numpy_oracle()]
    print(json.dumps(checks[-1]), flush=True)
    if not args.cpu_only:
        with tempfile.TemporaryDirectory(prefix="mitsuba-distortion-regression-") as temporary:
            mi, dr, np, handles = initialize_runtime(RuntimeConfig(args.build_dir, args.gpu),
                                                    Path(temporary), {})
            assert int(getattr(mi.VoxelReconstruction, "distortion_version", 0)) >= 1, \
                "Rebuild this checkout before running distortion regression"
            for check in (check_values_and_baseline, check_single_hit_and_rgb_evidence, check_gradients,
                          check_clipped_depth_and_weights, check_chunks_and_replay,
                          check_training_components):
                checks.append(check(mi, dr, np))
                print(json.dumps(checks[-1]), flush=True)
    print(json.dumps({"status": "passed", "checks": checks}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
