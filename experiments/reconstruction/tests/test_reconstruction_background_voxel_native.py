"""Independent cell-preimage checks for optional pure projective background voxels.

World-space halfspaces determine background membership. The oracle never reads
background ellipsoid parameters; core ellipsoids use the existing independent
projective oracle. CUDA checks are serialized by this standalone entry point.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preview import RuntimeConfig
from reconstruction_common import initialize_runtime, restore_model
from reconstruction_optimizer import FalcorOptimizer, OptimizerConfig
from reconstruction_replay import NativeLossReplay
from test_reconstruction_dssim import dssim_oracle
from test_reconstruction_projective_native import Y00, make_ray, native, oracle as ellipsoid_oracle


GEOMETRY = ("centers", "log_scales", "rotation_tangent")


def fixture(np):
    core, layers, grid = 4, 2, 8
    coords = [(x, y, z) for z in range(grid) for y in range(grid) for x in range(grid)
              if sum(not layers <= c < layers + core for c in (x, y, z)) <= 1]
    count = len(coords)
    rng = np.random.default_rng(824)
    axes = rng.normal(size=(count, 3))
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    angles = rng.uniform(-.4, .4, count)
    return {"model_metadata": {"resolution": grid, "core_resolution": core,
        "grid_min": [-2.] * 3, "grid_max": [2.] * 3,
        "scene_mapping": "projective_shell", "background_layers": layers,
        "background_refinement": "preserve_physical_width",
        "background_use_ellipsoid": False, "sh_count": 1, "opacity_sh_count": 1},
        "model_state": {
            "cell_indices": np.array([x + grid * (y + grid * z) for x, y, z in coords], np.uint32),
            "centers": np.tile([.53, .46, .48], count),
            "log_scales": np.tile(np.log([.32, .27, .29]), count),
            "rotations": np.column_stack((np.cos(angles/2), axes*np.sin(angles/2)[:, None])).ravel(),
            "radiance_sh": (rng.uniform(.15, .85, (count, 3))/Y00).ravel(),
            "opacity_sh": np.full(count, math.log(.17/.83)/Y00)}}


def background_mask(payload, np):
    md = payload["model_metadata"]
    grid, core = md["resolution"], md["core_resolution"]
    pad = (grid-core)//2
    cells = payload["model_state"]["cell_indices"]
    coords = np.column_stack((cells % grid, cells//grid % grid, cells//grid**2))
    return np.any((coords < pad) | (coords >= pad+core), axis=1)


def subset(payload, selection):
    result = copy.deepcopy(payload)
    count = len(payload["model_state"]["cell_indices"])
    result["model_state"] = {name: value.reshape(count, -1)[selection].ravel().copy()
                             for name, value in payload["model_state"].items()}
    return result


def oracle(payload, origin, direction, np, *, maxt=math.inf, max_hits=64, threshold=0.):
    md, state = payload["model_metadata"], payload["model_state"]
    grid, core = md["resolution"], md["core_resolution"]
    pad, size = (grid-core)//2, 2/core
    far = float(md.get("background_far_distance", 0.))
    width = 2*pad/core / (1-1/far if far else 1.)
    o, d = np.asarray(origin, float), np.asarray(direction, float)
    length = np.linalg.norm(d)
    if length == 0 or maxt <= 0:
        return dict(rgb=np.zeros(3), alpha=0., hits=[], distortion=0.)
    d = d/length
    bg = background_mask(payload, np)
    core_ids = np.flatnonzero(~bg)
    core_result = ellipsoid_oracle(subset(payload, ~bg), origin, direction, np,
                                  maxt=maxt, max_hits=128)
    hits = [(t, int(core_ids[row]), a, c, z, star)
            for t, row, a, c, z, star in core_result["hits"]]
    basis = np.array([Y00, -d[1]*Y00*math.sqrt(3), -d[2]*Y00*math.sqrt(3),
                      -d[0]*Y00*math.sqrt(3)])
    sh, osh = md["sh_count"], md["opacity_sh_count"]
    radiance = state["radiance_sh"].reshape(-1, sh, 3)
    opacity = state["opacity_sh"].reshape(-1, osh)
    for row in np.flatnonzero(bg):
        cell_id = int(state["cell_indices"][row])
        cell = np.array([cell_id % grid, cell_id//grid % grid, cell_id//grid**2])
        low, high = -1-pad*size+cell*size, -1-pad*size+(cell+1)*size
        axis = int(np.flatnonzero((cell < pad) | (cell >= pad+core))[0])
        sign = -1 if cell[axis] < pad else 1
        h0, hd = sign*o[axis], sign*d[axis]
        bounds, valid = [0., maxt*length], True

        def clip(a, b, strict=False):
            nonlocal valid
            if abs(b) < 1e-14:
                valid &= a > 0 if strict else a >= -1e-12
            elif b > 0:
                bounds[0] = max(bounds[0], -a/b)
            else:
                bounds[1] = min(bounds[1], -a/b)

        clip(h0-1, hd, True)
        qlow, qhigh = sorted((sign*low[axis], sign*high[axis]))
        min_h = width/(1+width-qlow)
        max_h = math.inf if abs(1+width-qhigh) < 1e-12 else width/(1+width-qhigh)
        # Spatial cells own their lower mapped boundary. On negative faces
        # increasing h reverses that ordering: the radial lower bound is open,
        # while the radial upper bound belongs to the nearer physical cell.
        clip(h0-min_h, hd, sign < 0 or abs(min_h-1) < 1e-12)
        if math.isfinite(max_h):
            outer_boundary = far and math.isclose(max_h, far, rel_tol=1e-12)
            clip(max_h-h0, -hd, sign > 0 and not outer_boundary)
        if far:
            clip(far-h0, -hd)
        for j in range(3):
            if j == axis:
                continue
            clip(o[j]-low[j]*h0, d[j]-low[j]*hd)
            clip(high[j]*h0-o[j], high[j]*hd-d[j], cell[j] != pad+core-1)
            if j < axis:
                clip(h0-o[j], hd-d[j], True)
                clip(h0+o[j], hd+d[j], True)
        begin, end = bounds
        if not valid or end <= begin:
            continue

        def mapped(t):
            if math.isinf(t):
                q = d/(sign*d[axis])
                q[axis] = sign*(1+width)
                return q
            p = o+t*d
            h = sign*p[axis]
            q = p/h
            # This algebra remains stable for a finite outer cube near h=1.
            q[axis] = sign*(1+width*(h-1)/h)
            return q

        midpoint = (mapped(begin)+mapped(end))*.5
        h = width/(1+width-sign*midpoint[axis])
        world = midpoint*h
        world[axis] = sign*h
        t = float((world-o)@d)
        depth = t/(t+1)
        alpha = 1/(1+math.exp(-float(opacity[row]@basis[:osh])))
        color = np.maximum(0., basis[:sh]@radiance[row])
        hits.append((begin, int(row), alpha, color, depth, .5))
    hits.sort(key=lambda hit: (hit[0], hit[1]))
    color, transmittance, used, weights = np.zeros(3), 1., [], []
    for hit in hits:
        if len(used) >= max_hits or transmittance <= threshold:
            break
        weight = transmittance*hit[2]
        color += weight*hit[3]
        transmittance *= 1-hit[2]
        used.append(hit)
        weights.append(weight)
    distortion = sum(2*weights[i]*weights[j]*abs(used[i][4]-used[j][4])
                     for i in range(len(used)) for j in range(i+1, len(used)))
    return dict(rgb=color, alpha=1-transmittance, hits=used, distortion=distortion)


def two_background_cells(np):
    payload = fixture(np)
    cells = payload["model_state"]["cell_indices"]
    payload = subset(payload, np.isin(cells, [4+8*(4+8*z) for z in (6, 7)]))
    payload["model_state"]["radiance_sh"] = np.array([1., 1., 1., .6, .8, 1.])/Y00
    payload["model_state"]["opacity_sh"][:] = 0.
    # Neither ray intersects these tiny, distant ellipsoids.
    payload["model_state"]["centers"][:] = 4.
    payload["model_state"]["log_scales"][:] = math.log(.0001)
    return payload


def check_numpy():
    import numpy as np
    payload = two_background_cells(np)
    origin, direction = (.13, .22, 0.), (0., 0., 1.)
    result = oracle(payload, origin, direction, np)
    np.testing.assert_allclose(result["rgb"], [.65, .7, .75], atol=1e-12)
    assert result["alpha"] == .75 and len(result["hits"]) == 2
    # Positive intervals carry full per-hit alpha, even when arbitrarily short.
    first_positive = float(np.nextafter(np.float32(1.), np.float32(math.inf)))
    for maxt in (first_positive, 1.00001, 1.25, 2.):
        one = oracle(payload, origin, direction, np, maxt=maxt)
        assert one["alpha"] == .5 and len(one["hits"]) == 1
    for maxt in (0., 1.):
        assert not oracle(payload, origin, direction, np, maxt=maxt)["hits"]
    changed = copy.deepcopy(payload)
    for name in ("centers", "log_scales", "rotations"):
        changed["model_state"][name][:] += 12.
    altered = oracle(changed, origin, direction, np)
    np.testing.assert_array_equal(result["rgb"], altered["rgb"])
    assert result["distortion"] == altered["distortion"]
    return {"independent_preimage_fixed_alpha_midpoint_oracle": "passed"}


def comparison(actual, expected, np):
    assert actual.keys() == expected.keys(), (actual.keys(), expected.keys())
    for key in actual:
        if key != "gradients":
            np.testing.assert_allclose(np.asarray(actual[key]), np.asarray(expected[key]),
                                       atol=5e-6, rtol=4e-5, err_msg=key)
    for name in actual["gradients"]:
        np.testing.assert_allclose(np.asarray(actual["gradients"][name]),
            np.asarray(expected["gradients"][name]), atol=4e-5, rtol=4e-4, err_msg=name)


def with_record_layout(model, mi, dr, ray, compact):
    options = {"rgb_only": 1., "max_hits": 64., "max_candidates": 4.,
        "use_octree": 1., "distortion_loss_weight": .7, "transmittance_threshold": 0.,
        "collect_rgb_deletion_evidence": 1., "collect_growth_evidence": 0.,
        "max_growth_hits": 8., "dssim_loss_weight": .2, "image_width": 3.,
        "image_height": 2., "compact_path_records": float(compact)}
    result = model.loss_and_grad(ray, mi.Color3f(.3, .4, .5), mi.Float(0),
                                mi.Color3f(0), mi.Float(1), options)
    dr.eval(result)
    dr.sync_thread()
    return result


def check_world_preimages(mi, dr, np):
    payload = fixture(np)
    model = restore_model(payload, mi)
    assert model.native_background_use_ellipsoid() is False
    rays = [((.8, 0., 0.), (.6, 0., .8)), ((0., 0., 0.), (1., 1., 0.)),
            ((0., 0., 0.), (-1., -1., -1.)), ((1., 0., 0.), (0., 1., 0.)),
            ((1.8, -.7, .2), (-.2, .8, .1)), ((3., .15, .23), (-1., .02, .04))]
    cases = 0
    nearest_far = float(np.nextafter(np.float32(1.), np.float32(math.inf)))
    for far in (0., nearest_far, 1.125, 2.5):
        payload["model_metadata"]["background_far_distance"] = far
        model.set_background_far_distance(far)
        # The existing projective cursor represents absolute world t in
        # float32. With H only one ULP above 1, an oblique or distant-origin
        # chart can collapse before any cell is visited (e.g. origin
        # (3,.15,.23), direction (-1,.02,.04), on its outgoing -X chart).
        # This mode change retains that cursor contract. Test the nearest H
        # on its supported axis-aligned cases; ordinary H keeps all oblique
        # rays above. Do not report the extreme-H general case as supported.
        selected = [] if far == nearest_far else list(rays)
        for axis in range(3):
            for sign in (-1., 1.):
                origin = np.array([.13, .22, .17])
                direction = np.zeros(3)
                direction[axis] = sign
                selected.append((tuple(origin), tuple(direction)))
                origin[axis] = sign*(2*far if far else 5.)
                selected.extend(((tuple(origin), tuple(direction)),
                                 (tuple(origin), tuple(-direction))))
                if not far:
                    origin[axis] = sign*2.
                    tangent = np.zeros(3)
                    tangent[(axis+1) % 3] = 1.
                    selected.append((tuple(origin), tuple(tangent)))
                if far:
                    origin[axis] = sign*far
                    tangent = np.zeros(3)
                    tangent[(axis+1) % 3] = 1.
                    selected.extend(((tuple(origin), tuple(direction)),
                                     (tuple(origin), tuple(-direction)),
                                     (tuple(origin), tuple(tangent))))
        for maxt in (math.inf, .7, 2.3, 0.):
            ray = make_ray(mi, np, [o for o, _ in selected], [d for _, d in selected], maxt)
            expected = [oracle(payload, o, d, np, maxt=maxt) for o, d in selected]
            for octree in (False, True):
                result = native(model, mi, dr, ray, use_octree=octree, evidence=True)
                rgb, alpha, hits = model.trace(ray, mi.Color3f(0), 0., 64, 0., octree)
                dr.eval(rgb, alpha, hits)
                context = f"H={far}, maxt={maxt}, octree={octree}"
                np.testing.assert_array_equal(np.asarray(hits), [len(e["hits"]) for e in expected], err_msg=context)
                np.testing.assert_allclose(np.asarray(rgb).T, [e["rgb"] for e in expected], atol=5e-6, rtol=4e-5, err_msg=context)
                np.testing.assert_allclose(np.asarray(alpha), [e["alpha"] for e in expected], atol=5e-6, rtol=4e-5, err_msg=context)
                np.testing.assert_allclose(np.asarray(result["raw_rgb"]), np.asarray(rgb), atol=3e-6)
                np.testing.assert_allclose(np.asarray(result["raw_alpha"]), np.asarray(alpha), atol=3e-6)
                np.testing.assert_allclose(np.asarray(result["distortion_loss"]), [e["distortion"] for e in expected], atol=4e-6, rtol=5e-5, err_msg=context)
                cases += len(selected)
    return {"six_face_finite_infinite_negative_seam_zero_interval_trace_train_cases": cases,
            "nearest_representable_H_axis_aligned_origin_and_outer_boundary": "passed",
            "known_limitations": "For H one float32 ULP above 1, existing float32 absolute world-t chart bounds can collapse for oblique rays or distant origins; general support for those extreme-H cases is not asserted."}


def check_parameters_and_gradients(mi, dr, np):
    payload = fixture(np)
    count = len(payload["model_state"]["cell_indices"])
    payload["model_metadata"].update(sh_count=4, opacity_sh_count=4, background_far_distance=2.5)
    rng = np.random.default_rng(149)
    color = np.zeros((count, 4, 3))
    color[:, 0] = payload["model_state"]["radiance_sh"].reshape(count, 3)
    color[:, 1:] = rng.uniform(-.08, .08, (count, 3, 3))
    opacity = np.zeros((count, 4))
    opacity[:, 0] = payload["model_state"]["opacity_sh"]
    opacity[:, 1:] = rng.uniform(-.1, .1, (count, 3))
    payload["model_state"].update(radiance_sh=color.ravel(), opacity_sh=opacity.ravel())
    rays = [((.8, .02*y, .02*x), (.6, .03*x, .8)) for y in range(2) for x in range(3)]
    ray = make_ray(mi, np, [o for o, _ in rays], [d for _, d in rays])
    model = restore_model(payload, mi)
    result = native(model, mi, dr, ray, evidence=True, dssim_weight=.2, image_size=(3, 2))
    for compact in (False, True):
        comparison(with_record_layout(model, mi, dr, ray, compact), result, np)
    bg = background_mask(payload, np)
    for name in GEOMETRY:
        np.testing.assert_array_equal(np.asarray(result["gradients"][name]).reshape(count, 3)[bg], 0.)
    np.testing.assert_array_equal(np.asarray(result["geometry_valid"])[bg], 0)
    assert any(np.any(np.asarray(result["gradients"][name]).reshape(count, 3)[~bg]) for name in GEOMETRY)
    expected = [oracle(payload, o, d, np) for o, d in rays]
    np.testing.assert_allclose(np.asarray(result["raw_rgb"]).T, [e["rgb"] for e in expected], atol=4e-6)

    def loss(checkpoint):
        values = [oracle(checkpoint, o, d, np) for o, d in rays]
        rgb = np.array([e["rgb"] for e in values]).reshape(2, 3, 3)
        target = np.broadcast_to([.3, .4, .5], rgb.shape)
        return (np.sum((rgb-target)**2) + .7*sum(e["distortion"] for e in values)
                + .2*dssim_oracle(rgb, target, np).sum())

    hit_rows = sorted({h[1] for e in expected for h in e["hits"]})
    selected = [next(row for row in hit_rows if not bg[row])]
    selected += [row for row in hit_rows if bg[row]]
    checks = 0
    for name, width in (("radiance_sh", 12), ("opacity_sh", 4)):
        for row in selected:
            for component in range(width):
                index, epsilon = row*width+component, 2e-5
                plus, minus = copy.deepcopy(payload), copy.deepcopy(payload)
                plus["model_state"][name][index] += epsilon
                minus["model_state"][name][index] -= epsilon
                numerical = (loss(plus)-loss(minus))/(2*epsilon)
                actual = np.asarray(result["gradients"][name])[index]
                np.testing.assert_allclose(actual, numerical, atol=2e-5, rtol=4e-3, err_msg=f"{name}[{index}]")
                checks += 1
    changed = copy.deepcopy(payload)
    for name, width in (("centers", 3), ("log_scales", 3), ("rotations", 4)):
        rows = changed["model_state"][name].reshape(count, width)
        rows[bg] = (np.tile([.5, .5, .5, .5], (bg.sum(), 1)) if width == 4
                    else (7. if name == "centers" else -12.))
    altered = restore_model(changed, mi)
    comparison(native(altered, mi, dr, ray, evidence=True, dssim_weight=.2, image_size=(3, 2)), result, np)
    return {"color_opacity_SH_RGB_distortion_DSSIM_finite_differences": checks,
            "background_geometry_independence_zero_gradients_core_still_trains": "passed",
            "compact_and_legacy_path_records_all_outputs_gradients_evidence": "passed"}


def check_deletion_and_replay(mi, dr, np):
    payload = two_background_cells(np)
    origin, direction = (.13, .22, 0.), (0., 0., 1.)
    ray = make_ray(mi, np, [origin], [direction])
    model = restore_model(payload, mi)
    expected = oracle(payload, origin, direction, np)
    result = native(model, mi, dr, ray, evidence=True)
    target = np.array([.3, .4, .5])
    for row in range(2):
        without = oracle(subset(payload, np.arange(2) != row), origin, direction, np)
        gain = np.sum((expected["rgb"]-target)**2)-np.sum((without["rgb"]-target)**2)
        np.testing.assert_allclose(np.asarray(result["deletion_gain"])[row], gain, atol=3e-6)
    assert np.asarray(result["deletion_gain"])[0] > .1
    np.testing.assert_array_equal(np.asarray(result["deletion_observed"]), [1, 1])
    np.testing.assert_array_equal(np.asarray(result["deletion_unknown"]), [0, 0])
    perfect = native(model, mi, dr, ray, evidence=True, reference=expected["rgb"])
    assert np.all(np.asarray(perfect["deletion_gain"]) < 0.)
    for cap, threshold in ((1, 0.), (64, .6)):
        truncated = native(model, mi, dr, ray, evidence=True, max_hits=cap, threshold=threshold)
        np.testing.assert_array_equal(np.asarray(truncated["hit_count"]), [1])
        np.testing.assert_array_equal(np.asarray(truncated["deletion_gain"]), [0., 0.])
        np.testing.assert_array_equal(np.asarray(truncated["deletion_observed"]), [0, 0])
        np.testing.assert_array_equal(np.asarray(truncated["deletion_unknown"]), [1, 0])
        from reconstruction_deletion_rgb import initialize_pending, observe_numpy
        pending = initialize_pending(2, 1, 8, np)
        observe_numpy(pending, truncated, 1, 0., np)
        np.testing.assert_array_equal(pending["support_views"], [0, 0])
        np.testing.assert_array_equal(pending["observed_views"], [0, 0])
    # A short positive cell segment still consumes exactly one fixed alpha.
    first_positive = float(np.nextafter(np.float32(1.), np.float32(math.inf)))
    for maxt, hits in ((0., 0), (1., 0), (first_positive, 1), (1.0001, 1), (1.25, 1), (2., 1)):
        short = make_ray(mi, np, [origin], [direction], maxt)
        got = native(model, mi, dr, short)
        np.testing.assert_array_equal(np.asarray(got["hit_count"]), [hits])
        np.testing.assert_allclose(np.asarray(got["raw_alpha"]), [hits*.5], atol=2e-6)
    replay = NativeLossReplay(dr)
    for enabled in (False, False, True, True, False):
        before, old = int(model.topology_revision), model.native_background_use_ellipsoid()
        model.set_background_use_ellipsoid(enabled)
        assert model.native_background_use_ellipsoid() is enabled
        assert int(model.topology_revision) == before+int(old != enabled)
        direct = native(model, mi, dr, ray, evidence=True)
        comparison(native(model, mi, dr, ray, evidence=True, replay=replay), direct, np)
        np.testing.assert_array_equal(np.asarray(direct["hit_count"]), [0 if enabled else 2])
    assert replay.statistics()["topology_changes"] == 2, replay.statistics()
    legacy = copy.deepcopy(payload)
    legacy["model_metadata"].pop("background_use_ellipsoid")
    assert restore_model(legacy, mi).native_background_use_ellipsoid() is True
    return {"exact_positive_negative_background_removal_gain": "passed",
            "truncated_unknown_does_not_cast_deletion_vote": "passed",
            "short_positive_interval_fixed_alpha_zero_length_exclusion": "passed",
            "mode_switch_invalidates_frozen_replay_legacy_default": "passed"}


def check_adam_freeze(mi, dr, np):
    payload = fixture(np)
    cells = payload["model_state"]["cell_indices"]
    payload = subset(payload, np.isin(cells, [3+8*(3+8*3), 4+8*(4+8*6)]))
    bg = background_mask(payload, np)
    for mode in ("dense", "device", "compact", "compact_reference"):
        model = restore_model(copy.deepcopy(payload), mi)
        optimizer = FalcorOptimizer(model, OptimizerConfig(geometry_direction_guard=False), sparse_updates=mode != "dense")
        for name, width in optimizer.dimensions.items():
            optimizer.state[name] = {"mean": mi.Float(np.full(2*width, .3)),
                                     "squared_mean": mi.Float(np.full(2*width, .09)),
                                     "steps": mi.UInt32([3, 3])}
        before = {name: np.array(value, copy=True) for name, value in model.snapshot().items()}
        moments = optimizer.state_dict()["groups"]
        # Intentionally supply valid geometry and nonzero gradients: optimizer
        # freezing must also survive stale momentum and external gradient input.
        out = {"gradients": {name: mi.Float(np.full(2*width, .4)) for name, width in optimizer.dimensions.items()},
               "appearance_valid": mi.UInt32([1, 1]), "geometry_valid": mi.UInt32([1, 1]),
               "background_carve_valid": mi.UInt32([0, 0])}
        if mode == "compact":
            out["voxel_ids"] = mi.UInt32([0, 1])
        if mode == "compact_reference":
            optimizer._step_compact_reference(model, out, sample_count=1, epoch=0,
                coarse_to_fine=True, update_allowed=mi.Bool(True))
        else:
            optimizer.step(model, out, coarse_to_fine=True, update_allowed=mi.Bool(True))
        dr.sync_thread()
        after = model.snapshot()
        state = optimizer.state_dict()["groups"]
        for parameter, width in (("centers", 3), ("log_scales", 3), ("rotations", 4)):
            actual = np.asarray(after[parameter]).reshape(2, width)
            old = before[parameter].reshape(2, width)
            np.testing.assert_array_equal(actual[bg], old[bg], err_msg=f"{mode} {parameter}")
            assert np.any(actual[~bg] != old[~bg]), (mode, parameter)
        for name in GEOMETRY:
            for field, width in (("mean", 3), ("squared_mean", 3), ("steps", 1)):
                np.testing.assert_array_equal(state[name][field].reshape(2, width)[bg],
                    moments[name][field].reshape(2, width)[bg], err_msg=f"{mode} {name}.{field}")
        assert np.any(np.asarray(after["radiance_sh"]).reshape(2, -1)[bg] != before["radiance_sh"].reshape(2, -1)[bg])
        assert np.any(np.asarray(after["opacity_sh"]).reshape(2, -1)[bg] != before["opacity_sh"].reshape(2, -1)[bg])
        if mode in ("device", "compact"):
            # Reuse the same model, Adam instance, gradient shapes and nonzero
            # momentum while crossing modes. A frozen update must rerecord its
            # native geometry mask and resume ordinary background Adam on true.
            for enabled in (True, True, True, False, False, False):
                model.set_background_use_ellipsoid(enabled)
                old = {key: np.array(value, copy=True) for key, value in model.snapshot().items()}
                old_moments = optimizer.state_dict()["groups"]
                optimizer.step(model, out, coarse_to_fine=True, update_allowed=mi.Bool(True))
                dr.sync_thread()
                current = model.snapshot()
                current_moments = optimizer.state_dict()["groups"]
                for parameter, width in (("centers", 3), ("log_scales", 3), ("rotations", 4)):
                    actual = np.asarray(current[parameter]).reshape(2, width)[bg]
                    previous = old[parameter].reshape(2, width)[bg]
                    if enabled:
                        assert np.any(actual != previous), (mode, parameter, enabled)
                    else:
                        np.testing.assert_array_equal(actual, previous, err_msg=f"{mode} switched {parameter}")
                for name in GEOMETRY:
                    steps = current_moments[name]["steps"][bg]
                    np.testing.assert_array_equal(steps, old_moments[name]["steps"][bg]+int(enabled))
                    if not enabled:
                        for field in ("mean", "squared_mean"):
                            np.testing.assert_array_equal(current_moments[name][field].reshape(2, 3)[bg],
                                old_moments[name][field].reshape(2, 3)[bg])
            assert optimizer.statistics()["replays"] >= 2, optimizer.statistics()
    return {"dense_device_compact_reference_Adam_freezes_background_parameters_and_momentum": "passed",
            "device_compact_Adam_replay_invalidates_geometry_mask_on_mode_switch": "passed"}


def check_core_and_bounded_baseline(mi, dr, np):
    payload = fixture(np)
    ray = make_ray(mi, np, [(.13, .22, .0), (.3, -.2, .1)],
                   [(0., 0., 1.), (-.4, .3, .7)], .7)
    for mapping in ("projective_shell", "bounded"):
        changed = copy.deepcopy(payload)
        if mapping == "bounded":
            changed["model_metadata"] = {
                key: value for key, value in changed["model_metadata"].items()
                if not key.startswith("background_") and key != "core_resolution"}
            changed["model_metadata"]["scene_mapping"] = mapping
        model = restore_model(changed, mi)
        model.set_background_use_ellipsoid(True)
        before = native(model, mi, dr, ray, evidence=True)
        model.set_background_use_ellipsoid(False)
        comparison(native(model, mi, dr, ray, evidence=True), before, np)
    return {"core_and_bounded_ellipsoid_behavior_unchanged": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    reports = [check_numpy()]
    print(json.dumps(reports[-1]), flush=True)
    if not args.cpu_only:
        with tempfile.TemporaryDirectory(prefix="background-voxel-native-") as directory:
            mi, dr, np, _ = initialize_runtime(RuntimeConfig(args.build_dir, args.gpu), Path(directory), {})
            assert hasattr(mi.VoxelReconstruction, "set_background_use_ellipsoid"), "Rebuild native background voxel support before testing"
            for check in (check_world_preimages, check_parameters_and_gradients,
                          check_deletion_and_replay, check_adam_freeze,
                          check_core_and_bounded_baseline):
                reports.append(check(mi, dr, np))
                print(json.dumps(reports[-1]), flush=True)
    print(json.dumps({"status": "passed", "checks": reports}), flush=True)


if __name__ == "__main__":
    main()
