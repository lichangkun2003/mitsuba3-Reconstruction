"""Independent cube-surface oracle for once-per-layer background sampling.

This is a sampling model, not an integration of every intersected background
cell. Each inverse-depth midpoint selects the positive cube EXIT, then exactly
one face texel. Empty texels do not search for another hit. Core ellipsoids keep
their existing world-space oracle. The optional CUDA checks run only from the
standalone entry point, never when this module is imported.
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
from reconstruction_replay import NativeLossReplay
from test_reconstruction_background_voxel_native import (
    background_mask, comparison, fixture as voxel_fixture, oracle as voxel_oracle,
    subset, two_background_cells,
)
from test_reconstruction_projective_native import (
    Y00, make_ray, native, oracle as ellipsoid_oracle,
)


def fixture(np, *, far=0., core=True):
    payload = voxel_fixture(np)
    payload["model_metadata"].update(background_layered_sampling=True,
                                      background_far_distance=float(far))
    return payload if core else subset(payload, background_mask(payload, np))


def shell_samples(metadata, origin, direction, np, *, maxt=math.inf):
    """Analytic ray/AABB far roots; no native cursor, mapping helper or DDA."""
    n, grid = metadata["core_resolution"], metadata["resolution"]
    layers = (grid-n)//2
    far = float(metadata.get("background_far_distance", 0.))
    o, raw = np.asarray(origin, float), np.asarray(direction, float)
    length = float(np.linalg.norm(raw))
    if length == 0. or maxt <= 0.:
        return []
    d = raw/length
    result = []
    for layer in range(layers):
        u = (layer+.5)/layers
        h = 1./(1.-u*(1.-1./far if far else 1.))
        near, end, valid = -math.inf, math.inf, True
        for axis in range(3):
            if d[axis] == 0.:
                valid &= abs(o[axis]) <= h
            else:
                a, b = (-h-o[axis])/d[axis], (h-o[axis])/d[axis]
                near, end = max(near, min(a, b)), min(end, max(a, b))
        # One point of contact has no cube interior. A camera on the shell may
        # nevertheless have exit t=0 when its ray points out of the cube.
        if not valid or end <= near or end < 0. or end > maxt*length:
            continue
        point = o+end*d
        # The nearest exiting positive slab bound determines the surface.
        # np.argmax supplies the specified X, then Y, then Z seam ownership.
        axis = int(np.argmax(np.abs(point)))
        positive = point[axis] >= 0.
        cell = np.clip(np.floor((point/h+1.)*(n/2)).astype(int), 0, n-1)+layers
        cell[axis] = layers+n+layer if positive else layers-1-layer
        spatial = int(cell[0]+grid*(cell[1]+grid*cell[2]))
        result.append(dict(layer=layer, distance=end, depth=end/(end+1.),
                           spatial=spatial, axis=axis, positive=bool(positive), h=h))
    return sorted(result, key=lambda item: item["distance"])


def oracle(payload, origin, direction, np, *, maxt=math.inf, max_hits=64,
           threshold=0.):
    md, state = payload["model_metadata"], payload["model_state"]
    d = np.asarray(direction, float)
    length = float(np.linalg.norm(d))
    if length == 0. or maxt <= 0.:
        return dict(rgb=np.zeros(3), alpha=0., hits=[], distortion=0.)
    d /= length
    bg = background_mask(payload, np)
    core_rows = np.flatnonzero(~bg)
    core_result = ellipsoid_oracle(subset(payload, ~bg), origin, direction, np,
                                  maxt=maxt, max_hits=128)
    hits = [(t, int(core_rows[row]), alpha, rgb, depth, star)
            for t, row, alpha, rgb, depth, star in core_result["hits"]]
    basis = np.array([Y00, -d[1]*Y00*math.sqrt(3), -d[2]*Y00*math.sqrt(3),
                      -d[0]*Y00*math.sqrt(3)])
    sh, osh = md["sh_count"], md["opacity_sh_count"]
    assert sh in (1, 4) and osh in (1, 4), "Oracle intentionally covers DC and degree-one SH"
    radiance = state["radiance_sh"].reshape(-1, sh, 3)
    opacity = state["opacity_sh"].reshape(-1, osh)
    rows = {int(spatial): row for row, spatial in enumerate(state["cell_indices"])}
    for sample in shell_samples(md, origin, direction, np, maxt=maxt):
        row = rows.get(sample["spatial"])
        if row is None:
            continue
        assert bg[row]
        alpha = 1./(1.+math.exp(-float(opacity[row]@basis[:osh])))
        rgb = np.maximum(0., basis[:sh]@radiance[row])
        hits.append((sample["distance"], row, alpha, rgb, sample["depth"], .5))
    hits.sort(key=lambda hit: (hit[0], hit[1]))
    color, transmittance, used, weights = np.zeros(3), 1., [], []
    for hit in hits:
        if len(used) >= max_hits or transmittance <= threshold:
            break
        weight = transmittance*hit[2]
        color += weight*hit[3]
        transmittance *= 1.-hit[2]
        used.append(hit)
        weights.append(weight)
    distortion = sum(2*weights[i]*weights[j]*abs(used[i][4]-used[j][4])
                     for i in range(len(used)) for j in range(i+1, len(used)))
    return dict(rgb=color, alpha=1.-transmittance, hits=used, distortion=distortion)


def check_numpy():
    import numpy as np
    payload = two_background_cells(np)
    payload["model_metadata"]["background_layered_sampling"] = True
    o, d = (.13, .22, 0.), (0., 0., 1.)
    samples = shell_samples(payload["model_metadata"], o, d, np)
    np.testing.assert_allclose([s["distance"] for s in samples], [4/3, 4.], atol=1e-12)
    result = oracle(payload, o, d, np)
    np.testing.assert_allclose(result["rgb"], [.65, .7, .75], atol=1e-12)
    assert result["alpha"] == .75 and len(result["hits"]) == 2
    assert not oracle(payload, o, d, np, maxt=1.2)["hits"]
    assert len(oracle(payload, o, d, np, maxt=2.)["hits"]) == 1
    finite = copy.deepcopy(payload)
    finite["model_metadata"]["background_far_distance"] = 2.5
    np.testing.assert_allclose(
        [s["h"] for s in shell_samples(finite["model_metadata"], o, d, np)],
        [1/0.85, 1/0.55], atol=1e-12)
    # In the old occupied-cell model the oblique ray visits multiple tangential
    # texels. Removing both fixed samples must not fall back to those texels.
    full = fixture(np, core=False)
    o, d = (.8, .07, .11), (.6, .03, .8)
    selected = [s["spatial"] for s in shell_samples(full["model_metadata"], o, d, np)]
    holes = subset(full, ~np.isin(full["model_state"]["cell_indices"], selected))
    assert len(oracle(full, o, d, np)["hits"]) == 2
    assert not oracle(holes, o, d, np)["hits"]
    assert len(voxel_oracle(holes, o, d, np)["hits"]) >= 2
    # Outside cameras deliberately use only far roots, not the near-side layer.
    outside = shell_samples(full["model_metadata"], (6., .1, .2), (-1., 0., 0.), np)
    np.testing.assert_allclose([s["distance"] for s in outside], [6+4/3, 10.])
    assert all(not s["positive"] for s in outside)
    assert not shell_samples(full["model_metadata"], (6., .1, .2), (1., 0., 0.), np)
    seam = shell_samples(full["model_metadata"], (0., 0., 0.), (1., 1., 1.), np)
    assert all(s["axis"] == 0 for s in seam)
    return {"independent_cube_exit_inverse_depth_midpoint_oracle": "passed",
            "selected_cell_holes_do_not_fall_back_to_other_layer_texels": "passed",
            "outside_camera_uses_far_exit_only": "passed"}


def check_trace_and_training(mi, dr, np):
    rays = [((.8, .07, .11), (.6, .03, .8)),
            ((0., 0., 0.), (1., 1., 0.)),
            ((0., 0., 0.), (-1., -1., -1.)),
            ((6., .15, .23), (-1., .02, .04)),
            ((6., .15, .23), (1., .02, .04)),
            ((1.8, -.7, .2), (-.2, .8, .1)),
            ((0., 0., 4.), (0., 0., 1.)),
            ((.13, .22, .17), (0., 0., 3.))]
    for axis in range(3):
        for sign in (-1., 1.):
            direction = np.zeros(3)
            direction[axis] = sign
            rays.append(((.13, .22, .17), tuple(direction)))
    cases = 0
    for with_core in (False, True):
        for far in (0., 1.125, 2.5, 8.):
            payload = fixture(np, far=far, core=with_core)
            model = restore_model(payload, mi)
            assert model.native_background_layered_sampling() is True
            assert model.native_background_use_ellipsoid() is False
            for maxt in (math.inf, .7, 1.5, 3., 0.):
                ray = make_ray(mi, np, [o for o, _ in rays], [d for _, d in rays], maxt)
                expected = [oracle(payload, o, d, np, maxt=maxt) for o, d in rays]
                for octree in (False, True):
                    context = f"core={with_core}, H={far}, maxt={maxt}, octree={octree}"
                    actual = native(model, mi, dr, ray, use_octree=octree, evidence=True)
                    rgb, alpha, hits = model.trace(ray, mi.Color3f(0), 0., 64, 0., octree)
                    dr.eval(rgb, alpha, hits)
                    np.testing.assert_array_equal(np.asarray(hits), [len(e["hits"]) for e in expected], err_msg=context)
                    np.testing.assert_allclose(np.asarray(rgb).T, [e["rgb"] for e in expected], atol=6e-6, rtol=5e-5, err_msg=context)
                    np.testing.assert_allclose(np.asarray(alpha), [e["alpha"] for e in expected], atol=5e-6, err_msg=context)
                    np.testing.assert_array_equal(np.asarray(actual["hit_count"]), np.asarray(hits), err_msg=context)
                    np.testing.assert_allclose(np.asarray(actual["raw_rgb"]), np.asarray(rgb), atol=4e-6, err_msg=context)
                    np.testing.assert_allclose(np.asarray(actual["raw_alpha"]), np.asarray(alpha), atol=4e-6, err_msg=context)
                    np.testing.assert_allclose(np.asarray(actual["distortion_loss"]), [e["distortion"] for e in expected], atol=5e-6, rtol=5e-5, err_msg=context)
                    if not with_core:
                        assert np.all(np.asarray(hits) <= 2), context
                    cases += len(rays)
    return {"oblique_six_faces_seams_finite_infinite_mixed_core_trace_train_cases": cases}


def sparse_1024_fixture(np, rays, *, far):
    core, layers = 1024, 8
    grid = core+2*layers
    extent = 1+2*layers/core
    md = {"resolution": grid, "core_resolution": core,
          "grid_min": [-extent]*3, "grid_max": [extent]*3,
          "scene_mapping": "projective_shell", "background_layers": layers,
          "background_refinement": "preserve_physical_width",
          "background_use_ellipsoid": False, "background_layered_sampling": True,
          "background_far_distance": far, "sh_count": 1, "opacity_sh_count": 1}
    # This uses O(rays*layers) addresses, never an N^3 or 6*N^2*layers table.
    cells = sorted({sample["spatial"] for o, d in rays
                    for sample in shell_samples(md, o, d, np)})
    count = len(cells)
    rng = np.random.default_rng(724)
    return {"model_metadata": md, "model_state": {
        "cell_indices": np.asarray(cells, np.uint32),
        "centers": np.full(count*3, 9.),
        "log_scales": np.full(count*3, math.log(1e-7)),
        "rotations": np.tile([1., 0., 0., 0.], count),
        "radiance_sh": (rng.uniform(.1, .9, (count, 3))/Y00).ravel(),
        "opacity_sh": np.full(count, math.log(.17/.83)/Y00)}}


def check_sparse_1024(mi, dr, np):
    rays = [((.83, .07, .12), (.6, .03, .8)),
            ((0., 0., 0.), (1., 1., 1.)),
            ((0., 0., 0.), (-1., -1., 0.))]
    for axis in range(3):
        for sign in (-1., 1.):
            direction = np.zeros(3)
            direction[axis] = sign*3.
            rays.append(((.13, .22, .17), tuple(direction)))
    cases, largest = 0, 0
    for far in (0., 1.125, 8.):
        payload = sparse_1024_fixture(np, rays, far=far)
        largest = max(largest, len(payload["model_state"]["cell_indices"]))
        assert largest <= len(rays)*8
        model = restore_model(payload, mi)
        for maxt in (math.inf, .5, 4.):
            ray = make_ray(mi, np, [o for o, _ in rays], [d for _, d in rays], maxt)
            expected = [oracle(payload, o, d, np, maxt=maxt, max_hits=16) for o, d in rays]
            for octree in (False, True):
                result = native(model, mi, dr, ray, use_octree=octree, max_hits=16)
                rgb, alpha, hits = model.trace(ray, mi.Color3f(0), 0., 16, 0., octree)
                dr.eval(rgb, alpha, hits)
                context = f"1024 sparse H={far}, maxt={maxt}, octree={octree}"
                np.testing.assert_array_equal(np.asarray(hits), [len(v["hits"]) for v in expected], err_msg=context)
                assert np.all(np.asarray(hits) <= 8), context
                if math.isinf(maxt):
                    np.testing.assert_array_equal(np.asarray(hits), 8, err_msg=context)
                np.testing.assert_allclose(np.asarray(rgb).T, [v["rgb"] for v in expected], atol=6e-6, rtol=5e-5, err_msg=context)
                np.testing.assert_allclose(np.asarray(result["raw_rgb"]), np.asarray(rgb), atol=4e-6, err_msg=context)
                np.testing.assert_allclose(np.asarray(result["distortion_loss"]), [v["distortion"] for v in expected], atol=5e-6, rtol=5e-5, err_msg=context)
                for name in ("centers", "log_scales", "rotation_tangent"):
                    np.testing.assert_array_equal(np.asarray(result["gradients"][name]), 0.)
                cases += len(rays)
    return {"sparse_1024_eight_layers_six_faces_ties_nonunit_ray_maxt_cases": cases,
            "largest_test_field_voxels": largest}


def check_holes_geometry_and_capacity(mi, dr, np):
    payload = fixture(np, core=False)
    o, d = (.8, .07, .11), (.6, .03, .8)
    samples = shell_samples(payload["model_metadata"], o, d, np)
    chosen = [sample["spatial"] for sample in samples]
    ray = make_ray(mi, np, [o], [d])
    for removed in ([], chosen[:1], chosen):
        sparse = subset(payload, ~np.isin(payload["model_state"]["cell_indices"], removed))
        model = restore_model(sparse, mi)
        result = native(model, mi, dr, ray, evidence=True)
        expected = oracle(sparse, o, d, np)
        np.testing.assert_array_equal(np.asarray(result["hit_count"]), [2-len(removed)])
        np.testing.assert_allclose(np.asarray(result["raw_rgb"]).ravel(), expected["rgb"], atol=4e-6)
        for name in ("centers", "log_scales", "rotation_tangent"):
            np.testing.assert_array_equal(np.asarray(result["gradients"][name]), 0.)
        np.testing.assert_array_equal(np.asarray(result["geometry_valid"]), 0)
    changed = copy.deepcopy(payload)
    changed["model_state"]["centers"][:] = 5.
    changed["model_state"]["log_scales"][:] = -12.
    count = len(changed["model_state"]["cell_indices"])
    changed["model_state"]["rotations"][:] = np.tile([.5]*4, count)
    comparison(native(restore_model(changed, mi), mi, dr, ray, evidence=True),
               native(restore_model(payload, mi), mi, dr, ray, evidence=True), np)
    model = restore_model(payload, mi)
    for cap, threshold in ((1, 0.), (64, .9)):
        result = native(model, mi, dr, ray, evidence=True, max_hits=cap, threshold=threshold)
        expected = oracle(payload, o, d, np, max_hits=cap, threshold=threshold)
        assert len(expected["hits"]) == 1
        np.testing.assert_array_equal(np.asarray(result["hit_count"]), [1])
        np.testing.assert_allclose(np.asarray(result["raw_rgb"]).ravel(), expected["rgb"], atol=4e-6)
        np.testing.assert_array_equal(np.asarray(result["deletion_observed"]), 0)
        assert int(np.asarray(result["deletion_unknown"]).sum()) == 1
    return {"fixed_texel_holes_skip_layer_and_geometry_is_ignored": "passed",
            "shared_hit_cap_and_transmittance_stop_keep_deletion_unknown": "passed"}


def check_gradients_deletion_and_layouts(mi, dr, np):
    payload = fixture(np, far=2.5)
    count = len(payload["model_state"]["cell_indices"])
    rng = np.random.default_rng(963)
    color = np.zeros((count, 4, 3))
    color[:, 0] = payload["model_state"]["radiance_sh"].reshape(count, 3)
    color[:, 1:] = rng.uniform(-.08, .08, (count, 3, 3))
    opacity = np.zeros((count, 4))
    opacity[:, 0] = payload["model_state"]["opacity_sh"]
    opacity[:, 1:] = rng.uniform(-.1, .1, (count, 3))
    payload["model_metadata"].update(sh_count=4, opacity_sh_count=4)
    payload["model_state"].update(radiance_sh=color.ravel(), opacity_sh=opacity.ravel())
    rays = [((.8, .02*y+.03, .02*x+.01), (.6, .03*x+.01, .8))
            for y in range(2) for x in range(3)]
    ray = make_ray(mi, np, [o for o, _ in rays], [d for _, d in rays])
    model = restore_model(payload, mi)
    expected = [oracle(payload, o, d, np) for o, d in rays]
    result = native(model, mi, dr, ray, evidence=True)
    for compact in (False, True):
        options = {"rgb_only": 1., "max_hits": 64., "max_candidates": 4.,
            "use_octree": 1., "distortion_loss_weight": .7, "transmittance_threshold": 0.,
            "collect_rgb_deletion_evidence": 1., "collect_growth_evidence": 0.,
            "max_growth_hits": 8., "compact_path_records": float(compact)}
        other = model.loss_and_grad(ray, mi.Color3f(.3, .4, .5), mi.Float(0),
                                   mi.Color3f(0), mi.Float(1), options)
        dr.eval(other)
        comparison(other, result, np)
    bg = background_mask(payload, np)
    for name in ("centers", "log_scales", "rotation_tangent"):
        np.testing.assert_array_equal(np.asarray(result["gradients"][name]).reshape(count, 3)[bg], 0.)
    np.testing.assert_array_equal(np.asarray(result["geometry_valid"])[bg], 0)
    hit_rows = sorted({hit[1] for item in expected for hit in item["hits"]})
    assert any(bg[row] for row in hit_rows) and any(not bg[row] for row in hit_rows)
    target = np.array([.3, .4, .5])

    def loss(checkpoint):
        values = [oracle(checkpoint, o, d, np) for o, d in rays]
        return sum(np.sum((item["rgb"]-target)**2)+.7*item["distortion"] for item in values)

    checked = 0
    selected = [next(row for row in hit_rows if not bg[row])]
    selected += [row for row in hit_rows if bg[row]][:6]
    for name, width in (("radiance_sh", 12), ("opacity_sh", 4)):
        for row in selected:
            for component in range(width):
                index, epsilon = row*width+component, 2e-5
                plus, minus = copy.deepcopy(payload), copy.deepcopy(payload)
                plus["model_state"][name][index] += epsilon
                minus["model_state"][name][index] -= epsilon
                numerical = (loss(plus)-loss(minus))/(2*epsilon)
                np.testing.assert_allclose(np.asarray(result["gradients"][name])[index], numerical,
                    atol=3e-5, rtol=5e-3, err_msg=f"{name}[{index}]")
                checked += 1
    # Deleting a sampled texel makes this one layer empty. It must not select a
    # second cell from the same layer, so the existing complete-path suffix is
    # the correct counterfactual in this new fixed-sample mode.
    for row in hit_rows:
        without = subset(payload, np.arange(count) != row)
        changed = [oracle(without, o, d, np) for o, d in rays]
        gain = sum(np.sum((a["rgb"]-target)**2)-np.sum((b["rgb"]-target)**2)
                   for a, b in zip(expected, changed))
        np.testing.assert_allclose(np.asarray(result["deletion_gain"])[row], gain, atol=5e-6, rtol=5e-5)
    return {"RGB_SH_opacity_distortion_finite_differences": checked,
            "mixed_core_and_background_exact_deletion_counterfactuals": len(hit_rows),
            "compact_and_legacy_records_agree": "passed"}


def check_mode_default_and_replay(mi, dr, np):
    payload = fixture(np, core=False)
    o, d = (.8, .07, .11), (.6, .03, .8)
    ray = make_ray(mi, np, [o], [d])
    legacy = copy.deepcopy(payload)
    legacy["model_metadata"].pop("background_layered_sampling")
    old = restore_model(legacy, mi)
    assert old.native_background_layered_sampling() is False
    old_expected = voxel_oracle(legacy, o, d, np)
    assert len(old_expected["hits"]) > 2
    baseline = native(old, mi, dr, ray)
    np.testing.assert_allclose(np.asarray(baseline["raw_rgb"]).ravel(), old_expected["rgb"], atol=5e-6)
    model = restore_model(payload, mi)
    replay = NativeLossReplay(dr)
    for enabled in (True, True, False, False, True):
        before, prior = int(model.topology_revision), model.native_background_layered_sampling()
        model.set_background_layered_sampling(enabled)
        assert model.native_background_layered_sampling() is enabled
        assert int(model.topology_revision) == before+int(prior != enabled)
        direct = native(model, mi, dr, ray, evidence=True)
        comparison(native(model, mi, dr, ray, evidence=True, replay=replay), direct, np)
        expected = oracle(payload, o, d, np) if enabled else old_expected
        np.testing.assert_array_equal(np.asarray(direct["hit_count"]), [len(expected["hits"])])
        np.testing.assert_allclose(np.asarray(direct["raw_rgb"]).ravel(), expected["rgb"], atol=5e-6)
    assert replay.statistics()["topology_changes"] == 2, replay.statistics()
    return {"missing_metadata_preserves_existing_voxel_behavior": "passed",
            "mode_switch_invalidates_native_replay": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    reports = [check_numpy()]
    print(json.dumps(reports[-1]), flush=True)
    if not args.cpu_only:
        with tempfile.TemporaryDirectory(prefix="background-layered-native-") as directory:
            mi, dr, np, _ = initialize_runtime(RuntimeConfig(args.build_dir, args.gpu), Path(directory), {})
            assert getattr(mi.VoxelReconstruction, "background_layered_sampling_version", 0) >= 1, \
                "Rebuild native layered background sampling before testing"
            for check in (check_trace_and_training, check_sparse_1024, check_holes_geometry_and_capacity,
                          check_gradients_deletion_and_layouts, check_mode_default_and_replay):
                reports.append(check(mi, dr, np))
                print(json.dumps(reports[-1]), flush=True)
    print(json.dumps({"status": "passed", "checks": reports}), flush=True)


if __name__ == "__main__":
    main()
