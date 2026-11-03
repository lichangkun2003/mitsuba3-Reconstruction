"""World-space mixed-leaf traversal against an independent ray/AABB oracle.

Small synthetic fields only. CPU checks never initialize Mitsuba/CUDA; GPU
checks run only when this standalone script is invoked without --cpu-only.
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

Y00 = .5 / math.sqrt(math.pi)
ENV = (.07, .11, .13)
TARGET = (.3, .4, .5)


def fixture(np, core=8, levels=5, *, include_core=True, subdivision=0, sh_count=4,
            background_geometry="occupied_cell"):
    cells = []
    if include_core:
        for z in (core//2-1, core//2):
            cells.append(core//2+core*(core//2+core*z))
    cells += list(range(core**3, core**3+56*levels*8**subdivision))
    count = len(cells)
    rng = np.random.default_rng(293)
    sh = np.zeros((count, sh_count, 3), np.float32)
    sh[:, 0] = rng.uniform(.2, .7, (count, 3))/Y00
    sh[:, 1:] = rng.uniform(-.07, .07, (count, sh_count-1, 3))
    state = {
        "cell_indices": np.asarray(cells, np.uint64),
        "centers": np.full(count*3, .5, np.float32),
        "log_scales": np.full(count*3, math.log(.45*2/core), np.float32),
        "rotations": np.tile(np.array([1, 0, 0, 0], np.float32), count),
        "radiance_sh": sh.ravel(),
        "opacity_sh": np.full(count, math.log(.06/.94)/Y00, np.float32),
    }
    # Legacy occupied-cell fixtures deliberately place tiny ellipsoids outside
    # their leaves: ignoring their geometry is required checkpoint behavior.
    bg = state["cell_indices"] >= core**3
    state["centers"].reshape(-1, 3)[bg] = 5.
    state["log_scales"].reshape(-1, 3)[bg] = -12.
    result = {"model_metadata": {
        "resolution": core, "core_resolution": core,
        "grid_min": [-1.]*3, "grid_max": [1.]*3,
        "scene_mapping": "world_shell", "background_shell_levels": levels,
        "background_subdivision_level": subdivision,
        "background_refinement": "uniform", "world_shell_version": 2,
        "background_geometry": background_geometry,
        "sh_count": sh_count, "opacity_sh_count": 1,
    }, "model_state": state}
    if background_geometry == "ellipsoid":
        result["model_metadata"]["world_shell_version"] = 3
        lows, highs, _ = prepare_boxes(result, np)
        state["centers"].reshape(-1, 3)[bg] = [.44, .56, .47]
        state["log_scales"].reshape(-1, 3)[bg] = np.log((highs[bg] - lows[bg]) * [.34, .25, .42])
        axis = np.array([.4, .7, -.2])
        axis /= np.linalg.norm(axis)
        state["rotations"].reshape(-1, 4)[bg] = np.r_[math.cos(.37 / 2), axis * math.sin(.37 / 2)]
    return result


def rotation_matrix(quaternion, np):
    q = np.asarray(quaternion, float)
    length = np.linalg.norm(q)
    w, x, y, z = q / length if np.isfinite(length) and length > 1e-10 else (1., 0., 0., 0.)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def boxes(payload, np):
    """Enumerate physical boxes; no octree keys, coordinate compression or DDA."""
    md = payload["model_metadata"]
    n = md["resolution"]
    shell_coordinates = [(x, y, z) for z in range(4) for y in range(4) for x in range(4)
                         if not (1 <= x <= 2 and 1 <= y <= 2 and 1 <= z <= 2)]
    result = []
    for row, key in enumerate(payload["model_state"]["cell_indices"]):
        key = int(key)
        if key < n**3:
            low = -1+np.array([key % n, key//n % n, key//n**2])*2/n
            size = 2/n
        else:
            subdivision = int(md.get("background_subdivision_level", 0))
            coarse, child = divmod(key-n**3, 8**subdivision)
            k, j = divmod(coarse, 56)
            size = float(2**k)
            low = -2*size+np.array(shell_coordinates[j])*size
            # Follow octant digits from parent to child in world space. This
            # deliberately does not use the production bit-deinterleave helper.
            for level in range(subdivision-1, -1, -1):
                octant = (child//(8**level)) % 8
                size *= .5
                low += size*np.array((octant % 2, octant//2 % 2, octant//4))
        result.append((row, low, low+size, key < n**3))
    return result


def prepare_boxes(payload, np):
    leaves = boxes(payload, np)
    return (np.array([item[1] for item in leaves]), np.array([item[2] for item in leaves]),
            np.array([item[3] for item in leaves], bool))


def oracle(payload, origin, direction, np, *, maxt=math.inf, max_hits=128, threshold=0.,
           active_sh_count=None, prepared=None, fp32_intersections=False):
    state, md = payload["model_state"], payload["model_metadata"]
    # Ray3f stores maxt as Float32 just like o/d. Preserve that input contract
    # even when the independent intersection calculation uses doubles.
    maxt = float(np.float32(maxt))
    o, d = np.asarray(origin, float), np.asarray(direction, float)
    length = float(np.linalg.norm(d))
    if length == 0 or maxt <= 0:
        return dict(rgb=np.array(ENV), alpha=0., hits=[], distortion=0.)
    d = d/length
    if fp32_intersections:
        # Explicit numerical oracle for a shared-edge ray whose real interval
        # is less than one FP32 time ULP. Keep the ordinary oracle in double.
        raw = np.asarray(direction, np.float32)
        length = float(np.linalg.norm(raw))
        d = raw/np.float32(length)
    x, y, z = -d
    basis = np.array([Y00, y*Y00*math.sqrt(3), z*Y00*math.sqrt(3), x*Y00*math.sqrt(3),
        .5*math.sqrt(15/math.pi)*x*y, .5*math.sqrt(15/math.pi)*y*z,
        .25*math.sqrt(5/math.pi)*(3*z*z-1), .5*math.sqrt(15/math.pi)*x*z,
        .25*math.sqrt(15/math.pi)*(x*x-y*y)])
    sh_count = md["sh_count"]
    active_sh_count = sh_count if active_sh_count is None else active_sh_count
    lows, highs, core_mask = prepare_boxes(payload, np) if prepared is None else prepared
    begins, ends = np.zeros(len(lows)), np.full(len(lows), maxt*length)
    valid = np.ones(len(lows), bool)
    for axis in range(3):
        if d[axis] == 0:
            valid &= (lows[:, axis] <= o[axis]) & (o[axis] < highs[:, axis])
        else:
            if fp32_intersections:
                inverse = np.float32(1)/np.float32(d[axis])
                first = (lows[:, axis].astype(np.float32)-np.float32(o[axis]))*inverse
                last = (highs[:, axis].astype(np.float32)-np.float32(o[axis]))*inverse
            else:
                first, last = (lows[:, axis]-o[axis])/d[axis], (highs[:, axis]-o[axis])/d[axis]
            begins = np.maximum(begins, np.minimum(first, last))
            ends = np.minimum(ends, np.maximum(first, last))
    hits = []
    for row in np.flatnonzero(valid & (ends > begins)):
        low, high, core = lows[row], highs[row], core_mask[row]
        begin, end = begins[row], ends[row]
        depth_t = (begin+end)/2
        if core or md.get("background_geometry", "occupied_cell") == "ellipsoid":
            center = low+(high-low)*state["centers"].reshape(-1, 3)[row]
            scale = np.exp(state["log_scales"].reshape(-1, 3)[row].astype(float))
            rotation = rotation_matrix(state["rotations"].reshape(-1, 4)[row], np)
            u, v = (rotation.T @ (o-center))/scale, (rotation.T @ d)/scale
            a, b, c = v@v, 2*(u@v), u@u-1
            discriminant = b*b-4*a*c
            if discriminant < 0:
                continue
            t0, t1 = (-b-math.sqrt(discriminant))/(2*a), (-b+math.sqrt(discriminant))/(2*a)
            if max(t0, begin) > min(t1, end):
                continue
            depth_t = float(np.clip(-b/(2*a), begin, end))
        alpha = 1/(1+math.exp(-float(state["opacity_sh"][row])*Y00))
        coefficients = state["radiance_sh"].reshape(-1, sh_count, 3)[row, :active_sh_count]
        color = np.maximum(0, basis[:active_sh_count]@coefficients)
        hits.append((begin, int(row), alpha, color, depth_t/(depth_t+1)))
    hits.sort(key=lambda hit: (hit[0], hit[1]))
    transmittance, rgb, used, weights = 1., np.zeros(3), [], []
    for hit in hits:
        if len(used) >= max_hits or transmittance <= threshold:
            break
        w = transmittance*hit[2]
        rgb += w*hit[3]
        transmittance *= 1-hit[2]
        used.append(hit)
        weights.append(w)
    rgb += transmittance*np.array(ENV)
    distortion = sum(2*weights[i]*weights[j]*abs(used[i][4]-used[j][4])
                     for i in range(len(used)) for j in range(i+1, len(used)))
    return dict(rgb=rgb, alpha=1-transmittance, hits=used, distortion=distortion)


def make_ray(mi, np, rays, maxt=math.inf):
    o = np.asarray([r[0] for r in rays], np.float32)
    d = np.asarray([r[1] for r in rays], np.float32)
    ray = mi.Ray3f(mi.Point3f(*(mi.Float(o[:, a].copy()) for a in range(3))),
                  mi.Vector3f(*(mi.Float(d[:, a].copy()) for a in range(3))))
    ray.maxt = maxt
    return ray


def evaluate(model, mi, dr, ray, *, growth=False, coefficient=.7):
    result = model.loss_and_grad(ray, mi.Color3f(*TARGET), mi.Float(0),
        mi.Color3f(*ENV), mi.Float(1), {
            "rgb_only": 1., "max_hits": 128., "max_candidates": 4.,
            "transmittance_threshold": 0., "distortion_loss_weight": coefficient,
            "collect_growth_evidence": float(growth), "max_growth_hits": 8.,
            "collect_rgb_deletion_evidence": 1., "use_octree": 1.,
        })
    dr.eval(result)
    dr.sync_thread()
    return result


def ellipsoid_case(np, shell, subdivision):
    """One coarse background leaf with rotated, anisotropic world geometry."""
    payload = fixture(np, 1024, 8, include_core=False, background_geometry="ellipsoid")
    row, count = shell * 56, len(payload["model_state"]["cell_indices"])
    payload["model_state"] = {key: value.reshape(count, -1)[row:row+1].ravel().copy()
                              for key, value in payload["model_state"].items()}
    payload["model_metadata"]["background_subdivision_level"] = subdivision
    child = (8**subdivision - 1) // 2
    payload["model_state"]["cell_indices"][:] = 1024**3 + row*8**subdivision + child
    _, low, high, _ = boxes(payload, np)[0]
    edge = high - low
    payload["model_state"]["log_scales"] = np.log(edge * [.34, .25, .42]).astype(np.float32)
    payload["model_state"]["opacity_sh"][:] = 0.
    rotation = rotation_matrix(payload["model_state"]["rotations"], np)
    center = low + edge * payload["model_state"]["centers"]
    scale = np.exp(payload["model_state"]["log_scales"].astype(float))
    direction = rotation[:, 2]
    origins = [center + rotation[:, 0]*scale[0]*offset + rotation[:, 1]*scale[1]*.15
               - 2*edge[0]*direction for offset in (.85, 1.08)]
    rays = [(origins[0], direction), (origins[1], direction),
            (origins[0] + 4*edge[0]*direction, -direction), (center, direction)]
    rays = [(tuple(np.float32(origin)), tuple(np.float32(d))) for origin, d in rays]
    bounded = copy.deepcopy(payload)
    bounded["model_metadata"] = {key: value for key, value in bounded["model_metadata"].items()
                                  if not key.startswith("background_") and key != "world_shell_version"}
    bounded["model_metadata"].update(resolution=2, core_resolution=2, scene_mapping="bounded",
                                      grid_min=low.tolist(), grid_max=(low + 2*edge).tolist())
    bounded["model_state"]["cell_indices"][:] = 0
    return payload, bounded, rays, float(edge[0])


def check_ellipsoid_cpu():
    import numpy as np
    checked = 0
    for shell, subdivision in ((0, 0), (3, 1), (7, 0), (7, 2)):
        payload, _, rays, edge = ellipsoid_case(np, shell, subdivision)
        expected = [oracle(payload, *ray, np) for ray in rays]
        assert [len(value["hits"]) for value in expected] == [1, 0, 1, 1]
        assert not oracle(payload, *rays[0], np, maxt=.1*edge)["hits"]
        legacy = copy.deepcopy(payload)
        legacy["model_metadata"].update(world_shell_version=2, background_geometry="occupied_cell")
        assert len(oracle(legacy, *rays[1], np)["hits"]) == 1
        checked += len(rays)
    return {"independent_rotated_background_ellipsoid_hits_misses_different_leaf_sizes": checked}


def check_ellipsoid_geometry(mi, dr, np):
    """Use a same-size bounded leaf as the geometry-proxy reference.

    Hard-hit RGB is discontinuous in geometry, so its finite differences are
    not an oracle for the intentional boundary proxy. The bounded renderer
    exercises the same proxy without background masking or mixed leaf sizes.
    """
    from reconstruction_common import model_metadata, restore_model
    checked = 0
    for shell, subdivision in ((0, 0), (3, 1), (7, 0), (7, 2)):
        payload, bounded, rays, edge = ellipsoid_case(np, shell, subdivision)
        model, reference = restore_model(payload, mi), restore_model(bounded, mi)
        assert model.native_background_use_ellipsoid()
        ray = make_ray(mi, np, rays)
        expected = [oracle(payload, *r, np) for r in rays]
        traced = model.trace(ray, mi.Color3f(*ENV), 0., 128)
        actual = evaluate(model, mi, dr, ray, coefficient=0.)
        baseline = evaluate(reference, mi, dr, ray, coefficient=0.)
        dr.eval(traced)
        np.testing.assert_array_equal(np.asarray(traced[2]), [1, 0, 1, 1])
        np.testing.assert_array_equal(np.asarray(actual["hit_count"]), np.asarray(traced[2]))
        np.testing.assert_allclose(np.asarray(traced[0]).T, [out["rgb"] for out in expected], atol=3e-6)
        np.testing.assert_allclose(np.asarray(actual["raw_rgb"]), np.asarray(traced[0]), atol=3e-6)
        assert np.asarray(actual["geometry_valid"]).any()
        for key in ("centers", "log_scales", "rotation_tangent"):
            gradient = np.asarray(actual["gradients"][key])
            assert np.isfinite(gradient).all() and np.any(gradient != 0), (shell, subdivision, key)
            np.testing.assert_allclose(gradient, np.asarray(baseline["gradients"][key]),
                atol=8e-5, rtol=2e-3, err_msg=f"shell={shell}/S={subdivision}/{key}")
        # A finite ray that ends before entering the leaf must miss in both paths.
        short = make_ray(mi, np, [rays[0]], .1*edge)
        short_trace = model.trace(short, mi.Color3f(*ENV), 0., 128)
        short_loss = evaluate(model, mi, dr, short, coefficient=0.)
        dr.eval(short_trace)
        np.testing.assert_array_equal(np.asarray(short_trace[2]), [0])
        np.testing.assert_array_equal(np.asarray(short_loss["hit_count"]), [0])
        # Legacy checkpoint geometry remains a filled leaf and receives no
        # geometric gradients, even when its stored ellipsoid misses the ray.
        legacy = copy.deepcopy(payload)
        legacy["model_metadata"].update(world_shell_version=2, background_geometry="occupied_cell")
        old_model = restore_model(legacy, mi)
        assert not old_model.native_background_use_ellipsoid()
        old = evaluate(old_model, mi, dr, make_ray(mi, np, [rays[1]]), coefficient=0.)
        np.testing.assert_array_equal(np.asarray(old["hit_count"]), [1])
        for key in ("centers", "log_scales", "rotation_tangent"):
            np.testing.assert_array_equal(np.asarray(old["gradients"][key]), 0.)
        saved = {"model_metadata": model_metadata(model), "model_state": {
            key: np.asarray(value).copy() for key, value in model.snapshot().items()}}
        assert saved["model_metadata"]["background_geometry"] == "ellipsoid"
        assert saved["model_metadata"]["world_shell_version"] == 7
        assert restore_model(saved, mi).native_background_use_ellipsoid()
        checked += len(rays)
    return {"ellipsoid_trace_loss_hit_miss_and_bounded_proxy_parity": checked,
            "legacy_occupied_cell_and_geometry_metadata_roundtrip": "passed"}


def check_ellipsoid_optimizer(mi, dr, np):
    from reconstruction_common import restore_model
    from reconstruction_optimizer import FalcorOptimizer, OptimizerConfig
    compared = 0
    for shell, subdivision in ((0, 0), (7, 2)):
        payload, _, _, edge = ellipsoid_case(np, shell, subdivision)
        # Force lower and upper clamps; a foreground-sized bound would shrink
        # coarse background axes by orders of magnitude.
        payload["model_state"]["log_scales"] = np.log(edge * np.array([.5e-4, .3, 5.])).astype(np.float32)
        snapshots, moments = [], []
        for mode in ("native", "dense", "compact_reference"):
            model = restore_model(payload, mi)
            optimizer = FalcorOptimizer(model, OptimizerConfig(max_scale_voxels=.4),
                sparse_updates=mode != "dense", replay_updates=False)
            out = {"gradients": {key: mi.Float([.2] * width) for key, width in optimizer.dimensions.items()},
                   "appearance_valid": mi.UInt32([1]), "geometry_valid": mi.UInt32([1]),
                   "background_carve_valid": mi.UInt32([0])}
            if mode == "compact_reference":
                optimizer._step_compact_reference(model, out, sample_count=1, epoch=30,
                                                   coarse_to_fine=True, update_allowed=None)
            else:
                optimizer.step(model, out, sample_count=1, epoch=30, coarse_to_fine=True)
            dr.sync_thread()
            snapshot = {key: np.asarray(value).copy() for key, value in model.snapshot().items()}
            state = optimizer.state_dict()
            for key in ("centers", "log_scales", "rotation_tangent"):
                np.testing.assert_array_equal(state["groups"][key]["steps"], 1)
                assert np.any(state["groups"][key]["mean"] != 0)
            for key in ("centers", "log_scales", "rotations"):
                assert np.any(snapshot[key] != payload["model_state"][key]), (mode, key)
            np.testing.assert_allclose(np.exp(snapshot["log_scales"][[0, 2]]),
                                       edge*np.array([1e-4, .4]), atol=1e-7, rtol=2e-6)
            snapshots.append(snapshot)
            moments.append(state)
        for snapshot, state in zip(snapshots[1:], moments[1:]):
            for key in snapshots[0]:
                np.testing.assert_allclose(snapshot[key], snapshots[0][key], atol=3e-6, rtol=3e-5)
            for key, group in moments[0]["groups"].items():
                for name, values in group.items():
                    np.testing.assert_allclose(state["groups"][key][name], values, atol=1e-7, rtol=3e-5)
            compared += 1
    return {"ellipsoid_background_native_dense_compact_Adam_and_leaf_scale_clamps": compared}


def check_cpu():
    import numpy as np
    for levels in (1, 5, 8):
        payload = fixture(np, levels=levels, include_core=False)
        volume = sum(float(np.prod(high-low)) for _, low, high, _ in boxes(payload, np))
        assert volume == (2*2**levels)**3-8
        result = oracle(payload, (.13, .22, 0), (0, 0, 1), np)
        assert len(result["hits"]) == levels
        np.testing.assert_allclose(result["alpha"], 1-.94**levels, atol=2e-8)
    parent = fixture(np, core=1024, levels=8, include_core=False)
    axis_rays = [((.13, .22, 0), (0, 0, sign)) for sign in (-1, 1)]
    expected = [oracle(parent, *ray, np) for ray in axis_rays]
    children_checked = 0
    for subdivision in (1, 2):
        payload = inherited_children(parent, subdivision, np)
        prepared = prepare_boxes(payload, np)
        volume = np.sum(np.prod(prepared[1]-prepared[0], axis=1))
        assert volume == (2*2**8)**3-8
        for ray, baseline in zip(axis_rays, expected):
            out = oracle(payload, *ray, np, prepared=prepared)
            assert len(out["hits"]) == 8*2**subdivision
            np.testing.assert_allclose(out["rgb"], baseline["rgb"], atol=3e-8, rtol=2e-7)
            np.testing.assert_allclose(out["alpha"], baseline["alpha"], atol=3e-8)
        children_checked += len(prepared[0])
    return {"independent_world_box_volume_and_axial_oracle": "passed",
            "independent_subdivision_boxes": children_checked,
            "axis_optical_thickness_initialization_invariance": "passed"}


def inherited_children(parent, subdivision, np):
    """Independent reference split; only the axis-aligned alpha invariant is exact."""
    result = copy.deepcopy(parent)
    md = result["model_metadata"]
    assert not md.get("background_subdivision_level", 0)
    n = md["resolution"]
    assert np.all(parent["model_state"]["cell_indices"] >= n**3)
    count, children = len(parent["model_state"]["cell_indices"]), 8**subdivision
    for key, value in parent["model_state"].items():
        if key == "cell_indices":
            result["model_state"][key] = np.arange(n**3, n**3+count*children, dtype=np.uint64)
        else:
            result["model_state"][key] = np.repeat(value.reshape(count, -1), children, axis=0).ravel()
    opacity = result["model_state"]["opacity_sh"].astype(float)*Y00
    alpha = 1/(1+np.exp(-opacity))
    child_alpha = -np.expm1(np.log1p(-alpha)/(2**subdivision))
    result["model_state"]["opacity_sh"] = (np.log(child_alpha/(1-child_alpha))/Y00).astype(np.float32)
    md["background_subdivision_level"] = subdivision
    return result


def check_subdivided_trace(mi, dr, np):
    from reconstruction_common import restore_model, model_metadata
    rays = [((.13, .22, 0), (0, 0, 1)), ((.13, .22, 0), (0, 0, -1)),
            ((0, 0, 0), (1, 1, 1)), ((0, 0, 0), (-1, -1, -1)),
            ((1, .2, .3), (0, 1, 0)), ((-1, .2, .3), (0, -1, 0)),
            ((256, 0, 0), (0, 1, 0)), ((-256, .13, .22), (1, 0, 0)),
            ((257, .13, .22), (-1, 0, 0)), ((1.8, -.7, .2), (-.2, .8, .1)),
            ((3, .15, .23), (-1, .023, .041)), ((.4, -.25, .75), (.613, -.287, .119)),
            # Resolve both sides of the near-rational shared-edge ray below.
            ((.4, -.251, .75), (.6, -.3, .1)), ((.4, -.249, .75), (.6, -.3, .1))]
    rng = np.random.default_rng(971)
    rays += [(tuple(rng.uniform(-1, 1, 3)), tuple(rng.normal(size=3))) for _ in range(20)]
    rays = [(tuple(np.float32(o)), tuple(np.float32(d))) for o, d in rays]
    count, gradient_checks = 0, 0
    for subdivision in (1, 2):
        payload = fixture(np, 1024, 8, subdivision=subdivision, sh_count=9)
        prepared = prepare_boxes(payload, np)
        model = restore_model(payload, mi)
        assert model.native_background_subdivision_level() == subdivision
        mode_colors = []
        for active_sh in (1, 9):
            model.set_active_sh_count(active_sh)
            for maxt in (math.inf, 2.3):
                expected = [oracle(payload, *ray, np, maxt=maxt,
                    active_sh_count=active_sh, prepared=prepared) for ray in rays]
                ray = make_ray(mi, np, rays, maxt)
                traced = model.trace(ray, mi.Color3f(*ENV), 0., 128)
                dr.eval(traced)
                np.testing.assert_array_equal(np.asarray(traced[2]), [len(x["hits"]) for x in expected],
                    err_msg=f"S{subdivision}/SH{active_sh}/maxt={maxt}")
                np.testing.assert_allclose(np.asarray(traced[0]).T, [x["rgb"] for x in expected], atol=4e-6, rtol=5e-5)
                np.testing.assert_allclose(np.asarray(traced[1]), [x["alpha"] for x in expected], atol=4e-6)
                result = evaluate(model, mi, dr, ray)
                np.testing.assert_allclose(np.asarray(result["raw_rgb"]), np.asarray(traced[0]), atol=4e-6)
                np.testing.assert_array_equal(np.asarray(result["hit_count"]), np.asarray(traced[2]))
                np.testing.assert_allclose(np.asarray(result["distortion_loss"]),
                    [x["distortion"] for x in expected], atol=4e-6, rtol=3e-4)
                bg = payload["model_state"]["cell_indices"] >= 1024**3
                for name in ("centers", "log_scales", "rotation_tangent"):
                    assert not np.asarray(result["gradients"][name]).reshape(-1, 3)[bg].any()
                gradient = np.asarray(result["gradients"]["radiance_sh"]).reshape(-1, 9, 3)
                if active_sh == 1:
                    assert not gradient[:, 1:].any()
                else:
                    assert np.any(gradient[:, 1:])
                if math.isinf(maxt):
                    mode_colors.append(np.asarray(traced[0]).copy())
                count += len(rays)
            single = evaluate(model, mi, dr, make_ray(mi, np, [rays[0]]))
            row = oracle(payload, *rays[0], np, active_sh_count=active_sh,
                         prepared=prepared)["hits"][0][1]
            for key, stride, component in (("opacity_sh", 1, 0),
                                           ("radiance_sh", 27, 0), ("radiance_sh", 27, 25)):
                index = row*stride+component
                values = []
                parameters = []
                for sign in (-1, 1):
                    perturbed = dict(payload)
                    perturbed["model_state"] = dict(payload["model_state"])
                    perturbed["model_state"][key] = payload["model_state"][key].copy()
                    perturbed["model_state"][key][index] += sign*.002
                    out = oracle(perturbed, *rays[0], np, active_sh_count=active_sh, prepared=prepared)
                    values.append(float(np.sum((out["rgb"]-TARGET)**2)+.7*out["distortion"]))
                    parameters.append(float(perturbed["model_state"][key][index]))
                numerical = (values[1]-values[0])/(parameters[1]-parameters[0])
                analytic = float(np.asarray(single["gradients"][key])[index])
                np.testing.assert_allclose(analytic, numerical, atol=2e-6, rtol=3e-3,
                    err_msg=f"S{subdivision}/SH{active_sh}/{key}[{index}]")
                gradient_checks += 1
        assert np.any(mode_colors[0] != mode_colors[1])
        saved = {"model_metadata": model_metadata(model), "model_state": {
            key: np.asarray(value).copy() for key, value in model.snapshot().items()}}
        restored = restore_model(saved, mi)
        assert restored.native_background_subdivision_level() == subdivision
        restored_trace = restored.trace(ray, mi.Color3f(*ENV), 0., 128)
        dr.eval(restored_trace)
        np.testing.assert_allclose(np.asarray(restored_trace[0]), np.asarray(traced[0]), atol=1e-7)
    return {"H8_N1024_S1_S2_trace_training_SH1_SH9_oracle_rays": count,
            "subdivided_background_SH1_SH9_finite_differences": gradient_checks,
            "subdivision_metadata_roundtrip_and_frozen_geometry": "passed"}


def check_fp32_shared_edge(mi, dr, np):
    """Check a sub-ULP shared-edge interval under the renderer's FP32 arithmetic.

    The decimal direction (.6,-.3,.1) and FP32 inputs produce double-precision
    intervals of 4.211e-8 at t=1.69558 and 2.106e-7 at t=8.47791. Both are less
    than one FP32 time ULP. Reciprocal/multiply FP32 slab arithmetic reverses
    the first boundary ordering and collapses the second. This test has exact
    hit assertions for that arithmetic; the main double oracle remains strict.
    """
    from reconstruction_common import restore_model
    cases = [((.4, -.25, .75), (.6, -.3, .1), (25, 53)),
             ((3, .15, .23), (-1, .02, .04), (19, 39))]
    tested = 0
    for origin_values, direction_values, counts in cases:
        origin = tuple(np.float32(origin_values))
        direction = tuple(np.float32(direction_values))
        for subdivision, hit_count in zip((1, 2), counts):
            payload = fixture(np, 1024, 8, subdivision=subdivision, sh_count=9)
            prepared = prepare_boxes(payload, np)
            model = restore_model(payload, mi)
            ray = make_ray(mi, np, [(origin, direction)])
            normalized = ray.d/dr.norm(ray.d)
            inverse = dr.rcp(normalized)
            dr.eval(normalized, inverse)
            cpu_d = np.asarray(direction, np.float32)/np.linalg.norm(np.asarray(direction, np.float32))
            np.testing.assert_array_equal(np.asarray(normalized).ravel(), cpu_d)
            np.testing.assert_array_equal(np.asarray(inverse).ravel(), np.float32(1)/cpu_d)
            for active_sh in (1, 9):
                model.set_active_sh_count(active_sh)
                expected = oracle(payload, origin, direction, np, active_sh_count=active_sh,
                                  prepared=prepared, fp32_intersections=True)
                assert len(expected["hits"]) == hit_count
                result = evaluate(model, mi, dr, ray)
                trace = model.trace(ray, mi.Color3f(*ENV), 0., 128)
                dr.eval(trace)
                np.testing.assert_array_equal(np.asarray(trace[2]), [hit_count])
                np.testing.assert_array_equal(np.asarray(result["hit_count"]), [hit_count])
                np.testing.assert_allclose(np.asarray(trace[0]).ravel(), expected["rgb"], atol=4e-6, rtol=5e-5)
                np.testing.assert_allclose(np.asarray(result["raw_rgb"]), np.asarray(trace[0]), atol=4e-6)
                np.testing.assert_allclose(np.asarray(trace[1]), [expected["alpha"]], atol=4e-6)
                wanted = sorted(hit[1] for hit in expected["hits"])
                observed = np.flatnonzero(np.asarray(result["deletion_observed"])).tolist()
                assert observed == wanted, (subdivision, active_sh, observed, wanted)
                tested += 1
    return {"sub_ULP_shared_edge_FP32_slab_oracle_and_hit_ID_sets": tested}


def check_invalid_subdivision(mi, dr, np):
    empty = {key: (mi.UInt64([]) if key == "cell_indices" else mi.Float([]))
             for key in fixture(np)["model_state"]}

    def construct(n, h, subdivision, state=empty):
        return mi.VoxelReconstruction.from_state(n, [-1.]*3, [1.]*3, state,
            4, 1, n, h, subdivision)

    checked = 0
    for n, h, subdivision in ((8, 0, 1), (8, 2, 3), (1024, 8, 10), (1024, 11, 0)):
        try:
            construct(n, h, subdivision)
        except (RuntimeError, ValueError):
            checked += 1
        else:
            raise AssertionError((n, h, subdivision))
    for n, h, subdivision in ((8, 0, 0), (8, 8, 2), (1024, 10, 6), (1024, 10, 9)):
        model = construct(n, h, subdivision)
        assert model.native_background_subdivision_level() == subdivision
        ray = make_ray(mi, np, [((0, 0, 0), (0, 0, 1))])
        traced = model.trace(ray, mi.Color3f(*ENV), 0., 128)
        dr.eval(traced)
        np.testing.assert_array_equal(np.asarray(traced[2]), [0])
    from reconstruction_common import restore_model
    model = restore_model(fixture(np, 8, 2, subdivision=1), mi)
    revision = model.topology_revision
    before = {key: np.asarray(value).copy() for key, value in model.snapshot().items()}
    for h, subdivision in ((2, 3), (0, 1), (11, 1), (2, 7)):
        try:
            model.set_world_shells(h, subdivision)
        except (RuntimeError, ValueError):
            checked += 1
        else:
            raise AssertionError((h, subdivision))
        assert model.native_background_shell_levels() == 2
        assert model.native_background_subdivision_level() == 1
        assert model.topology_revision == revision
        for key, value in model.snapshot().items():
            np.testing.assert_array_equal(np.asarray(value), before[key])
    single = fixture(np, 8, 1)["model_state"]
    size = len(single["cell_indices"])
    single = {key: value.reshape(size, -1)[:1].ravel().copy() for key, value in single.items()}
    single["cell_indices"][:] = 8**3+56*2*8  # first invalid ID at H2/S1
    state = {key: mi.UInt64(value) if key == "cell_indices" else mi.Float(value)
             for key, value in single.items()}
    try:
        construct(8, 2, 1, state)
    except (RuntimeError, ValueError):
        checked += 1
    else:
        raise AssertionError("Accepted out-of-range subdivided background ID")
    return {"native_invalid_subdivision_or_ID_rejections": checked,
            "valid_empty_models_and_setter_rollback": "passed"}


def high_address_case(np):
    """Six physical leaves, including three pairs sharing their low 32 bits."""
    payload = fixture(np, 1024, 1, sh_count=9)
    state, metadata = payload["model_state"], payload["model_metadata"]
    count = len(state["cell_indices"])
    payload["model_state"] = state = {
        key: values.reshape(count, -1)[:6].ravel().copy() for key, values in state.items()}
    state["cell_indices"] = np.array([
        17, 1024**3 + 37, 1024**3 + 239,
        2**32 + 17, 2**32 + 1024**3 + 37, 2**32 + 1024**3 + 239], np.uint64)
    metadata.update(background_shell_levels=8, background_subdivision_level=8,
                    background_geometry="ellipsoid", world_shell_version=4)
    low, high, _ = prepare_boxes(payload, np)
    edges = high - low
    state["centers"][:] = .5
    state["log_scales"] = np.log(edges * .35).astype(np.float32).ravel()
    state["opacity_sh"][:] = 0.
    coefficients = state["radiance_sh"].reshape(6, 9, 3)
    coefficients[:] = 0.
    coefficients[:, 0] = np.array([[.2 + .08*r, .5 - .05*r, .1 + .025*r] for r in range(6)]) / Y00
    rays, limits = [], []
    for lower, edge in zip(low, edges):
        center = lower + .5*edge
        for offset in (0., .49):
            origin = center + edge * [offset, 0., -2.]
            rays.append((tuple(np.float32(origin)), (0., 0., 1.)))
            limits.append(float(4*edge[0]))
    return payload, rays, limits


def check_high_address_cpu():
    import numpy as np
    payload, rays, limits = high_address_case(np)
    ids = payload["model_state"]["cell_indices"]
    assert np.all(ids[3:] > 2**32)
    np.testing.assert_array_equal(ids[:3], ids[3:] - np.uint64(2**32))
    for index, (ray, limit) in enumerate(zip(rays, limits)):
        result = oracle(payload, *ray, np, maxt=limit)
        assert [hit[1] for hit in result["hits"]] == ([] if index % 2 else [index // 2])
    return {"S8_sparse_UInt64_distinct_core_background_and_background_alias_oracle_rays": len(rays)}


def check_high_address_trace(mi, dr, np):
    from reconstruction_common import model_metadata, restore_model
    payload, rays, limits = high_address_case(np)
    expected = [oracle(payload, *ray, np, maxt=limit) for ray, limit in zip(rays, limits)]
    model = restore_model(payload, mi)
    ray = make_ray(mi, np, rays)
    ray.maxt = mi.Float(limits)
    traced = model.trace(ray, mi.Color3f(*ENV), 0., 128)
    result = evaluate(model, mi, dr, ray)
    dr.eval(traced)
    np.testing.assert_array_equal(np.asarray(traced[2]), [len(value["hits"]) for value in expected])
    np.testing.assert_allclose(np.asarray(traced[0]).T, [value["rgb"] for value in expected], atol=3e-6)
    np.testing.assert_allclose(np.asarray(traced[1]), [value["alpha"] for value in expected], atol=3e-6)
    np.testing.assert_array_equal(np.asarray(result["hit_count"]), np.asarray(traced[2]))
    np.testing.assert_allclose(np.asarray(result["raw_rgb"]), np.asarray(traced[0]), atol=3e-6)
    np.testing.assert_array_equal(np.flatnonzero(np.asarray(result["deletion_observed"])), np.arange(6))
    snapshot = {key: np.asarray(value).copy() for key, value in model.snapshot().items()}
    assert snapshot["cell_indices"].dtype == np.dtype(np.uint64)
    np.testing.assert_array_equal(snapshot["cell_indices"], payload["model_state"]["cell_indices"])
    saved = {"model_metadata": model_metadata(model), "model_state": snapshot}
    assert saved["model_metadata"]["world_shell_version"] == 7
    assert saved["model_metadata"]["background_subdivision_level"] == 8
    restored = restore_model(saved, mi)
    restored_trace = restored.trace(ray, mi.Color3f(*ENV), 0., 128)
    dr.eval(restored_trace)
    np.testing.assert_array_equal(np.asarray(restored_trace[2]), np.asarray(traced[2]))
    np.testing.assert_allclose(np.asarray(restored_trace[0]), np.asarray(traced[0]), atol=1e-7)
    return {"S8_sparse_UInt64_trace_training_hit_IDs_snapshot_roundtrip": len(rays)}


def check_trace(mi, dr, np):
    from reconstruction_common import restore_model, model_metadata
    count = 0
    for core, levels in ((8, 1), (8, 5), (8, 8), (1024, 8)):
        payload = fixture(np, core, levels)
        model = restore_model(payload, mi)
        radius = 2**levels
        rays = [((.13, .22, 0), (0, 0, 1)), ((0, 0, 0), (1, 1, 1)),
                ((0, 0, 0), (-1, -1, -1)), ((1, .2, .3), (0, 1, 0)),
                ((-1, .2, .3), (0, -1, 0)), ((radius, 0, 0), (0, 1, 0)),
                ((-radius, .13, .22), (1, 0, 0)), ((radius+1, .13, .22), (-1, 0, 0)),
                ((1.8, -.7, .2), (-.2, .8, .1)), ((3, .15, .23), (-1, .02, .04))]
        rng = np.random.default_rng(151+levels)
        rays += [(tuple(rng.uniform(-2.2*radius, 2.2*radius, 3)), tuple(rng.normal(size=3)))
                 for _ in range(45)]
        rays += [(tuple(rng.uniform(-1, 1, 3)), tuple(rng.normal(size=3))) for _ in range(45)]
        for maxt in (math.inf, 2.3):
            # Match native float32 input, while independently computing intersections in double.
            rays = [(tuple(np.float32(o)), tuple(np.float32(d))) for o, d in rays]
            expected = [oracle(payload, *r, np, maxt=maxt) for r in rays]
            ray = make_ray(mi, np, rays, maxt)
            rgb, alpha, hits = model.trace(ray, mi.Color3f(*ENV), 0., 128)
            dr.eval(rgb, alpha, hits)
            np.testing.assert_allclose(np.asarray(rgb).T, [x["rgb"] for x in expected], atol=3e-6, rtol=3e-5)
            np.testing.assert_allclose(np.asarray(alpha), [x["alpha"] for x in expected], atol=3e-6)
            np.testing.assert_array_equal(np.asarray(hits), [len(x["hits"]) for x in expected])
            loss = evaluate(model, mi, dr, ray)
            np.testing.assert_allclose(np.asarray(loss["raw_rgb"]), np.asarray(rgb), atol=3e-6)
            np.testing.assert_array_equal(np.asarray(loss["hit_count"]), np.asarray(hits))
            np.testing.assert_allclose(np.asarray(loss["distortion_loss"]),
                                       [x["distortion"] for x in expected], atol=3e-6, rtol=2e-4)
            bg = payload["model_state"]["cell_indices"] >= core**3
            for key in ("centers", "log_scales", "rotation_tangent"):
                assert not np.asarray(loss["gradients"][key]).reshape(-1, 3)[bg].any(), key
            count += len(rays)
        # Native snapshot + metadata roundtrip must retain the world root.
        saved = {"model_metadata": model_metadata(model), "model_state": {
            k: np.asarray(v).copy() for k, v in model.snapshot().items()}}
        restored = restore_model(saved, mi)
        assert restored.native_background_shell_levels() == levels
        restored_trace = restored.trace(ray, mi.Color3f(*ENV), 0., 128)
        dr.eval(restored_trace)
        np.testing.assert_allclose(np.asarray(restored_trace[0]), np.asarray(rgb), atol=1e-7)
    return {"mixed_size_trace_training_oracle_rays": count,
            "background_geometry_zero_and_snapshot_roundtrip": "passed"}


def check_derivatives(mi, dr, np):
    from reconstruction_common import restore_model
    payload = fixture(np, 8, 5, include_core=False)
    origin, direction = (.13, .22, 0), (.31, -.12, 1.)
    origin, direction = tuple(np.float32(origin)), tuple(np.float32(direction))
    model = restore_model(payload, mi)
    actual = evaluate(model, mi, dr, make_ray(mi, np, [(origin, direction)]))
    expected = oracle(payload, origin, direction, np)
    checked = 0
    for hit in expected["hits"]:
        row = hit[1]
        for key, stride, components in (("opacity_sh", 1, (0,)),
                                         ("radiance_sh", 12, (0, 3, 7, 11))):
            for component in components:
                index = row*stride+component
                plus, minus = copy.deepcopy(payload), copy.deepcopy(payload)
                plus["model_state"][key][index] += .002
                minus["model_state"][key][index] -= .002
                denominator = float(plus["model_state"][key][index]-minus["model_state"][key][index])
                def objective(p):
                    out = oracle(p, origin, direction, np)
                    return np.sum((out["rgb"]-TARGET)**2)+.7*out["distortion"]
                numerical = (objective(plus)-objective(minus))/denominator
                analytic = float(np.asarray(actual["gradients"][key])[index])
                np.testing.assert_allclose(analytic, numerical, atol=2e-6, rtol=2e-3,
                                           err_msg=f"{key}[{index}]")
                checked += 1
    assert checked >= 25
    return {"background_opacity_and_directional_SH_finite_differences": checked}


def check_depth20_near_planes(mi, dr, np):
    """Adding a large root offset must not round across a core cell face."""
    from reconstruction_common import restore_model
    payload = fixture(np, 1024, 10, include_core=False)
    core, size, zcell, ycell = 1024, 2/1024, 512, 576
    xs = (-.50001, -.49999, -1e-5, 1e-5, .49999, .50001, -.99999, .99999)
    core_state = {k: [] for k in payload["model_state"]}
    rays = []
    for x in xs:
        column = int(math.floor((x+1)/size))
        fractional = (x+1)/size-column
        center_x = .01 if fractional < .5 else .99
        core_state["cell_indices"].append(column+core*(ycell+core*zcell))
        core_state["centers"].extend((center_x, .5, .5))
        core_state["log_scales"].extend([math.log(.45*size)]*3)
        core_state["rotations"].extend((1., 0., 0., 0.))
        core_state["radiance_sh"].extend([.8/Y00, .3/Y00, .2/Y00]+[0.]*9)
        core_state["opacity_sh"].append(0.)
        y = -1+(ycell+.5)*size
        # Inside the core, outside the core, and inside a background shell.
        # Both signs and oblique rays retain the same tiny distance to the face.
        for z, dz in ((-.01, 1.), (.02, -1.), (-1.01, 1.), (1.01, -1.)):
            for dx in (0., 1e-7, -1e-7):
                rays.append(((x, y, z), (dx, 0., dz)))
    for key, data in payload["model_state"].items():
        values = np.asarray(core_state[key], dtype=data.dtype)
        payload["model_state"][key] = np.concatenate((values, data))
    rays = [(tuple(np.float32(o)), tuple(np.float32(d))) for o, d in rays]
    expected = [oracle(payload, *r, np) for r in rays]
    assert all(any(h[1] < len(xs) for h in item["hits"]) for item in expected)
    model = restore_model(payload, mi)
    ray = make_ray(mi, np, rays)
    traced = model.trace(ray, mi.Color3f(*ENV), 0., 128)
    dr.eval(traced)
    loss = evaluate(model, mi, dr, ray)
    np.testing.assert_array_equal(np.asarray(traced[2]), [len(r["hits"]) for r in expected])
    np.testing.assert_allclose(np.asarray(traced[0]).T, [r["rgb"] for r in expected], atol=3e-6, rtol=3e-5)
    np.testing.assert_array_equal(np.asarray(loss["hit_count"]), np.asarray(traced[2]))
    np.testing.assert_allclose(np.asarray(loss["raw_rgb"]), np.asarray(traced[0]), atol=3e-6)
    return {"depth20_near_core_faces_parallel_and_oblique_rays": len(rays)}


def check_growth(mi, dr, np):
    from reconstruction_common import restore_model
    payload = fixture(np, 8, 5, include_core=False)
    model = restore_model(payload, mi)
    proposal = fixture(np, 8, 1)["model_state"]
    widths = {"cell_indices": 1, "centers": 3, "log_scales": 3, "rotations": 4,
              "radiance_sh": 12, "opacity_sh": 1}
    proposal = {k: v[:widths[k]].copy() for k, v in proposal.items()}
    proposal["opacity_sh"][:] = 0.
    ray = make_ray(mi, np, [((.125, .125, -.8), (0, 0, 1))])
    before = evaluate(model, mi, dr, ray, coefficient=0.)
    model.set_growth_candidates({k: mi.UInt64(v) if k == "cell_indices" else mi.Float(v)
                                 for k, v in proposal.items()})
    with_growth = evaluate(model, mi, dr, ray, growth=True, coefficient=0.)
    np.testing.assert_array_equal(np.asarray(before["raw_rgb"]), np.asarray(with_growth["raw_rgb"]))
    inserted = copy.deepcopy(payload)
    inserted["model_state"] = {k: np.concatenate((v, proposal[k]))
                               for k, v in payload["model_state"].items()}
    inserted_model = restore_model(inserted, mi)
    changed = evaluate(inserted_model, mi, dr, ray, coefficient=0.)
    expected = np.asarray(before["rgb_loss"])-np.asarray(changed["rgb_loss"])
    np.testing.assert_allclose(np.asarray(with_growth["growth_gain"]), expected, atol=2e-6, rtol=2e-4)
    np.testing.assert_array_equal(np.asarray(with_growth["growth_observed"]), [1])
    return {"core_growth_overlay_on_mixed_world_tree": "passed"}


def main():
    from preview import RuntimeConfig
    from reconstruction_common import initialize_runtime
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    reports = [check_cpu(), check_ellipsoid_cpu(), check_high_address_cpu()]
    print(json.dumps(reports[-1]), flush=True)
    if not args.cpu_only:
        with tempfile.TemporaryDirectory(prefix="world-shell-native-") as directory:
            mi, dr, np, _ = initialize_runtime(RuntimeConfig(args.build_dir, args.gpu), Path(directory), {})
            assert getattr(mi.VoxelReconstruction, "world_shell_version", 0) >= 4, "Rebuild native world-shell support"
            assert getattr(mi.VoxelReconstruction, "cell_index_bits", 0) == 64, "Rebuild native UInt64 cell support"
            for check in (check_trace, check_derivatives, check_depth20_near_planes, check_growth,
                          check_subdivided_trace, check_fp32_shared_edge, check_invalid_subdivision,
                          check_ellipsoid_geometry, check_ellipsoid_optimizer, check_high_address_trace):
                reports.append(check(mi, dr, np))
                print(json.dumps(reports[-1]), flush=True)
    print(json.dumps({"status": "passed", "checks": reports}), flush=True)


if __name__ == "__main__":
    main()
