"""Independent float64 checks of Falcor's explicit boundary surrogate.

Hard visibility has no ordinary useful finite-difference derivative. Instead we
freeze the forward minimizer and surface normalization, finite-difference the
implicit ellipsoid function (including right-multiplied rotations), and apply
the documented Falcor clamp/gating/PDF rules. No dataset or image is required.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


def matrix(np, quaternion):
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def multiply(np, a, b):
    return np.concatenate(([a[0]*b[0]-np.dot(a[1:], b[1:])],
        a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:], b[1:])))


def frozen_g_gradient(np, state, index, point):
    """Numerical derivative, independent of the C++ closed-form gradient."""
    center, logs, q = (state[k][index].copy() for k in ("centers", "log_scales", "rotations"))
    size = state["size"]
    origin = state["cells"][index]*size

    def implicit(c, s, rotation):
        local = matrix(np, rotation).T @ (point-origin-c*size)
        return np.sum((local/np.exp(s))**2)-1

    answer = {}
    eps = 1e-6
    for key in ("centers", "log_scales", "rotation_tangent"):
        values = []
        for axis in range(3):
            unit = np.eye(3)[axis]*eps
            if key == "centers":
                plus, minus = implicit(center+unit, logs, q), implicit(center-unit, logs, q)
            elif key == "log_scales":
                plus, minus = implicit(center, logs+unit, q), implicit(center, logs-unit, q)
            else:
                dq = np.concatenate(([math.cos(eps/2)], np.eye(3)[axis]*math.sin(eps/2)))
                inv = dq.copy()
                inv[1:] *= -1
                plus = implicit(center, logs, multiply(np, q, dq))
                minus = implicit(center, logs, multiply(np, q, inv))
            values.append((plus-minus)/(2*eps))
        answer[key] = np.array(values)
    return answer


DEFAULTS = dict(alpha_loss_weight=.5, foreground_alpha_min=.95, background_alpha_max=1e-4,
                geometry_tau_voxel_fraction=.12, geometry_grad_clamp=5.,
                alpha_geometry_weight=.1, background_carve_weight=.01,
                geometry_two_sided_boundary=0., geometry_boundary_band_tau=1.,
                geometry_transition_only=1., geometry_hit_expansion_band_tau=0.,
                max_hits=16., max_candidates=16., transmittance_threshold=.01)


def oracle(np, state, origin, direction, target_rgb, target_alpha, pdf, settings):
    """Enumerate clipped cells, then differentiate their frozen hard path."""
    cfg = DEFAULTS | settings
    background = np.ones(3)
    direction = direction / np.linalg.norm(direction)
    encountered = []
    for i, cell in enumerate(state["cells"]):
        lower, upper = cell*state["size"], (cell+1)*state["size"]
        entry, end = 0., 100.
        for axis in range(3):
            if direction[axis] == 0:
                if not lower[axis] <= origin[axis] < upper[axis]:
                    end = -1
                    break
            else:
                bounds = (lower[axis]-origin[axis])/direction[axis], (upper[axis]-origin[axis])/direction[axis]
                entry, end = max(entry, min(bounds)), min(end, max(bounds))
        if end < entry:
            continue
        rotation = matrix(np, state["rotations"][i])
        scale = np.exp(state["log_scales"][i])
        world_center = lower + state["centers"][i]*state["size"]
        u = rotation.T @ (origin+entry*direction-world_center)/scale
        v = rotation.T @ ((end-entry)*direction)/scale
        star = np.clip(-np.dot(u, v)/np.dot(v, v), 0, 1) if np.dot(v, v) > 1e-20 else 0.
        point = origin+(entry+star*(end-entry))*direction
        unit = u+star*v
        g = np.dot(unit, unit)-1
        # Fixtures stay away from tangency, so sign(g) unambiguously determines hit.
        radius = np.linalg.norm(unit)
        distance_scale = 1/(2*np.linalg.norm(unit/radius/scale)) if radius > 1e-6 else 0.
        score = max(g*distance_scale, 0)/state["size"][0] + .05*np.linalg.norm(point-origin)/np.linalg.norm(state["size"]*state["resolution"])
        encountered.append(dict(index=i, point=point, g=g, ds=distance_scale, entry=entry,
                                score=score, hit=g<=0, radiance=state["rgb"][i], opacity=state["opacity"][i]))
    encountered.sort(key=lambda item: item["entry"])
    hits, misses, color, transmittance = [], [], np.zeros(3), 1.
    for record in encountered:
        record.update(before=transmittance, front=len(hits))
        if record["hit"]:
            hits.append(record)
            color += transmittance*record["opacity"]*record["radiance"]
            transmittance *= 1-record["opacity"]
            if len(hits) >= cfg["max_hits"] or transmittance <= cfg["transmittance_threshold"]:
                break
        else:
            misses.append(record)
    misses = sorted(misses, key=lambda item: item["score"])[:int(cfg["max_candidates"])]
    raw_rgb, raw_alpha = color+transmittance*background, 1-transmittance
    rgb, alpha = 1+(raw_rgb-1)*pdf, raw_alpha*pdf
    target = target_rgb*target_alpha + (1-target_alpha)
    dc, da = 2*(rgb-target), 2*cfg["alpha_loss_weight"]*(alpha-target_alpha)
    result = {key: np.zeros_like(state["centers"]) for key in ("centers", "log_scales", "rotation_tangent")}
    result.update(geometry_valid=np.zeros(len(state["cells"])), background_carve_valid=np.zeros(len(state["cells"])),
                  rgb=rgb, alpha=alpha, raw_rgb=raw_rgb, raw_alpha=raw_alpha, hit_count=len(hits))
    suffix_rgb, suffix_t = [background.copy()], [1.]
    for record in reversed(hits):
        suffix_rgb.insert(0, record["opacity"]*record["radiance"]+(1-record["opacity"])*suffix_rgb[0])
        suffix_t.insert(0, (1-record["opacity"])*suffix_t[0])
    is_bg = target_alpha <= cfg["background_alpha_max"]
    for record in hits+misses:
        index, hard = record["index"], record["hit"]
        if record["ds"] == 0 or (is_bg and not hard):
            continue
        suffix = record["front"]+int(hard)
        before, opacity = record["before"], record["opacity"]
        distance = record["g"]*record["ds"]
        tau = max(cfg["geometry_tau_voxel_fraction"]*state["size"][0], 1e-8)
        w = 1/(1+math.exp(distance/tau))
        slope = -w*(1-w)/tau
        derivative = np.dot(dc, before*opacity*(record["radiance"]-suffix_rgb[suffix]))*slope
        if hard and is_bg:
            derivative = -cfg["background_carve_weight"]*w/tau
        derivative += cfg["alpha_geometry_weight"]*da*before*opacity*suffix_t[suffix]*slope
        dg = np.clip(derivative*record["ds"], -cfg["geometry_grad_clamp"], cfg["geometry_grad_clamp"])
        contains = True
        for corner in np.ndindex(2, 2, 2):
            offset = (np.array(corner)-state["centers"][index])*state["size"]
            contains &= np.sum((matrix(np, state["rotations"][index]).T @ offset / np.exp(state["log_scales"][index]))**2) <= 1
        if cfg["geometry_two_sided_boundary"]:
            band = tau*cfg["geometry_boundary_band_tau"]
            t = np.clip(1-abs(distance)/band, 0, 1) if band > 0 else 0.
            if not (hard and is_bg):
                dg *= t*t*(3-2*t)
            if hard and dg > 0 and (is_bg or contains):
                dg = 0.
        elif cfg["geometry_transition_only"]:
            if hard and dg > 0:
                band = tau*cfg["geometry_hit_expansion_band_tau"]
                t = np.clip(1+distance/band, 0, 1) if band > 0 and not is_bg and not contains else 0.
                dg *= t*t*(3-2*t)
            elif not hard:
                dg = max(dg, 0.)
        partials = frozen_g_gradient(np, state, index, record["point"])
        for key, values in partials.items():
            result[key][index] += dg*pdf*values
        result["geometry_valid"][index] += 1
        result["background_carve_valid"][index] += hard and is_bg
    return result


def create_state(np, *, cells=((0, 0, 0),), resolution=1, radiance=(.2, .4, .3), scales=(.23, .31, .28), angle=.4):
    count = len(cells)
    return dict(cells=np.array(cells), resolution=resolution, size=np.ones(3)/resolution,
                centers=np.tile([.5, .5, .5], (count, 1)), log_scales=np.tile(np.log(scales), (count, 1)),
                rotations=np.tile([math.cos(angle/2), 0, 0, math.sin(angle/2)], (count, 1)),
                rgb=np.tile(radiance, (count, 1)), opacity=np.full(count, .6))


def native_model(np, mi, state):
    import drjit as dr
    cells, resolution = state["cells"], state["resolution"]
    buffer_f = dr.scalar.ArrayXf if mi.variant() == "scalar_rgb" else mi.Float
    buffer_u = dr.scalar.ArrayXu if mi.variant() == "scalar_rgb" else mi.UInt32
    data = {key: buffer_f(state[key].astype(np.float32).ravel()) for key in ("centers", "log_scales", "rotations")}
    data["cell_indices"] = buffer_u((cells[:, 0]+resolution*(cells[:, 1]+resolution*cells[:, 2])).astype(np.uint32))
    y00 = 1/(2*math.sqrt(math.pi))
    data["radiance_sh"] = buffer_f((state["rgb"]/y00).astype(np.float32).ravel())
    data["opacity_sh"] = buffer_f((np.log(state["opacity"]/(1-state["opacity"]))/y00).astype(np.float32))
    return mi.VoxelReconstruction.from_state(resolution, [0., 0., 0.], [1., 1., 1.], data, 1, 1)


def run_tests(np, mi, dr):
    # Off-axis rays give nonzero tangent-rotation gradients. Inputs are promoted
    # from float32 before the float64 oracle to isolate arithmetic differences.
    state = create_state(np)
    cases = [
        ("hard_hit_shrink", state, [.65, .58, -1.], [.95]*3, 1., 1., {}),
        ("hard_hit_expansion_blocked", state, [.65, .58, -1.], [0.]*3, 1., 1., {}),
        ("near_miss_expand", state, [.79, .56, -1.], [.1]*3, 1., 1., {}),
        ("near_miss_shrink_blocked", state, [.79, .56, -1.], [1.2]*3, 1., 1., {}),
        ("background_carve", state, [.65, .58, -1.], [0.]*3, 0., 1., {}),
        ("background_no_miss", state, [.79, .56, -1.], [0.]*3, 0., 1., {}),
        ("center_skip", create_state(np, angle=0.), [.5, .5, -1.], [.95]*3, 1., 1., {}),
        ("pdf_after_clamp", state, [.65, .58, -1.], [.95]*3, 1., 3., {"geometry_grad_clamp": .002}),
        ("two_sided_near_hit", state, [.72, .56, -1.], [0.]*3, 1., 1., {"geometry_two_sided_boundary": 1.}),
        ("two_sided_miss_shrink", state, [.75, .56, -1.], [1.2]*3, 1., 1., {"geometry_two_sided_boundary": 1.}),
        ("two_sided_deep_zero", state, [.65, .58, -1.], [.95]*3, 1., 1., {"geometry_two_sided_boundary": 1., "geometry_boundary_band_tau": .01}),
        ("legacy_expansion_band", state, [.72, .56, -1.], [0.]*3, 1., 1., {"geometry_hit_expansion_band_tau": 1.}),
        ("contains_cell_blocks_expansion", create_state(np, scales=(2., 2., 2.)), [.65, .58, -1.], [0.]*3, 1., 1., {"geometry_transition_only": 1., "geometry_hit_expansion_band_tau": 1000.}),
    ]
    tilted = create_state(np)
    tilted["rotations"][0] = np.concatenate(([math.cos(.3)], np.array([1., 2., 3.])/math.sqrt(14)*math.sin(.3)))
    cases.append(("all_rotation_axes", tilted, [.65, .58, -1.], [.95]*3, 1., 1., {}))
    # The miss occurs between two actual hits: it must see the correct suffix,
    # not the full composite or just the environment.
    multi = create_state(np, cells=((0, 0, 0), (0, 0, 1), (0, 0, 2)), resolution=3, scales=(.12, .13, .11))
    multi["centers"][1, 0] = .1
    multi["rgb"][:] = [[.2, .4, .7], [.8, .1, .3], [.3, .8, .2]]
    cases += [("candidate_suffix", multi, [.21, .19, -1.], [.1]*3, 1., 1., {}),
              ("hit_truncation", multi, [.21, .19, -1.], [.9]*3, 1., 1., {"max_hits": 1.})]
    nearest = create_state(np, cells=((0, 0, 0), (0, 0, 1), (0, 0, 2)), resolution=3, scales=(.04, .06, .06), angle=0.)
    nearest["centers"][:, 0] = [.1, .5, .7]
    cases.append(("nearest_candidate_capacity", nearest, [.29, .17, -1.], [.1]*3, 1., 1., {"max_candidates": 1.}))
    names = []
    max_error = 0.
    for name, spec, origin, target, alpha, pdf, settings in cases:
        # Use the actual stored float32 geometry in the independent reference.
        spec = {key: (value.astype(np.float32).astype(np.float64) if isinstance(value, np.ndarray) and key != "cells" else value)
                for key, value in spec.items()}
        model = native_model(np, mi, spec)
        origin, target = [np.array(value, dtype=np.float32).astype(np.float64) for value in (origin, target)]
        direction = np.array([0., 0., 1.])
        expected = oracle(np, spec, origin, direction, target, alpha, pdf, settings)
        ray = mi.Ray3f(mi.Point3f(origin.tolist()), mi.Vector3f(direction.tolist()))
        ray.maxt = 100.
        out = model.loss_and_grad(ray, mi.Color3f(target.tolist()), mi.Float(alpha),
                                  mi.Color3f(1.), mi.Float(pdf), settings)
        dr.eval(out)
        dr.sync_thread()
        for key in ("centers", "log_scales", "rotation_tangent"):
            actual = np.asarray(out["gradients"][key]).reshape(-1, 3)
            np.testing.assert_allclose(actual, expected[key], atol=8e-6, rtol=4e-4, err_msg=f"{name}: {key}")
            max_error = max(max_error, float(np.max(np.abs(actual-expected[key]))))
        for key in ("geometry_valid", "background_carve_valid", "hit_count"):
            np.testing.assert_array_equal(np.asarray(out[key]).ravel(), np.asarray(expected[key]).ravel(), err_msg=f"{name}: {key}")
        for key in ("rgb", "raw_rgb", "alpha", "raw_alpha"):
            np.testing.assert_allclose(np.asarray(out[key]).ravel(), np.asarray(expected[key]).ravel(), atol=2e-6, rtol=2e-5, err_msg=f"{name}: {key}")
        if name == "hard_hit_shrink":
            assert np.linalg.norm(expected["rotation_tangent"]) > .001, "Fixture must exercise rotation"
            assert np.sum(expected["log_scales"]) > 0, "Gradient descent must shrink this hit"
        if name == "near_miss_expand":
            assert expected["hit_count"] == 0 and np.sum(expected["log_scales"]) < 0
        if name == "all_rotation_axes":
            assert np.all(np.abs(expected["rotation_tangent"]) > 1e-4)
        if name == "nearest_candidate_capacity":
            np.testing.assert_array_equal(expected["geometry_valid"], [0, 0, 1])
        if name.endswith("blocked") or name in ("center_skip", "background_no_miss", "two_sided_deep_zero", "contains_cell_blocks_expansion"):
            assert np.max(np.abs(expected["log_scales"])) == 0, name
        print("PASS", name, flush=True)
        names.append(name)
    if mi.variant() == "cuda_ad_rgb":
        # Divergent SIMD lanes must retain independent paths/candidate slots;
        # their atomics must equal a sum of scalar reference derivatives.
        spec = create_state(np)
        spec = {key: (value.astype(np.float32).astype(np.float64) if isinstance(value, np.ndarray) and key != "cells" else value)
                for key, value in spec.items()}
        origins = np.array([[.65, .58, -1], [.79, .56, -1], [1.2, .5, -1],
                            [.5, .5, -1], [.72, .56, -1], [.79, .56, -1]], dtype=np.float32)
        directions = np.tile([0., 0., 1.], (len(origins), 1))
        targets = np.array([[.95]*3, [.1]*3, [.5]*3, [.2]*3, [.8]*3, [.1]*3], dtype=np.float32)
        alphas = np.array([1., 1., 1., 1., 0., 0.], dtype=np.float32)
        weights = np.array([1., 2., 1., 1., .5, 2.], dtype=np.float32)
        references = [oracle(np, spec, o.astype(np.float64), d, c.astype(np.float64), float(a), float(w), {})
                      for o, d, c, a, w in zip(origins, directions, targets, alphas, weights)]
        ray = mi.Ray3f(mi.Point3f(*(mi.Float(origins[:, i]) for i in range(3))),
                       mi.Vector3f(*(mi.Float(directions[:, i]) for i in range(3))))
        ray.maxt = 100.
        target = mi.Color3f(*(mi.Float(targets[:, i]) for i in range(3)))
        out = native_model(np, mi, spec).loss_and_grad(ray, target, mi.Float(alphas), mi.Color3f(1.), mi.Float(weights))
        dr.eval(out)
        dr.sync_thread()
        for key in ("centers", "log_scales", "rotation_tangent"):
            expected = sum(value[key] for value in references)
            np.testing.assert_allclose(np.asarray(out["gradients"][key]).reshape(-1, 3), expected, atol=1e-5, rtol=4e-4,
                                       err_msg=f"divergent_ray_batch: {key}")
        for key in ("geometry_valid", "background_carve_valid"):
            np.testing.assert_array_equal(np.asarray(out[key]), sum(value[key] for value in references))
        np.testing.assert_array_equal(np.asarray(out["hit_count"]), [value["hit_count"] for value in references])
        print("PASS divergent_ray_batch", flush=True)
        names.append("divergent_ray_batch")
    return dict(cases=names, maximum_geometry_absolute_error=max_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--variant", choices=("cuda_ad_rgb", "scalar_rgb"), default="cuda_ad_rgb")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from reconstruction_common import initialize_runtime
    from reconstruction_preview import RuntimeConfig
    output = Path(__file__).resolve().parents[2]/"outputs"/"reconstruction-tests"/"gradients"
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    mi, dr, np, handles = initialize_runtime(RuntimeConfig(build_dir=str(args.build_dir.resolve())), output, report)
    mi.set_variant(args.variant)
    report["runtime_validation_variant"] = report.pop("actual_variant")
    report.update(actual_variant=mi.variant(), variant=mi.variant(), status="running")
    print("Gradient test actual variant:", mi.variant(), flush=True)
    try:
        report.update(run_tests(np, mi, dr))
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        (output/f"{args.variant}.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print("PASS", len(report["cases"]), "independent geometry cases", flush=True)


if __name__ == "__main__":
    main()
