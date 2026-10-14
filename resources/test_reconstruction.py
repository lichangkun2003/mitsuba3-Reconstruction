"""Validate the source-built voxel renderer against an independent float64 oracle.

The oracle enumerates occupied cells and solves ray/ellipsoid quadratics; it
does not implement the renderer's DDA. CUDA is the default and never falls back
to CPU. --variant scalar_rgb runs a separate auxiliary CPU check.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import traceback


def write_cloud(path, points, colors=None, *, binary=False):
    """Write renderer-space points in the input PLY's NeRF coordinate frame."""
    header = ["ply", "format binary_little_endian 1.0" if binary else "format ascii 1.0",
              f"element vertex {len(points)}", "property float x", "property float y",
              "property float z"]
    if colors is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header += ["end_header"]
    with path.open("wb") as stream:
        stream.write(("\n".join(header) + "\n").encode("ascii"))
        for index, (x, y, z) in enumerate(points):
            values = [float(x), -float(z), float(y)]
            if colors is not None:
                values += [int(value) for value in colors[index]]
            if binary:
                stream.write(struct.pack("<fff" + ("BBB" if colors is not None else ""), *values))
            else:
                stream.write((" ".join(str(value) for value in values) + "\n").encode("ascii"))


def rotation_matrix(np, quaternion):
    w, x, y, z = quaternion
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def sh_basis(np, direction, count):
    x, y, z = direction
    basis = np.array([
        1/(2*math.sqrt(math.pi)),
        y*math.sqrt(3)/(2*math.sqrt(math.pi)),
        z*math.sqrt(3)/(2*math.sqrt(math.pi)),
        x*math.sqrt(3)/(2*math.sqrt(math.pi)),
        x*y*math.sqrt(15)/(2*math.sqrt(math.pi)),
        y*z*math.sqrt(15)/(2*math.sqrt(math.pi)),
        (3*z*z - 1)*math.sqrt(5)/(4*math.sqrt(math.pi)),
        x*z*math.sqrt(15)/(2*math.sqrt(math.pi)),
        (x*x - y*y)*math.sqrt(15)/(4*math.sqrt(math.pi)),
    ], dtype=np.float64)
    if count not in (1, 4, 9):
        raise AssertionError(f"Test oracle supports SH counts 1, 4, 9; got {count}")
    return basis[:count]


def unpack_snapshot(np, model, resolution, grid_min, grid_max):
    raw = model.snapshot()
    indices = np.asarray(raw["cell_indices"], dtype=np.uint32).reshape(-1)
    count = len(indices)
    if count == 0 or len(np.unique(indices)) != count:
        raise AssertionError("Snapshot must contain unique occupied cells")
    state = {"indices": indices, "resolution": resolution,
             "grid_min": np.asarray(grid_min, dtype=np.float64),
             "grid_max": np.asarray(grid_max, dtype=np.float64)}
    for name, width in (("centers", 3), ("log_scales", 3), ("rotations", 4)):
        state[name] = np.asarray(raw[name], dtype=np.float64).reshape(count, width)
        if not np.isfinite(state[name]).all():
            raise AssertionError(f"Non-finite snapshot field {name}")
    state["radiance_sh"] = np.asarray(raw["radiance_sh"], dtype=np.float64).reshape(count, -1, 3)
    state["opacity_sh"] = np.asarray(raw["opacity_sh"], dtype=np.float64).reshape(count, -1)
    state["voxel_size"] = (state["grid_max"] - state["grid_min"]) / resolution
    state["cells"] = np.stack((indices % resolution, (indices // resolution) % resolution,
                               indices // (resolution*resolution)), axis=1).astype(np.float64)
    state["world_centers"] = state["grid_min"] + (state["cells"] + state["centers"])*state["voxel_size"]
    state["scales"] = np.exp(state["log_scales"])
    state["matrices"] = [rotation_matrix(np, q) for q in state["rotations"]]
    if not np.all((state["centers"] >= 0.00999) & (state["centers"] <= 0.99001)):
        raise AssertionError("Initialization center bounds were violated")
    np.testing.assert_allclose(np.linalg.norm(state["rotations"], axis=1), 1, atol=2e-6)
    return state


def oracle(np, state, origins, directions, maxt, background, threshold, max_hits):
    """Independent brute-force intersection, sorted by occupied-cell entry."""
    colors, alphas, counts = [], [], []
    for origin, direction, ray_max in zip(origins, directions, maxt):
        intersections = []
        for index, cell in enumerate(state["cells"]):
            lower = state["grid_min"] + cell*state["voxel_size"]
            upper = lower + state["voxel_size"]
            enter, leave = 0.0, float(ray_max)
            for axis in range(3):
                if direction[axis] == 0:
                    # Match ownership of an axis-parallel ray on a grid plane.
                    if not lower[axis] <= origin[axis] < upper[axis]:
                        enter, leave = 1.0, 0.0
                        break
                else:
                    a = (lower[axis] - origin[axis])/direction[axis]
                    b = (upper[axis] - origin[axis])/direction[axis]
                    enter, leave = max(enter, min(a, b)), min(leave, max(a, b))
            if leave <= enter:
                continue
            transform = state["matrices"][index].T
            relative = transform @ (origin - state["world_centers"][index]) / state["scales"][index]
            travel = transform @ direction / state["scales"][index]
            a = float(travel @ travel)
            b = 2*float(relative @ travel)
            c = float(relative @ relative) - 1
            discriminant = b*b - 4*a*c
            if discriminant < 0:
                continue
            root = math.sqrt(max(0, discriminant))
            near, far = (-b-root)/(2*a), (-b+root)/(2*a)
            if max(near, enter) <= min(far, leave):
                intersections.append((enter, index))
        intersections.sort()
        transmittance, rgb, hits = 1.0, np.zeros(3, dtype=np.float64), 0
        view = -direction / np.linalg.norm(direction)
        radiance_basis = sh_basis(np, view, state["radiance_sh"].shape[1])
        opacity_basis = sh_basis(np, view, state["opacity_sh"].shape[1])
        for _, index in intersections[:max_hits]:
            radiance = np.maximum(radiance_basis @ state["radiance_sh"][index], 0)
            logit = float(opacity_basis @ state["opacity_sh"][index])
            opacity = 1/(1+math.exp(-logit))
            rgb += transmittance*opacity*radiance
            transmittance *= 1-opacity
            hits += 1
            if transmittance <= threshold:
                break
        colors.append(rgb + transmittance*background)
        alphas.append(1-transmittance)
        counts.append(hits)
    return np.asarray(colors), np.asarray(alphas), np.asarray(counts)


def native_trace(np, mi, dr, model, origins, directions, maxt, background, threshold, max_hits):
    if mi.variant() == "scalar_rgb":
        values = []
        for origin, direction, end in zip(origins, directions, maxt):
            ray = mi.Ray3f(mi.Point3f(*(float(v) for v in origin)),
                          mi.Vector3f(*(float(v) for v in direction)))
            ray.maxt = float(end)
            rgb, alpha, count = model.trace(ray, mi.Color3f(*(float(v) for v in background)), threshold, max_hits)
            values.append((np.asarray(rgb, dtype=np.float64), float(alpha), int(count)))
        return (np.stack([v[0] for v in values]), np.asarray([v[1] for v in values]),
                np.asarray([v[2] for v in values]))
    ray = mi.Ray3f(mi.Point3f(*(mi.Float(origins[:, i]) for i in range(3))),
                  mi.Vector3f(*(mi.Float(directions[:, i]) for i in range(3))))
    ray.maxt = mi.Float(maxt)
    rgb, alpha, count = model.trace(ray, mi.Color3f(*(float(v) for v in background)), threshold, max_hits)
    dr.eval(rgb, alpha, count)
    dr.sync_thread()
    return (np.asarray(rgb, dtype=np.float64).reshape(3, -1).T,
            np.asarray(alpha, dtype=np.float64).reshape(-1), np.asarray(count).reshape(-1))


def run_tests(np, mi, dr, output):
    grid_min, grid_max = (-1., -1., -1.), (1., 1., 1.)
    cases = []

    def model_from(path, resolution):
        model = mi.VoxelReconstruction(str(path), resolution, grid_min, grid_max, .05, 9, 9)
        return model, unpack_snapshot(np, model, resolution, grid_min, grid_max)

    def compare(name, model, state, origins, directions, *, maxt=None,
                background=(.7, .8, .9), threshold=.01, max_hits=16):
        # Native float32 inputs and float64 oracle must receive identical rays.
        origins = np.asarray(origins, dtype=np.float32).astype(np.float64)
        directions = np.asarray(directions, dtype=np.float32).astype(np.float64)
        maxt = np.full(len(origins), np.inf) if maxt is None else np.asarray(maxt, dtype=np.float32)
        background = np.asarray(background, dtype=np.float32).astype(np.float64)
        expected = oracle(np, state, origins, directions, maxt, background, threshold, max_hits)
        actual = native_trace(np, mi, dr, model, origins, directions, maxt, background, threshold, max_hits)
        for values in actual:
            if not np.isfinite(values).all():
                raise AssertionError(f"{name}: non-finite renderer result")
        np.testing.assert_array_equal(actual[2], expected[2], err_msg=f"{name}: hit counts")
        np.testing.assert_allclose(actual[0], expected[0], rtol=2e-5, atol=3e-6, err_msg=f"{name}: RGB")
        np.testing.assert_allclose(actual[1], expected[1], rtol=2e-5, atol=3e-6, err_msg=f"{name}: alpha")
        cases.append({"name": name, "rays": len(origins),
                      "rgb_max_abs_error": float(np.max(np.abs(actual[0]-expected[0]))),
                      "alpha_max_abs_error": float(np.max(np.abs(actual[1]-expected[1]))),
                      "max_hits_observed": int(np.max(actual[2]))})
        print(f"PASS {name}: {len(origins)} rays", flush=True)
        return actual

    single = output / "single_rgb.ply"
    write_cloud(single, [[.25, .25, .25]], [[128, 64, 255]])
    model, state = model_from(single, 4)
    encoded = np.asarray([128, 64, 255])/255
    linear = np.where(encoded <= .04045, encoded/12.92, ((encoded+.055)/1.055)**2.4)
    np.testing.assert_array_equal(state["indices"], [42])
    np.testing.assert_allclose(state["centers"], .5, atol=1e-7)
    np.testing.assert_allclose(state["scales"], .3, atol=1e-7)
    np.testing.assert_allclose(state["radiance_sh"][0, 0]*sh_basis(np, [0, 0, 1], 1)[0], linear, atol=2e-7)
    np.testing.assert_array_equal(state["radiance_sh"][:, 1:], 0)
    np.testing.assert_array_equal(state["opacity_sh"], 0)
    origins = [[-2,.25,.25], [2,.25,.25], [.25,-2,.25], [.25,.25,-2],
               [.25,.25,.25], [-1,.25,.25], [1,.25,.25], [-2,.49,.49],
               [-2,.75,.25], [-2,.25,.25], [-2,.25,.25], [.5,.25,.25],
               [0,.25,.25], [1.1,.25,.25], [-2,0,.25], [-2,1,.25], [-2,.52,.25]]
    directions = [[1,0,0],[-1,0,0],[0,1,0],[0,0,1],[1,0,0],[1,0,0],[-1,0,0],
                  [1,0,0],[1,0,0],[1,0,0],[1,0,0],[-1,0,0],[1,0,0],[1,0,0],
                  [1,0,0],[1,0,0],[1,0,0]]
    actual = compare("single_analytic_axes_boundaries_inside_nearmiss", model, state, origins, directions,
                     maxt=[np.inf]*9 + [1., 2.3] + [np.inf]*6)
    np.testing.assert_allclose(actual[1][:7], .5, atol=1e-7)
    np.testing.assert_array_equal(actual[2][7:10], 0)
    # This ray intersects the full ellipsoid but lies outside its occupied cell.
    np.testing.assert_array_equal(actual[2][-1], 0)

    binary = output / "single_rgb_binary.ply"
    write_cloud(binary, [[.25,.25,.25]], [[128,64,255]], binary=True)
    binary_model, binary_state = model_from(binary, 4)
    for name in ("indices", "centers", "log_scales", "rotations", "radiance_sh", "opacity_sh"):
        np.testing.assert_allclose(binary_state[name], state[name], atol=2e-7)
    compare("binary_ply_equivalence", binary_model, binary_state, origins[:2], directions[:2])

    edge = output / "outer_grid_faces.ply"
    write_cloud(edge, [[.25,.75,.25], [.25,-.75,.25]], [[255,128,32], [255,128,32]])
    edge_model, edge_state = model_from(edge, 4)
    edge_result = compare("parallel_outer_grid_face_ownership", edge_model, edge_state,
                          [[-2,1,.25], [-2,-1,.25]], [[1,0,0], [1,0,0]])
    np.testing.assert_array_equal(edge_result[2], [0, 1])

    # At an exact XY corner Falcor crosses X first, visits the zero-length
    # intermediate cell, then crosses Y. A plain closed-box enumeration would
    # also visit the other side cell, so this needs a separate analytic oracle.
    tie_path = output / "xy_tie_priority.ply"
    write_cloud(tie_path, [[-.01,-.01,.25], [.01,-.01,.25], [-.01,.01,.25], [.01,.01,.25]],
                [[255,0,0], [0,255,0], [0,0,255], [255,255,255]])
    tie_model, _ = model_from(tie_path, 4)
    diagonal = float(np.float32(1/math.sqrt(2)))
    tie_result = native_trace(np, mi, dr, tie_model,
                             np.asarray([[-.4,-.4,.25], [.4,.4,.25]], dtype=np.float32),
                             np.asarray([[diagonal,diagonal,0], [-diagonal,-diagonal,0]], dtype=np.float32),
                             np.full(2, np.inf), np.zeros(3), 0, 16)
    tie_rgb = np.asarray([[.625,.375,.125], [.625,.5,.75]])
    np.testing.assert_array_equal(tie_result[2], [3,3])
    np.testing.assert_allclose(tie_result[0], tie_rgb, atol=3e-6)
    np.testing.assert_allclose(tie_result[1], .875, atol=3e-6)
    cases.append({"name": "xy_tie_falcor_x_before_y", "rays": 2,
                  "rgb_max_abs_error": float(np.max(np.abs(tie_result[0]-tie_rgb))),
                  "alpha_max_abs_error": float(np.max(np.abs(tie_result[1]-.875))),
                  "max_hits_observed": int(np.max(tie_result[2]))})
    print("PASS xy_tie_falcor_x_before_y: 2 rays", flush=True)

    # Invalid rays are inactive lanes, not a reason to contaminate other rays.
    invalid_origins = np.asarray([[np.nan,.25,.25], [np.inf,.25,.25],
                                  [.25,.25,.25], [.25,.25,.25], [.25,.25,.25],
                                  [.25,.25,.25], [.25,.25,.25], [.25,.25,.25]])
    invalid_directions = np.asarray([[1,0,0], [1,0,0], [0,0,0], [np.nan,0,1],
                                     [np.inf,0,1], [1,0,0], [1,0,0], [1,0,0]])
    invalid_end = np.asarray([np.inf]*5 + [0, -1, np.nan])
    invalid_background = np.asarray([.2,.4,.6])
    invalid_result = native_trace(np, mi, dr, model, invalid_origins, invalid_directions,
                                 invalid_end, invalid_background, .01, 16)
    np.testing.assert_allclose(invalid_result[0], np.tile(invalid_background, (8,1)), atol=1e-7)
    np.testing.assert_array_equal(invalid_result[1], 0)
    np.testing.assert_array_equal(invalid_result[2], 0)
    if not all(np.isfinite(values).all() for values in invalid_result):
        raise AssertionError("Inactive invalid rays must return finite background only")
    cases.append({"name": "invalid_rays_return_background", "rays": 8,
                  "rgb_max_abs_error": float(np.max(np.abs(invalid_result[0]-invalid_background))),
                  "alpha_max_abs_error": 0., "max_hits_observed": 0})
    print("PASS invalid_rays_return_background: 8 rays", flush=True)

    pca = output / "pca_rgb.ply"
    points = np.array([[.07,.10,.12], [.18,.17,.18], [.27,.28,.26], [.41,.38,.34],
                       [.20,.29,.14], [.29,.21,.37]], dtype=np.float32)
    write_cloud(pca, points, [[20,100,220]]*len(points))
    pca_model, pca_state = model_from(pca, 4)
    if len(pca_state["indices"]) != 1:
        raise AssertionError("PCA fixture should occupy exactly one cell")
    local = (points.astype(np.float64) - pca_state["world_centers"][0]) @ pca_state["matrices"][0]
    if np.max(np.sum((local/pca_state["scales"][0])**2, axis=1)) > 1.00001:
        raise AssertionError("PCA initialized ellipsoid does not enclose its source points")
    rng = np.random.default_rng(20261014)
    random_origins = rng.uniform(-1.3, 1.3, (192, 3))
    targets = rng.uniform(-.1, .6, (192, 3))
    random_directions = targets-random_origins
    random_directions /= np.linalg.norm(random_directions, axis=1)[:, None]
    compare("pca_random_float64_reference", pca_model, pca_state, random_origins, random_directions)

    chain = output / "colored_chain.ply"
    chain_points = [[-1+(i+.5)/16, 1/32, 1/32] for i in range(32)]
    chain_colors = [[8*i, 255-8*i, 64+(i % 8)*20] for i in range(32)]
    write_cloud(chain, chain_points, chain_colors)
    chain_model, chain_state = model_from(chain, 32)
    chain_origins, chain_directions = [[-2,1/32,1/32],[2,1/32,1/32]], [[1,0,0],[-1,0,0]]
    capped = compare("colored_chain_16_hit_limit", chain_model, chain_state, chain_origins,
                     chain_directions, threshold=0, max_hits=16)
    np.testing.assert_array_equal(capped[2], 16)
    early = compare("colored_chain_transmittance_stop", chain_model, chain_state, chain_origins,
                    chain_directions, threshold=.2, max_hits=16)
    np.testing.assert_array_equal(early[2], 3)
    compare("colored_chain_1_hit_limit", chain_model, chain_state, chain_origins,
            chain_directions, threshold=0, max_hits=1)

    gaussian = output / "gaussian_coverage.ply"
    gaussian_header = ["ply", "format ascii 1.0", "element vertex 2"]
    gaussian_header += [f"property float {name}" for name in
                        ("x","y","z","opacity","scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3")]
    # The threshold sphere touches eight cells while missing every cell center.
    scale = math.log(.06)
    row = f".25 -.25 .25 0 {scale} {scale} {scale} 1 0 0 0\n"
    row += f"-.75 .75 -.75 -10 {scale} {scale} {scale} 1 0 0 0\n"
    gaussian.write_text("\n".join(gaussian_header+["end_header"])+"\n"+row, encoding="ascii")
    gaussian_model, gaussian_state = model_from(gaussian, 8)
    radius_squared = .06**2 * 2*math.log(.5/.05)
    expected_cells = []
    for index in range(8**3):
        cell = np.array([index % 8, (index//8) % 8, index//64])
        lower, upper = -1+cell*.25, -1+(cell+1)*.25
        distance = np.maximum(np.maximum(lower-.25, .25-upper), 0)
        if distance @ distance <= radius_squared:
            expected_cells.append(index)
    np.testing.assert_array_equal(gaussian_state["indices"], expected_cells)
    np.testing.assert_allclose(gaussian_state["centers"], .5)
    np.testing.assert_allclose(gaussian_state["scales"], .15, atol=2e-7)
    np.testing.assert_array_equal(gaussian_state["radiance_sh"], 0)
    np.testing.assert_array_equal(gaussian_state["opacity_sh"], 0)
    compare("gaussian_coverage_black_initialization", gaussian_model, gaussian_state,
            origins[:7], directions[:7])
    return {"cases": cases, "rays_checked": sum(case["rays"] for case in cases),
            "statistics": {"single": model.statistics(), "pca": pca_model.statistics(),
                           "chain": chain_model.statistics(), "gaussian": gaussian_model.statistics()},
            "limits": ["Initial SH coefficients only; nonzero higher-order SH not exercised.",
                       "No optimization, gradient, topology, or full-image convergence validation."]}


def main():
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True,
                        help="Source runtime directory (build/Release on Windows)")
    parser.add_argument("--variant", choices=("cuda_ad_rgb", "scalar_rgb"), default="cuda_ad_rgb")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", type=Path,
                        default=repository.parent / "outputs" / "reconstruction-tests")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "python_executable": sys.executable, "requested_variant": args.variant}
    report_path = output / f"native-{args.variant}.json"
    handles = []
    try:
        runtime = args.build_dir.resolve()
        python_directory = runtime / "python"
        for package in ("mitsuba", "drjit"):
            if not (python_directory/package/"__init__.py").is_file():
                raise RuntimeError(f"Incomplete source runtime: {python_directory/package}")
        sys.path.insert(0, str(python_directory))
        if os.name == "nt":
            for directory in (runtime, python_directory/"drjit", python_directory/"mitsuba"):
                handles.append(os.add_dll_directory(str(directory)))
        for variable, dirname in (("DRJIT_CACHE_DIR", "drjit"), ("CUDA_CACHE_PATH", "cuda"),
                                  ("OPTIX_CACHE_PATH", "optix")):
            cache = output/"cache"/dirname
            cache.mkdir(parents=True, exist_ok=True)
            os.environ[variable] = str(cache)
        if args.variant == "cuda_ad_rgb":
            query = subprocess.run(["nvidia-smi", f"--id={args.gpu}",
                                    "--query-gpu=index,name,uuid,driver_version",
                                    "--format=csv,noheader,nounits"], check=True, capture_output=True,
                                   text=True, timeout=30)
            rows = list(csv.reader(query.stdout.strip().splitlines(), skipinitialspace=True))
            if len(rows) != 1 or len(rows[0]) != 4:
                raise RuntimeError(f"Unexpected nvidia-smi result: {query.stdout!r}")
            report["gpu"] = dict(zip(("index","name","uuid","driver"), rows[0]))
            os.environ["CUDA_VISIBLE_DEVICES"] = report["gpu"]["uuid"]
        import numpy as np
        import drjit as dr
        import mitsuba as mi
        for module in (mi, dr):
            if not Path(module.__file__).resolve().is_relative_to(python_directory):
                raise RuntimeError(f"Expected source runtime; imported {module.__file__}")
        mi.set_variant(args.variant)
        report.update({"mitsuba_version": mi.__version__, "drjit_version": dr.__version__,
                       "mitsuba_file": mi.__file__, "drjit_file": dr.__file__, "variant": mi.variant()})
        if args.variant == "cuda_ad_rgb" and dr.backend_v(mi.Float) != dr.JitBackend.CUDA:
            raise RuntimeError("The actual array backend is not CUDA")
        if not hasattr(mi, "VoxelReconstruction"):
            raise RuntimeError("Source runtime lacks VoxelReconstruction; rebuild the native module")
        print(json.dumps(report, indent=2), flush=True)
        report.update(run_tests(np, mi, dr, output))
        report["status"] = "passed"
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
        print("Report:", report_path, flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
