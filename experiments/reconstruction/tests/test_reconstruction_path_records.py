"""Compare compact RGB path records against the original native record layout.

Both executions use the same immutable model, rays, image loss and topology
evidence settings. Integer counters and forward images must match exactly;
floating reductions allow the existing GPU atomic accumulation tolerance.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preview import RuntimeConfig
from reconstruction_common import initialize_runtime, restore_model
from reconstruction_replay import NativeLossReplay
from test_reconstruction_distortion import fixture as core_fixture
from test_reconstruction_projective_native import fixture as shell_fixture, make_ray


def expanded_fixture(np, *, projective):
    checkpoint = shell_fixture(np) if projective else core_fixture(np)
    count = len(checkpoint["model_state"]["cell_indices"])
    old = checkpoint["model_state"]["radiance_sh"].reshape(count, 3)
    coefficients = np.random.default_rng(850).normal(0, .07, (count, 9, 3))
    coefficients[:, 0, :] = old
    checkpoint["model_state"]["radiance_sh"] = coefficients.ravel()
    checkpoint["model_metadata"]["sh_count"] = 9
    # Include clamped dark radiance and nearly opaque/transparent hits.
    coefficients[0, 0] = -.2
    checkpoint["model_state"]["opacity_sh"][0] = 60.
    checkpoint["model_state"]["opacity_sh"][-1] = -60.
    return checkpoint


def inputs(mi, np, *, projective, width=17, height=13):
    rng = np.random.default_rng(956)
    count = width * height
    if projective:
        origins = rng.uniform(-2.5, 2.5, (count, 3))
        directions = rng.normal(size=(count, 3))
        origins[:4] = [[0, 0, 0], [3, .15, .23], [1, 0, 0], [-3, -.3, .2]]
        directions[:4] = [[1, 1, 0], [-1, .02, .04], [0, 1, 0], [1, .1, -.1]]
    else:
        x, y = np.meshgrid(np.linspace(1.01, 1.99, width), np.linspace(1.01, 1.99, height))
        origins = np.column_stack((x.ravel(), y.ravel(), np.full(count, -1.)))
        directions = rng.normal(0., .012, (count, 3)) + [0, 0, 1]
    reference = rng.uniform(.05, .95, (count, 3)).astype(np.float32)
    ray = make_ray(mi, np, origins, directions)
    target = mi.Color3f(*(mi.Float(reference[:, c].copy()) for c in range(3)))
    return ray, target


def run_native(model, ray, target, mi, dr, options, *, layout, replay=None):
    settings = dict(rgb_only=1., max_hits=16., max_candidates=16.,
                    transmittance_threshold=.00001, distortion_loss_weight=.2,
                    dssim_loss_weight=.2, image_width=17., image_height=13.)
    settings.update(options)
    settings["compact_path_records"] = float(layout)
    args = (ray, target, mi.Float(1), mi.Color3f(.9, .8, 1.), mi.Float(1), settings)
    result = model.loss_and_grad(*args) if replay is None else replay(model, *args)
    dr.eval(result)
    dr.sync_thread()
    return result


def host_result(result, np):
    return {name: ({key: np.array(value) for key, value in item.items()}
                   if name == "gradients" else np.array(item))
            for name, item in result.items() if name != "stage_profile"}


def compare(expected, actual, np):
    assert set(expected) == set(actual), (set(expected), set(actual))
    for name in expected:
        pairs = expected[name].items() if name == "gradients" else [(name, expected[name])]
        for key, value in pairs:
            other = actual[name][key] if name == "gradients" else actual[name]
            label = f"{name}.{key}" if name == "gradients" else name
            if value.dtype.kind in "bui" or name in ("rgb", "raw_rgb", "alpha", "raw_alpha", "hit_count"):
                np.testing.assert_array_equal(value, other, err_msg=label)
            else:
                np.testing.assert_allclose(value, other, atol=3e-5, rtol=3e-4,
                                           equal_nan=True, err_msg=label)


def run_checks(mi, dr, np):
    checks = 0
    for projective in (False, True):
        payload = expanded_fixture(np, projective=projective)
        model = restore_model(payload, mi)
        if not projective:
            proposal = copy.deepcopy(core_fixture(np, 1)["model_state"])
            proposal["cell_indices"][:] = 53
            proposal["radiance_sh"] = payload["model_state"]["radiance_sh"][:27].copy()
            model.set_growth_candidates({name: (mi.UInt32(value) if name == "cell_indices" else mi.Float(value))
                                         for name, value in proposal.items()})
        ray, target = inputs(mi, np, projective=projective)
        scenarios = [
            {},
            {"compact_gradient_capacity": float(model.active_voxel_count)},
            {"max_hits": 1.},
            {"max_hits": 3., "transmittance_threshold": .6},
            {"max_candidates": 0.},
            {"use_octree": 0., "max_candidates": 3.},
            {"dssim_loss_weight": 0., "distortion_loss_weight": 0.},
            {"collect_rgb_deletion_evidence": 1., "collect_growth_evidence": 1.},
        ]
        for active_sh in (1, 9):
            model.set_active_sh_count(active_sh)
            for options in scenarios:
                old = host_result(run_native(model, ray, target, mi, dr, options, layout=0), np)
                new = host_result(run_native(model, ray, target, mi, dr, options, layout=1), np)
                compare(old, new, np)
                checks += 1
        # Frozen calls must consume updated parameters, including recomputed SH.
        replay = NativeLossReplay(dr)
        for scale in (1., .93):
            model.set_parameters({"radiance_sh": mi.Float(payload["model_state"]["radiance_sh"] * scale),
                                  "opacity_sh": mi.Float(payload["model_state"]["opacity_sh"] * scale)})
            for _ in range(2):
                old = host_result(run_native(model, ray, target, mi, dr, {}, layout=0), np)
                new = host_result(run_native(model, ray, target, mi, dr, {}, layout=1, replay=replay), np)
                compare(old, new, np)
                checks += 1
        assert replay.statistics()["replays"] >= 3
    return {"core_projective_SH1_SH9_record_layout_differentials": checks,
            "forward_and_integer_counters": "exact", "status": "passed"}


def memory_sample(mi, dr, np, layout, *, frozen=False):
    width, height = 1559, 1039
    model = restore_model(expanded_fixture(np, projective=False), mi)
    ray, target = inputs(mi, np, projective=False, width=width, height=height)
    dr.eval(ray, target)
    dr.sync_thread()
    dr.flush_malloc_cache()
    replay = NativeLossReplay(dr) if frozen else None
    timings, watermarks = [], []
    for _ in range(4 if frozen else 1):
        start = time.perf_counter()
        result = run_native(model, ray, target, mi, dr,
                            {"image_width": float(width), "image_height": float(height)},
                            layout=layout, replay=replay)
        timings.append(time.perf_counter() - start)
        assert np.isfinite(np.asarray(result["loss"])).all()
        watermarks.append(dr.detail.malloc_watermark(dr.JitBackend.CUDA))
        # Match the training lifetime: consume/release a view's outputs before
        # executing the next view, while keeping its frozen recording alive.
        del result
    return {"layout": layout, "image_size": [width, height], "seconds": timings,
            "allocator_watermark_bytes": dr.detail.malloc_watermark(dr.JitBackend.CUDA),
            "round_watermarks": watermarks,
            "replay": None if replay is None else replay.statistics(),
            "scope": "fresh-process allocator watermark, not dedicated VRAM"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--memory-layout", type=int, choices=(0, 1))
    parser.add_argument("--memory-frozen", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="mitsuba-path-records-") as temporary:
        mi, dr, np, handles = initialize_runtime(RuntimeConfig(args.build_dir, args.gpu), Path(temporary), {})
        assert int(getattr(mi.VoxelReconstruction, "path_record_layout_version", 0)) >= 1, "Rebuild this checkout"
        result = (run_checks(mi, dr, np) if args.memory_layout is None else
                  memory_sample(mi, dr, np, args.memory_layout, frozen=args.memory_frozen))
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
