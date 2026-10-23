"""CUDA refinement versus the independent FP64 NumPy fitter and lifecycle."""

from __future__ import annotations

import copy
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preview import RuntimeConfig
from reconstruction_common import initialize_runtime, restore_model
from reconstruction_optimizer import FalcorOptimizer
from reconstruction_refinement_gpu import refine_state_gpu
from reconstruction_topology import CoarseToFineConfig, TopologyManager, TopologySettings, refine_state
from reconstruction_topology_gpu import device_checkpoint


def fixture(np, *, count=41, seed=217, sh=9, opacity=4):
    rng = np.random.default_rng(seed)
    data = {"model_metadata": {"resolution": 8, "grid_min": [-2., -4., -1.],
            "grid_max": [6., 4., 3.], "sh_count": sh, "opacity_sh_count": opacity},
            "model_state": {
                "cell_indices": rng.choice(8 ** 3, count, replace=False).astype(np.uint32),
                "centers": rng.uniform(.05, .95, count * 3).astype(np.float32),
                "log_scales": np.log(rng.uniform(.05, 1.8, count * 3)).astype(np.float32),
                "rotations": rng.normal(size=count * 4).astype(np.float32),
                "radiance_sh": rng.normal(size=count * sh * 3).astype(np.float32),
                "opacity_sh": rng.normal(size=count * opacity).astype(np.float32)}}
    if count >= 3:
        # Both the fully enclosed analytic branch and near-tangent child boxes.
        data["model_state"]["centers"][:9] = [.5] * 9
        data["model_state"]["log_scales"][:9] = np.log([2., 2., 2., .25, .25, .25, .5, .5, .5])
        data["model_state"]["rotations"][:12] = [1., 0., 0., 0.] * 3
    return data


def compare(reference, actual, dr, np):
    assert reference["model_metadata"] == actual["model_metadata"]
    for name, expected in reference["model_state"].items():
        assert dr.is_array_v(type(actual["model_state"][name])), name
        values = np.array(actual["model_state"][name])
        if name == "cell_indices":
            np.testing.assert_array_equal(expected, values, err_msg=name)
        else:
            np.testing.assert_allclose(expected, values, rtol=4e-6, atol=3e-7, err_msg=name)


def expect_failure(function, *args, match):
    try:
        function(*args)
    except RuntimeError as error:
        assert match in str(error), str(error)
    else:
        raise AssertionError(f"Expected {match}")


def geometry_cases(dr, np):
    schedule = CoarseToFineConfig()
    for seed in (217, 218, 219):
        data = fixture(np, seed=seed)
        for attenuation in (.01, .65, 1.):
            cfg = replace(schedule, opacity_optical_depth_scale=attenuation,
                          child_face_overlap=.2 if seed == 218 else .1)
            compare(refine_state(data, cfg), refine_state_gpu(data, cfg), dr, np)
    # Invalid quaternion has the same identity fallback as the CPU fitter.
    data = fixture(np)
    data["model_state"]["rotations"][12:16] = 0
    data["model_state"]["rotations"][16:20] = np.nan
    compare(refine_state(data, schedule), refine_state_gpu(data, schedule), dr, np)
    # Threshold uses the full opacity-SH bound, while transfer preserves DC only.
    cfg = replace(schedule, parent_opacity_threshold=.6)
    data = fixture(np)
    compare(refine_state(data, cfg), refine_state_gpu(data, cfg), dr, np)
    # Repeated levels preserve child-major ordering and shell/core metadata.
    data = fixture(np, count=7, sh=4, opacity=1)
    data["model_metadata"].update(core_resolution=4, scene_mapping="radial_seed_shell")
    reference, actual = data, data
    for _ in range(2):
        reference = refine_state(reference, schedule)
        actual = refine_state_gpu(actual, schedule)
        compare(reference, actual, dr, np)
    # Large positive/negative logits exercise the existing optical-depth limits.
    data = fixture(np, count=5, opacity=1)
    data["model_state"]["opacity_sh"][:] = [-1000., -50., 0., 50., 1000.]
    compare(refine_state(data, schedule), refine_state_gpu(data, schedule), dr, np)
    # Capacity failure rejects the whole replacement and leaves its input intact.
    before = copy.deepcopy(data)
    for function in (refine_state, refine_state_gpu):
        expect_failure(function, data, schedule, 1, match="exceeds max_active_voxels")
    for key in before["model_state"]:
        np.testing.assert_array_equal(before["model_state"][key], data["model_state"][key])
    data["model_state"]["opacity_sh"][:] = -1000.
    for function in (refine_state, refine_state_gpu):
        expect_failure(function, data, replace(schedule, parent_opacity_threshold=.1),
                       match="no representable children")
    empty = fixture(np, count=0)
    compare(refine_state(empty, schedule), refine_state_gpu(empty, schedule), dr, np)


def lifecycle_case(mi, dr, np, *, save):
    cfg = TopologySettings(use_rgb_growth=True, use_rgb_deletion=True,
                           enable_growth=False, enable_deletion=False)
    schedule = CoarseToFineConfig(start_resolution=8, target_resolution=16, total_epochs=4,
                                 level_weights=(1, 1), save_each_level=save)
    results = []
    for gpu in (False, True):
        model = restore_model(fixture(np, count=7, opacity=1), mi)
        manager = TopologyManager(model, 3, replace(cfg, gpu_topology=gpu), schedule)
        optimizer = FalcorOptimizer(model)
        state = optimizer.state_dict()
        for group in state["groups"].values():
            group["mean"][:] = .2
            group["squared_mean"][:] = .3
            group["steps"][:] = 5
        optimizer.load_state_dict(state)
        assert manager.finish_epoch(model, optimizer, 1)[0] is None
        if gpu and not save:
            with patch("reconstruction_topology_gpu.GPUTopologyLifecycle.materialize",
                       side_effect=AssertionError("unexpected lifecycle host copy")), patch(
                       "reconstruction_topology.model_checkpoint", side_effect=AssertionError("unexpected model host copy")):
                replacement, event = manager.finish_epoch(model, optimizer, 2)
        else:
            replacement, event = manager.finish_epoch(model, optimizer, 2)
        assert event["refined"] and not event["finished"]
        if gpu:
            assert event["refinement_backend"] == "cuda"
        before = manager.before_refine_checkpoint
        assert (before is not None) == save
        if save:
            assert before["model_metadata"]["resolution"] == 8
            assert before["topology_state"]["resolution"] == 8
            for group in before["optimizer_state"]["groups"].values():
                np.testing.assert_array_equal(group["steps"], 5)
        model = restore_model(replacement, mi)
        manager.prepare_rgb_growth(model, 2)
        assert manager.finish_epoch(model, optimizer, 3)[0] is None
        assert manager.finish_epoch(model, optimizer, 4)[1]["finished"]
        for group in optimizer.state_dict()["groups"].values():
            for field in group.values():
                assert not np.any(field)
        lifecycle = manager.state_dict()
        lifecycle["settings"].pop("gpu_topology")
        results.append((device_checkpoint(model), lifecycle))
        optimizer.clear_replay()
    compare({"model_metadata": results[0][0]["model_metadata"], "model_state": {
        k: np.array(v) for k, v in results[0][0]["model_state"].items()}}, results[1][0], dr, np)
    for key, value in results[0][1].items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, results[1][1][key], err_msg=key)
        else:
            assert value == results[1][1][key], key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true", help="Time an 8192-parent synthetic refinement")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="mitsuba-gpu-refinement-") as temporary:
        mi, dr, np, handles = initialize_runtime(RuntimeConfig(), Path(temporary), {})
        geometry_cases(dr, np)
        lifecycle_case(mi, dr, np, save=False)
        lifecycle_case(mi, dr, np, save=True)
        print(json.dumps({"fp64_fitter_geometry_opacity_and_order": "passed", "repeated_levels": "passed",
                          "capacity_empty_and_no_children": "passed", "device_lifecycle_and_adam_reset": "passed",
                          "optional_old_level_checkpoint": "passed"}))
        if args.benchmark:
            data = fixture(np, count=256, opacity=1)
            data["model_metadata"].update(resolution=32, grid_max=[30., 28., 15.])
            data["model_state"] = {name: np.tile(value, 32) for name, value in data["model_state"].items()}
            data["model_state"]["cell_indices"] = np.arange(8192, dtype=np.uint32)
            schedule = CoarseToFineConfig()
            start = time.perf_counter()
            reference = refine_state(data, schedule)
            cpu_seconds = time.perf_counter() - start
            start = time.perf_counter()
            actual = refine_state_gpu(data, schedule)
            dr.sync_thread()
            gpu_seconds = time.perf_counter() - start
            compare(reference, actual, dr, np)
            print(json.dumps({"benchmark": "synthetic_refinement_only", "parents": 8192,
                              "children": len(actual["model_state"]["cell_indices"]),
                              "cpu_seconds": cpu_seconds, "gpu_seconds": gpu_seconds,
                              "note": "includes input upload; excludes native model/octree rebuild"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
