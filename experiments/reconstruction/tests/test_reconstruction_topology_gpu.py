"""Small CUDA/NumPy lifecycle parity: vote, revalidate, delete, grow and resume."""

from __future__ import annotations

import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preview import RuntimeConfig
from reconstruction_common import initialize_runtime, restore_model
from reconstruction_optimizer import FalcorOptimizer
from reconstruction_topology import CoarseToFineConfig, TopologyManager, TopologySettings, model_checkpoint


def fixture(np):
    count, sh = 4, 4
    return {"model_metadata": {"resolution": 8, "grid_min": [0.] * 3, "grid_max": [8.] * 3,
            "sh_count": sh, "opacity_sh_count": 1}, "model_state": {
        "cell_indices": np.array([73, 75, 77, 90], np.uint32),
        "centers": np.tile([.5, .5, .5], count).astype(np.float32),
        "log_scales": np.tile(np.log([.8, .7, .6]), count).astype(np.float32),
        "rotations": np.tile([1., 0., 0., 0.], count).astype(np.float32),
        "radiance_sh": np.arange(count * sh * 3, dtype=np.float32) / 100,
        "opacity_sh": np.array([0., 0., 0., -30.], np.float32)}}


def configuration(gpu):
    return TopologySettings(gpu_topology=gpu, use_rgb_growth=True, use_rgb_deletion=True,
        use_low_opacity_deletion=True, growth_warmup_epochs=0, growth_fine_warmup_epochs=0,
        growth_rgb_normalization="observed", growth_interval=1, growth_wait_epochs=1,
        growth_inherit_opacity=True, growth_initial_opacity=.3, growth_opacity_scale=.8,
        growth_min_supporting_parents=2, growth_rgb_min_support_views=2,
        deletion_interval=1, deletion_warmup_epochs=0, deletion_protection_epochs=2,
        deletion_rgb_min_support_views=2, deletion_rgb_confirmation_windows=1,
        deletion_opacity_confirmation_windows=2, deletion_opacity_threshold=.01,
        deletion_cooldown_epochs=3, max_active_voxels=4)


def arrays_equal(first, second, np, *, path=""):
    if isinstance(first, dict):
        # Backend/transfer reporting does not change lifecycle semantics.
        reporting = {"topology_backend", "topology_transfer"}
        assert set(first) - reporting == set(second) - reporting, path
        for name in set(first) - reporting:
            arrays_equal(first[name], second[name], np, path=f"{path}/{name}")
    elif isinstance(first, np.ndarray):
        if np.issubdtype(first.dtype, np.floating):
            np.testing.assert_allclose(first, second, rtol=3e-6, atol=2e-7, err_msg=path)
        else:
            np.testing.assert_array_equal(first, second, err_msg=path)
    elif isinstance(first, float):
        assert math.isclose(first, second, rel_tol=3e-6, abs_tol=2e-7), (path, first, second)
    else:
        assert first == second, (path, first, second)


def simulate(mi, dr, np, *, gpu, host_staged=False, staging_options=None):
    model = restore_model(fixture(np), mi)
    cfg = configuration(gpu)
    schedule = CoarseToFineConfig(enabled=False, total_epochs=4)
    manager = TopologyManager(model, 3, cfg, schedule)
    optimizer = FalcorOptimizer(model)
    state = optimizer.state_dict()
    for group in state["groups"].values():
        group["mean"][:] = np.arange(len(group["mean"]), dtype=np.float32) / 100
        group["squared_mean"][:] = .125
        group["steps"][:] = [1, 2, 3, 4]
    optimizer.load_state_dict(state)
    events = []
    for epoch in range(4):
        manager.prepare_rgb_growth(model, epoch)
        proposals = manager.pending_growth
        cells = np.asarray(proposals["cells"]) if proposals is not None else np.empty(0, np.uint32)
        n = len(cells)
        for view in range(3):
            growth_gain = np.where((cells == 74) | (cells == 76), .25, -.5).astype(np.float32)
            # A view with unknown hits still votes using reliable hits.
            out = {"growth_gain": mi.Float(growth_gain),
                   "growth_observed": dr.full(mi.UInt32, 2, n),
                   "growth_unknown": dr.full(mi.UInt32, 1 if view == 1 else 0, n)}
            manager.observe_rgb_growth(model, out, view, epoch, 16)
            active_cells = np.asarray(model.snapshot()["cell_indices"])
            # Removing parent 77 invalidates hole 76 at the same boundary.
            deletion_gain = np.where(active_cells == 77, .5, -.5).astype(np.float32)
            deletion = {"deletion_gain": mi.Float(deletion_gain),
                        "deletion_observed": dr.full(mi.UInt32, 1, manager.count),
                        "deletion_unknown": dr.zeros(mi.UInt32, manager.count)}
            manager.observe_rgb_deletion(model, deletion, view, epoch, 16)
            if epoch == 1 and view == 0:
                saved = manager.state_dict()
                resumed = TopologyManager(model, 3, cfg, schedule)
                resumed.load_state_dict(saved)
                manager = resumed
                manager.prepare_rgb_growth(model, epoch)
                manager.observe_rgb_growth(model, out, view, epoch, 16)
                manager.observe_rgb_deletion(model, deletion, view, epoch, 16)
        if gpu:
            if host_staged:
                manager.topology_memory_guard = lambda *_: dict(staging_options)
            # Any accidental complete host model snapshot is a regression.
            with patch("reconstruction_topology.model_checkpoint", side_effect=AssertionError("GPU topology copied model")):
                replacement, event = manager.finish_epoch(model, optimizer, epoch + 1)
        else:
            replacement, event = manager.finish_epoch(model, optimizer, epoch + 1)
        if replacement is not None:
            if gpu and host_staged:
                from reconstruction_parameter_transfer import DeferredRemapField
                assert all(isinstance(value, DeferredRemapField)
                           for value in replacement["model_state"].values())
                model = None
                for name in list(replacement["model_state"]):
                    replacement["model_state"][name] = replacement["model_state"][name].materialize()
            elif gpu:
                assert all(dr.is_array_v(type(value)) for value in replacement["model_state"].values())
            model = restore_model(replacement, mi)
        events.append(event)
    lifecycle = manager.state_dict()
    lifecycle["settings"].pop("gpu_topology")
    return {"events": {str(i): event for i, event in enumerate(events)},
            "model": model_checkpoint(model), "optimizer": optimizer.state_dict(), "lifecycle": lifecycle}


def delete_all(mi, dr, np, *, host_staged=False, staging_options=None):
    data = fixture(np)
    data["model_state"]["opacity_sh"][:] = -30.
    cfg = configuration(True)
    cfg.deletion_opacity_confirmation_windows = 1
    cfg.growth_rgb_min_support_views = 1
    model = restore_model(data, mi)
    manager = TopologyManager(model, 1, cfg, CoarseToFineConfig(enabled=False, total_epochs=3))
    optimizer = FalcorOptimizer(model)
    if host_staged:
        manager.topology_memory_guard = lambda *_: dict(staging_options)
    manager.prepare_rgb_growth(model, 0)
    n = len(manager.pending_growth["cells"])
    manager.observe_rgb_growth(model, {"growth_gain": dr.full(mi.Float, 1., n),
        "growth_observed": dr.full(mi.UInt32, 1, n), "growth_unknown": dr.zeros(mi.UInt32, n)}, 0, 0, 16)
    replacement, event = manager.finish_epoch(model, optimizer, 1)
    assert event["deleted"] == 4 and manager.count == optimizer.count == 0
    assert event["growth_rgb"]["rgb_qualified"] > 0
    assert event["growth_rgb"]["geometry_revalidated"] == 0
    if host_staged:
        model = None
        for name in list(replacement["model_state"]):
            replacement["model_state"][name] = replacement["model_state"][name].materialize()
    model = restore_model(replacement, mi)
    manager.prepare_rgb_growth(model, 1)
    for key in ("growth", "deletion"):
        out = {f"{key}_gain": mi.Float([]), f"{key}_observed": mi.UInt32([]), f"{key}_unknown": mi.UInt32([])}
        method = manager.observe_rgb_growth if key == "growth" else manager.observe_rgb_deletion
        method(model, out, 0, 1, 16)
    replacement, event = manager.finish_epoch(model, optimizer, 2)
    assert replacement is None and event["active_voxel_count"] == 0
    assert manager.state_dict()["count"] == 0


def main():
    with tempfile.TemporaryDirectory(prefix="mitsuba-gpu-lifecycle-") as temporary:
        report = {}
        mi, dr, np, handles = initialize_runtime(RuntimeConfig(), Path(temporary), report)
        from reconstruction_memory import TrainingMemoryError, probe_cuda_allocator
        from train import refinement_memory_snapshot
        policy = probe_cuda_allocator(dr)["policy"]

        def allocation_guard(required_bytes):
            # Dr.Jit need not leave its CUDA context current between calls.
            # Use the same physical-memory fallback as production staging,
            # retaining its device reserve instead of bypassing the guard.
            memory = refinement_memory_snapshot(report.get("gpu"))
            free = None if memory is None else min(memory["free_bytes"],
                memory.get("dedicated_free_bytes", memory["free_bytes"]))
            if free is None or int(required_bytes) + 1024**3 > free:
                raise TrainingMemoryError({"stage": "topology_staged_test_upload", "supported": False,
                    "required_bytes": int(required_bytes) + 1024**3, "available_bytes": free,
                    "reason": "Staged topology regression requires enough real free GPU memory."})

        staging_options = {"migration_mode": "host_staged", "allocation_guard": allocation_guard,
                           "allocator_policy": policy, "scratch_directory": temporary}
        reference = simulate(mi, dr, np, gpu=False)
        actual = simulate(mi, dr, np, gpu=True)
        arrays_equal(reference, actual, np)
        staged = simulate(mi, dr, np, gpu=True, host_staged=True, staging_options=staging_options)
        arrays_equal(reference, staged, np)
        delete_all(mi, dr, np)
        delete_all(mi, dr, np, host_staged=True, staging_options=staging_options)
        print(json.dumps({"cpu_gpu_votes_remap_revalidation_resume": "passed",
                          "host_staged_votes_remap_revalidation_resume": "passed",
                          "delete_all_empty_lifecycle": "passed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
