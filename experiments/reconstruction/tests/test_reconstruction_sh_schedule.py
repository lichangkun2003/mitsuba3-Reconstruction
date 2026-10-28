"""Progressive radiance SH: schedule boundaries, masked rendering, and Adam state.

The rendering oracle stores nine coefficients but explicitly zeros disabled
bands. Optimizer checks deliberately retain nonzero disabled gradients and
momentum, so merely clearing gradients cannot satisfy the freeze contract.
Only tiny generated scenes are used; no dataset or experiment output is read.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from preview import RuntimeConfig
from reconstruction_common import initialize_runtime, restore_model, model_metadata
from reconstruction_optimizer import FalcorOptimizer, OptimizerConfig
from reconstruction_replay import NativeLossReplay
from test_reconstruction_distortion import fixture, ray, evaluate
from test_reconstruction_dssim import evaluate as evaluate_image


def check_schedule():
    from reconstruction_sh_schedule import (active_sh_count_for_epoch, resolve_resume_sh_schedule,
                                            sh_schedule_protocol, validate_sh_schedule)
    from reconstruction_memory import estimate_training_memory

    config = SimpleNamespace(
        training=SimpleNamespace(num_epochs=1500, sh_degree_progression=True,
                                 sh_degree_weights=(4, 3, 3)),
        model=SimpleNamespace(sh_count=9))
    # The argument counts completed epochs, so 600 means the start of epoch 601.
    for completed, count in ((0, 1), (599, 1), (600, 4), (1049, 4), (1050, 9), (1499, 9)):
        assert active_sh_count_for_epoch(config, completed, 9) == count
    for total in (1, 2, 10, 11, 1000, 1500):
        config.training.num_epochs = total
        for completed in range(total):
            expected = 1 if completed < total * 4 // 10 else (4 if completed < total * 7 // 10 else 9)
            assert active_sh_count_for_epoch(config, completed, 9) == expected
            assert active_sh_count_for_epoch(config, completed, 4) == min(expected, 4)
            assert active_sh_count_for_epoch(config, completed, 1) == 1
    config.training.num_epochs = 1500
    config.training.sh_degree_weights = (40, 30, 30)
    assert active_sh_count_for_epoch(config, 599, 9) == 1
    assert active_sh_count_for_epoch(config, 600, 9) == 4
    assert active_sh_count_for_epoch(config, 1050, 9) == 9
    config.training.sh_degree_weights = (8, 0, 2)
    validate_sh_schedule(config)
    for completed, count in ((0, 1), (1199, 1), (1200, 9), (1499, 9)):
        assert active_sh_count_for_epoch(config, completed, 9) == count
    assert sh_schedule_protocol(config)["transition_completed_epochs"] == [1200, 1200]
    for total in (1, 2, 10, 11, 1001, 1500):
        config.training.num_epochs = total
        boundary = total * 8 // 10
        assert sh_schedule_protocol(config)["transition_completed_epochs"] == [boundary, boundary]
        for completed in range(total):
            expected = 1 if completed < boundary else 9
            assert active_sh_count_for_epoch(config, completed, 9) == expected
            assert active_sh_count_for_epoch(config, completed, 4) == min(expected, 4)
            assert active_sh_count_for_epoch(config, completed, 1) == 1
    config.training.num_epochs = 1500
    config.training.sh_degree_progression = False
    assert active_sh_count_for_epoch(config, 0, 9) == 9
    for invalid in ((0, 3, 3), (4, 3), (4, 3, 3, 1), (True, 3, 3), (4., 3, 3),
                    (8, -1, 2), (8, 0, 0), (0, 0, 2), (8, False, 2), (8, 0., 2)):
        config.training.sh_degree_weights = invalid
        try:
            validate_sh_schedule(config)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid SH weights accepted: {invalid}")
    config.training.sh_degree_progression = True
    config.training.sh_degree_weights = (4, 3, 3)
    validate_sh_schedule(config)
    resolve_resume_sh_schedule(config, {"config": {"training": {"num_epochs": 1500}}}, {})
    assert config.training.sh_degree_progression is False
    saved = {"config": {"training": {"num_epochs": 1500, "sh_degree_progression": True,
                                       "sh_degree_weights": [4, 3, 3]}}}
    resolve_resume_sh_schedule(config, saved, {})
    assert config.training.sh_degree_progression is True
    assert tuple(config.training.sh_degree_weights) == (4, 3, 3)
    for overrides in ({"training": {"sh_degree_progression": False}},
                      {"training": {"sh_degree_weights": [6, 3, 1]}}):
        try:
            resolve_resume_sh_schedule(config, saved, overrides)
        except ValueError:
            pass
        else:
            raise AssertionError("Resume accepted a different SH schedule")
    config.training.num_epochs = 1600
    try:
        resolve_resume_sh_schedule(config, saved, {})
    except ValueError:
        pass
    else:
        raise AssertionError("Resume changed the total SH schedule budget")
    import train
    # Exercise both user entry points, while preserving legacy checkpoint defaults.
    assert train.TrainingSettings().sh_degree_weights == (4, 3, 3)
    preset = json.loads((Path(__file__).resolve().parents[1] / "configs" / "bicycle_1024.json")
                        .read_text(encoding="utf-8"))
    for values in ({}, preset):
        fresh = copy.deepcopy(train.CONFIG)
        with patch.object(sys, "argv", ["train.py"]):
            arguments = train.parse_arguments()
        train.apply_arguments(fresh, arguments, values)
        validate_sh_schedule(fresh)
        assert fresh.training.sh_degree_progression is True
        assert tuple(fresh.training.sh_degree_weights) == (8, 0, 2)
        assert sh_schedule_protocol(fresh)["transition_completed_epochs"] == [1200, 1200]
        resumed = copy.deepcopy(fresh)
        resolve_resume_sh_schedule(resumed, saved, {})
        assert tuple(resumed.training.sh_degree_weights) == (4, 3, 3)
        assert active_sh_count_for_epoch(resumed, 600, 9) == 4
        simultaneous = {"config": {"training": {"num_epochs": 1500, "sh_degree_progression": True,
                                                "sh_degree_weights": [8, 0, 2]}}}
        resolve_resume_sh_schedule(resumed, simultaneous, {"training": {"sh_degree_weights": [8, 0, 2]}})
        assert tuple(resumed.training.sh_degree_weights) == (8, 0, 2)
        assert active_sh_count_for_epoch(resumed, 1199, 9) == 1
        assert active_sh_count_for_epoch(resumed, 1200, 9) == 9
    memory_options = dict(active_voxels=17, sh_count=9, opacity_sh_count=1,
                          compact_capacity=0, ray_count=35, max_hits=8, max_candidates=4)
    ordinary = estimate_training_memory(**memory_options)
    progressive = estimate_training_memory(**memory_options, radiance_degree_steps=True)
    assert progressive["known_logical_bytes"] - ordinary["known_logical_bytes"] == 17 * 8
    return {"total_epoch_40_70_percent_boundaries_and_stored_capacity": "passed",
            "DC_first_80_percent_then_both_higher_bands_and_rounding": "passed",
            "fresh_default_and_bicycle_preset_80_percent_with_legacy_resume_preserved": "passed",
            "schedule_validation_legacy_resume_and_progressive_resume_contract": "passed",
            "degree_clock_memory_budget_includes_two_extra_UInt32_per_voxel": "passed"}


def sh_fixture(np, count=3):
    result = fixture(np, count)
    coefficients = np.zeros((count, 9, 3), dtype=np.float32)
    coefficients[:, 0, :] = result["model_state"]["radiance_sh"].reshape(count, 3)
    # Nonzero high bands are essential: a zero-initialized fixture cannot tell
    # whether inactive SH affects the forward pass or merely stops learning.
    for voxel in range(count):
        coefficients[voxel, 1:, :] = np.linspace(-.14, .18, 24).reshape(8, 3) + voxel * .01
    result["model_metadata"]["sh_count"] = 9
    result["model_state"]["radiance_sh"] = coefficients.ravel()
    return result


def masked_fixture(snapshot, active_count, np):
    result = copy.deepcopy(snapshot)
    result["model_state"]["radiance_sh"].reshape(-1, 9, 3)[:, active_count:, :] = 0.
    return result


def install_proposal(model, snapshot, mi):
    model.set_growth_candidates({
        name: (mi.UInt32(value) if name == "cell_indices" else mi.Float(value))
        for name, value in snapshot["model_state"].items()})


def compare_render_gradients(actual, expected, active_count, np):
    for name in ("rgb", "raw_rgb", "alpha", "raw_alpha", "hit_count", "loss",
                 "rgb_loss", "dssim_loss", "appearance_valid", "geometry_valid",
                 "growth_gain", "growth_observed", "growth_unknown",
                 "deletion_gain", "deletion_observed", "deletion_unknown"):
        if name in actual:
            np.testing.assert_allclose(np.asarray(actual[name]), np.asarray(expected[name]),
                                       atol=3e-7, rtol=3e-5, err_msg=name)
    for name, gradient in actual["gradients"].items():
        actual_gradient = np.asarray(gradient)
        expected_gradient = np.asarray(expected["gradients"][name])
        if name == "radiance_sh":
            actual_gradient = actual_gradient.reshape(-1, 9, 3)
            expected_gradient = expected_gradient.reshape(-1, 9, 3)
            np.testing.assert_array_equal(actual_gradient[:, active_count:, :], 0.)
            actual_gradient = actual_gradient[:, :active_count, :]
            expected_gradient = expected_gradient[:, :active_count, :]
        np.testing.assert_allclose(actual_gradient, expected_gradient, atol=4e-7, rtol=4e-5,
                                   err_msg="gradients." + name)


def check_rendering_and_replay(mi, dr, np):
    snapshot = sh_fixture(np)
    model = restore_model(snapshot, mi)
    assert int(model.active_sh_count) == 9
    assert int(mi.VoxelReconstruction.sh_schedule_version) >= 1
    for bad_count in (0, 10):
        try:
            model.set_active_sh_count(bad_count)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("Invalid active SH count was accepted")
    proposal = sh_fixture(np, 1)
    proposal["model_state"]["cell_indices"][:] = 53
    install_proposal(model, proposal, mi)
    replay = NativeLossReplay(dr)
    colors = []
    options = {"collect_growth_evidence": 1., "collect_rgb_deletion_evidence": 1.,
               "max_growth_hits": 4., "max_candidates": 4.}
    for active_count in (1, 4, 9, 1, 9):
        model.set_active_sh_count(active_count)
        unchanged_revision = int(model.topology_revision)
        model.set_active_sh_count(active_count)
        assert int(model.topology_revision) == unchanged_revision
        reference = restore_model(masked_fixture(snapshot, active_count, np), mi)
        install_proposal(reference, masked_fixture(proposal, active_count, np), mi)
        for octree in (False, True):
            arguments = (ray(mi, np), mi.Color3f(1.), .00001, 8, .0001, octree)
            rendered, expected = model.trace(*arguments), reference.trace(*arguments)
            dr.eval(rendered, expected)
            dr.sync_thread()
            np.testing.assert_allclose(np.asarray(rendered[0]), np.asarray(expected[0]),
                                       atol=3e-7, rtol=3e-5)
            settings = dict(options, use_octree=float(octree))
            expected = evaluate(reference, mi, dr, np, coefficient=0., native_options=settings)
            actual = evaluate(model, mi, dr, np, coefficient=0., native_options=settings)
            compare_render_gradients(actual, expected, active_count, np)
            for _ in range(2):
                replayed = evaluate(model, mi, dr, np, coefficient=0., native_options=settings,
                                    replay=replay)
                compare_render_gradients(replayed, expected, active_count, np)
        # D-SSIM supplies an image adjoint to the native backpropagation path.
        # It must obey the same band mask without reintroducing high-band RGB.
        expected_image = evaluate_image(reference, mi, dr, np)
        actual_image = evaluate_image(model, mi, dr, np)
        compare_render_gradients(actual_image, expected_image, active_count, np)
        colors.append(np.asarray(actual["rgb"]).copy())
        assert np.asarray(actual["growth_observed"]).sum() > 0
        assert np.asarray(actual["deletion_observed"]).sum() > 0
    assert not np.allclose(colors[0], colors[1], atol=1e-5)
    assert not np.allclose(colors[1], colors[2], atol=1e-5)
    np.testing.assert_array_equal(colors[0], colors[3])
    np.testing.assert_array_equal(colors[2], colors[4])
    assert replay.statistics()["replays"] >= 8
    return {"nonzero_inactive_SH_render_gradients_and_topology_evidence": "passed",
            "dda_octree_and_native_replay_across_SH_stages": "passed",
            "direct_DC_to_all_bands_invalidates_replay_and_restores_full_color": "passed"}


def optimizer_snapshot(model, optimizer, np):
    parameters = {name: np.asarray(value).copy() for name, value in model.snapshot().items()}
    state = optimizer.state_dict()
    return parameters, state


def check_refinement_and_metadata(mi, dr, np):
    from reconstruction_topology import CoarseToFineConfig, refine_state
    from reconstruction_refinement_gpu import refine_state_gpu
    from reconstruction_topology_gpu import device_checkpoint

    source = sh_fixture(np)
    parent_coefficients = source["model_state"]["radiance_sh"].reshape(-1, 9, 3)
    parent_rows = {int(cell): row for row, cell in enumerate(source["model_state"]["cell_indices"])}
    for progressive in (False, True):
        checkpoint = copy.deepcopy(source)
        if progressive:
            checkpoint["model_metadata"]["active_sh_count"] = 4
        model = restore_model(checkpoint, mi)
        assert int(model.active_sh_count) == (4 if progressive else 9)
        host_metadata = model_metadata(model)
        device_metadata = device_checkpoint(model)["model_metadata"]
        for metadata in (host_metadata, device_metadata):
            assert ("active_sh_count" in metadata) == progressive
            if progressive:
                assert metadata["active_sh_count"] == 4
        refined = []
        for refine in (refine_state, refine_state_gpu):
            result = refine(checkpoint, CoarseToFineConfig())
            dr.sync_thread()
            cells = np.asarray(result["model_state"]["cell_indices"], dtype=np.uint32)
            assert len(cells) > 0
            xyz = np.column_stack((cells % 8, cells // 8 % 8, cells // 64)) // 2
            parent_cells = xyz[:, 0] + 4 * xyz[:, 1] + 16 * xyz[:, 2]
            rows = [parent_rows[int(cell)] for cell in parent_cells]
            expected = parent_coefficients[rows].copy()
            if not progressive:
                expected[:, 1:, :] = 0.
            coefficients = np.asarray(result["model_state"]["radiance_sh"]).reshape(-1, 9, 3)
            np.testing.assert_array_equal(coefficients, expected)
            assert ("active_sh_count" in result["model_metadata"]) == progressive
            if progressive:
                assert result["model_metadata"]["active_sh_count"] == 4
            refined.append((cells.copy(), coefficients.copy()))
        np.testing.assert_array_equal(refined[0][0], refined[1][0])
        np.testing.assert_array_equal(refined[0][1], refined[1][1])
    return {"progressive_checkpoint_metadata_and_CPU_GPU_refinement_preserve_all_SH": "passed",
            "legacy_refinement_remains_DC_only": "passed"}


def check_optimizer_freezing(mi, dr, np):
    modes = ("dense", "device", "device_replay", "compact_replay", "compact_reference")
    models = [restore_model(sh_fixture(np), mi) for _ in modes]
    config = OptimizerConfig(lr_opacity=0., lr_center=0., lr_log_scale=0., lr_rotation=0.)
    optimizers = [FalcorOptimizer(model, copy.deepcopy(config), sparse_updates=mode != "dense",
                                  replay_updates="replay" in mode, radiance_degree_steps=True)
                  for mode, model in zip(modes, models)]
    expected_clocks = np.zeros((3, 3), dtype=np.uint32)
    for iteration, active_count in enumerate((1, 1, 9, 9, 1, 1, 4, 4, 9, 9, 1)):
        expected_clocks[:, :int(active_count ** .5)] += 1
        states = []
        for mode, model, optimizer in zip(modes, models, optimizers):
            model.set_active_sh_count(active_count)
            previous, previous_state = optimizer_snapshot(model, optimizer, np)
            out = {"gradients": {name: dr.full(mi.Float, .3 + iteration * .02, 3 * width)
                                  for name, width in optimizer.dimensions.items()},
                   "appearance_valid": mi.UInt32([1, 1, 1]),
                   "geometry_valid": mi.UInt32([0, 0, 0]),
                   "background_carve_valid": mi.UInt32([0, 0, 0])}
            if mode == "compact_replay":
                out["voxel_ids"] = mi.UInt32([0, 1, 2])
            if mode == "compact_reference":
                optimizer._step_compact_reference(model, out, sample_count=1, epoch=iteration,
                                                 coarse_to_fine=True, update_allowed=mi.Bool(True))
            else:
                optimizer.step(model, out, epoch=iteration, update_allowed=mi.Bool(True))
            dr.sync_thread()
            current, current_state = optimizer_snapshot(model, optimizer, np)
            old_coefficients = previous["radiance_sh"].reshape(3, 9, 3)
            new_coefficients = current["radiance_sh"].reshape(3, 9, 3)
            np.testing.assert_array_equal(new_coefficients[:, active_count:, :],
                                          old_coefficients[:, active_count:, :], err_msg=mode)
            assert np.any(new_coefficients[:, :active_count, :] != old_coefficients[:, :active_count, :])
            for key in ("mean", "squared_mean"):
                old = np.asarray(previous_state["groups"]["radiance_sh"][key]).reshape(3, 9, 3)
                new = np.asarray(current_state["groups"]["radiance_sh"][key]).reshape(3, 9, 3)
                np.testing.assert_array_equal(new[:, active_count:, :], old[:, active_count:, :],
                                              err_msg=mode + "." + key)
            np.testing.assert_array_equal(np.asarray(current_state["groups"]["radiance_sh"]["steps"])
                                          .reshape(3, 3), expected_clocks, err_msg=mode)
            if iteration <= 2:
                # Two DC updates must not age either higher band. Both degree
                # clocks start at one on their simultaneous first update.
                clocks = np.asarray(current_state["groups"]["radiance_sh"]["steps"]).reshape(3, 3)
                np.testing.assert_array_equal(clocks[:, 1:], 1 if iteration == 2 else 0,
                                              err_msg=mode + ".simultaneous_degree_clock_start")
            states.append((current, current_state))
        for mode, (parameters, state) in zip(modes[1:], states[1:]):
            for name, value in states[0][0].items():
                np.testing.assert_allclose(parameters[name], value, atol=3e-7, rtol=3e-5,
                                           err_msg=mode + "." + name)
            for name, group in states[0][1]["groups"].items():
                for key, value in group.items():
                    np.testing.assert_allclose(state["groups"][name][key], value,
                                               atol=3e-7, rtol=3e-5, err_msg=mode + "." + name + "." + key)
    assert optimizers[2].statistics()["replays"] > 0
    assert optimizers[3].statistics()["replays"] > 0
    # Saved clocks survive restore; old per-voxel clocks expand to every band.
    saved = optimizers[2].state_dict()
    restored = FalcorOptimizer(models[2], copy.deepcopy(config))
    restored.load_state_dict(saved)
    for name, group in saved["groups"].items():
        for key, value in group.items():
            np.testing.assert_array_equal(restored.state_dict()["groups"][name][key], value)
    legacy = copy.deepcopy(saved)
    legacy.pop("radiance_degree_steps", None)
    legacy["groups"]["radiance_sh"]["steps"] = np.array([7, 11, 13], dtype=np.uint32)
    restored.load_state_dict(legacy)
    np.testing.assert_array_equal(restored.state_dict()["groups"]["radiance_sh"]["steps"]
                                  .reshape(3, 3), np.repeat([[7], [11], [13]], 3, axis=1))
    before_remap = restored.state_dict()["groups"]
    restored.remap([2, 0], 3)
    after_remap = restored.state_dict()["groups"]
    for name, group in before_remap.items():
        for key, value in group.items():
            rows = np.asarray(value).reshape(3, -1)
            np.testing.assert_array_equal(np.asarray(after_remap[name][key]).reshape(3, -1),
                                          np.concatenate((rows[[2, 0]], np.zeros_like(rows[:1]))))
    return {"dense_device_compact_reference_Adam_and_replay_parity": "passed",
            "direct_DC_to_SH2_starts_both_new_Adam_degree_clocks_at_one": "passed",
            "inactive_parameters_moments_and_degree_clocks_frozen_with_stale_momentum": "passed",
            "degree_clock_checkpoint_legacy_migration_and_topology_remap": "passed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    results = check_schedule()
    if not args.cpu_only:
        with tempfile.TemporaryDirectory(prefix="mitsuba-sh-schedule-") as output:
            mi, dr, np, handles = initialize_runtime(RuntimeConfig(args.build_dir, args.gpu), Path(output), {})
            print("Checking SH rendering and replay", flush=True)
            results.update(check_rendering_and_replay(mi, dr, np))
            print("Checking SH Adam freeze and state migration", flush=True)
            results.update(check_optimizer_freezing(mi, dr, np))
            print("Checking SH metadata and refinement", flush=True)
            results.update(check_refinement_and_metadata(mi, dr, np))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
