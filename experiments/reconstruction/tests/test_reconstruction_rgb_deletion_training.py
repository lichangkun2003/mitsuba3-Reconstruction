"""Tiny CUDA train/view/vote/delete/resume integration, with no dataset.

Three 4x4 views at 1 spp observe one dark voxel against white. Two complete
evidence windows confirm its removal, while persistent low opacity removes
an unseen voxel independently. Two opaque unseen voxels retain optimizer state.
The frozen tail then continues continuous optimization.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preview import RuntimeConfig
from reconstruction_checkpoint import load_checkpoint, save_checkpoint
from reconstruction_common import UniformSampler, initialize_runtime, restore_model
from reconstruction_optimizer import FalcorOptimizer
from reconstruction_replay import NativeLossReplay
from reconstruction_topology import TopologyManager, model_checkpoint
from train import CONFIG, compute_view_shard, train_view, validate


EVIDENCE = ("deletion_gain", "deletion_observed", "deletion_unknown")


def view_at(index, x=1.2):
    return {"index": index, "camera": {
        "position": [x, 1.5, -1.], "forward": [0., 0., 1.],
        "right": [1., 0., 0.], "up": [0., 1., 0.], "camera_angle_x": .05}}


def compare_evidence(first, second, np):
    for name in EVIDENCE:
        if name == "deletion_gain":
            np.testing.assert_allclose(np.asarray(first[name]), np.asarray(second[name]),
                                       atol=2e-6, rtol=3e-6, err_msg=name)
        else:
            np.testing.assert_array_equal(np.asarray(first[name]), np.asarray(second[name]),
                                          err_msg=name)


def run(mi, dr, np, temporary):
    config = copy.deepcopy(CONFIG)
    config.render.width = config.render.height = 4
    config.render.spp = 1
    config.training.num_epochs = config.schedule.total_epochs = 5
    config.schedule.enabled = False
    config.schedule.enable_coarse_growth = False
    config.topology.enable_growth = config.topology.use_rgb_growth = False
    config.topology.enable_deletion = config.topology.use_rgb_deletion = True
    config.topology.use_low_opacity_deletion = True
    config.topology.deletion_opacity_threshold = .005
    config.topology.deletion_opacity_confirmation_windows = 2
    config.topology.deletion_warmup_epochs = 0
    config.topology.deletion_protection_epochs = 0
    config.topology.deletion_interval = 1
    config.topology.deletion_rgb_confirmation_windows = 2
    config.topology.deletion_rgb_min_support_views = 3
    config.topology.deletion_freeze_last_epochs = 3
    config.training.profile_stages = False
    config.training.chunk_size = 0
    for name in ("lr_radiance", "lr_opacity", "lr_center", "lr_log_scale", "lr_rotation"):
        setattr(config.optimizer, name, 0.)
    validate(config)
    count, sh = 4, config.model.sh_count
    checkpoint = {
        "model_metadata": {"resolution": 8, "grid_min": [0.] * 3, "grid_max": [8.] * 3,
                           "sh_count": sh, "opacity_sh_count": 1},
        "model_state": {
            "cell_indices": np.asarray([73, 75, 77, 79], np.uint32),
            "centers": np.tile([.5, .5, .5], count).astype(np.float32),
            "log_scales": np.tile(np.log([.8, .7, .6]), count).astype(np.float32),
            "rotations": np.tile([1., 0., 0., 0.], count).astype(np.float32),
            "radiance_sh": np.zeros(count * sh * 3, np.float32),
            "opacity_sh": np.zeros(count, np.float32),
        }}
    checkpoint["model_state"]["opacity_sh"][1] = (
        math.log(.001 / (1. - .001)) * (2. * math.sqrt(math.pi)))
    model = restore_model(checkpoint, mi)
    optimizer = FalcorOptimizer(model, config.optimizer)
    # Distinct nonzero moments detect accidental resets or wrong survivor indices.
    seeded_optimizer = optimizer.state_dict()
    for group in seeded_optimizer["groups"].values():
        group["mean"][:] = np.linspace(-.002, .002, len(group["mean"]), dtype=np.float32)
        group["squared_mean"][:] = np.linspace(.01, .02, len(group["squared_mean"]), dtype=np.float32)
        group["steps"][:] = [2, 3, 4, 5]
    optimizer.load_state_dict(seeded_optimizer)
    manager = TopologyManager(model, 3, config.topology, config.schedule)
    replay, sampler = NativeLossReplay(dr), UniformSampler(np)
    reference = np.ones((4, 4, 3), np.float32)

    # Chunking and uneven pixel shards must preserve unnormalized native sums.
    full, full_loss, _ = train_view(
        model, view_at(0), reference, config, sampler, np.random.default_rng(57),
        mi, dr, np, native_replay=replay, collect_deletion_evidence=True)
    config.training.chunk_size = 5
    chunked, chunked_loss, _ = train_view(
        model, view_at(0), reference, config, sampler, np.random.default_rng(57),
        mi, dr, np, native_replay=replay, collect_deletion_evidence=True)
    compare_evidence(full, chunked, np)
    np.testing.assert_allclose(full_loss, chunked_loss, atol=2e-6, rtol=3e-6)
    config.training.chunk_size = 0
    offsets, weights, token = sampler.sample(0, reference, np.random.default_rng(57))
    partials = [compute_view_shard(
        model, view_at(0), reference, config, offsets, weights, token, mi, dr,
        pixel_begin=begin, pixel_end=end, native_replay=replay,
        collect_deletion_evidence=True) for begin, end in ((0, 7), (7, 16))]
    combined = {name: partials[0][0][name] + partials[1][0][name] for name in EVIDENCE}
    dr.eval(combined)
    compare_evidence(full, combined, np)

    rng, events = np.random.default_rng(71), []
    for epoch in range(2):
        assert manager.collect_deletion(epoch)
        config.training.chunk_size = 0 if epoch == 0 else 5
        for view_id in range(3):
            out, loss, alpha = train_view(
                model, view_at(view_id), reference, config, sampler, rng, mi, dr, np,
                native_replay=replay, collect_deletion_evidence=manager.collect_deletion(epoch))
            assert math.isfinite(loss) and alpha is None
            np.testing.assert_array_equal(np.asarray(out["deletion_observed"]), [16, 0, 0, 0])
            np.testing.assert_array_equal(np.asarray(out["deletion_unknown"]), [0, 0, 0, 0])
            manager.observe_rgb_deletion(model, out, view_id, epoch, 16)
            optimizer.step(model, out, epoch=epoch, coarse_to_fine=False)
            if epoch == 1 and view_id == 0:
                # Serialize one full confirmation window plus one view from
                # the next. Restoring must neither forget nor count it twice.
                saved_model = model_checkpoint(model)
                path = save_checkpoint(
                    Path(temporary) / "deletion-mid-window.npz",
                    saved_model["model_state"], saved_model["model_metadata"],
                    training_state={"topology": manager.state_dict()},
                    optimizer_state=optimizer.state_dict(), config=asdict(config))
                loaded = load_checkpoint(path)
                replay.clear()
                model = restore_model(loaded, mi)
                optimizer = FalcorOptimizer(model, config.optimizer)
                optimizer.load_state_dict(loaded["optimizer_state"])
                manager = TopologyManager(model, 3, config.topology, config.schedule)
                manager.load_state_dict(loaded["training_state"]["topology"])
                assert manager.pending_rgb_deletion["seen_views"] == {0}
                np.testing.assert_array_equal(manager.rgb_deletion_streak, [1, 0, 0, 0])
                np.testing.assert_array_equal(manager.low_opacity_deletion_streak, [0, 1, 0, 0])
                assert manager.low_opacity_deletion_last_boundary == 1
                manager.observe_rgb_deletion(model, out, view_id, epoch, 16)
                assert manager.pending_rgb_deletion["seen_views"] == {0}
        before_optimizer = optimizer.state_dict()
        before_model = model_checkpoint(model)
        replacement, event = manager.finish_epoch(model, optimizer, epoch + 1)
        events.append(event)
        assert event["deletion_rgb"]["support_votes"] == 3, event
        assert event["deletion_rgb"]["status"] == "complete", event
        assert event["deletion_opacity"]["low_opacity_voxels"] == 1, event
        if epoch == 0:
            assert replacement is None and event["deleted"] == 0
            np.testing.assert_array_equal(manager.rgb_deletion_streak, [1, 0, 0, 0])
            np.testing.assert_array_equal(manager.low_opacity_deletion_streak, [0, 1, 0, 0])
        else:
            assert replacement is not None and event["deleted"] == 2, event
            assert event["deletion_rgb"]["deleted"] == event["deletion_opacity"]["deleted"] == 1
            np.testing.assert_array_equal(replacement["model_state"]["cell_indices"], [77, 79])
            for name, values in before_model["model_state"].items():
                width = len(values) // count
                np.testing.assert_array_equal(replacement["model_state"][name],
                                              values.reshape(count, width)[2:].ravel(), err_msg=name)
            after_optimizer = optimizer.state_dict()
            for name, group in before_optimizer["groups"].items():
                for field, values in group.items():
                    width = len(values) // count
                    np.testing.assert_array_equal(after_optimizer["groups"][name][field],
                                                  values.reshape(count, width)[2:].ravel(),
                                                  err_msg=f"{name}.{field}")
            replay.clear()
            model = restore_model(replacement, mi)
            assert int(model.active_voxel_count) == manager.count == optimizer.count == 2

    # Restored lifecycle state still freezes deletion at global epoch three;
    # an observed dark survivor nevertheless receives normal radiance updates.
    saved_topology = manager.state_dict()
    manager = TopologyManager(model, 3, config.topology, config.schedule)
    manager.load_state_dict(saved_topology)
    assert manager.deletion_frozen(3) and not manager.collect_deletion(2)
    evidence_views = manager.statistics()["rgb_deletion_device_views"]
    before = model_checkpoint(model)["model_state"]
    optimizer.config.lr_radiance = .01
    for view_id in range(3):
        out, loss, alpha = train_view(
            model, view_at(view_id, 5.2), reference, config, sampler, rng, mi, dr, np,
            native_replay=replay, collect_deletion_evidence=manager.collect_deletion(2))
        assert math.isfinite(loss) and alpha is None
        assert all(name not in out for name in EVIDENCE)
        manager.observe_rgb_deletion(model, out, view_id, 2, 16)
        optimizer.step(model, out, epoch=2, coarse_to_fine=False)
    replacement, event = manager.finish_epoch(model, optimizer, 3)
    assert replacement is None and event["deleted"] == 0 and event["grown"] == 0
    assert event["deletion_frozen"] and not event["deletion_evidence_requested"]
    assert manager.statistics()["rgb_deletion_device_views"] == evidence_views
    after = model_checkpoint(model)["model_state"]
    np.testing.assert_array_equal(after["cell_indices"], before["cell_indices"])
    assert np.max(np.abs(after["radiance_sh"] - before["radiance_sh"])) > 0
    events.append(event)
    stats = replay.statistics()
    assert stats["recordings"] > 0 and stats["replays"] > 0, stats
    return {"train_view_full_chunk_uneven_shard_evidence_parity": "passed",
            "rgb_and_persistent_low_opacity_remove_disjoint_voxels": "passed",
            "mid_window_npz_resume_and_duplicate_view_guard": "passed",
            "survivor_parameters_and_adam_moments_retained": "passed",
            "frozen_tail_keeps_continuous_optimization": "passed",
            "epochs": len(events), "deleted": sum(event["deleted"] for event in events),
            "final_voxels": int(model.active_voxel_count), "replay": stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="mitsuba-rgb-deletion-training-") as temporary:
        mi, dr, np, handles = initialize_runtime(
            RuntimeConfig(args.build_dir, args.gpu), Path(temporary), {})
        print(json.dumps(run(mi, dr, np, temporary)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
