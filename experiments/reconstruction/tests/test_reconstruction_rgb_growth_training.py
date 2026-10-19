"""Tiny CUDA integration: online topology -> train_view -> Adam -> commit/resume.

Two one-cell holes, three generated cameras, 4x4 at 1 spp. The test uses no
dataset and writes runtime metadata only in an automatically removed temp dir.
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

from preview import RuntimeConfig
from reconstruction_common import UniformSampler, initialize_runtime, restore_model
from reconstruction_optimizer import FalcorOptimizer
from reconstruction_replay import NativeLossReplay
from reconstruction_topology import TopologyManager, model_checkpoint
from train import CONFIG, train_view, validate


def run(mi, dr, np):
    config = copy.deepcopy(CONFIG)
    config.render.width = config.render.height = 4
    config.render.spp = 1
    config.training.num_epochs = 5
    config.schedule.enabled = False
    config.schedule.total_epochs = 5
    config.topology.growth_warmup_epochs = 0
    config.topology.growth_interval = 1
    assert config.topology.growth_rgb_normalization == "observed"
    config.topology.growth_rgb_min_support_views = 3
    config.training.profile_stages = False
    # Exercise the actual optimizer without allowing this wiring check to
    # change the hand-constructed geometric eligibility between events.
    for name in ("lr_radiance", "lr_opacity", "lr_center", "lr_log_scale", "lr_rotation"):
        setattr(config.optimizer, name, 0.)
    validate(config)
    n, sh = 3, config.model.sh_count
    checkpoint = {"model_metadata": {"resolution": 8, "grid_min": [0.] * 3,
                    "grid_max": [8.] * 3, "sh_count": sh, "opacity_sh_count": 1},
                  "model_state": {
                      "cell_indices": np.asarray([73, 75, 77], np.uint32),
                      "centers": np.tile([.5, .5, .5], n).astype(np.float32),
                      "log_scales": np.tile(np.log([.8, .7, .6]), n).astype(np.float32),
                      "rotations": np.tile([1., 0., 0., 0.], n).astype(np.float32),
                      "radiance_sh": np.zeros(n * sh * 3, np.float32),
                      "opacity_sh": np.zeros(n, np.float32)}}
    model = restore_model(checkpoint, mi)
    optimizer = FalcorOptimizer(model, config.optimizer)
    manager = TopologyManager(model, 3, config.topology, config.schedule)
    replay = NativeLossReplay(dr, enabled=True)
    sampler, rng = UniformSampler(np), np.random.default_rng(57)
    reference = np.full((4, 4, 3), .5, dtype=np.float32)
    events = []
    for epoch, cell in enumerate((74, 76)):
        replay.clear()
        assert manager.prepare_rgb_growth(model, epoch)
        pending = manager.pending_growth
        candidate_count = len(pending["cells"])
        assert cell in pending["cells"]
        assert candidate_count == pending["all_candidate_count"]
        assert int(model.growth_candidate_count) == candidate_count
        if epoch == 0:
            # Both holes are installed together. Cameras only observe the
            # first hole in this event, so evidence (not a budget) selects it.
            assert 76 in pending["cells"] and candidate_count >= 2
        config.training.chunk_size = 0 if epoch == 0 else 5
        expected_gain_sum = np.zeros(candidate_count, dtype=np.float64)
        expected_observed_views = np.zeros(candidate_count, dtype=np.uint32)
        for view_id in range(3):
            view = {"index": view_id, "camera": {
                "position": [(cell % 8) + .2, 1.5, -1.],
                "forward": [0., 0., 1.], "right": [1., 0., 0.],
                "up": [0., 1., 0.], "camera_angle_x": .05}}
            gradients, loss, alpha = train_view(
                model, view, reference, config, sampler, rng, mi, dr, np,
                native_replay=replay, collect_growth_evidence=True,
                collect_deletion_evidence=False)
            assert math.isfinite(loss) and alpha is None
            assert set(("growth_gain", "growth_observed", "growth_unknown")) <= gradients.keys()
            assert dr.width(gradients["growth_gain"]) == candidate_count
            gain = np.asarray(gradients["growth_gain"])
            observed = np.asarray(gradients["growth_observed"])
            unknown = np.asarray(gradients["growth_unknown"])
            reliable = (observed > 0) & (unknown == 0) & np.isfinite(gain)
            expected_gain_sum[reliable] += gain[reliable] / observed[reliable]
            expected_observed_views += reliable.astype(np.uint32)
            manager.observe_rgb_growth(model, gradients, view_id, epoch, 16)
            optimizer.step(model, gradients, epoch=epoch, coarse_to_fine=False)
            if epoch == 1 and view_id == 0:
                # Resume after one full view: native overlays are rebuilt from
                # portable pending state, and that view cannot be voted twice.
                topology_state = manager.state_dict()
                optimizer_state = optimizer.state_dict()
                replay.clear()
                model = restore_model(model_checkpoint(model), mi)
                optimizer = FalcorOptimizer(model, config.optimizer)
                optimizer.load_state_dict(optimizer_state)
                manager = TopologyManager(model, 3, config.topology, config.schedule)
                manager.load_state_dict(topology_state)
                assert manager.prepare_rgb_growth(model, epoch)
                assert manager.pending_growth["seen_views"] == {0}
        pending = manager.state_dict()["pending_growth"]
        assert pending["normalization"] == "observed"
        np.testing.assert_allclose(pending["gain_sum"], expected_gain_sum, atol=2e-7, rtol=3e-6)
        np.testing.assert_array_equal(pending["observed_views"], expected_observed_views)
        replacement, event = manager.finish_epoch(model, optimizer, epoch + 1)
        assert event["grown"] == 1, event
        assert event["growth_rgb"]["support_votes"] == 3, event
        assert event["growth_rgb"]["unknown_votes"] == 0, event
        assert int(model.growth_candidate_count) == 0
        assert replacement is not None
        for group in optimizer.state_dict()["groups"].values():
            assert int(group["steps"][-1]) == 0
        replay.clear()
        model = restore_model(replacement, mi)
        assert int(model.active_voxel_count) == n + epoch + 1
        events.append(event)
    stats = replay.statistics()
    assert stats["recordings"] > 0 and stats["replays"] > 0, stats
    assert manager.statistics()["rgb_growth_device_views"] > 0
    return {"native_prepare_train_view_vote_commit_resume": "passed",
            "epochs": len(events), "grown": sum(event["grown"] for event in events),
            "final_voxels": int(model.active_voxel_count), "replay": stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="mitsuba-rgb-growth-training-") as temporary:
        mi, dr, np, handles = initialize_runtime(RuntimeConfig(args.build_dir, args.gpu), Path(temporary), {})
        print(json.dumps(run(mi, dr, np)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
