"""Generated CUDA checks for same-view pixel sharding without two physical GPUs.

Two sequential shards emulate the reduction contract on one GPU. This verifies
sampling, reduction, contour history and updates, not NCCL or multi-GPU speed.
No project dataset or experiment is opened; runtime artifacts are temporary.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reconstruction_common import ContourSampler, initialize_runtime, restore_model
from reconstruction_optimizer import FalcorOptimizer
from preview import RuntimeConfig
from reconstruction_reference import GPUReferenceCache
from reconstruction_replay import NativeLossReplay
from train import compute_view_shard, shard_pixel_range, train_view
from test_reconstruction_performance import compare_accumulators, fixture


def combine_shards(first, second, dr):
    """The production collective must implement precisely these reductions."""
    a, a_loss, a_finite, a_raw, a_alpha = first
    b, b_loss, b_finite, b_raw, b_alpha = second
    out = {"gradients": {name: value + b["gradients"][name]
                         for name, value in a["gradients"].items()}}
    for name in ("appearance_valid", "geometry_valid", "background_carve_valid"):
        out[name] = a[name] + b[name]
    for name in ("foreground_evidence", "background_evidence"):
        out[name] = dr.maximum(a[name], b[name])
    assert (a_raw is None) == (b_raw is None)
    assert (a_alpha is None) == (b_alpha is None)
    return (out, a_loss + b_loss, a_finite & b_finite,
            None if a_raw is None else a_raw + b_raw,
            None if a_alpha is None else a_alpha + b_alpha)


class SequentialCollective:
    """Capture rank zero, then deliver the same reduced arrays to both replicas."""

    world_size = 2

    def __init__(self, rank, peer, dr, np, result=None):
        self.rank, self.peer, self.dr, self.np = rank, peer, dr, np
        self.result = result
        self.calls = 0
        self.local = None

    def reduce_view(self, *partial):
        self.calls += 1
        self.local = partial
        if self.result is None:
            self.result = combine_shards(partial, self.peer, self.dr)
        else:
            # The earlier peer calculation used this rank's model and samples.
            compare_accumulators(partial[0], self.peer[0], self.np)
            self.np.testing.assert_allclose(self.np.asarray(partial[1]),
                                           self.np.asarray(self.peer[1]), atol=2e-6, rtol=2e-5)
            for actual, expected in zip(partial[3:], self.peer[3:]):
                assert (actual is None) == (expected is None)
                if actual is not None:
                    self.np.testing.assert_allclose(self.np.asarray(actual), self.np.asarray(expected),
                                                   atol=2e-7, rtol=2e-6)
        return self.result


def compare_sampler(first, second, np):
    assert first.views.keys() == second.views.keys()
    for view_id, state in first.views.items():
        np.testing.assert_array_equal(state["pixels"], second.views[view_id]["pixels"])
        np.testing.assert_allclose(state["mean_alpha"], second.views[view_id]["mean_alpha"],
                                   atol=2e-7, rtol=2e-6)


def check_partition():
    assert shard_pixel_range(323, 0, 2) == (0, 161)
    assert shard_pixel_range(323, 1, 2) == (161, 323)
    for count, world in ((1, 1), (3, 3), (323, 2), (323, 7)):
        ranges = [shard_pixel_range(count, rank, world) for rank in range(world)]
        assert ranges[0][0] == 0 and ranges[-1][1] == count
        assert all(a[1] == b[0] for a, b in zip(ranges, ranges[1:]))
        assert max(end - begin for begin, end in ranges) - min(end - begin for begin, end in ranges) <= 1
    for args in ((0, 0, 1), (1, 0, 2), (3, -1, 2), (3, 2, 2), (3, 0, 0), (True, 0, 1)):
        try:
            shard_pixel_range(*args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted an invalid partition: {args}")


def check_training(contour, chunk, mi, dr, np):
    config, payload, view, _ = fixture(np)
    config.render.width, config.render.height = 17, 19
    config.training.chunk_size = chunk
    count = config.render.width * config.render.height
    rgba = np.zeros((config.render.height, config.render.width, 4), dtype=np.float32)
    rgba[..., :3] = [.6, .25, .1]
    rgba[3:16, 3:14, 3] = 1.
    rgba[3:16, 3, 3] = .4
    rgba[3:16, 13, 3] = .7
    models = [restore_model(payload, mi) for _ in range(3)]
    optimizers = [FalcorOptimizer(model, copy.deepcopy(config.optimizer)) for model in models]
    samplers = [ContourSampler(np, enabled=contour) for _ in models]
    if contour:
        for sampler in samplers:
            sampler.sample(view["index"], rgba[..., 3], np.random.default_rng(9))
            sampler.views["0"]["mean_alpha"][:] = np.linspace(.1, .9, 16)
    rngs = [np.random.default_rng(42) for _ in models]
    cache = GPUReferenceCache(config.render.width, config.render.height, mi, dr, np, capacity_mib=1)
    reference = cache.get("generated-sharding-reference.png", rgba)
    replays = [NativeLossReplay(dr, enabled=chunk == 0) for _ in models]
    max_gradient_error = 0.
    try:
        for step in range(3):
            need_alpha = step != 1
            deletion = step != 1
            # Peer's sampling is performed on clones: real ranks below each
            # consume the original full-view RNG exactly once.
            peer_sampler = ContourSampler(np, enabled=contour,
                uniform_fraction=samplers[2].uniform_fraction, history_rate=samplers[2].history_rate)
            peer_sampler.load_state_dict(copy.deepcopy(samplers[2].state_dict()))
            peer_rng = copy.deepcopy(rngs[2])
            offsets, pdf, token = peer_sampler.sample(view["index"], rgba[..., 3], peer_rng)
            begin, end = shard_pixel_range(count, 1, 2)
            peer = compute_view_shard(models[2], view, rgba, config, offsets, pdf, token, mi, dr,
                pixel_begin=begin, pixel_end=end, need_rendered_alpha=need_alpha,
                collect_deletion_evidence=deletion, gpu_reference=reference,
                native_replay=replays[2], zero_alpha=True)
            root_collective = SequentialCollective(0, peer, dr, np)
            root = train_view(models[1], view, rgba, config, samplers[1], rngs[1], mi, dr, np,
                need_rendered_alpha=need_alpha, collect_deletion_evidence=deletion,
                gpu_reference=reference, native_replay=replays[1], collective=root_collective)
            peer_collective = SequentialCollective(1, peer, dr, np, root_collective.result)
            worker = train_view(models[2], view, rgba, config, samplers[2], rngs[2], mi, dr, np,
                need_rendered_alpha=need_alpha, collect_deletion_evidence=deletion,
                gpu_reference=reference, native_replay=replays[2], collective=peer_collective)
            full = train_view(models[0], view, rgba, config, samplers[0], rngs[0], mi, dr, np,
                need_rendered_alpha=need_alpha, collect_deletion_evidence=deletion,
                gpu_reference=reference, native_replay=replays[0])
            assert root_collective.calls == peer_collective.calls == 1
            for actual in (root, worker):
                max_gradient_error = max(max_gradient_error,
                                         compare_accumulators(full[0], actual[0], np))
                np.testing.assert_allclose(full[1], actual[1], atol=2e-7, rtol=2e-6)
                assert (full[2] is None) == (actual[2] is None)
                if need_alpha:
                    np.testing.assert_allclose(full[2], actual[2], atol=2e-7, rtol=2e-6)
            assert all(rng.bit_generator.state == rngs[0].bit_generator.state for rng in rngs[1:])
            for sampler in samplers[1:]:
                compare_sampler(samplers[0], sampler, np)
            for optimizer, model, result in zip(optimizers, models, (full, root, worker)):
                optimizer.step(model, result[0], sample_count=1, epoch=step)
            for name, value in models[0].snapshot().items():
                for model in models[1:]:
                    np.testing.assert_allclose(np.asarray(value), np.asarray(model.snapshot()[name]),
                                               atol=4e-6, rtol=3e-5, err_msg=f"step {step} parameter {name}")
            expected_state = optimizers[0].state_dict()["groups"]
            for optimizer in optimizers[1:]:
                actual_state = optimizer.state_dict()["groups"]
                for name, state in expected_state.items():
                    for key, value in state.items():
                        if key == "steps":
                            np.testing.assert_array_equal(value, actual_state[name][key])
                        else:
                            np.testing.assert_allclose(value, actual_state[name][key],
                                atol=4e-6, rtol=5e-5, err_msg=f"step {step} Adam {name}.{key}")
        return {"contour": contour, "chunk_size": chunk, "pixels": count,
                "updates": 3, "max_gradient_absolute_error": max_gradient_error,
                "status": "passed"}
    finally:
        for replay in replays:
            replay.clear()
        cache.clear()


def check_global_nonfinite(mi, dr, np):
    config, payload, view, rgba = fixture(np)
    model = restore_model(payload, mi)

    class OverflowCollective:
        rank, world_size = 0, 2

        def reduce_view(self, accumulator, loss, finite, raw, alpha):
            # Emulate a finite-input collective whose result overflows.
            accumulator["gradients"]["centers"] += mi.Float(float("inf"))
            return accumulator, loss, mi.Bool(True), raw, alpha

    try:
        train_view(model, view, rgba, config, ContourSampler(np), np.random.default_rng(42),
                   mi, dr, np, collective=OverflowCollective())
    except RuntimeError as error:
        assert "Nonfinite" in str(error)
    else:
        raise AssertionError("A nonfinite global gradient was accepted")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    check_partition()
    with tempfile.TemporaryDirectory(prefix="mitsuba-sharding-regression-") as directory:
        report = {}
        mi, dr, np, handles = initialize_runtime(RuntimeConfig(args.build_dir, args.gpu), Path(directory), report)
        checks = [check_training(contour, chunk, mi, dr, np)
                  for contour in (False, True) for chunk in (0, 37)]
        check_global_nonfinite(mi, dr, np)
        dr.sync_thread()
        print(json.dumps({"status": "passed", "scope": "single GPU sequential pixel shards",
                          "checks": checks, "global_nonfinite_rejected": True}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
