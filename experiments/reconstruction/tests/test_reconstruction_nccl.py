"""Real two-GPU NCCL and reconstruction equivalence test inside HPC Slurm.

Run this once in a one-task, two-GPU allocation before production training.
The parent imports no CUDA framework. Workers use generated arrays/scenes and
temporary directories; no dataset or reconstruction experiment is created.
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

from train_multigpu import run_workers, validate_allocation


def _known_reductions(context, collective, mi, dr, np):
    from reconstruction_collective import GRADIENT_KEYS, SUM_COUNTER_KEYS, MAX_COUNTER_KEYS
    from train import shard_pixel_range
    for step in range(4):
        context.heartbeat(f"known-reduction-{step}", force=True)
        count = 0 if step == 3 else 3
        source = {"gradients": {}}
        for i, (name, dimension) in enumerate(zip(GRADIENT_KEYS, (27, 9, 3, 3, 3))):
            base = np.arange(count * dimension, dtype=np.float32) * .125 + step + i
            source["gradients"][name] = mi.Float(base * (context.rank + 1))
        for i, name in enumerate(SUM_COUNTER_KEYS):
            source[name] = mi.UInt32((np.arange(count, dtype=np.uint32) + i + step) * (context.rank + 1))
        for i, name in enumerate(MAX_COUNTER_KEYS):
            source[name] = mi.UInt32((np.arange(count, dtype=np.uint32) + i + context.rank) % 2)
        raw, alpha = None, None
        image = np.arange(7, dtype=np.float32) * .125
        if step in (0, 2):
            begin, end = shard_pixel_range(len(image), context.rank, context.world_size)
            shard = np.zeros_like(image)
            shard[begin:end] = image[begin:end]
            raw, alpha = mi.Float(shard), mi.Float(shard * .5)
        local_finite = not (step == 1 and context.rank == 1)
        result, loss, finite, reduced_raw, reduced_alpha = collective.reduce_view(
            source, mi.Float(step + context.rank + 1), mi.Bool(local_finite), raw, alpha)
        for i, (name, dimension) in enumerate(zip(GRADIENT_KEYS, (27, 9, 3, 3, 3))):
            base = np.arange(count * dimension, dtype=np.float32) * .125 + step + i
            np.testing.assert_array_equal(np.asarray(result["gradients"][name]), base * 3)
        for i, name in enumerate(SUM_COUNTER_KEYS):
            np.testing.assert_array_equal(np.asarray(result[name]), (np.arange(count, dtype=np.uint32) + i + step) * 3)
        for name in MAX_COUNTER_KEYS:
            np.testing.assert_array_equal(np.asarray(result[name]), np.ones(count, dtype=np.uint32))
        np.testing.assert_array_equal(np.asarray(loss), [2 * step + 3])
        np.testing.assert_array_equal(np.asarray(finite), [step != 1])
        if raw is None:
            assert reduced_raw is None and reduced_alpha is None
        else:
            np.testing.assert_array_equal(np.asarray(reduced_raw), image)
            np.testing.assert_array_equal(np.asarray(reduced_alpha), image * .5)
    return {"known_reductions": "passed", "steps": 4,
            "checks": ["gradient_sum", "validity_sum", "evidence_max", "finite_min",
                       "scalar_loss_sum", "uneven_disjoint_images", "empty_topology"]}


def _compare_native(context, collective, directory, mi, dr, np):
    from reconstruction_common import ContourSampler, restore_model
    from reconstruction_optimizer import FalcorOptimizer
    from reconstruction_reference import GPUReferenceCache
    from reconstruction_replay import NativeLossReplay
    from train import train_view
    from test_reconstruction_performance import compare_accumulators, fixture

    maximum_gradient_error = 0.
    maximum_parameter_error = 0.
    for contour in (False, True):
        config, payload, view, rgba = fixture(np)
        config.render.width = config.render.height = 15  # 112/113 pixels per rank.
        config.training.chunk_size = 37  # Exercise shard-tail chunks as well.
        rgba = rgba[:15, :15].copy()
        models = [restore_model(payload, mi), restore_model(payload, mi)]
        optimizers = [FalcorOptimizer(model, copy.deepcopy(config.optimizer)) for model in models]
        samplers = [ContourSampler(np, enabled=contour), ContourSampler(np, enabled=contour)]
        rngs = [np.random.default_rng(42), np.random.default_rng(42)]
        replays = [NativeLossReplay(dr), NativeLossReplay(dr)]
        cache = GPUReferenceCache(15, 15, mi, dr, np, capacity_mib=1)
        reference = cache.get(Path(directory) / f"generated-reference-{contour}.png", rgba)
        if contour:
            for sampler in samplers:
                sampler.sample(view["index"], rgba[..., 3], np.random.default_rng(9))
                sampler.views[str(view["index"])]["mean_alpha"][:] = np.linspace(.1, .9, 16)
        for step in range(3):
            context.heartbeat(f"native-contour-{contour}-step-{step}", force=True)
            camera_view = copy.deepcopy(view)
            camera_view["camera"]["position"][0] += step * .01
            results = []
            for index in (0, 1):
                results.append(train_view(models[index], camera_view, rgba, config, samplers[index],
                    rngs[index], mi, dr, np, need_rendered_alpha=True,
                    collect_deletion_evidence=True, gpu_reference=reference,
                    native_replay=replays[index], collective=collective if index else None))
            maximum_gradient_error = max(maximum_gradient_error,
                compare_accumulators(results[0][0], results[1][0], np))
            np.testing.assert_allclose(results[0][1], results[1][1], atol=2e-7, rtol=2e-6)
            np.testing.assert_allclose(results[0][2], results[1][2], atol=2e-6, rtol=2e-5)
            assert rngs[0].bit_generator.state == rngs[1].bit_generator.state
            if contour:
                np.testing.assert_array_equal(samplers[0].views["0"]["pixels"], samplers[1].views["0"]["pixels"])
                np.testing.assert_allclose(samplers[0].views["0"]["mean_alpha"],
                    samplers[1].views["0"]["mean_alpha"], atol=2e-6, rtol=2e-5)
            for optimizer, model, result in zip(optimizers, models, results):
                optimizer.step(model, result[0], sample_count=1, epoch=step)
            for name, value in models[0].snapshot().items():
                first, second = np.asarray(value), np.asarray(models[1].snapshot()[name])
                if name == "cell_indices":
                    np.testing.assert_array_equal(first, second)
                else:
                    maximum_parameter_error = max(maximum_parameter_error,
                        float(np.max(np.abs(first - second))))
                    np.testing.assert_allclose(first, second, atol=2e-6, rtol=2e-5, err_msg=name)
            first_state, second_state = [optimizer.state_dict()["groups"] for optimizer in optimizers]
            for group, states in first_state.items():
                for name, value in states.items():
                    if name == "steps":
                        np.testing.assert_array_equal(value, second_state[group][name])
                    else:
                        np.testing.assert_allclose(value, second_state[group][name], atol=2e-6, rtol=3e-5,
                                                   err_msg=f"Adam {group}.{name}")
        # AllReduce gives both replicas the same global gradients. Verify that
        # the independent rank optimizers finish with identical parameter bits.
        final_state = {name: np.asarray(value) for name, value in models[1].snapshot().items()}
        root_state = context.broadcast(final_state if context.rank == 0 else None)
        for name, value in final_state.items():
            np.testing.assert_array_equal(value, root_state[name], err_msg=f"replica {name}")
        for replay in replays:
            replay.clear()
        cache.clear()
    return {"native_training": "passed", "steps_per_sampling_mode": 3,
            "sampling_modes": ["uniform", "contour"], "pixels": 225, "chunk_size": 37,
            "max_gradient_absolute_error": maximum_gradient_error,
            "max_parameter_absolute_error": maximum_parameter_error,
            "checks": ["loss", "gradients", "validity_and_evidence", "alpha", "RNG",
                       "contour_history", "Adam_parameters_and_moments", "replica_bit_equality"]}


def _worker_main(context, arguments):
    # Importing these inside each spawned worker keeps CUDA out of the parent.
    from reconstruction_collective import NcclCollective
    from reconstruction_common import initialize_runtime
    from preview import RuntimeConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--nccl-library")
    args = parser.parse_args(arguments)
    with tempfile.TemporaryDirectory(prefix=f"mitsuba-nccl-rank-{context.rank}-") as directory:
        mi, dr, np, handles = initialize_runtime(RuntimeConfig(build_dir=args.build_dir),
            Path(directory), {}, cuda_device=context.rank, cpu_threads=context.cpus_per_rank)
        context.heartbeat("NCCL-initialize", force=True)
        with NcclCollective(context.rank, context.world_size, context.connection, mi, dr, np,
                            library=args.nccl_library) as collective:
            checks = [_known_reductions(context, collective, mi, dr, np),
                      _compare_native(context, collective, directory, mi, dr, np)]
            context.barrier()
            print(json.dumps({"status": "passed", "rank": context.rank,
                              "checks": checks, "collective": collective.statistics()}), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--build-dir")
    parser.add_argument("--nccl-library")
    parser.add_argument("--worker-timeout", type=float, default=600.)
    args = parser.parse_args(argv)
    allocation = validate_allocation()
    arguments = []
    for name in ("build_dir", "nccl_library"):
        if getattr(args, name):
            arguments.extend(("--" + name.replace("_", "-"), getattr(args, name)))
    return run_workers(arguments, allocation, worker_timeout=args.worker_timeout, worker_main=_worker_main)


if __name__ == "__main__":
    raise SystemExit(main())
