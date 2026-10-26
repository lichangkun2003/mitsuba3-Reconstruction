"""Real N-GPU NCCL and reconstruction equivalence test inside HPC Slurm.

Run this once in a one-task GPU allocation before production training.
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
        image = np.arange(max(7, context.world_size + 1), dtype=np.float32) * .125
        if step in (0, 2):
            begin, end = shard_pixel_range(len(image), context.rank, context.world_size)
            shard = np.zeros_like(image)
            shard[begin:end] = image[begin:end]
            raw, alpha = mi.Float(shard), mi.Float(shard * .5)
        local_finite = not (step == 1 and context.rank == context.world_size - 1)
        result, loss, finite, reduced_raw, reduced_alpha = collective.reduce_view(
            source, mi.Float(step + context.rank + 1), mi.Bool(local_finite), raw, alpha)
        for i, (name, dimension) in enumerate(zip(GRADIENT_KEYS, (27, 9, 3, 3, 3))):
            base = np.arange(count * dimension, dtype=np.float32) * .125 + step + i
            np.testing.assert_array_equal(np.asarray(result["gradients"][name]),
                                          base * (context.world_size * (context.world_size + 1) // 2))
        for i, name in enumerate(SUM_COUNTER_KEYS):
            np.testing.assert_array_equal(np.asarray(result[name]),
                (np.arange(count, dtype=np.uint32) + i + step) * (context.world_size * (context.world_size + 1) // 2))
        for i, name in enumerate(MAX_COUNTER_KEYS):
            expected = (np.ones(count, dtype=np.uint32) if context.world_size > 1
                        else (np.arange(count, dtype=np.uint32) + i) % 2)
            np.testing.assert_array_equal(np.asarray(result[name]), expected)
        np.testing.assert_array_equal(np.asarray(loss),
            [context.world_size * step + context.world_size * (context.world_size + 1) // 2])
        np.testing.assert_array_equal(np.asarray(finite), [step != 1])
        if raw is None:
            assert reduced_raw is None and reduced_alpha is None
        else:
            np.testing.assert_array_equal(np.asarray(reduced_raw), image)
            np.testing.assert_array_equal(np.asarray(reduced_alpha), image * .5)
    return {"known_reductions": "passed", "steps": 4,
            "checks": ["gradient_sum", "validity_sum", "evidence_max", "finite_min",
                       "scalar_loss_sum", "uneven_disjoint_images", "empty_topology"]}


def _known_batches(context, collective, mi, dr, np):
    from reconstruction_collective import GRADIENT_KEYS, SUM_COUNTER_KEYS, MAX_COUNTER_KEYS
    # Include a short final batch: its padding rank must contribute zero and
    # must not change the caller's actual-view normalization.
    for actual_views in sorted({1, context.world_size, max(1, context.world_size - 1)}):
        active = context.rank < actual_views
        local = context.rank + 1 if active else 0
        accumulator = {"gradients": {key: mi.Float([local, local * 2]) for key in GRADIENT_KEYS}}
        accumulator.update({key: mi.UInt32([int(active), 0]) for key in SUM_COUNTER_KEYS + MAX_COUNTER_KEYS})
        total = actual_views * (actual_views + 1) // 2
        result, loss, finite = collective.reduce_batch(accumulator, mi.Float(local), mi.Bool(True))
        for key in GRADIENT_KEYS:
            np.testing.assert_array_equal(np.asarray(result["gradients"][key]), [total, total * 2])
        np.testing.assert_array_equal(np.asarray(result["appearance_valid"]), [actual_views, 0])
        np.testing.assert_array_equal(np.asarray(loss), [total])
        assert bool(np.asarray(finite)[0])
    fields = {"gain": mi.Float([context.rank + .25, context.rank + .5]),
              "observed": mi.UInt32([context.rank + 1, context.rank + 2]),
              "empty": dr.empty(mi.Float, 0)}
    for source_rank in range(context.world_size):
        result = collective.broadcast_arrays(fields, root=source_rank, uint_keys=("observed",))
        np.testing.assert_array_equal(np.asarray(result["gain"]), [source_rank + .25, source_rank + .5])
        np.testing.assert_array_equal(np.asarray(result["observed"]), [source_rank + 1, source_rank + 2])
        assert dr.width(result["empty"]) == 0
    np.testing.assert_array_equal(np.asarray(fields["gain"]), [context.rank + .25, context.rank + .5])
    return {"view_batches": "passed", "checks": ["full_and_short_batch_sums",
            "validity_sum", "per_camera_root_broadcast", "source_buffers_preserved", "empty_arrays"]}


def _rgb_batches(context, collective, mi, dr, np):
    from reconstruction_collective import GRADIENT_KEYS, SUM_COUNTER_KEYS, MAX_COUNTER_KEYS
    # Compare new compact/dense paths against legacy dense SUM on real NCCL,
    # including a tail, an empty topology, and an all-rows-active fallback.
    width = max(64, context.world_size * 4)
    for dense, count, tail in ((False, width, False), (False, width, True),
                               (True, width, False), (False, 0, False)):
        context.heartbeat(f"rgb-batch-{dense}-{count}-{tail}", force=True)
        active = not tail or context.rank < max(1, context.world_size - 1)
        selected = (np.arange(count) if dense else np.asarray([context.rank + 1, 0])) if active else np.asarray([], np.int64)
        selected = selected[selected < count]
        host = {key: np.zeros(count, np.uint32) for key in SUM_COUNTER_KEYS + MAX_COUNTER_KEYS}
        host["appearance_valid"][selected] = context.rank + 1
        host["geometry_valid"][selected] = (context.rank + 1) * 2
        if count and context.rank == 0:
            host["appearance_valid"][-1] = 1  # Valid-zero row still required by Adam.
        value = {key: mi.UInt32(array) for key, array in host.items()}
        value["gradients"] = {}
        for key, dimension in zip(GRADIENT_KEYS, (27, 1, 3, 3, 3)):
            rows = np.zeros((count, dimension), np.float32)
            rows[selected] = np.arange(1, dimension + 1, dtype=np.float32) * (context.rank + 1) * .125
            if count and key == "centers" and context.rank == 0:
                rows[-2] = .5  # Unexpected nonzero invalid-row data is preserved too.
            value["gradients"][key] = mi.Float(rows.reshape(-1))
        value.update(rgb_loss_sum=mi.Float(context.rank + .125 if active else 0),
                     distortion_loss_sum=mi.Float(.125 if active else 0))
        local_loss = value["rgb_loss_sum"] + value["distortion_loss_sum"]
        reference, loss, finite, _, _ = collective.reduce_view(value, local_loss, mi.Bool(True))
        for mode, compact_output in (("auto", False), ("auto", True), ("dense", True)):
            actual, actual_loss, actual_finite = collective.reduce_batch(
                value, local_loss, mi.Bool(True), rgb_only=True, mode=mode, compact_output=compact_output)
            ids = np.asarray(actual["voxel_ids"]) if "voxel_ids" in actual else None
            if ids is not None:
                assert compact_output and mode == "auto"
                assert np.all(ids < count) and np.all(ids[1:] > ids[:-1])
            for key, dimension in zip(GRADIENT_KEYS, (27, 1, 3, 3, 3)):
                actual_values = np.asarray(actual["gradients"][key])
                if ids is not None:
                    # Expand only in this numerical oracle, never in training.
                    expanded = np.zeros((count, dimension), np.float32)
                    expanded[ids] = actual_values.reshape(len(ids), dimension)
                    actual_values = expanded.reshape(-1)
                np.testing.assert_allclose(actual_values, np.asarray(reference["gradients"][key]),
                                           rtol=2e-6, atol=2e-7)
            for key in SUM_COUNTER_KEYS + MAX_COUNTER_KEYS:
                actual_values = np.asarray(actual[key])
                if ids is not None:
                    expanded = np.zeros(count, np.uint32)
                    expanded[ids] = actual_values
                    actual_values = expanded
                np.testing.assert_array_equal(actual_values, np.asarray(reference[key]))
            for key in ("rgb_loss_sum", "distortion_loss_sum"):
                np.testing.assert_allclose(np.asarray(actual[key]), np.asarray(reference[key]), rtol=2e-6)
            np.testing.assert_allclose(np.asarray(actual_loss), np.asarray(loss), rtol=2e-6)
            np.testing.assert_array_equal(np.asarray(actual_finite), np.asarray(finite))
    for fault in ("nonfinite", "legacy_evidence"):
        value = {key: dr.zeros(mi.UInt32, width) for key in SUM_COUNTER_KEYS + MAX_COUNTER_KEYS}
        value["gradients"] = {key: dr.zeros(mi.Float, width * dimension)
                              for key, dimension in zip(GRADIENT_KEYS, (27, 1, 3, 3, 3))}
        local_finite = mi.Bool(True)
        if context.rank == context.world_size - 1:
            if fault == "nonfinite":
                bad = np.zeros(width, np.float32)
                bad[width - 1] = np.nan  # No valid counter, still must not disappear.
                value["gradients"]["opacity_sh"] = mi.Float(bad)
                local_finite &= dr.all(dr.isfinite(value["gradients"]["opacity_sh"]))
            else:
                value["background_evidence"] = dr.full(mi.UInt32, 1, width)
        for mode in ("auto", "dense"):
            reduced, loss, finite = collective.reduce_batch(value, mi.Float(2.), local_finite,
                                                            rgb_only=True, mode=mode, compact_output=True)
            # A JIT consumer reads the external NCCL result before any host
            # conversion; one bad peer rejects the complete batch on all ranks.
            accepted = finite & dr.all(dr.isfinite(reduced["gradients"]["opacity_sh"]))
            marker = dr.select(accepted, loss, -1.)
            dr.eval(marker)
            np.testing.assert_array_equal(np.asarray(marker), [-1.])
    return {"rgb_batches": "passed", "checks": ["compact_vs_dense", "dense_union_fallback",
        "tail_idle_rank", "empty_topology", "valid_zero_rows", "unexpected_nonzero_invalid_rows",
        "loss_components", "direct_compact_output", "async_consumer_ordering", "peer_nonfinite_guard", "legacy_evidence_guard"]}


def _large_dense_batches(context, collective, mi, dr, np):
    """Exercise production-sized NCCL messages, returning only eight flags.

    A tiny generated scene cannot cover NCCL's large-message protocol. Every
    gradient here has a rank-specific, opaque known value. Check all elements
    on the GPU, retain only one scalar result per batch, and read them together
    at the end. Do not retain whole output tables across batches: each transfer
    owns approximately 312 MiB of source/destination storage before temporary
    reductions and allocator rounding, keeping live allocations below 1 GiB.
    """
    from reconstruction_collective import GRADIENT_KEYS, MAX_COUNTER_KEYS

    count, batches = 1_048_576, 8
    dimensions = (27, 1, 3, 3, 3)
    rank_factor = context.rank + 1
    sum_factor = context.world_size * (context.world_size + 1) // 2

    def submit(step):
        source = {"gradients": {
            name: dr.opaque(mi.Float, (step + 1) * (index + 1) * .125 * rank_factor,
                            shape=count * dimension)
            for index, (name, dimension) in enumerate(zip(GRADIENT_KEYS, dimensions))}}
        source.update(appearance_valid=dr.opaque(mi.UInt32, rank_factor, shape=count),
                      geometry_valid=dr.opaque(mi.UInt32, 2 * rank_factor, shape=count),
                      background_carve_valid=dr.zeros(mi.UInt32, count),
                      rgb_loss_sum=dr.opaque(mi.Float, 2. * rank_factor),
                      distortion_loss_sum=dr.opaque(mi.Float, 10. * rank_factor))
        source.update({name: dr.zeros(mi.UInt32, count) for name in MAX_COUNTER_KEYS})
        reduced, loss, finite = collective.reduce_batch(
            source, dr.opaque(mi.Float, 3. * rank_factor), mi.Bool(True),
            rgb_only=True, mode="dense")
        valid = finite & (loss == 3. * sum_factor)
        for index, name in enumerate(GRADIENT_KEYS):
            expected = (step + 1) * (index + 1) * .125 * sum_factor
            valid &= dr.all(reduced["gradients"][name] == expected)
        valid &= dr.all(reduced["appearance_valid"] == sum_factor)
        valid &= dr.all(reduced["geometry_valid"] == 2 * sum_factor)
        valid &= dr.all(reduced["background_carve_valid"] == 0)
        for name in MAX_COUNTER_KEYS:
            valid &= dr.all(reduced[name] == 0)
        valid &= (reduced["rgb_loss_sum"] == 2. * sum_factor)
        valid &= (reduced["distortion_loss_sum"] == 10. * sum_factor)
        dr.eval(valid)
        return mi.UInt32(valid)

    flags = []
    for step in range(batches):
        context.heartbeat(f"large-dense-batch-{step + 1}", force=True)
        flags.append(submit(step))
    np.testing.assert_array_equal(np.asarray(dr.concat(flags)), np.ones(batches, np.uint32),
                                  err_msg="Large dense NCCL transfer or following GPU consumer is incorrect")
    return {"large_dense_batches": "passed", "batches": batches, "voxels": count,
            "gradient_dimensions": list(dimensions),
            "largest_gradient_elements": count * max(dimensions),
            "gradient_payload_bytes_per_rank_per_batch": count * sum(dimensions) * 4,
            "host_readback_uint32_values": batches,
            "checks": ["opaque_rank_specific_inputs", "all_elements_verified_on_gpu",
                       "continuous_large_messages", "single_final_scalar_readback"]}


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


def _continuous_rgb_native(context, collective, directory, mi, dr, np):
    """Keep native/NCCL/Adam consumers queued across many complete batches.

    In particular, do not convert any device array to NumPy inside the batch
    loop: doing so would mask missing external-write dependencies. The legacy
    reduction is an explicitly synchronized oracle, run in a separate pass so
    its waits cannot accidentally synchronize the dense/auto pass under test.
    """
    from reconstruction_batch_epoch import _finite_accumulator
    from reconstruction_checks import DeferredViewChecks
    from reconstruction_common import UniformSampler, restore_model
    from reconstruction_optimizer import FalcorOptimizer
    from reconstruction_reference import GPUReferenceCache
    from reconstruction_replay import NativeLossReplay
    from reconstruction_view_batch import zero_view_partial
    from train import compute_view_shard
    from test_reconstruction_performance import fixture

    config, payload, original_view, rgba = fixture(np)
    config.training.rgb_only = True
    config.model.opacity_sh_count = 1
    config.gradient.distortion_loss_weight = .1
    payload["model_metadata"]["opacity_sh_count"] = 1
    payload["model_state"]["opacity_sh"] = payload["model_state"]["opacity_sh"][:1].copy()
    # Preserve the original visible cell and its world-space size, then add
    # off-camera cells. A one-voxel fixture always chooses dense fallback and
    # would never exercise direct compact NCCL output followed by native Adam.
    payload["model_metadata"].update(resolution=8, grid_max=[8., 8., 8.])
    cells = np.array([0] + [63 + 64 * z for z in range(8)], np.uint32)
    for name, value in tuple(payload["model_state"].items()):
        payload["model_state"][name] = cells if name == "cell_indices" else np.tile(value, len(cells))
    # Keep the generated geometry near its initial shape over this long test;
    # the test targets ordering/transport, not convergence or topology changes.
    for name in ("lr_radiance", "lr_opacity", "lr_center", "lr_log_scale", "lr_rotation"):
        setattr(config.optimizer, name, getattr(config.optimizer, name) * .1)
    reference_rgb = rgba[..., :3].copy() + np.float32(context.rank * .01)
    cache = GPUReferenceCache(16, 16, mi, dr, np, capacity_mib=1, rgb_only=True)
    reference = cache.get(Path(directory) / "continuous-generated-reference.png", reference_rgb)
    batch_count, ray_count = 64, 16 * 16
    expected_views = sum(1 if step % 7 == 6 else context.world_size for step in range(batch_count))
    results = {}
    for mode in ("legacy", "dense", "auto"):
        context.heartbeat(f"continuous-native-{mode}", force=True)
        model = restore_model(copy.deepcopy(payload), mi)
        optimizer = FalcorOptimizer(model, copy.deepcopy(config.optimizer))
        replay = NativeLossReplay(dr)
        sampler, rng = UniformSampler(np), np.random.default_rng(9100 + context.rank)
        checks = DeferredViewChecks(mi, dr, np)
        direct_batches = 0
        for step in range(batch_count):
            # Repeated partial batches exercise idle ranks and both exact Adam
            # normalization factors, including switching back to full batches.
            actual_views = 1 if step % 7 == 6 else context.world_size
            if context.rank < actual_views:
                view = copy.deepcopy(original_view)
                view["index"] = context.rank
                view["camera"]["position"][0] += context.rank * .015 + (step % 5 - 2) * .005
                offsets, inverse_pdf, token = sampler.sample(view["index"], reference_rgb, rng)
                accumulator, loss, finite, _, _ = compute_view_shard(
                    model, view, reference_rgb, config, offsets, inverse_pdf, token, mi, dr,
                    collect_deletion_evidence=False, collect_growth_evidence=False,
                    gpu_reference=reference, native_replay=replay)
            else:
                accumulator, loss, finite, _, _ = zero_view_partial(model, mi, dr)
                accumulator.update(rgb_loss_sum=mi.Float(0), distortion_loss_sum=mi.Float(0))
            finite = _finite_accumulator(accumulator, loss, finite, dr)
            if mode == "legacy":
                reduced, loss, finite, _, _ = collective.reduce_view(accumulator, loss, finite)
            else:
                reduced, loss, finite = collective.reduce_batch(
                    accumulator, loss, finite, rgb_only=True, mode=mode, compact_output=True)
                direct_batches += int("voxel_ids" in reduced)
            finite = _finite_accumulator(reduced, loss, finite, dr)
            divisor = actual_views * ray_count
            checks.add(step, dr.concat([loss / divisor, mi.Float(finite),
                reduced["rgb_loss_sum"] / divisor, reduced["distortion_loss_sum"] / divisor]))
            update = optimizer.step(model, reduced, sample_count=actual_views,
                                    epoch=0, coarse_to_fine=False, update_allowed=finite)
            checks.updated(update["accepted"])
            # Deliberately no checks.read(), np.asarray(), scalar indexing,
            # snapshot, or explicit stream wait at this batch boundary.
            if step % 16 == 15:
                context.heartbeat(f"continuous-native-{mode}-batch-{step + 1}")

        measurements = checks.read(rgb_only=True, distortion_weight=.1)
        assert len(measurements) == batch_count
        parameters = {name: np.array(value) for name, value in model.snapshot().items()}
        state = optimizer.state_dict()["groups"]
        assert optimizer.statistics()["calls"] == batch_count
        assert optimizer.statistics()["replays"] > 0
        assert replay.statistics()["replays"] > 0
        if mode == "auto" and context.world_size > 1:
            assert direct_batches == batch_count, "Continuous test did not exercise direct compact Adam"
        assert any(np.any(group["steps"] > 1) for group in state.values()), "No actual Adam updates occurred"
        for name, value in parameters.items():
            assert np.isfinite(value).all(), f"{mode}: nonfinite final parameter {name}"
        for group, fields in state.items():
            for name, value in fields.items():
                assert np.isfinite(value).all(), f"{mode}: nonfinite Adam {group}.{name}"
        # Test real rank agreement only after the entire unsynchronized pass.
        root = context.broadcast((parameters, state) if context.rank == 0 else None)
        for name, value in parameters.items():
            np.testing.assert_array_equal(value, root[0][name], err_msg=f"{mode} replica {name}")
        for group, fields in state.items():
            for name, value in fields.items():
                np.testing.assert_array_equal(value, root[1][group][name],
                                              err_msg=f"{mode} replica Adam {group}.{name}")
        results[mode] = (parameters, state, measurements)
        replay.clear()
        optimizer.clear_replay()

    maximum_parameter_error = 0.
    for mode in ("dense", "auto"):
        parameters, state, measurements = results[mode]
        for name, value in parameters.items():
            expected = results["legacy"][0][name]
            if name == "cell_indices":
                np.testing.assert_array_equal(value, expected)
            else:
                maximum_parameter_error = max(maximum_parameter_error, float(np.max(np.abs(value - expected))))
                np.testing.assert_allclose(value, expected, rtol=2e-4, atol=2e-6,
                                           err_msg=f"{mode} continuous parameter {name}")
        for group, fields in state.items():
            for name, value in fields.items():
                expected = results["legacy"][1][group][name]
                if name == "steps":
                    np.testing.assert_array_equal(value, expected)
                else:
                    np.testing.assert_allclose(value, expected, rtol=3e-4, atol=2e-6,
                                               err_msg=f"{mode} continuous Adam {group}.{name}")
        for step, (actual, expected) in enumerate(zip(measurements, results["legacy"][2])):
            for name in ("loss", "rgb_loss", "distortion_loss"):
                np.testing.assert_allclose(actual[name], expected[name], rtol=2e-4, atol=2e-6,
                                           err_msg=f"{mode} continuous batch {step} {name}")
    cache.clear()
    return {"continuous_rgb_native": "passed", "batches_per_mode": batch_count,
            "views_per_mode": expected_views, "modes": ["legacy", "dense", "auto"],
            "max_parameter_absolute_error": maximum_parameter_error,
            "checks": ["no_per_batch_numpy_readback", "native_replay_to_nccl_to_frozen_adam",
                       "direct_compact_to_adam",
                       "repeated_idle_tail_ranks", "deferred_finite_guard", "legacy_synchronized_parity",
                       "Adam_parameters_moments_and_clocks", "replica_bit_equality"]}


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
                      _known_batches(context, collective, mi, dr, np),
                      _rgb_batches(context, collective, mi, dr, np),
                      _large_dense_batches(context, collective, mi, dr, np),
                      _compare_native(context, collective, directory, mi, dr, np),
                      _continuous_rgb_native(context, collective, directory, mi, dr, np)]
            context.barrier()
            print(json.dumps({"status": "passed", "rank": context.rank,
                              "checks": checks, "collective": collective.statistics()}), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--build-dir")
    parser.add_argument("--nccl-library")
    parser.add_argument("--worker-timeout", type=float, default=600.)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)
    allocation = validate_allocation(workers=args.workers)
    arguments = []
    for name in ("build_dir", "nccl_library"):
        if getattr(args, name):
            arguments.extend(("--" + name.replace("_", "-"), getattr(args, name)))
    return run_workers(arguments, allocation, worker_timeout=args.worker_timeout, worker_main=_worker_main)


if __name__ == "__main__":
    raise SystemExit(main())
