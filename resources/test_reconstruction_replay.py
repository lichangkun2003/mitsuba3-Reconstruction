"""Generated CUDA regression for native replay, topology, and Adam updates.

Uses a sparse 4^3 scene and 16x16 rays, without a dataset or output experiment.
All temporary runtime artifacts are removed. Invoke with the source environment.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reconstruction_common import ContourSampler, generate_ray_batch, initialize_runtime, prepare_camera, restore_model
from reconstruction_optimizer import FalcorOptimizer
from reconstruction_preview import RuntimeConfig
from reconstruction_reference import GPUReferenceCache
from reconstruction_replay import NativeLossReplay
from reconstruction_train import TrainConfig, settings, train_view


def fixture(np):
    config = TrainConfig()
    config.render.width = config.render.height = 16
    config.optimizer.opacity_warmup_epochs = 0
    config.optimizer.opacity_ramp_epochs = 0
    coordinates = [(x, y, z) for z in range(4) for y in (1, 2) for x in (1, 2)]
    n, y00 = len(coordinates), 1 / (2 * math.sqrt(math.pi))
    state = {
        "cell_indices": np.array([x + 4 * (y + 4 * z) for x, y, z in coordinates], np.uint32),
        "centers": np.tile(np.array([.48, .52, .50], np.float32), n),
        "log_scales": np.tile(np.log(np.array([.105, .087, .095], np.float32)), n),
        "rotations": np.tile(np.array([1., 0., 0., 0.], np.float32), n),
        "radiance_sh": np.tile(np.linspace(-.015, .015, 27, dtype=np.float32), n),
        "opacity_sh": np.tile(np.linspace(-.01, .01, 9, dtype=np.float32), n),
    }
    state["radiance_sh"].reshape(n, 27)[:, :3] = [.2 / y00, .3 / y00, .4 / y00]
    state["opacity_sh"].reshape(n, 9)[:, 0] = -.4 / y00
    payload = dict(model_state=state, model_metadata=dict(resolution=4,
        grid_min=[0., 0., 0.], grid_max=[1., 1., 1.], sh_count=9, opacity_sh_count=9))
    camera = dict(position=[.5, .5, -1.5], forward=[0., 0., 1.],
                  right=[1., 0., 0.], up=[0., 1., 0.], camera_angle_x=.55)
    rgba = np.zeros((16, 16, 4), np.float32)
    rgba[..., :3] = [.6, .25, .1]
    rgba[3:13, 3:13, 3] = 1.
    rgba[3:13, 3, 3] = .4
    return config, payload, camera, rgba


def run_checks(mi, dr, np):
    config, payload, camera, rgba = fixture(np)
    options = settings(config) | {"collect_deletion_evidence": False}
    replay = NativeLossReplay(dr)
    checks = []
    offsets = np.random.default_rng(194).random((256, 2)).astype(np.float32)
    inverse_pdf = np.linspace(.8, 1.2, 256, dtype=np.float32)

    def inputs_for(current_camera=camera, reference=rgba, begin=0, end=256):
        ray = generate_ray_batch(config.render, current_camera,
            np.arange(begin, end, dtype=np.uint32), offsets[begin:end], mi, dr,
            prepared_camera=prepare_camera(config.render, current_camera, mi, dr))
        ref = reference.reshape(-1, 4)[begin:end]
        # Leave rays lazy and background literal, like real training inputs.
        return (ray, mi.Color3f(*(mi.Float(ref[:, c].copy()) for c in range(3))),
                mi.Float(ref[:, 3].copy()), mi.Color3f(1), mi.Float(inverse_pdf[begin:end]))

    def flatten(result):
        for name, value in result.items():
            if name == "stage_profile":
                continue
            if isinstance(value, dict):
                for field, item in flatten(value):
                    yield name + "." + field, item
            else:
                yield name, value

    def host(result):
        dr.eval(*(value for _, value in flatten(result)))
        dr.sync_thread()
        return {name: np.asarray(value).copy() for name, value in flatten(result)}

    def compare(name, expected, actual):
        assert expected.keys() == actual.keys(), name
        maximum = 0.
        for field, a in expected.items():
            b = actual[field]
            assert a.shape == b.shape, (name, field, a.shape, b.shape)
            assert np.isfinite(a).all() and np.isfinite(b).all(), (name, field)
            if np.issubdtype(a.dtype, np.integer):
                np.testing.assert_array_equal(a, b, err_msg=name + ":" + field)
            else:
                np.testing.assert_allclose(a, b, rtol=3e-5, atol=1e-5,
                                           err_msg=name + ":" + field)
                if a.size:
                    maximum = max(maximum, float(np.max(np.abs(a.astype(float) - b))))
        item = {"check": name, "max_absolute_error": maximum, "status": "passed"}
        checks.append(item)
        print(json.dumps(item), flush=True)

    def verify(name, model, inputs, current_options=options, wrapper=replay, changed_from=None):
        expected = host(model.loss_and_grad(*inputs, current_options))
        actual = host(wrapper(model, *inputs, current_options))
        compare(name, expected, actual)
        if changed_from is not None:
            assert any(not np.array_equal(expected[k], changed_from[k])
                       for k in ("rgb", "alpha", "loss")), name + " fixture did not change"
        return actual

    model = restore_model(payload, mi)
    inputs = inputs_for()
    baseline = verify("record", model, inputs)
    assert int(np.max(baseline["hit_count"])) >= 2
    assert np.any(baseline["geometry_valid"] > 0)
    before = replay.statistics()["recordings"]
    verify("repeat", model, inputs)
    assert replay.statistics()["recordings"] == before
    assert replay.statistics()["replays"] == 1

    camera2 = copy.deepcopy(camera)
    camera2["position"][0] += .12
    changed_rays = inputs_for(camera2)
    verify("changed_camera", model, changed_rays, changed_from=baseline)
    rgba2 = rgba.copy()
    rgba2[..., :3] = [.1, .7, .45]
    rgba2[..., 3] = np.flip(rgba2[..., 3], axis=1)
    changed_reference = inputs_for(reference=rgba2)
    verify("changed_reference", model, changed_reference, changed_from=baseline)
    assert replay.statistics()["recordings"] == before

    replacement = {k: v.copy() for k, v in payload["model_state"].items() if k != "cell_indices"}
    replacement["centers"][::3] += .08
    replacement["opacity_sh"][::9] += .7
    replacement["radiance_sh"].reshape(-1, 27)[:, :3] *= [.7, 1.3, .8]
    model.set_parameters({k: mi.Float(v) for k, v in replacement.items()})
    updated = verify("replaced_parameter_handles", model, inputs, changed_from=baseline)
    assert replay.statistics()["recordings"] == before

    moved = copy.deepcopy(payload)
    moved["model_state"].update(replacement)
    moved["model_state"]["cell_indices"] -= np.uint32(1)
    moved_model = restore_model(moved, mi)
    verify("same_count_changed_occupancy", moved_model, inputs, changed_from=updated)
    assert replay.statistics()["model_changes"] == 1
    assert replay.statistics()["entries"] == 1

    # All non-array native settings must produce a distinct recording. Keep
    # deletion counters and contour/growth alpha arrays available to consumers.
    deletion_options = options | {"collect_deletion_evidence": True}
    deletion = verify("deletion_enabled", moved_model, inputs, deletion_options)
    assert "foreground_evidence" in deletion and "background_evidence" in deletion
    assert deletion["raw_alpha"].size == 256 and deletion["alpha"].size == 256
    verify("changed_capacities", moved_model, inputs,
           options | {"max_hits": 3, "max_candidates": 5})
    assert replay.statistics()["entries"] == 3

    # Diagnostic code synchronizes and reads timer data: it must never execute
    # inside freeze, even after a cached non-diagnostic call exists.
    recorded = replay.statistics()["recordings"]
    diagnostic = replay(moved_model, *inputs, deletion_options | {"profile_stages": True})
    assert set(diagnostic["stage_profile"]) == {"forward", "hit_backward", "candidate_backward"}
    compare("profile_bypass", deletion, host(diagnostic))
    assert replay.statistics()["recordings"] == recorded
    assert replay.statistics()["bypasses"] == 1
    disabled = NativeLossReplay(dr, enabled=False)
    verify("disabled_bypass", moved_model, inputs, wrapper=disabled)
    assert disabled.statistics()["recordings"] == 0 and disabled.statistics()["bypasses"] == 1

    # KernelHistory changes native JIT flags. Do not assume it shares a frozen
    # layout; validate correctness and that repeated same-layout calls replay.
    for index in range(2):
        with dr.kernel_history() as history:
            captured = replay(moved_model, *inputs, options)
            dr.eval(captured)
            dr.sync_thread()
        assert list(history), "Replay produced no kernel history"
        compare("kernel_history_" + str(index), host(moved_model.loss_and_grad(*inputs, options)), host(captured))

    # Small final batches must not reuse a literal ray stride from full frames.
    chunks = NativeLossReplay(dr)
    pieces = []
    for begin in range(0, 256, 37):
        end = min(begin + 37, 256)
        piece_inputs = inputs_for(begin=begin, end=end)
        pieces.append(verify(f"chunk_{begin}_{end}", model, piece_inputs, wrapper=chunks))
    assert chunks.statistics()["entries"] == 2  # 37 rays and 34-ray tail
    assert chunks.statistics()["recordings"] == 2
    full = host(model.loss_and_grad(*inputs, options))
    composed = {}
    ray_fields = {"rgb", "raw_rgb", "alpha", "raw_alpha", "hit_count", "loss"}
    for field in full:
        arrays = [piece[field] for piece in pieces]
        if field in ray_fields:
            composed[field] = np.concatenate(arrays, axis=-1)
        elif field.endswith("_evidence"):
            composed[field] = np.maximum.reduce(arrays)
        else:
            composed[field] = np.add.reduce(arrays)
    compare("full_vs_replayed_chunks", full, composed)

    # Exercise the actual consumer of raw/weighted alpha. Contour sampling must
    # keep its proposal history and PDF compensation when the native call replays.
    view = {"index": 0, "camera": camera}
    for chunk in (0, 37):
        current_config = copy.deepcopy(config)
        current_config.training.chunk_size = chunk
        results, samplers, rng_states = [], [], []
        for enabled in (False, True):
            contour = ContourSampler(np, enabled=True)
            contour.sample(0, rgba[..., 3], np.random.default_rng(9))
            contour.views["0"]["mean_alpha"][:] = np.linspace(.1, .9, 16)
            current_model = restore_model(payload, mi)
            current_replay = NativeLossReplay(dr, enabled=enabled)
            references = GPUReferenceCache(16, 16, mi, dr, np, capacity_mib=int(enabled))
            rng = np.random.default_rng(42)
            for step in range(2):
                result = train_view(current_model, view, rgba, current_config, contour,
                    rng, mi, dr, np, need_rendered_alpha=True,
                    collect_deletion_evidence=True, native_replay=current_replay,
                    gpu_reference=references.get("generated-contour.png", rgba))
            if enabled:
                assert current_replay.statistics()["replays"] > 0
                assert references.statistics()["uploads"] == references.statistics()["hits"] == 1
            results.append(result)
            samplers.append(contour)
            rng_states.append(copy.deepcopy(rng.bit_generator.state))
        compare("contour_accumulator_chunk_" + str(chunk), host(results[0][0]), host(results[1][0]))
        np.testing.assert_allclose(results[0][1], results[1][1], atol=2e-7, rtol=2e-6)
        np.testing.assert_allclose(results[0][2], results[1][2], atol=2e-7, rtol=2e-6)
        np.testing.assert_allclose(samplers[0].views["0"]["mean_alpha"],
                                   samplers[1].views["0"]["mean_alpha"], atol=2e-7)
        assert rng_states[0] == rng_states[1], "Replay/cache changed training RNG consumption"

    # Adam updates all parameter groups; every replay must see replaced handles
    # and reproduce both model parameters and moment state over several views.
    models = [restore_model(payload, mi), restore_model(payload, mi)]
    optimizers = [FalcorOptimizer(m, copy.deepcopy(config.optimizer)) for m in models]
    adam_replay = NativeLossReplay(dr)
    for step, current in enumerate((inputs, changed_rays, changed_reference), 1):
        results = [models[0].loss_and_grad(*current, options), adam_replay(models[1], *current, options)]
        compare("adam_native_" + str(step), host(results[0]), host(results[1]))
        for m, optimizer, result in zip(models, optimizers, results):
            optimizer.step(m, result, sample_count=1, epoch=1, coarse_to_fine=True)
        compare("adam_model_" + str(step), host(models[0].snapshot()), host(models[1].snapshot()))
        compare("adam_moments_" + str(step), host(optimizers[0].state), host(optimizers[1].state))
    assert adam_replay.statistics()["recordings"] == 1

    # Growth/deletion/refinement rebuild native models. Clear must release
    # previous recordings even when the next topology happens to have equal size.
    replay.clear()
    assert replay.statistics()["entries"] == replay.statistics()["cached_recordings"] == 0
    for name, current in (("reduced_topology", copy.deepcopy(payload)),
                          ("empty_topology", copy.deepcopy(payload)),
                          ("refined_grid", copy.deepcopy(payload)),
                          ("changed_bounds", copy.deepcopy(payload)),
                          ("changed_sh_order", copy.deepcopy(payload))):
        if name in ("reduced_topology", "empty_topology"):
            for field, value in current["model_state"].items():
                rows = value.reshape(16, -1)
                current["model_state"][field] = (rows[::2] if name == "reduced_topology" else rows[:0]).reshape(-1).copy()
        elif name == "refined_grid":
            current["model_metadata"]["resolution"] = 8
        elif name == "changed_bounds":
            current["model_metadata"]["grid_max"] = [1.2, 1.1, 1.3]
        else:
            current["model_metadata"].update(sh_count=1, opacity_sh_count=1)
            current["model_state"]["radiance_sh"] = current["model_state"]["radiance_sh"].reshape(16, 27)[:, :3].copy().ravel()
            current["model_state"]["opacity_sh"] = current["model_state"]["opacity_sh"].reshape(16, 9)[:, :1].copy().ravel()
        changed_model = restore_model(current, mi)
        verify(name, changed_model, inputs)
        verify(name + "_replay", changed_model, inputs)
        assert replay.statistics()["entries"] == (0 if name == "empty_topology" else 1)
        if name == "empty_topology":
            assert replay.statistics()["empty_model_bypasses"] == 2

    bounded = NativeLossReplay(dr, max_entries=2)
    for hits in (1, 2, 3, 1):
        verify("bounded_" + str(hits), model, inputs, options | {"max_hits": hits}, bounded)
    assert bounded.statistics()["entries"] == 2 and bounded.statistics()["evictions"] == 2
    try:
        bounded(model, *inputs, options | {"not_a_native_option": 1})
    except RuntimeError as error:
        messages = []
        while error is not None:
            messages.append(str(error))
            error = error.__cause__ or error.__context__
        assert "not_a_native_option" in "\n".join(messages), messages
    else:
        raise AssertionError("Invalid native options were silently accepted")
    return {"status": "passed", "checks": checks, "replay": replay.statistics(),
            "adam_replay": adam_replay.statistics(), "chunk_replay": chunks.statistics(),
            "bounded_replay": bounded.statistics()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="mitsuba-replay-regression-") as directory:
        mi, dr, np, handles = initialize_runtime(RuntimeConfig(args.build_dir, args.gpu), Path(directory), {})
        print(json.dumps(run_checks(mi, dr, np)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
