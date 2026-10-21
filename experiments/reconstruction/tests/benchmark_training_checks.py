"""Compare synchronized/deferred view checks with identical frozen CUDA training.

Generated occupied sphere, RGB references, camera order and sampling RNG; no
dataset/download/output artifact. Both modes update Adam after every view and
use the same GPU reference cache. Warm-up and measurement alternate modes.
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

from preview import RuntimeConfig
from reconstruction_common import UniformSampler, initialize_runtime, restore_model
from reconstruction_optimizer import FalcorOptimizer
from reconstruction_checks import DeferredViewChecks
from reconstruction_reference import GPUReferenceCache
from reconstruction_replay import NativeLossReplay
from train import CONFIG, train_view


def fixture(grid, np):
    z, y, x = np.mgrid[:grid, :grid, :grid]
    xyz = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    choose = np.linalg.norm((xyz + .5) / grid - .5, axis=1) < .32
    cells = np.flatnonzero(choose).astype(np.uint32)
    count = len(cells)
    radiance = np.zeros((count, 27), np.float32)
    radiance[:, :3] = np.array([.3, .4, .5]) / .28209479177387814
    return dict(model_metadata=dict(resolution=grid, grid_min=[0.]*3, grid_max=[1.]*3,
                                   sh_count=9, opacity_sh_count=1),
                model_state=dict(cell_indices=cells,
                    centers=np.full(count*3, .5, np.float32),
                    log_scales=np.full(count*3, np.log(.6/grid), np.float32),
                    rotations=np.tile(np.array([1., 0., 0., 0.], np.float32), count),
                    radiance_sh=radiance.reshape(-1), opacity_sh=np.zeros(count, np.float32)))


def benchmark(width, args, mi, dr, np):
    config = copy.deepcopy(CONFIG)
    config.render.width = config.render.height = width
    config.training.profile_views = config.training.profile_interval_epochs = 0
    config.training.profile_stages = False
    config.gradient.distortion_loss_weight = .1
    config.schedule.enabled = False
    payload = fixture(args.grid, np)
    models = [restore_model(payload, mi) for _ in range(2)]
    optimizers = [FalcorOptimizer(model, copy.deepcopy(config.optimizer)) for model in models]
    rngs = [np.random.default_rng(42) for _ in models]
    samplers = [UniformSampler(np) for _ in models]
    replays = [NativeLossReplay(dr, enabled=True) for _ in models]
    cache = GPUReferenceCache(width, width, mi, dr, np, capacity_mib=512, rgb_only=True)
    reference = np.full((width, width, 3), .7, np.float32)
    gpu_reference = cache.get("generated-reference", reference)
    views = []
    for index in range(args.views):
        angle = 2*np.pi*index/args.views
        forward = np.array([np.sin(angle), 0., np.cos(angle)])
        right = np.array([np.cos(angle), 0., -np.sin(angle)])
        views.append(dict(index=index, camera=dict(position=(.5-1.5*forward).tolist(),
            forward=forward.tolist(), right=right.tolist(), up=[0., 1., 0.], camera_angle_x=.55)))
    times = [[], []]
    losses = [[], []]
    try:
        for epoch in range(args.warmup + args.epochs):
            for mode in (0, 1) if epoch % 2 == 0 else (1, 0):
                model, optimizer = models[mode], optimizers[mode]
                checks = DeferredViewChecks(mi, dr, np) if mode and not args.repeat_sync else None
                dr.sync_thread()
                started = time.perf_counter()
                total_loss = 0.
                for view in views:
                    out, loss, _ = train_view(model, view, reference, config, samplers[mode], rngs[mode],
                        mi, dr, np, gpu_reference=gpu_reference, native_replay=replays[mode],
                        deferred_checks=checks, collect_deletion_evidence=False)
                    updated = optimizer.step(model, out, epoch=epoch, coarse_to_fine=True,
                                             update_allowed=out.get("_update_allowed"))
                    if checks is not None:
                        checks.updated(updated["accepted"])
                    else:
                        total_loss += loss
                if checks is not None:
                    total_loss = sum(row["loss"] for row in checks.read(rgb_only=True, distortion_weight=.1))
                dr.sync_thread()
                elapsed = time.perf_counter()-started
                if epoch >= args.warmup:
                    times[mode].append(elapsed)
                    losses[mode].append(total_loss/args.views)
        parameter_errors = {}
        second = models[1].snapshot()
        for key, value in models[0].snapshot().items():
            if key == "cell_indices":
                continue
            first_host = np.asarray(value).astype(np.float64)
            difference = first_host-np.asarray(second[key]).astype(np.float64)
            parameter_errors[key] = dict(max_absolute=float(np.max(np.abs(difference))),
                relative_l2=float(np.linalg.norm(difference)/max(np.linalg.norm(first_host), 1e-30)),
                baseline_max_absolute=float(np.max(np.abs(first_host))))
        error = max(item["max_absolute"] for item in parameter_errors.values())
        return dict(width=width, height=width, views=args.views, active_voxels=len(payload["model_state"]["cell_indices"]),
                    comparison="sync_vs_sync" if args.repeat_sync else "sync_vs_deferred",
                    warmup_epochs=args.warmup, measured_epochs=args.epochs,
                    sync_seconds=times[0], deferred_seconds=times[1],
                    sync_mean=float(np.mean(times[0])), deferred_mean=float(np.mean(times[1])),
                    speedup=float(np.mean(times[0])/np.mean(times[1])),
                    max_parameter_absolute_error=error, parameter_errors=parameter_errors, loss_histories=losses,
                    replay=[replay.statistics() for replay in replays])
    finally:
        for replay in replays:
            replay.clear()
        cache.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--widths", type=int, nargs="+", default=[128, 800])
    parser.add_argument("--grid", type=int, default=64)
    parser.add_argument("--views", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--repeat-sync", action="store_true", help="Run sync twice to measure floating-point/atomic repeatability")
    args = parser.parse_args()
    if min(args.grid, args.views, args.warmup, args.epochs, *args.widths) < 1:
        raise ValueError("Grid, view, warm-up, measurement and image sizes must be positive")
    with tempfile.TemporaryDirectory(prefix="mitsuba-checks-benchmark-") as temporary:
        mi, dr, np, handles = initialize_runtime(RuntimeConfig(args.build_dir, args.gpu), Path(temporary), {})
        for width in args.widths:
            print(json.dumps(benchmark(width, args, mi, dr, np)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
