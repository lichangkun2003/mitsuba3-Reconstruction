"""Diagnose reconstruction memory from a checkpoint without resuming training.

Loads only checkpoint model arrays, optionally refines 512 -> 1024 once, then
resets Adam and executes a few checked RGB views. Optional growth diagnostics
run real evidence windows and commit their topology changes in memory. GPU
lifecycle fields match a fresh refinement; no model/checkpoint writes occur.
Every diagnostic boundary synchronizes the GPU, so timings are not throughput
measurements. Run this in an isolated process with an external timeout: the
memory guard runs between stages and cannot stop a single oversized allocation.
Dr.Jit allocator watermarks can include cached/pinned/shared allocations and
backend-wide device accounting; they are not dedicated VRAM measurements.
Whole-GPU memory reported by nvidia-smi is recorded separately.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction_checkpoint import FORMAT, FORMAT_VERSION, validate_model_state
from reconstruction_checks import DeferredViewChecks
from reconstruction_common import (
    REPOSITORY, WORKSPACE, ReferenceImageCache, UniformSampler, initialize_runtime,
    load_views, prepare_dataset, restore_model, workspace_path,
)
from reconstruction_optimizer import FalcorOptimizer
from reconstruction_reference import GPUReferenceCache
from reconstruction_refinement_gpu import refine_state_gpu
from reconstruction_replay import NativeLossReplay
from reconstruction_topology import TopologyManager
from reconstruction_topology_gpu import GPUTopologyLifecycle
from train import CONFIG, train_view, update


class MemoryGuard(RuntimeError):
    pass


class DiagnosticInterrupted(RuntimeError):
    pass


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--refine", action="store_true", help="Refine a saved core-512 model once to core 1024")
    parser.add_argument("--views", type=int, default=3, help="Evenly spaced training cameras; default: 3")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--growth", action="store_true",
                        help="After ordinary warmup views, run real growth/deletion evidence and finish_epoch")
    parser.add_argument("--growth-views", type=int, default=0,
                        help="0: all training cameras; positive: complete diagnostic window on an evenly spaced subset")
    parser.add_argument("--growth-windows", type=int, default=1)
    parser.add_argument("--post-growth-views", type=int, default=1,
                        help="Ordinary checked RGB/Adam views after all completed growth windows; 0 disables")
    parser.add_argument("--growth-boundary", type=int,
                        help="First one-based due boundary; default finds the first scheduled growth window")
    parser.add_argument("--growth-prepare-only", action="store_true",
                        help="With --growth, stop after candidate preparation/cache release without tracing evidence")
    parser.add_argument("--growth-profile-interval-views", type=int, default=16,
                        help="Full memory snapshots on first two/last and every N growth views; progress is always saved")
    parser.add_argument("--stop-file", help="Stop at the next diagnostic stage if this file exists; default: OUTPUT/STOP")
    parser.add_argument("--retain-previous-gradient", action="store_true",
                        help="Keep the previous view's gradient alive through the next train_view call")
    parser.add_argument("--gpu-reference-cache-mib", type=int, default=4096)
    parser.add_argument("--reference-cache-mib", type=int, default=4096)
    parser.add_argument("--warm-all-references", action=argparse.BooleanOptionalAction, default=True,
                        help="Warm all training reference images (bicycle: 169), including CPU cache")
    parser.add_argument("--chunk-size", type=int, help="Override saved ray chunk size for an explicit comparison")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--build-dir")
    parser.add_argument("--max-device-gib", type=float, default=0.,
                        help="Optional stage-boundary allocator/GPU memory guard; default 0 disables")
    parser.add_argument("--output-dir", help="New/empty directory outside the source checkout; default: workspace .cache")
    args = parser.parse_args(argv)
    if min(args.views, args.rounds, args.growth_windows, args.growth_profile_interval_views) < 1:
        parser.error("views, rounds, growth-windows and growth-profile-interval-views must be positive")
    if min(args.growth_views, args.post_growth_views) < 0 or (args.growth_boundary is not None and args.growth_boundary < 1):
        parser.error("growth-views/post-growth-views must be nonnegative and growth-boundary must be positive")
    if args.growth_prepare_only and not args.growth:
        parser.error("growth-prepare-only requires --growth")
    if min(args.gpu, args.gpu_reference_cache_mib, args.reference_cache_mib) < 0:
        parser.error("GPU index and cache budgets must be nonnegative")
    if args.chunk_size is not None and args.chunk_size < 0:
        parser.error("chunk-size must be nonnegative")
    if not 0 <= args.max_device_gib < float("inf"):
        parser.error("max-device-gib must be finite and nonnegative")
    if args.output_dir and workspace_path(args.output_dir, WORKSPACE).is_relative_to(REPOSITORY):
        parser.error("output-dir must be outside the source checkout")
    return args


def load_model_only(path, np):
    """Read and hash six model arrays; never decompress unused optimizer tables."""
    with np.load(path, allow_pickle=False) as archive:
        raw = archive["metadata_json"]
        if raw.dtype != np.uint8 or raw.ndim != 1:
            raise ValueError("Invalid checkpoint metadata encoding")
        header = json.loads(raw.tobytes().decode("utf-8"))
        if header.get("format") != FORMAT or header.get("version") != FORMAT_VERSION:
            raise ValueError("Unsupported checkpoint format")
        manifest = header["arrays"]
        if set(archive.files) != set(manifest) | {"metadata_json"}:
            raise ValueError("Archive entries do not match the checkpoint manifest")

        def decode(value):
            if not isinstance(value, dict):
                return value
            kind = value.get("type")
            if kind == "array":
                key = value["key"]
                array = archive[key]
                record = manifest[key]
                if (array.dtype.hasobject or array.dtype.str != record["dtype"]
                        or list(array.shape) != record["shape"]):
                    raise ValueError(f"Invalid model array: {key}")
                digest = hashlib.sha256(memoryview(array).cast("B") if array.size else b"").hexdigest()
                if digest != record["sha256"]:
                    raise ValueError(f"Model array checksum mismatch: {key}")
                return array
            if kind == "dict":
                return {key: decode(item) for key, item in value["items"].items()}
            if kind in ("list", "tuple"):
                items = [decode(item) for item in value["items"]]
                return tuple(items) if kind == "tuple" else items
            raise ValueError(f"Unsupported metadata node: {kind}")

        payload = header["payload"]["items"]
        result = {key: decode(payload[key]) for key in ("model_state", "model_metadata", "config")}
        state = payload["training_state"]["items"]
        result["completed_epochs"] = int(state["completed_epochs"])
        result["rng_state"] = decode(state["rng_state"])
    validate_model_state(result["model_state"], result["model_metadata"])
    return result


def nvidia_memory(gpu_id):
    try:
        result = subprocess.run([
            "nvidia-smi", "--id=" + str(gpu_id),
            "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True, timeout=8)
        total, used, free, utilization = (float(value.strip()) for value in result.stdout.splitlines()[0].split(","))
        return dict(total_bytes=int(total * 1024**2), used_bytes=int(used * 1024**2),
                    free_bytes=int(free * 1024**2), utilization_percent=utilization)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {"error": str(error)}


def host_memory():
    if sys.platform != "win32":
        return {}
    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load_percent", ctypes.c_ulong)] + [
            (key, ctypes.c_ulonglong) for key in ("total_physical", "available_physical",
                "commit_limit", "available_commit", "total_virtual", "available_virtual", "reserved")]
    result = Memory()
    result.length = ctypes.sizeof(result)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(result)):
        return {"error": "GlobalMemoryStatusEx failed"}
    return {key: getattr(result, key) for key, _ in result._fields_ if key not in ("length", "reserved")}


class Diagnostics:
    def __init__(self, output, report, dr, gpu_id, limit):
        self.output, self.report, self.dr, self.gpu_id = output, report, dr, gpu_id
        self.limit = limit * 1024**3

    def write(self):
        (self.output / "memory-report.json").write_text(json.dumps(self.report, indent=2) + "\n", encoding="utf-8")

    def begin(self, name):
        stop_file = self.report.get("stop_file")
        if stop_file and Path(stop_file).exists():
            raise DiagnosticInterrupted(f"Stop requested by {stop_file}; next stage {name} was not started")
        self.report["current_stage"] = name
        self.write()  # Preserve the last entered stage if a driver call stalls.
        return time.perf_counter()

    def capture(self, name, started=None, **extra):
        self.dr.sync_thread()
        elapsed = time.perf_counter() - started if started is not None else None
        peak = int(self.dr.detail.malloc_watermark(self.dr.JitBackend.CUDA))
        row = dict(stage=name, operation_wall_seconds=elapsed, cuda_allocator_peak_bytes=peak,
                   whos_tail="\n".join(self.dr.whos(as_string=True).splitlines()[-24:]),
                   gpu=nvidia_memory(self.gpu_id), host=host_memory(), **extra)
        self.report["stages"].append(row)
        self.report["peak_cuda_allocator_bytes"] = max(peak, self.report.get("peak_cuda_allocator_bytes", 0))
        self.report["peak_observed_gpu_used_bytes"] = max(row["gpu"].get("used_bytes", 0),
                                                       self.report.get("peak_observed_gpu_used_bytes", 0))
        self.write()
        print(json.dumps({key: row[key] for key in ("stage", "operation_wall_seconds", "cuda_allocator_peak_bytes", "gpu")}), flush=True)
        if self.limit and max(peak, row["gpu"].get("used_bytes", 0)) > self.limit:
            raise MemoryGuard(f"Memory guard exceeded after {name}; no subsequent stage will run")
        return row


def fresh_gpu_lifecycle(model, views, config, dr):
    """Reproduce post-refinement GPU row fields with empty CPU placeholders."""
    proxy = SimpleNamespace(resolution=int(model.resolution),
        core_resolution=int(getattr(model, "core_resolution", model.resolution)),
        background_refinement=getattr(model, "background_refinement", ""),
        background_layers=int(getattr(model, "background_layers", 0)),
        opacity_sh_count=int(model.opacity_sh_count), active_voxel_count=0)
    manager = TopologyManager(proxy, len(views), config.topology, config.schedule,
                              config.optimizer.opacity_warmup_epochs, config.optimizer.opacity_ramp_epochs)
    lifecycle = GPUTopologyLifecycle(manager)
    manager.count = int(model.active_voxel_count)
    lifecycle.fields = {key: dr.zeros(type(value), manager.count) for key, value in lifecycle.fields.items()}
    lifecycle.candidates = dr.zeros(lifecycle.Bool, manager.count)
    manager._gpu_lifecycle = lifecycle
    dr.eval(*lifecycle.fields.values(), lifecycle.candidates)
    return manager


def lifecycle_storage(manager, dr):
    """Logical typed-array sizes; no device readback or assumed epoch dtype."""
    device = manager._gpu_lifecycle
    arrays = dict(device.fields, candidates=device.candidates,
                  cooldown_cells=device.cooldown_cells, cooldown_expiry=device.cooldown_expiry)
    fields = {}
    for name, value in arrays.items():
        cls = type(value)
        count, itemsize = int(dr.width(value)), int(dr.itemsize_v(cls))
        fields[name] = dict(dtype=cls.__module__ + "." + cls.__name__, count=count,
                            itemsize_bytes=itemsize, logical_bytes=count * itemsize)
    return dict(fields=fields, logical_device_bytes=sum(row["logical_bytes"] for row in fields.values()),
                note="Logical array width times scalar item size; includes still-literal arrays and "
                     "excludes allocator rounding, caches and temporary workspaces.")


def post_growth_updates(model, optimizer, replay, references, cache, sampler, rng,
                        views, config, epoch, args, diagnostics, mi, dr, np):
    """Validate that the final replacement supports subsequent normal updates."""
    record = diagnostics.report["post_growth_updates"] = dict(status="running", epoch=epoch + 1, views=[])
    diagnostics.report["progress"] = dict(phase="post_growth", completed_views=0, total_views=len(views))
    checks = DeferredViewChecks(mi, dr, np)
    previous = None
    for position, view in enumerate(views):
        label = f"post_growth_view_{position + 1}"
        diagnostics.report["progress"]["camera"] = int(view["index"])
        image = references.get(view["reference_path"])
        gpu_reference = cache.get(view["reference_path"], image)
        started = diagnostics.begin(label + "_backward")
        out, loss, alpha = train_view(model, view, image, config, sampler, rng, mi, dr, np,
            collect_growth_evidence=False, collect_deletion_evidence=False,
            gpu_reference=gpu_reference, native_replay=replay, deferred_checks=checks)
        backward = diagnostics.capture(label + "_backward", started,
                                       previous_gradient_retained=previous is not None)
        previous = None
        started = diagnostics.begin(label + "_adam")
        updated = optimizer.step(model, out, sample_count=1, epoch=epoch,
            coarse_to_fine=config.schedule.enabled, update_allowed=out.get("_update_allowed"))
        checks.updated(updated["accepted"])
        adam = diagnostics.capture(label + "_adam", started)
        if args.retain_previous_gradient:
            previous = out
        del out, loss, alpha, updated
        diagnostics.capture(label + "_after_release")
        record["views"].append(dict(camera=int(view["index"]),
            backward_seconds=backward["operation_wall_seconds"], adam_seconds=adam["operation_wall_seconds"]))
        diagnostics.report["progress"]["completed_views"] = position + 1
        diagnostics.write()
    previous = None
    started = diagnostics.begin("post_growth_validate")
    measurements = checks.read(rgb_only=True, distortion_weight=config.gradient.distortion_loss_weight)
    for row, measurement in zip(record["views"], measurements):
        row.update(measurement)
    diagnostics.capture("post_growth_validate", started)
    record.update(status="complete", active_voxels=int(model.active_voxel_count), optimizer_capacity=optimizer.count)
    diagnostics.report["progress"]["phase"] = "complete"
    diagnostics.write()


def growth_windows(state, topology, optimizer, replay, references, cache, sampler, rng,
                   views, config, completed, args, diagnostics, mi, dr, np):
    """Run production topology APIs, with genuine evidence and unchanged thresholds.

    The caller transfers its model through ``state`` and drops its own alias, so
    replacing a model here cannot accidentally retain its old full parameter set.
    """
    from train import release_topology_cache

    report = diagnostics.report
    model = state["model"]
    topology.level_start_epoch = (sum(budget for resolution, budget in topology.budgets.items()
                                      if resolution < topology.core_resolution)
                                 if config.schedule.enabled else 0)
    earliest = max(completed + 1, topology.level_start_epoch + 1)
    report["growth_diagnostic"] = dict(
        level_start_epoch=topology.level_start_epoch, window_view_count=len(views),
        camera_indices=[int(view["index"]) for view in views],
        coverage="all_training_cameras" if len(views) == report["total_training_views"] else "representative_subset",
        note="Only diagnostic boundary numbers advance across skipped epochs. Parameters/RNG are not "
             "trained through those epochs. A subset defines a smaller complete evidence window with "
             "dense evidence IDs; support/conflict/gain thresholds remain unchanged. Natural selection "
             "may grow zero rows; remap is exercised only if finish_epoch actually changes topology.",
        windows=[])

    def end_phase(name, started, sampled, **extra):
        if sampled:
            return diagnostics.capture(name, started, **extra)["operation_wall_seconds"]
        dr.sync_thread()
        return time.perf_counter() - started

    for window_index in range(args.growth_windows):
        if window_index == 0 and args.growth_boundary is not None:
            boundary = args.growth_boundary
            if boundary < earliest or not topology._growth_due(boundary):
                raise ValueError(f"Growth boundary {boundary} is not due or precedes this model level")
        else:
            boundary = next((value for value in range(earliest, config.schedule.total_epochs + 1)
                             if topology._growth_due(value)), None)
            if boundary is None:
                raise ValueError("No remaining growth boundary under the saved algorithm configuration")
        earliest = boundary + 1
        epoch = boundary - 1
        prefix = f"growth_{window_index + 1}_epoch_{boundary}"
        window = dict(boundary=boundary, active_voxels_before=int(model.active_voxel_count),
                      evidence_views=[], status="preparing")
        report["growth_diagnostic"]["windows"].append(window)
        report["progress"] = dict(window=window_index + 1, boundary=boundary,
                                  completed_views=0, total_views=len(views), phase="prepare")
        started = diagnostics.begin(prefix + "_before_prepare_release")
        replay.clear()
        release_topology_cache(model, dr)
        diagnostics.capture(prefix + "_before_prepare_release", started)
        started = diagnostics.begin(prefix + "_prepare")
        collect_growth = topology.prepare_rgb_growth(model, epoch)
        collect_deletion = topology.collect_deletion(epoch)
        window.update(growth_evidence_requested=collect_growth,
                      deletion_evidence_requested=collect_deletion,
                      candidate_count=int(model.growth_candidate_count),
                      prepare_stage_seconds=dict(topology.growth_prepare_stage_seconds))
        diagnostics.capture(prefix + "_prepare", started, candidate_count=int(model.growth_candidate_count))
        started = diagnostics.begin(prefix + "_after_prepare_release")
        release_topology_cache(model, dr)
        diagnostics.capture(prefix + "_after_prepare_release", started)
        if args.growth_prepare_only:
            window["status"] = "prepare_only"
            diagnostics.write()
            break
        checks = DeferredViewChecks(mi, dr, np)
        previous = None
        window["status"] = "collecting_evidence"
        for position, view in enumerate(views):
            sampled = (position < 2 or position + 1 == len(views)
                       or (position + 1) % args.growth_profile_interval_views == 0)
            label = f"{prefix}_view_{position + 1}"
            report["progress"].update(camera=int(view["index"]), evidence_id=position,
                                      completed_views=position, phase="evidence")
            image = references.get(view["reference_path"])
            gpu_reference = cache.get(view["reference_path"], image)
            started = diagnostics.begin(label + "_backward")
            out, loss, alpha = train_view(model, view, image, config, sampler, rng, mi, dr, np,
                collect_growth_evidence=collect_growth, collect_deletion_evidence=collect_deletion,
                gpu_reference=gpu_reference, native_replay=replay, deferred_checks=checks)
            backward_seconds = end_phase(label + "_backward", started, sampled,
                                         previous_gradient_retained=previous is not None)
            previous = None
            started = diagnostics.begin(label + "_observe")
            if collect_growth:
                topology.observe_rgb_growth(model, out, view_id=position, epoch=epoch,
                                            pixel_count=config.render.width * config.render.height)
            if collect_deletion:
                topology.observe_rgb_deletion(model, out, view_id=position, epoch=epoch,
                                              pixel_count=config.render.width * config.render.height)
            observe_seconds = end_phase(label + "_observe", started, sampled)
            started = diagnostics.begin(label + "_adam")
            updated = optimizer.step(model, out, sample_count=1, epoch=epoch,
                coarse_to_fine=config.schedule.enabled, update_allowed=out.get("_update_allowed"))
            checks.updated(updated["accepted"])
            adam_seconds = end_phase(label + "_adam", started, sampled)
            if args.retain_previous_gradient:
                previous = out
            del out, loss, alpha, updated
            row = dict(camera=int(view["index"]), evidence_id=position,
                       backward_seconds=backward_seconds, observe_seconds=observe_seconds,
                       adam_seconds=adam_seconds, memory_sampled=sampled)
            window["evidence_views"].append(row)
            report["progress"]["completed_views"] = position + 1
            diagnostics.write()
            print(json.dumps(dict(growth_boundary=boundary, view=position + 1, views=len(views), **row)), flush=True)
        previous = None
        started = diagnostics.begin(prefix + "_validate")
        measurements = checks.read(rgb_only=True, distortion_weight=config.gradient.distortion_loss_weight)
        for row, measurement in zip(window["evidence_views"], measurements):
            row.update(measurement)
        del checks
        diagnostics.capture(prefix + "_validate", started)
        report["progress"]["phase"] = "finish"
        started = diagnostics.begin(prefix + "_before_finish_release")
        replay.clear()
        release_topology_cache(model, dr)
        diagnostics.capture(prefix + "_before_finish_release", started)
        started = diagnostics.begin(prefix + "_finish")
        replacement, event = topology.finish_epoch(model, optimizer, boundary)
        topology.before_refine_checkpoint = None
        window["event"] = event
        window["model_replaced"] = replacement is not None
        diagnostics.capture(prefix + "_finish", started)
        if replacement is not None:
            started = diagnostics.begin(prefix + "_restore_replacement")
            model = restore_model(replacement, mi)
            state["model"] = model
            replacement = None
            diagnostics.capture(prefix + "_restore_replacement", started,
                                active_voxels=int(model.active_voxel_count))
        started = diagnostics.begin(prefix + "_after_finish_release")
        release_topology_cache(model, dr)
        diagnostics.capture(prefix + "_after_finish_release", started)
        if int(model.active_voxel_count) != optimizer.count or optimizer.count != topology.count:
            raise RuntimeError("Model, optimizer and topology row counts differ after finish_epoch")
        window.update(status="complete", active_voxels_after=int(model.active_voxel_count),
                      optimizer_capacity=optimizer.count, topology_execution=topology.statistics(),
                      lifecycle_storage_after=lifecycle_storage(topology, dr))
        report["progress"]["phase"] = "complete"
        diagnostics.write()
    state["model"] = model


def main(argv=None):
    args = parse_arguments(argv)
    checkpoint = workspace_path(args.checkpoint, WORKSPACE)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    output = (workspace_path(args.output_dir, WORKSPACE) if args.output_dir else
              WORKSPACE / ".cache" / ("reconstruction-memory-" + uuid.uuid4().hex[:12]))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status="running", checkpoint=str(checkpoint), arguments=vars(args), stages=[],
        stop_file=str(workspace_path(args.stop_file, WORKSPACE) if args.stop_file else output / "STOP"),
        scope="Synchronized memory diagnostic; not throughput or resumed optimization. Only checkpoint "
              "model arrays are loaded and checksummed; unused optimizer/topology arrays are not read. "
              "Adam and GPU lifecycle are reset, matching a new refinement. Optional growth windows "
              "use actual RGB evidence and commit naturally selected topology changes in memory only. "
              "Refinement omits the old "
              "optimizer and historical allocator pool, so it does not reproduce the full transition peak.",
        guard="Checked only after synchronized stages, using cumulative Dr.Jit CUDA allocation peak "
              "and sampled whole-GPU used memory. Cannot prevent an oversized allocation within a stage. "
              "Use an external process timeout. NVIDIA used memory includes other applications.",
        memory_metric_scope="Dr.Jit malloc_watermark is an allocator statistic, which can include cache, "
                            "pinned/shared allocations and allocations across devices within a backend. "
                            "Do not equate it with dedicated VRAM. Separately sampled nvidia-smi memory "
                            "describes the selected whole GPU, including other applications; snapshots "
                            "can miss transient peaks between stage boundaries. The report-wide allocator "
                            "maximum includes refinement, while per-stage statistics are reset after "
                            "restoring the model and releasing the refinement allocation cache.",
        timing="Each phase includes GPU completion. Memory queries/JSON output between phases are diagnostic overhead.")
    diagnostics = None
    replay = optimizer = cache = topology = None
    try:
        config = copy.deepcopy(CONFIG)
        config.runtime.gpu = args.gpu
        if args.build_dir:
            config.runtime.build_dir = args.build_dir
        mi, dr, np, handles = initialize_runtime(config.runtime, output, report)
        diagnostics = Diagnostics(output, report, dr, report["gpu"].get("uuid", args.gpu), args.max_device_gib)
        diagnostics.capture("runtime")
        started = diagnostics.begin("load_model_cpu")
        payload = load_model_only(checkpoint, np)
        runtime = config.runtime
        update(config, payload.pop("config"))
        config.runtime = runtime
        if not config.training.rgb_only:
            raise ValueError("This diagnostic requires an RGB-only checkpoint")
        if args.growth and not (config.topology.gpu_topology and config.topology.use_rgb_growth
                                and config.topology.use_rgb_deletion):
            raise ValueError("Growth diagnostics require the saved GPU RGB growth/deletion topology path")
        if args.chunk_size is not None:
            config.training.chunk_size = args.chunk_size
        config.training.profile_views = config.training.profile_interval_epochs = 0
        config.training.profile_stages = False
        config.training.gpu_reference_cache_mib = args.gpu_reference_cache_mib
        config.training.reference_cache_mib = args.reference_cache_mib
        completed = payload.pop("completed_epochs")
        rng_state = payload.pop("rng_state")
        core = int(payload["model_metadata"].get("core_resolution", payload["model_metadata"]["resolution"]))
        if args.refine and core != 512:
            raise ValueError("--refine requires a saved core-512 checkpoint")
        report.update(checkpoint_completed_epochs=completed, checkpoint_core_resolution=core,
                      checkpoint_voxels=len(payload["model_state"]["cell_indices"]))
        diagnostics.capture("load_model_cpu", started)
        started = diagnostics.begin("dataset")
        prepared = prepare_dataset(config, np, load_points=False)
        dataset = workspace_path(config.data.dataset, WORKSPACE)
        views = (prepared["views"] if prepared is not None else
                 load_views(dataset, config.data.cameras, config.render, np, data=config.data))
        del prepared
        if args.views > len(views):
            raise ValueError(f"Requested {args.views} views but dataset has {len(views)}")
        if args.growth and args.post_growth_views > len(views):
            raise ValueError(f"Requested {args.post_growth_views} post-growth views but dataset has {len(views)}")
        indices = ([0] if args.views == 1 else
                   [round(i * (len(views) - 1) / (args.views - 1)) for i in range(args.views)])
        selected = [views[i] for i in indices]
        growth_count = args.growth_views or len(views)
        if args.growth and growth_count > len(views):
            raise ValueError(f"Requested {growth_count} growth views but dataset has {len(views)}")
        growth_indices = ([0] if growth_count == 1 else
                          [round(i * (len(views) - 1) / (growth_count - 1)) for i in range(growth_count)])
        growth_views = [views[i] for i in growth_indices] if args.growth else []
        report.update(config=asdict(config), total_training_views=len(views), camera_indices=indices)
        diagnostics.capture("dataset", started)
        if args.refine:
            started = diagnostics.begin("refine_512_to_1024")
            replacement = refine_state_gpu(payload, config.schedule, config.topology.max_active_voxels)
            del payload
            gc.collect()
            diagnostics.capture("refine_512_to_1024", started,
                                active_voxels=len(replacement["model_state"]["cell_indices"]))
            payload = replacement
            del replacement
        started = diagnostics.begin("restore_model")
        model = restore_model(payload, mi)
        dr.eval(*model.snapshot().values())
        dr.sync_thread()
        del payload
        gc.collect()
        count = int(model.active_voxel_count)
        report.update(active_voxels=count, resolution=int(model.resolution),
                      nominal_model_adam_bytes=count * 472,
                      nominal_gradient_and_counters_bytes=count * 168)
        diagnostics.capture("restore_model", started, active_voxels=count)
        started = diagnostics.begin("release_refinement_cache")
        from train import release_topology_cache
        release_topology_cache(model, dr)
        dr.detail.malloc_clear_statistics()
        report["allocator_statistics_reset_after"] = "restore_model and flush_malloc_cache"
        diagnostics.capture("release_refinement_cache", started)
        started = diagnostics.begin("lifecycle_and_reset_adam")
        topology = fresh_gpu_lifecycle(model, growth_views if args.growth else views, config, dr)
        report["lifecycle_storage"] = lifecycle_storage(topology, dr)
        report["lifecycle_device_bytes"] = report["lifecycle_storage"]["logical_device_bytes"]
        optimizer = FalcorOptimizer(model, config.optimizer)
        # Adam zero state is lazy. The first update will allocate any buffers
        # still represented by literals; the subsequent watermark records it.
        diagnostics.capture("lifecycle_and_reset_adam", started)
        started = diagnostics.begin("warm_references")
        references = ReferenceImageCache(config.render.width, config.render.height, mi, np,
            capacity_mib=args.reference_cache_mib, rgb_only=True, background=config.render.background,
            reference_compositing=config.data.reference_compositing)
        cache = GPUReferenceCache(config.render.width, config.render.height, mi, dr, np,
                                  capacity_mib=args.gpu_reference_cache_mib, rgb_only=True)
        for view in views if args.warm_all_references else selected:
            image = references.get(view["reference_path"])
            cache.get(view["reference_path"], image)
        del image
        report.update(reference_cache=references.statistics(), gpu_reference_cache=cache.statistics())
        diagnostics.capture("warm_references", started)
        sampler = UniformSampler(np)
        report["sampling_execution"] = {"mode": "synchronous_uniform", "workers": 0}
        rng = np.random.default_rng()
        rng.bit_generator.state = rng_state
        replay = NativeLossReplay(dr, enabled=config.training.freeze_native)
        previous = None
        report["rounds"] = []
        for round_index in range(args.rounds):
            checks = DeferredViewChecks(mi, dr, np)
            rows = []
            for position, view in enumerate(selected):
                label = f"round_{round_index + 1}_view_{position + 1}"
                image = references.get(view["reference_path"])
                gpu_reference = cache.get(view["reference_path"], image)
                started = diagnostics.begin(label + "_backward")
                out, loss, alpha = train_view(model, view, image, config, sampler, rng, mi, dr, np,
                    collect_growth_evidence=False, collect_deletion_evidence=False,
                    gpu_reference=gpu_reference, native_replay=replay, deferred_checks=checks)
                backward = diagnostics.capture(label + "_backward", started,
                    previous_gradient_retained=previous is not None)
                # Old driver's assignment released the preceding output here,
                # after computing the next view and before its Adam update.
                previous = None
                started = diagnostics.begin(label + "_adam")
                updated = optimizer.step(model, out, sample_count=1, epoch=completed + round_index,
                    coarse_to_fine=config.schedule.enabled, update_allowed=out.get("_update_allowed"))
                checks.updated(updated["accepted"])
                adam = diagnostics.capture(label + "_adam", started)
                if args.retain_previous_gradient:
                    previous = out
                del out, loss, alpha, updated
                diagnostics.capture(label + "_after_release")
                rows.append(dict(camera=int(view["index"]), backward_seconds=backward["operation_wall_seconds"],
                                 adam_seconds=adam["operation_wall_seconds"]))
            measurements = checks.read(rgb_only=True, distortion_weight=config.gradient.distortion_loss_weight)
            report["rounds"].append([dict(row, **measurement) for row, measurement in zip(rows, measurements)])
            diagnostics.capture(f"round_{round_index + 1}_validated")
        previous = None
        if args.growth:
            state = {"model": model}
            del model  # The diagnostic helper must not leave an old-model alias in this caller.
            growth_windows(state, topology, optimizer, replay, references, cache, sampler, rng,
                           growth_views, config, completed, args, diagnostics, mi, dr, np)
            model = state.pop("model")
            report["final_active_voxels"] = int(model.active_voxel_count)
            report["final_lifecycle_storage"] = lifecycle_storage(topology, dr)
            if args.post_growth_views and not args.growth_prepare_only:
                post_indices = ([0] if args.post_growth_views == 1 else
                                [round(i * (len(views) - 1) / (args.post_growth_views - 1))
                                 for i in range(args.post_growth_views)])
                final_boundary = report["growth_diagnostic"]["windows"][-1]["boundary"]
                post_growth_updates(model, optimizer, replay, references, cache, sampler, rng,
                                    [views[i] for i in post_indices], config, final_boundary,
                                    args, diagnostics, mi, dr, np)
        report.update(status="complete", replay=replay.statistics(), optimizer=optimizer.statistics(),
                      gpu_reference_cache=cache.statistics())
        diagnostics.write()
        print(f"Memory report: {output / 'memory-report.json'}", flush=True)
        return 0
    except (KeyboardInterrupt, DiagnosticInterrupted) as error:
        report.update(status="interrupted", error=str(error) or "KeyboardInterrupt")
        if diagnostics:
            diagnostics.write()
        return 130
    except MemoryGuard as error:
        report.update(status="stopped_by_memory_guard", error=str(error))
        if diagnostics:
            diagnostics.write()
        return 2
    except BaseException as error:
        report.update(status="failed", error=str(error), traceback=traceback.format_exc())
        (output / "memory-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        if replay is not None:
            replay.clear()
        if optimizer is not None:
            optimizer.clear_replay()
        if cache is not None:
            cache.clear()
        if topology is not None:
            topology._gpu_lifecycle = None


if __name__ == "__main__":
    raise SystemExit(main())
