"""Portable timing summaries; rendering and hardware discovery stay in callers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import time


_PROFILE_FIELDS = ("wall_seconds", "operations", "jit_kernels", "cache_hits", "cache_misses",
                   "codegen_ms", "backend_compile_ms", "device_ms")
_NATIVE_STAGES = ("forward", "hit_backward", "candidate_backward")


def _number(value):
    return (float(value) if isinstance(value, (int, float))
            and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None)


def _distribution(values):
    values = sorted(value for item in values if (value := _number(item)) is not None)
    if not values:
        return dict(count=0, total=0.0, mean=None, min=None, max=None, p50=None, p95=None)

    def percentile(fraction):
        index = (len(values) - 1) * fraction
        low, high = math.floor(index), math.ceil(index)
        return values[low] + (values[high] - values[low]) * (index - low)

    total = math.fsum(values)
    return dict(count=len(values), total=total, mean=total / len(values),
                min=values[0], max=values[-1], p50=percentile(.5), p95=percentile(.95))


def _finite_json(value):
    """Unavailable/nonfinite measurements are null, never nonstandard JSON NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _finite_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _epoch_summary(entries):
    seconds = _distribution(entry["seconds"] for entry in entries)
    stages = {}
    for entry in entries:
        for name, value in entry["stages"].items():
            if value is not None:
                stages[name] = stages.get(name, 0.0) + value
    return {
        "completed_epochs": len(entries), "epoch_seconds": seconds,
        "stage_seconds": stages,
        "stage_fraction_of_epoch_time": {
            name: value / seconds["total"] if seconds["total"] else None
            for name, value in stages.items()},
        "active_voxels_before": _distribution(entry["active_voxels_before"] for entry in entries),
        "active_voxels_at_epoch_end": _distribution(entry["active_voxels"] for entry in entries),
    }


def _profile_statistics(records):
    return {
        "sampled_views": len(records),
        "wall_seconds": _distribution(record.get("wall_seconds") for record in records),
        "totals": {name: math.fsum(value for record in records
                                  if (value := _number(record.get(name))) is not None)
                   for name in _PROFILE_FIELDS if name != "wall_seconds"},
    }


def _profile_summary(records):
    result = _profile_statistics(records)
    # Stage measurements are a breakdown of a sampled view, never additional
    # work to add to its totals. Legacy/disabled profiles have no stage samples.
    result["native_stages"] = {}
    for name in _NATIVE_STAGES:
        samples = [record["native_stages"][name] for record in records
                   if name in record.get("native_stages", {})]
        if samples:
            result["native_stages"][name] = _profile_statistics(samples)
    return result


class PerformanceRecorder:
    """Collect small aggregates in memory and atomically publish a JSON report.

    Epochs/profiles belong to this process, including when resuming an experiment.
    Call write() periodically and in the caller's interruption/error handler.
    No synchronization, GPU sampling, or rendering is introduced here.
    """

    def __init__(self, path: Path, metadata: dict, *, started_at: float | None = None):
        self.path = Path(path)
        self.metadata = dict(metadata)
        self.started = time.perf_counter() if started_at is None else started_at
        self.started_utc = (datetime.now(timezone.utc)
                            - timedelta(seconds=time.perf_counter() - self.started)).isoformat()
        self.epochs = []
        self.profiles = []
        self.checkpoints = []
        self.phase_seconds = {}

    def record_epoch(self, entry: dict):
        topology = entry.get("topology", {})
        stages = {key: _number(value) for key, value in entry.get("view_stage_seconds", {}).items()}
        stages["epoch_topology"] = _number(entry.get("epoch_topology_seconds"))
        topology_event = bool(topology.get("refined") or topology.get("grown")
                              or topology.get("deleted") or topology.get("evidence_window_evaluated")
                              or entry.get("growth_evidence_requested"))
        self.epochs.append({
            "epoch": entry.get("epoch"),
            "resolution": str(entry.get("training_resolution", entry.get("resolution", "unknown"))),
            "seconds": _number(entry.get("seconds")), "stages": stages,
            "active_voxels": _number(entry.get("active_voxels")),
            "active_voxels_before": _number(entry.get("active_voxels_before")), "topology_event": topology_event,
        })

    def record_profile(self, record: dict):
        sample = {name: _number(record.get(name)) for name in _PROFILE_FIELDS}
        sample["resolution"] = str(record.get("training_resolution", record.get("resolution", "unknown")))
        sample["native_execution"] = record.get("native_execution", "unknown")
        stages = record.get("native_stages")
        if isinstance(stages, dict):
            sample["native_stages"] = {
                name: {field: _number(stages[name].get(field)) for field in _PROFILE_FIELDS}
                for name in _NATIVE_STAGES if isinstance(stages.get(name), dict)}
        self.profiles.append(sample)

    def record_checkpoint(self, path: Path, seconds: float):
        self.checkpoints.append({"path": str(path), "seconds": _number(seconds)})

    def set_phase_seconds(self, name: str, seconds: float):
        self.phase_seconds[name] = _number(seconds)

    def write(self, status: str, **extra):
        resolutions = sorted({entry["resolution"] for entry in self.epochs})
        profile_resolutions = sorted({record["resolution"] for record in self.profiles})
        report = {
            "schema_version": 1, "status": status, "metadata": self.metadata,
            "started_utc": self.started_utc, "updated_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - self.started,
            "clock": {
                "source": "time.perf_counter", "units": "seconds unless a key ends in _ms",
                "scope": "Current process only; resume excludes previous process timing.",
                "wall_time": "Host wall time includes tracing, transfers and asynchronous GPU waits; "
                             "stages are not pure CPU/GPU durations.",
                "epoch_time": "Completed epochs only, including their topology update. "
                              "Checkpoint saving and evaluation are measured separately. "
                              "An interrupted partial epoch is excluded from completed-epoch aggregates, "
                              "but remains in elapsed_seconds and phase_seconds.training_loop.",
                "phase_time": "Phase totals may contain the detailed epoch/checkpoint measurements; "
                              "do not add nested measurements together.",
                "resolution": "Training resolution before the epoch's topology update.",
                "profile": "Sampled views only; device_ms sums sampled device operation times. "
                           "Do not extrapolate these samples to all epochs or equate them with wall time. "
                           "Device time and wall time cannot be added or subtracted to obtain CPU time.",
                "native_stages": "Optional synchronized diagnostic breakdown of sampled views. "
                                 "Stage boundaries force evaluation, including full forward candidates, "
                                 "and change scheduling; they do not measure normal throughput. "
                                 "Native stage diagnostics bypass frozen replay. "
                                 "Stage values are nested within profile totals and must not be added again. "
                                 "Legacy/disabled records are excluded from stage sample counts. "
                                 "Fused atomic operations are not timed separately.",
                "missing_values": "Missing, negative or nonfinite timings are excluded from aggregates.",
            },
            "phase_seconds": self.phase_seconds, "checkpoints": self.checkpoints,
            "checkpoint_seconds": _distribution(item["seconds"] for item in self.checkpoints),
            "completed_epochs": len(self.epochs),
            "first_epoch": self.epochs[0]["epoch"] if self.epochs else None,
            "last_epoch": self.epochs[-1]["epoch"] if self.epochs else None,
            "overall": _epoch_summary(self.epochs),
            "by_resolution": {},
            "kernel_profiles": {
                "overall": _profile_summary(self.profiles),
                "by_native_execution": {execution: _profile_summary([
                    record for record in self.profiles if record["native_execution"] == execution])
                    for execution in sorted({record["native_execution"] for record in self.profiles})},
                "by_resolution": {resolution: _profile_summary([
                    record for record in self.profiles if record["resolution"] == resolution])
                    for resolution in profile_resolutions}},
        }
        for resolution in resolutions:
            entries = [entry for entry in self.epochs if entry["resolution"] == resolution]
            report["by_resolution"][resolution] = {
                **_epoch_summary(entries),
                "normal_epochs": _epoch_summary([entry for entry in entries if not entry["topology_event"]]),
                "topology_event_epochs": _epoch_summary([entry for entry in entries if entry["topology_event"]]),
            }
        report["details"] = extra
        report = _finite_json(report)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.path.parent,
                                             prefix=self.path.name + ".", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return report
