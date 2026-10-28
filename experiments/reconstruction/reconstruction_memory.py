"""CPU-only memory planning and an explicit runtime allocator capability probe.

The ledger sums logical array widths from the model, optimizer, topology, and
native loss implementations, then rounds each buffer using the selected CUDA
allocator policy. It is a planning upper bound for those named buffers, not a
measurement of dedicated VRAM: JIT fusion, aliases, lazy zeros, allocator
caches, and allocation lifetimes affect physical residency.
Uncounted indices and temporary work use explicit reserves. Refinement and
topology evidence windows can need additional storage and are not certified
by an ordinary-view plan.
Importing this module never imports or initializes a GPU runtime. The optional
probe acts only on an already initialized runtime supplied by its caller.
"""

from __future__ import annotations

import math
from collections.abc import Mapping


MIB = 1024**2
GIB = 1024**3


def probe_cuda_allocator(dr):
    """Identify actual CUDA buffer rounding before loading any model.

    Run once with a quiescent initialized CUDA runtime. This synchronizes and
    clears unused allocator buckets/statistics; it must not run inside normal
    training iterations. Unsupported deltas fail explicitly instead of
    guessing from a CMake option or the source tree's intended policy.
    """
    requested = 20 * MIB
    probe = warmup = None
    try:
        dr.sync_thread()
        # Undefined dr.empty() values can remain unmaterialized after eval.
        # An arange expression creates a real output allocation. Warm its
        # launch path before taking the baseline to exclude first-use setup.
        warmup = dr.arange(dr.cuda.Float, 1024)
        dr.eval(warmup)
        dr.sync_thread()
        warmup = None
        dr.flush_malloc_cache()
        dr.detail.malloc_clear_statistics()
        baseline = int(dr.detail.malloc_watermark(dr.JitBackend.CUDA))
        probe = dr.arange(dr.cuda.Float, requested // 4)
        dr.eval(probe)
        dr.sync_thread()
        # Drop temporary launch/cache allocations while retaining the output.
        # clear_statistics now reports currently allocated buckets, avoiding
        # the peak of a transient pinned launch buffer in the delta.
        dr.flush_malloc_cache()
        dr.detail.malloc_clear_statistics()
        allocated = int(dr.detail.malloc_watermark(dr.JitBackend.CUDA)) - baseline
    finally:
        # Drop this function's local array owner before cache flushing. No
        # model or pre-existing live allocation is freed here.
        del warmup, probe
        dr.sync_thread()
        dr.flush_malloc_cache()
    policies = {20 * MIB: "cuda_2m", 32 * MIB: "power_of_two"}
    if allocated not in policies:
        raise RuntimeError(
            f"Unsupported CUDA allocator probe: requested {requested} bytes but "
            f"the watermark grew by {allocated} bytes; expected 20 MiB for "
            "cuda_2m or 32 MiB for power_of_two. Ensure no concurrent CUDA "
            "allocations and verify the loaded Dr.Jit runtime before budgeting.")
    return {"policy": policies[allocated], "probe_requested_bytes": requested,
            "probe_allocated_bytes": allocated}


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def round_cuda_allocation(size, allocator_policy="cuda_2m"):
    """Round one non-shared CUDA allocation, including the initial 64B align.

    ``cuda_2m`` requires the project's patched runtime: allocations >=16 MiB
    round upward in 2 MiB units; smaller ones keep the legacy power-of-two
    buckets. ``power_of_two`` models the unmodified Dr.Jit allocator.
    """
    _integer("allocation size", size)
    if allocator_policy not in ("cuda_2m", "power_of_two"):
        raise ValueError("allocator_policy must be cuda_2m or power_of_two")
    if not size:
        return 0
    aligned = ((size + 63) // 64) * 64
    if allocator_policy == "cuda_2m" and aligned >= 16 * MIB:
        return ((aligned + 2 * MIB - 1) // (2 * MIB)) * (2 * MIB)
    return 1 << (aligned - 1).bit_length()


def estimate_training_memory(*, active_voxels, sh_count, opacity_sh_count,
                             compact_capacity, ray_count, max_hits, max_candidates,
                             distortion=True, lifecycle_epoch_bytes=4,
                             rgb_only=True, known_index_bytes=0,
                             allocator_policy="cuda_2m", reference_image_pixels=None):
    """Return named logical/allocated bytes for one view, excluding its cache.

    ``compact_capacity=0`` is the original N-row gradient layout. A positive
    capacity uses K=min(N, capacity) rows, without changing model or Adam
    storage. ``known_index_bytes`` is optional measured octree/spatial-index
    storage after allocator rounding; cell_indices are already counted
    separately and must not be included again. Index bytes are taken as given
    for both policies because their individual buffer widths are unavailable.
    The lifecycle uses three epoch fields, two UInt32 streaks,
    and one boolean per model row; epoch fields can use four or eight bytes.
    ``ray_count`` is the submitted batch size; ``reference_image_pixels`` is
    the complete cached image size and defaults to ray_count for whole views.
    """
    values = {
        "active_voxels": (active_voxels, 0), "sh_count": (sh_count, 1),
        "opacity_sh_count": (opacity_sh_count, 1), "compact_capacity": (compact_capacity, 0),
        "ray_count": (ray_count, 1), "max_hits": (max_hits, 1),
        "max_candidates": (max_candidates, 0), "known_index_bytes": (known_index_bytes, 0),
    }
    for name, (value, minimum) in values.items():
        _integer(name, value, minimum)
    if reference_image_pixels is None:
        reference_image_pixels = ray_count
    _integer("reference_image_pixels", reference_image_pixels, 1)
    if active_voxels > 2**32 - 1 or compact_capacity > 2**32 - 1 or ray_count > 2**32 - 1:
        raise ValueError("Model rows, compact capacity, and ray count must fit UInt32")
    if reference_image_pixels > 2**32 - 1:
        raise ValueError("Reference image pixels must fit UInt32")
    if isinstance(lifecycle_epoch_bytes, bool) or lifecycle_epoch_bytes not in (4, 8):
        raise ValueError("lifecycle_epoch_bytes must be 4 or 8")
    if not isinstance(distortion, bool) or not isinstance(rgb_only, bool):
        raise ValueError("distortion and rgb_only must be booleans")
    round_cuda_allocation(0, allocator_policy)

    n = active_voxels
    compact = compact_capacity > 0
    k = min(n, compact_capacity) if compact else n
    gradient_width = 3 * sh_count + opacity_sh_count + 9
    # Model rotations are quaternions (four floats), while gradients/moments
    # use three-dimensional rotation tangents.
    model_width = gradient_width + 1
    widths = {"radiance_sh": 3 * sh_count, "opacity_sh": opacity_sh_count,
              "centers": 3, "log_scales": 3, "rotation_tangent": 3}
    model_widths = {**widths, "rotations": 4}
    del model_widths["rotation_tangent"]
    # Each tuple is (single allocation logical bytes, allocation count).
    # Multiplicity avoids ever constructing N- or ray-sized Python arrays.
    buffers = {
        "model_parameters": [(n * width * 4, 1) for width in model_widths.values()],
        "model_cell_indices": [(n * 4, 1)],
        "adam_means": [(n * width * 4, 1) for width in widths.values()],
        "adam_squared_means": [(n * width * 4, 1) for width in widths.values()],
        "adam_step_counters": [(n * 4, 5)],
        "lifecycle_epochs": [(n * lifecycle_epoch_bytes, 3)],
        "lifecycle_deletion_streaks": [(n * 4, 2)],
        "lifecycle_candidate_mask": [(n, 1)],
        "gradients": [(k * width * 4, 1) for width in widths.values()],
        "gradient_validity_counters": [(k * 4, 3)],
        "compact_voxel_ids": [(k * 4 if compact else 0, 1)],
        "compact_slot_prefix": [(n * 4 if compact else 0, 1)],
        "compact_touched_scratch": [(n * 4 if compact else 0, 1)],
        # Native loss currently returns these N-row counters even for RGB-only
        # training. Their zero literals may be optimized, so this is logical
        # storage, not a claim that all zeros physically occupy device memory.
        "foreground_background_evidence_counters": [(n * 4, 2)],
        # A reference can remain alive after eviction. Account for one active
        # image independently of cache capacity; this also covers capacity=0.
        "active_reference_planes": [(reference_image_pixels * 4, 3 if rgb_only else 4)],
        # RGB/raw RGB, alpha/raw alpha, three loss planes, and hit_count.
        "render_result_planes": [(ray_count * 4, 12)],
    }
    if n:
        # Reconstruction.cpp: 9 hit floats (+2 distortion fields), hit IDs,
        # four reverse-suffix floats, and candidates with 6 floats + 2 UInt32s.
        buffers.update(
            hit_record_floats=[(ray_count * max_hits * (11 if distortion else 9) * 4, 1)],
            hit_record_ids=[(ray_count * max_hits * 4, 1)],
            reverse_suffix=[(ray_count * max_hits * 4 * 4, 1)],
            geometric_candidate_state=[(ray_count * 4, max_candidates * 8)],
        )
    else:
        buffers.update(hit_record_floats=[(0, 1)], hit_record_ids=[(0, 1)],
                       reverse_suffix=[(0, 1)], geometric_candidate_state=[(0, 1)])
    components = {name: sum(size * count for size, count in rows) for name, rows in buffers.items()}
    allocated = {name: sum(round_cuda_allocation(size, allocator_policy) * count for size, count in rows)
                 for name, rows in buffers.items()}
    legacy = {name: sum(round_cuda_allocation(size, "power_of_two") * count for size, count in rows)
              for name, rows in buffers.items()}
    for ledger in (components, allocated, legacy):
        ledger["known_octree_and_spatial_index"] = known_index_bytes
    return {
        "active_voxels": n, "gradient_rows": k, "compact_gradients": compact,
        "ray_count": ray_count, "reference_image_pixels": reference_image_pixels,
        "model_float_components": model_width,
        "gradient_float_components": gradient_width,
        "lifecycle_epoch_bytes": lifecycle_epoch_bytes,
        "allocator_policy": allocator_policy,
        "allocator_policy_requires_patched_runtime": allocator_policy == "cuda_2m",
        "buffer_groups": {name: [{"logical_bytes_each": size, "count": count}
                                 for size, count in rows] for name, rows in buffers.items()},
        "components_bytes": components,
        "known_logical_bytes": sum(components.values()),
        "allocated_components_bytes": allocated,
        "known_allocated_bytes": sum(allocated.values()),
        "legacy_allocated_components_bytes": legacy,
        "legacy_power_of_two_bytes": sum(legacy.values()),
        "scope": "ordinary view per-buffer allocation plan; not measured dedicated VRAM or a topology-boundary peak",
        "unmodeled_storage": [
            "ray-generation inputs, native loop temporaries, and optimizer transaction workspace",
            "allocator caches, alignment, JIT buffers, and retained external aliases",
            "octree/spatial index bytes not supplied through known_index_bytes",
            "growth/deletion evidence, candidate models, cooldowns, and refinement old/new model overlap",
        ],
    }


def _gpu_memory_bytes(gpu):
    value = gpu.get("memory_mib") if isinstance(gpu, Mapping) else None
    if isinstance(value, bool):
        return None
    try:
        amount = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(amount) or amount <= 0:
        return None
    return int(amount * MIB)


def plan_training_memory(*, active_voxels, sh_count, opacity_sh_count,
                         compact_capacity, ray_count, max_hits, max_candidates,
                         reference_cache_mib, gpu, distortion=True,
                         lifecycle_epoch_bytes=4, reference_image_count=None,
                         rgb_only=True, workspace_reserve_mib=1536,
                         device_reserve_mib=1024, known_index_bytes=0,
                         pressure_cache_cap_mib=256, cache_quantum_mib=128,
                         allocator_policy="cuda_2m", reference_image_pixels=None):
    """Recommend a reference-cache cap without changing training algorithms.

    ``gpu`` is the runtime report containing ``memory_mib`` (number or numeric
    string). The device reserve leaves room for desktop/driver/other processes;
    workspace reserve covers unmodeled reconstruction buffers. Neither is a
    measured peak guarantee. A small model retains its requested cache limit.
    When that limit cannot fit, pressure policy caps it at 256 MiB by default,
    rounded down in 128 MiB increments, or disables it if necessary. The
    reported available budget is separate from this deliberate pressure cap.

    Known image count bounds cache residency even when its configured limit is
    larger than the dataset. One extra active reference is already in the
    ledger. No GPU runtime is imported and no cache object is mutated.
    Pass the full image size through ``reference_image_pixels`` when ray_count
    describes a smaller submitted ray batch.
    """
    for name, value in (
        ("reference_cache_mib", reference_cache_mib),
        ("workspace_reserve_mib", workspace_reserve_mib),
        ("device_reserve_mib", device_reserve_mib),
        ("pressure_cache_cap_mib", pressure_cache_cap_mib),
    ):
        _integer(name, value)
    _integer("cache_quantum_mib", cache_quantum_mib, 1)
    if reference_image_count is not None:
        _integer("reference_image_count", reference_image_count)
    estimate = estimate_training_memory(
        active_voxels=active_voxels, sh_count=sh_count, opacity_sh_count=opacity_sh_count,
        compact_capacity=compact_capacity, ray_count=ray_count, max_hits=max_hits,
        max_candidates=max_candidates, distortion=distortion,
        lifecycle_epoch_bytes=lifecycle_epoch_bytes, rgb_only=rgb_only,
        known_index_bytes=known_index_bytes, allocator_policy=allocator_policy,
        reference_image_pixels=reference_image_pixels)
    image_pixels = estimate["reference_image_pixels"]
    image_bytes = image_pixels * (3 if rgb_only else 4) * 4
    image_allocated_bytes = round_cuda_allocation(image_pixels * 4, allocator_policy) * (3 if rgb_only else 4)

    def cache_residency(capacity_mib):
        image_capacity = capacity_mib * MIB // image_bytes
        if reference_image_count is not None:
            image_capacity = min(image_capacity, reference_image_count)
        return image_capacity * image_bytes, image_capacity * image_allocated_bytes

    total = _gpu_memory_bytes(gpu)
    workspace = workspace_reserve_mib * MIB
    device = device_reserve_mib * MIB
    fixed = estimate["known_allocated_bytes"] + workspace
    requested_resident, requested_allocated = cache_residency(reference_cache_mib)
    report = {
        **estimate, "gpu_total_bytes": total,
        "workspace_reserve_bytes": workspace, "device_reserve_bytes": device,
        "budget_bytes": None if total is None else max(0, total - device),
        "requested_reference_cache_mib": reference_cache_mib,
        "requested_reference_resident_bytes": requested_resident,
        "requested_reference_allocated_bytes": requested_allocated,
        "reference_image_bytes": image_bytes,
        "reference_image_allocated_bytes": image_allocated_bytes,
        "reference_image_count": reference_image_count,
        "pressure_cache_cap_mib": pressure_cache_cap_mib,
        "cache_quantum_mib": cache_quantum_mib,
        "available_reference_bytes": 0, "recommended_reference_cache_mib": 0,
        "recommended_reference_resident_bytes": 0,
        "recommended_reference_allocated_bytes": 0,
        "estimated_budget_bytes": fixed,
        "supported": False, "reason_code": None, "reason": None,
    }
    if total is None:
        report.update(reason_code="gpu_memory_unknown",
                      reason="GPU report has no usable memory_mib; cannot establish a reference-cache budget.")
        return report
    available = total - device - fixed
    report["available_reference_bytes"] = max(0, available)
    if available < 0:
        report.update(reason_code="model_exceeds_budget",
            reason=(f"Known allocator buckets plus workspace reserve need {fixed / GIB:.2f} GiB "
                    f"before cached references, but the GPU budget after device reserve is "
                    f"{max(0, total - device) / GIB:.2f} GiB. Disabling the reference cache "
                    "cannot fit this plan; reduce per-view workspace or model/optimizer storage."))
        return report
    if requested_allocated <= available:
        cap, reason_code = reference_cache_mib, "requested_cache_fits"
    else:
        cap = min(reference_cache_mib, pressure_cache_cap_mib, available // MIB)
        cap = (cap // cache_quantum_mib) * cache_quantum_mib
        if cap * MIB < image_bytes:
            cap = 0
        # Cache capacity uses logical bytes, but each RGB/RGBA plane has its
        # own allocator bucket. Never spend the available budget on padding.
        while cap and cache_residency(cap)[1] > available:
            cap = max(0, cap - cache_quantum_mib)
        reason_code = "reference_cache_reduced" if cap else "reference_cache_disabled"
    resident, resident_allocated = cache_residency(cap)
    report.update(supported=True, reason_code=reason_code,
        recommended_reference_cache_mib=cap,
        recommended_reference_resident_bytes=resident,
        recommended_reference_allocated_bytes=resident_allocated,
        estimated_budget_bytes=fixed + resident_allocated,
        reason=(f"Reference cache {reference_cache_mib} -> {cap} MiB; "
                f"allocator buckets, cached references, and workspace reserve total "
                f"{(fixed + resident_allocated) / GIB:.2f} GiB within a "
                f"{(total - device) / GIB:.2f} GiB budget. "
                "This is a plan, not a measured VRAM peak guarantee."))
    return report
