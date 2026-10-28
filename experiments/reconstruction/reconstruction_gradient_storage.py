"""Bounded native gradient rows without changing which voxels receive updates.

The native pass returns a device count and a fixed-capacity table. Overflow is
checked before any optimizer/topology consumer; an undersized pass is discarded
and retried. Only one scalar crosses to the host, never a gradient/model table.
"""

from __future__ import annotations


def gradient_capacity(model, ray_count, *, enabled=True, requested=0):
    if not enabled:
        return 0
    count = int(getattr(model, "active_voxel_count", 0))
    capacity = requested or max(65536, 4 * ray_count)
    if count <= capacity:
        return 0
    if getattr(model, "compact_gradient_version", 0) < 1:
        raise RuntimeError("Compact native gradients require compact_gradient_version >= 1; "
                           "rebuild cuda_ad_rgb and copy-targets-python from this checkout")
    return int(capacity)


def native_gradient_result(model, arguments, replay, capacity, np=None):
    """Return an exact batch; retry overflows before any persistent update."""
    if not capacity:
        return replay(model, *arguments) if replay is not None else model.loss_and_grad(*arguments)
    if np is None:
        import numpy as np
    revision = (id(model), int(getattr(model, "topology_revision", 0)))
    if replay is not None and getattr(replay, "_gradient_capacity_revision", None) == revision:
        capacity = max(capacity, replay._gradient_capacity_hint)
    maximum = int(model.active_voxel_count)
    capacity = min(capacity, maximum)
    while True:
        # Native settings use float32. Round large capacities upward so an odd
        # model count above 2**24 cannot silently lose its final gradient row.
        encoded_capacity = capacity if capacity < 2**24 else ((capacity + 255) // 256) * 256
        options = dict(arguments[-1], compact_gradient_capacity=encoded_capacity)
        call = (*arguments[:-1], options)
        result = replay(model, *call) if replay is not None else model.loss_and_grad(*call)
        if "voxel_count" not in result or "voxel_ids" not in result:
            raise RuntimeError("Native compact gradient output is unavailable; rebuild this checkout")
        required = int(np.asarray(result["voxel_count"]).reshape(-1)[0])
        if replay is not None:
            replay._gradient_count_readbacks += 1
        if not 0 <= required <= maximum:
            raise RuntimeError("Native compact gradient count exceeds the model voxel count")
        if required <= capacity:
            if replay is not None:
                replay._gradient_capacity_revision = revision
                replay._gradient_capacity_hint = capacity
            return result
        # No optimizer step or topology vote has consumed this truncated pass.
        # Release recordings as well: they can own the old scratch allocation.
        del result, call
        if replay is not None:
            replay._gradient_overflow_retries += 1
            replay.clear()
        next_capacity = min(maximum, max(2 * capacity, ((required + 65535) // 65536) * 65536))
        # Budget checks may inspect memory or release cached references. The
        # truncated result and its recording must already be gone, and no
        # larger gradient allocation or persistent update has happened yet.
        guard = getattr(replay, "gradient_capacity_guard", None)
        if callable(guard):
            guard(model, next_capacity)
        capacity = next_capacity
