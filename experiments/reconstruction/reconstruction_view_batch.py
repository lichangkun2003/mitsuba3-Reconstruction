"""Scheduling and checkpoint rules for one complete view per GPU.

View gradients are sums over image pixels. A batch combines these sums and
passes its real number of views to Adam as ``sample_count``; padded ranks do
not increase that divisor. These helpers do not initialize a CUDA runtime.
"""

from __future__ import annotations

from numbers import Integral

from reconstruction_collective import GRADIENT_KEYS, MAX_COUNTER_KEYS, SUM_COUNTER_KEYS


def _integer(value, name, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def iter_view_batches(order, world_size):
    """Visit each supplied view exactly once; the final batch is not padded."""
    world_size = _integer(world_size, "world_size")
    order = tuple(_integer(index, "view index", minimum=0) for index in order)
    if len(set(order)) != len(order):
        raise ValueError("A view order cannot contain duplicate views")
    for begin in range(0, len(order), world_size):
        yield order[begin:begin + world_size]


def batch_view_seeds(rng, batch_size):
    """Advance the canonical RNG identically on every rank.

    Each worker samples its assigned view with a separate RNG initialized from
    the corresponding seed. Sampling that local image must not consume this
    canonical stream, whose state is checkpointed by the primary rank.
    """
    batch_size = _integer(batch_size, "batch_size")
    return tuple(int(value) for value in rng.integers(0, 1 << 63, size=batch_size))


def batch_optimizer_steps_per_epoch(view_count, world_size):
    view_count = _integer(view_count, "view_count")
    world_size = _integer(world_size, "world_size")
    return (view_count + world_size - 1) // world_size


def view_batch_protocol(world_size):
    """Persist the update rule, independently of the number of processed views."""
    return {"version": 1, "mode": "view_batch",
            "views_per_update": _integer(world_size, "world_size"),
            "gradient_reduction": "mean_of_view_sums"}


def validate_batch_resume(training_state, world_size, view_count, completed_epochs):
    """Return the scalar Adam-update count after validating an epoch checkpoint.

    Old checkpoints describe one update per view (including old multi-GPU
    pixel sharding). They are valid for the single-GPU path, but cannot silently
    become a different mini-batch optimization trajectory when resumed.
    """
    completed_epochs = _integer(completed_epochs, "completed_epochs", minimum=0)
    per_epoch = batch_optimizer_steps_per_epoch(view_count, world_size)
    expected_views = completed_epochs * int(view_count)
    saved_views = _integer(training_state.get("global_step"), "global_step", minimum=0)
    if saved_views != expected_views:
        raise ValueError("Checkpoint global_step differs from its completed training views")
    saved = training_state.get("update_protocol")
    expected = view_batch_protocol(world_size)
    if saved is None:
        if world_size != 1:
            raise ValueError("Legacy per-view/pixel-shard checkpoints cannot resume as view-batch training; "
                             "start a new experiment for this optimizer update rule")
        steps = training_state.get("optimizer_step", expected_views)
    else:
        if saved != expected:
            raise ValueError("Checkpoint update protocol or views_per_update differs; "
                             "resume with the original GPU count or start a new experiment")
        if "optimizer_step" not in training_state:
            raise ValueError("View-batch checkpoint is missing optimizer_step")
        steps = training_state["optimizer_step"]
    steps = _integer(steps, "optimizer_step", minimum=0)
    if steps != completed_epochs * per_epoch:
        raise ValueError("Checkpoint optimizer_step differs from its completed view batches")
    return steps


def zero_view_partial(model, mi, dr):
    """An inactive rank's neutral input, with native gradient buffer widths.

    It contains no topology evidence: independent view votes must never be
    merged into one vote by the gradient collective. Callers add zero loss
    component scalars when distortion reporting is enabled on active ranks.
    """
    count = _integer(int(model.active_voxel_count), "active_voxel_count", minimum=0)
    widths = {"radiance_sh": 3 * int(model.sh_count),
              "opacity_sh": int(model.opacity_sh_count),
              "centers": 3, "log_scales": 3, "rotation_tangent": 3}
    accumulator = {"gradients": {
        name: dr.zeros(mi.Float, count * widths[name]) for name in GRADIENT_KEYS}}
    accumulator.update({name: dr.zeros(mi.UInt32, count)
                        for name in (*SUM_COUNTER_KEYS, *MAX_COUNTER_KEYS)})
    return accumulator, mi.Float(0), mi.Bool(True), None, None
