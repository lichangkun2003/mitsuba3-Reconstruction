"""Budgeted background-only refinement at validated epoch boundaries.

Priorities cover one complete training epoch. Their collection boundaries are
offset from the ordinary grow/prune cadence, and need no checkpoint history.
An ordinary topology transaction takes precedence; no stale row statistics are
ever reused after rows change.
"""

from __future__ import annotations

import time


class AdaptiveBackgroundController:
    def __init__(self, config, topology, views, dr):
        self.config, self.topology, self.views, self.dr = config, topology, views, dr
        self.enabled = config.data.background_refinement == "adaptive"
        self.collecting = False
        self.priority = None
        self.projection_cache = {}

    def begin_epoch(self, model, boundary):
        cfg = self.config.topology
        self.collecting = (self.enabled and boundary > cfg.deletion_warmup_epochs
            and not self.topology.growth_frozen(boundary)
            and (boundary - 1) % cfg.background_refine_interval == 0)
        self.priority = self.dr.zeros(self.dr.cuda.Float32, int(model.active_voxel_count)) if self.collecting else None
        return self.collecting

    def observe(self, result):
        priority = result.pop("adaptive_priority", None)
        if not self.collecting:
            return
        if priority is None or len(priority) != len(self.priority):
            raise RuntimeError("Adaptive background priority must cover every active model row")
        # Native statistics use the variant Float type. Keep the epoch
        # accumulator detached, even when the renderer uses cuda.ad.Float.
        self.priority += self.dr.cuda.Float32(priority)
        self.dr.eval(self.priority)

    def finish(self, model, optimizer, boundary, ordinary_state, event):
        try:
            return self._finish(model, optimizer, boundary, ordinary_state, event)
        finally:
            self.priority = None
            self.collecting = False

    def _finish(self, model, optimizer, boundary, ordinary_state, event):
        if not self.collecting:
            return ordinary_state, event
        if ordinary_state is not None:
            event["background_adaptive"] = {"status": "deferred_topology_changed"}
            return ordinary_state, event
        from reconstruction_adaptive_background import (
            select_adaptive_background_device, refine_adaptive_background_device)
        from reconstruction_topology_gpu import device_checkpoint, get_gpu_lifecycle, take_rows
        from reconstruction_memory import TrainingMemoryError
        from reconstruction_world_shell import ADAPTIVE_TAG

        dr, m, cfg = self.dr, self.topology, self.config.topology
        checkpoint = device_checkpoint(model)
        cells = checkpoint["model_state"]["cell_indices"]
        if not len(cells):
            event["background_adaptive"] = {"status": "no_candidates", "background_leaves": 0}
            return None, event
        background_count = int(dr.count((cells & ADAPTIVE_TAG) != 0)[0])
        capacity = min(cfg.background_refine_max_new_leaves,
                       cfg.background_max_active_voxels - background_count,
                       cfg.max_active_voxels - len(cells))
        detail = dict(status="no_candidates", background_leaves=background_count,
                      new_leaf_budget=max(0, capacity), priority_scope="complete_training_epoch",
                      pixel_threshold=cfg.background_refine_pixel_threshold)
        event["background_adaptive"] = detail
        if capacity <= 0:
            detail["status"] = "leaf_budget_reached"
            return None, event
        started = time.perf_counter()
        lifecycle = get_gpu_lifecycle(m, model)
        mature = lifecycle.mature(boundary)
        priority = dr.select(mature, self.priority, 0.)
        parents = select_adaptive_background_device(checkpoint, self.views, priority, dr,
            pixel_threshold=cfg.background_refine_pixel_threshold,
            max_new_leaves=capacity,
            max_depth=self.config.data.background_max_resolution.bit_length() - 2,
            projection_cache=self.projection_cache)
        limit = max(1, int(background_count * cfg.background_refine_fraction))
        if len(parents) > limit:
            parents = take_rows(parents, dr.arange(dr.cuda.UInt32, limit))
        # New leaves need time to establish geometry before another split.
        parents = take_rows(parents, dr.compress(dr.gather(dr.cuda.Bool, mature, parents)))
        detail["selected_parents"] = len(parents)
        selected_at = time.perf_counter()
        detail["selection_seconds"] = selected_at - started
        if not len(parents):
            detail["seconds"] = selected_at - started
            return None, event
        guard = getattr(m, "refinement_memory_guard", None)
        options = {} if guard is None else {
            "before_allocate": lambda request: guard(model, optimizer, request)}
        try:
            replacement = refine_adaptive_background_device(checkpoint, parents, m.schedule,
                min(cfg.max_active_voxels, len(cells) + capacity), dr, **options)
        except TrainingMemoryError as error:
            # No lifecycle, parameter or optimizer mutation has happened.
            # Keep training the current leaves and retry at a later boundary.
            detail.update(status="deferred_memory_budget", reason=str(error),
                          seconds=time.perf_counter() - started)
            return None, event
        fitted_at = time.perf_counter()
        detail["refinement_seconds"] = fitted_at - selected_at
        if not replacement.get("replaced", 0):
            detail["status"] = "no_representable_children"
            detail["seconds"] = fitted_at - started
            return None, event
        source = replacement.pop("source_rows")
        survivors = take_rows(source, dr.compress(source != 0xffffffff))
        count = len(replacement["model_state"]["cell_indices"])
        optimizer.remap(survivors, count)
        optimizer_at = time.perf_counter()
        detail["optimizer_remap_seconds"] = optimizer_at - fitted_at
        lifecycle.remap(survivors, count, boundary)
        index = dr.arange(dr.cuda.UInt32, count)
        lifecycle.fields["deletion_eligible"] = dr.select(index < len(survivors),
            lifecycle.fields["deletion_eligible"], boundary + max(
                cfg.deletion_protection_epochs, cfg.growth_protection_epochs))
        dr.eval(*lifecycle.fields.values())
        detail["lifecycle_remap_seconds"] = time.perf_counter() - optimizer_at
        detail["stage_timing"] = "host wall time; device work synchronized only where already required"
        m.pending_growth = m.pending_rgb_deletion = None
        m._growth_world_cache = m._growth_device = m._rgb_growth_device = m._rgb_deletion_device = None
        detail.update(status="refined", seconds=time.perf_counter() - started,
                      **{name: int(replacement.pop(name, 0)) for name in
                         ("selected", "replaced", "children", "added")})
        event.update(background_refined=True, active_voxel_count=count)
        return replacement, event
