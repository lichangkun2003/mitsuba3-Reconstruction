"""GPU-resident RGB topology lifecycle.

Only compact event reductions cross to the host at epoch boundaries. Model
rows, proposal statistics, cooldowns and Adam remaps stay in device arrays.
Serialization is an explicit host border; coarse-to-fine fitting stays on CUDA.
"""

from __future__ import annotations

import math
import time

from reconstruction_growth_rgb import RGB_STAT_FIELDS, RGB_GROWTH_EVIDENCE_VERSION
from reconstruction_deletion_rgb import RGB_DELETION_EVIDENCE_VERSION


ROW_WIDTHS = {"cell_indices": 1, "cells": 1, "parent_cells": 1, "parent_ids": 1,
              "centers": 3, "log_scales": 3, "rotations": 4}
EPOCH_FIELDS = ("growth_birth", "growth_wait_start", "deletion_eligible")
DELETION_VIEW_FIELDS = ("support_views", "conflict_views", "observed_views", "unknown_views", "invalid_views")


def compact_epoch_fields_safe(manager):
    """Prove signed 32-bit storage/arithmetic safe for the configured run.

    Only host state is inspected. Conservative fallback keeps unusual saved
    states and very long schedules on the original signed 64-bit path.
    """
    np, limit = manager.np, (1 << 31) - 1
    boundaries = (manager.schedule.total_epochs, manager.level_start_epoch)
    offsets = [manager.opacity_warmup, manager.opacity_ramp,
               manager.schedule.coarse_growth_interval]
    offsets.extend(value for name, value in vars(manager.settings).items()
                   if name.endswith("_epochs") or name.endswith("_interval"))
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
           or value < 0 for value in (*boundaries, *offsets)):
        return False
    # +1 covers newborn birth timestamps and boundary - (zero wait_start - 1).
    if max(map(int, boundaries)) + sum(map(int, offsets)) + 1 > limit:
        return False
    for name in EPOCH_FIELDS:
        values = np.asarray(getattr(manager, name))
        if not np.issubdtype(values.dtype, np.integer):
            return False
        if values.size and (int(values.min()) < 0 or int(values.max()) > limit):
            return False
    return True


def device_checkpoint(model):
    """Capture references to detached model buffers, without host copies."""
    import drjit as dr
    metadata = {"resolution": int(model.resolution), "grid_min": list(model.grid_min),
                "grid_max": list(model.grid_max), "sh_count": int(model.sh_count),
                "opacity_sh_count": int(model.opacity_sh_count)}
    core = int(getattr(model, "core_resolution", model.resolution))
    if core != metadata["resolution"]:
        metadata["core_resolution"] = core
    if getattr(model, "scene_mapping", "") in ("radial_seed_shell", "projective_shell"):
        metadata["scene_mapping"] = model.scene_mapping
    if getattr(model, "scene_mapping", "") == "projective_shell":
        metadata["background_refinement"] = "preserve_physical_width"
        metadata["background_layers"] = (metadata["resolution"] - core)//2
    if getattr(model, "background_refinement", "") == "fixed_layers":
        metadata.update(background_refinement="fixed_layers",
                        background_layers=int(model.background_layers))
    return {"model_state": {key: (dr.cuda.UInt32 if key == "cell_indices" else dr.cuda.Float32)(value)
                            for key, value in model.snapshot().items()},
            "model_metadata": metadata}


def take_rows(value, indices, width=1):
    import drjit as dr
    if not len(indices):
        return dr.empty(type(value), 0)
    UInt = dr.uint32_array_t(type(indices))
    positions = dr.arange(UInt, len(indices) * width)
    address = dr.gather(UInt, indices, positions // width) * width + positions % width
    return dr.gather(type(value), value, address)


def unique_first(cells):
    """First row of each sorted group (stable, including empty input)."""
    import drjit as dr
    UInt = type(cells)
    if not len(cells):
        return dr.empty(UInt, 0)
    index = dr.arange(UInt, len(cells))
    previous = dr.gather(UInt, cells, dr.maximum(index, 1) - 1, index > 0)
    return dr.compress((index == 0) | (cells != previous))


def concat_rows(first, second):
    import drjit as dr
    if not len(first):
        return second
    if not len(second):
        return first
    return dr.concat([first, second])


class GPUTopologyLifecycle:
    def __init__(self, manager):
        import drjit as dr
        import mitsuba as mi
        self.manager, self.dr, self.mi = manager, dr, mi
        self.Float = dr.cuda.Float32
        self.UInt = dr.cuda.UInt32
        self.Byte = dr.cuda.UInt8
        self.Int64 = dr.int64_array_t(self.UInt)
        self.Epoch = dr.int32_array_t(self.UInt) if compact_epoch_fields_safe(manager) else self.Int64
        self.Double = dr.float64_array_t(self.Float)
        self.Bool = dr.mask_t(self.UInt)
        self.fields = {
            name: (self.UInt if "streak" in name else self.Epoch)(getattr(manager, name))
            for name in ("growth_birth", "growth_wait_start", "deletion_eligible",
                         "rgb_deletion_streak", "low_opacity_deletion_streak")}
        self.candidates = self.Bool(manager.candidates)
        self.cooldown_cells = self.UInt(list(manager.cooldown))
        self.cooldown_expiry = self.Int64(list(manager.cooldown.values()))
        for name in ("pending_growth", "pending_rgb_deletion"):
            pending = getattr(manager, name)
            if pending is not None:
                self.upload_pending(pending)

    def upload_pending(self, pending):
        np, dr = self.manager.np, self.dr
        for name, value in list(pending.items()):
            if isinstance(value, np.ndarray):
                cls = (self.UInt if np.issubdtype(value.dtype, np.integer) else
                       self.Double if value.dtype == np.float64 else self.Float)
                pending[name] = cls(value.reshape(-1))
        if "candidate_state" in pending:
            pending["candidate_state"] = {
                name: (self.UInt if name == "cell_indices" else self.Float)(value.reshape(-1)
                       if isinstance(value, np.ndarray) else value)
                for name, value in pending["candidate_state"].items()}

    def statistics(self, pending, *, deletion=False):
        for name in RGB_STAT_FIELDS:
            # A deduplicated evidence window contributes at most one vote per
            # camera. Ray counts and gain sums retain their original widths.
            dtype = (self.Byte if deletion and self.manager.view_count <= 255 and name in DELETION_VIEW_FIELDS
                     else self.Float if name.endswith("gain_sum") else self.UInt)
            pending[name] = self.dr.zeros(
                dtype, len(pending["cells"]))

    def eligible(self, boundary):
        start = self.fields["growth_wait_start"]
        if not len(start):
            return self.candidates
        return ~self.candidates & ((start == 0) | ((boundary >= start - 1) &
            (boundary - (start - 1) >= self.manager.settings.growth_wait_epochs)))

    def forbidden(self, boundary):
        if not len(self.cooldown_cells):
            return self.cooldown_cells
        ids = self.dr.compress(boundary < self.cooldown_expiry)
        return take_rows(self.cooldown_cells, ids)

    def prepare(self, model, epoch):
        m, dr, boundary = self.manager, self.dr, int(epoch) + 1
        timings = m.growth_prepare_stage_seconds = {}
        if not m._growth_due(boundary):
            started = time.perf_counter()
            model.clear_growth_candidates()
            m.pending_growth = None
            timings["candidate_clear"] = time.perf_counter() - started
            return False
        pending = m.pending_growth
        if (pending is None or pending.get("mode") != "rgb" or
                pending["boundary"] != boundary or pending["resolution"] != m.resolution):
            from reconstruction_growth_candidates_gpu import generate_growth_gpu
            started = time.perf_counter()
            checkpoint = device_checkpoint(model)
            timings["snapshot"] = time.perf_counter() - started
            started = time.perf_counter()
            pending = generate_growth_gpu(m, checkpoint, boundary,
                eligible=self.eligible(boundary), cooldown_cells=self.forbidden(boundary))
            # Compact dynamic allocations in generation already establish the
            # row count; finish queued kernels here for honest boundary timing.
            dr.eval(*[v for v in pending.values() if dr.is_array_v(type(v))])
            dr.sync_thread()
            timings["candidate_generation"] = time.perf_counter() - started
            started = time.perf_counter()
            count = len(pending["cells"])
            # CPU reference freezes geometry to float32 before installing the
            # candidate payload. Commit and resume must retain those exact
            # rounded values even though eligibility was computed in FP64.
            for name in ("centers", "log_scales", "rotations"):
                pending[name] = pending["candidate_state"][name]
            pending.update(mode="rgb", normalization=m.settings.growth_rgb_normalization,
                           evidence_version=RGB_GROWTH_EVIDENCE_VERSION,
                           boundary=boundary, resolution=m.resolution, seen_views=set(),
                           all_candidate_count=count,
                           candidate_cell_count=len(unique_first(pending["cells"])),
                           sh_count=checkpoint["model_metadata"]["sh_count"],
                           opacity_sh_count=checkpoint["model_metadata"]["opacity_sh_count"])
            self.statistics(pending)
            m.pending_growth = pending
            timings["proposal_packing"] = time.perf_counter() - started
        m._rgb_normalization(pending)
        started = time.perf_counter()
        if len(pending["cells"]):
            model.set_growth_candidates(pending["candidate_state"])
            timings["candidate_install"] = time.perf_counter() - started
            return True
        model.clear_growth_candidates()
        timings["candidate_clear"] = time.perf_counter() - started
        return False

    def observe(self, model, out, view_id, epoch, pixel_count, *, deletion=False):
        m, dr, cfg, boundary = self.manager, self.dr, self.manager.settings, int(epoch) + 1
        if not (m._rgb_deletion_due(boundary) if deletion else m._growth_due(boundary)):
            return
        if (not 0 <= view_id < m.view_count or view_id != int(view_id)
                or int(model.active_voxel_count) != m.count):
            raise ValueError("Unexpected RGB topology view ID or model topology")
        if not isinstance(pixel_count, int) or pixel_count < 1:
            raise ValueError("RGB topology requires positive full-image pixel count")
        name = "pending_rgb_deletion" if deletion else "pending_growth"
        pending = getattr(m, name)
        if (pending is None or pending["boundary"] != boundary or pending["resolution"] != m.resolution):
            if not deletion:
                raise ValueError("RGB growth must be prepared before tracing the evidence epoch")
            pending = {"cells": dr.arange(self.UInt, m.count), "boundary": boundary,
                       "resolution": m.resolution, "seen_views": set(),
                       "evidence_version": RGB_DELETION_EVIDENCE_VERSION}
            self.statistics(pending, deletion=True)
            m.pending_rgb_deletion = pending
        if view_id in pending["seen_views"]:
            return
        if not len(pending["cells"]):
            pending["seen_views"].add(int(view_id))
            return
        prefix = "deletion" if deletion else "growth"
        gain, observed, unknown = [(self.Float if key == "gain" else self.UInt)(out[f"{prefix}_{key}"])
                                   for key in ("gain", "observed", "unknown")]
        if any(len(value) != len(pending["cells"]) for value in (gain, observed, unknown)):
            raise ValueError("RGB topology evidence count differs from frozen layout")
        normalization = "observed" if deletion else m._rgb_normalization(pending)
        epsilon = cfg.deletion_rgb_gain_epsilon if deletion else cfg.growth_rgb_gain_epsilon
        finite = dr.isfinite(gain)
        denominator = self.Float(dr.maximum(observed, 1)) if normalization == "observed" else None
        normalized = (dr.select(finite, gain, 0) / denominator if denominator is not None
                      else dr.select(finite, gain, 0) * dr.opaque(self.Float, 1. / pixel_count))
        eligible, threshold = finite & (observed > 0), dr.opaque(self.Float, epsilon)
        increments = {"raw_gain_sum": normalized, "gain_sum": dr.select(eligible, normalized, 0),
            "support_views": self.UInt(eligible & (normalized > threshold)),
            "conflict_views": self.UInt(eligible & (normalized < -threshold)),
            "observed_views": self.UInt(eligible), "unknown_views": self.UInt(unknown > 0),
            "invalid_views": self.UInt(~finite), "observed_rays": observed, "unknown_rays": unknown}
        if deletion:
            indices = dr.arange(self.UInt, len(pending["cells"]))
            for key in RGB_STAT_FIELDS:
                if type(pending[key]) == self.Byte:
                    # CUDA has byte storage/gathers but no byte scatter-add.
                    # Add in UInt32 before narrowing. These per-view counters
                    # increase at most once per view, with at most 255 views.
                    pending[key] = self.Byte(self.UInt(pending[key]) + increments[key])
                    dr.eval(pending[key])
                    continue
                # Replace the Python wrapper before mutation: a caller holding
                # the old wrapper must still see its old value. Without such an
                # alias the old wrapper dies here, permitting native in-place
                # writes instead of another full active-voxel allocation.
                pending[key] = type(pending[key])(pending[key])
                # Each index occurs once. Integer addition is non-atomic;
                # float atomics preserve the original add.ftz.f32 semantics.
                # NoConflicts uses add.f32, which would retain subnormal sums.
                mode = dr.ReduceMode.Direct if key.endswith("gain_sum") else dr.ReduceMode.NoConflicts
                dr.scatter_reduce(dr.ReduceOp.Add, pending[key], increments[key], indices, mode=mode)
                # Finish one field's expression before proceeding, without a
                # host synchronization or references to its previous wrapper.
                dr.eval(pending[key])
        else:
            for key in RGB_STAT_FIELDS:
                pending[key] = pending[key] + increments[key]
            dr.eval(*[pending[key] for key in RGB_STAT_FIELDS])
        pending["seen_views"].add(int(view_id))
        m._device_statistics[f"rgb_{prefix}_device_views"] += 1

    def sum(self, value):
        dr = self.dr
        if not len(value):
            return 0
        return dr.sum(self.Double(value) if dr.is_float_v(type(value))
                      else dr.uint64_array_t(self.UInt)(value))

    def count(self, mask):
        return self.sum(self.UInt(mask))

    def reduce_event(self, event):
        """Evaluate compact diagnostics together, then expose Python scalars."""
        dr, arrays = self.dr, []
        def collect(obj):
            for value in obj.values():
                if isinstance(value, dict):
                    collect(value)
                elif dr.is_array_v(type(value)):
                    arrays.append(value)
        collect(event)
        dr.eval(*arrays)
        def convert(obj):
            for name, value in list(obj.items()):
                if isinstance(value, dict):
                    convert(value)
                elif dr.is_array_v(type(value)):
                    obj[name] = value[0]
        convert(event)
        for name in ("growth_rgb", "deletion_rgb"):
            if name in event and "observed_rays" in event[name]:
                data = event[name]
                total = data["observed_rays"] + data["unknown_rays"]
                data["encountered_known_fraction"] = data["observed_rays"] / total if total else None

    def mature(self, boundary):
        birth = self.fields["growth_birth"]
        if not len(birth):
            return self.dr.empty(self.Bool, 0)
        return ((birth == 0) | (boundary - birth >= self.manager.settings.deletion_protection_epochs))

    def confirmation(self, qualified, boundary, *, opacity=False):
        m, dr, cfg = self.manager, self.dr, self.manager.settings
        prefix = "low_opacity_deletion" if opacity else "rgb_deletion"
        limit = cfg.deletion_opacity_confirmation_windows if opacity else cfg.deletion_rgb_confirmation_windows
        previous = self.fields[f"{prefix}_streak"]
        consecutive = getattr(m, f"{prefix}_last_boundary") == boundary - cfg.deletion_interval
        if not m.count:
            setattr(m, f"{prefix}_last_boundary", boundary)
            return dr.empty(self.Bool, 0), {"confirmation_windows": limit,
                "previous_window_consecutive": consecutive, "streak_reset": 0,
                "awaiting_confirmation": 0, "confirmed": 0, "deleted": 0, "max_streak": 0}
        streak = previous if consecutive else dr.zeros(self.UInt, m.count)
        streak = dr.select(qualified, dr.minimum(streak, limit - 1) + 1, 0)
        self.fields[f"{prefix}_streak"] = streak
        setattr(m, f"{prefix}_last_boundary", boundary)
        selected = streak >= limit
        return selected, {"confirmation_windows": limit, "previous_window_consecutive": consecutive,
            "streak_reset": self.count((previous > 0) & (~qualified | (not consecutive))),
            "awaiting_confirmation": self.count(qualified & ~selected),
            "confirmed": self.count(selected), "deleted": 0,
            "max_streak": dr.max(streak) if m.count else 0}

    def evaluate_rgb_deletion(self, boundary, event, *, region_eligible=None):
        m, dr, cfg = self.manager, self.dr, self.manager.settings
        if not m._rgb_deletion_due(boundary):
            return None
        p = m.pending_rgb_deletion
        if p is None or p["boundary"] != boundary or p["resolution"] != m.resolution:
            p = {"cells": dr.arange(self.UInt, m.count), "seen_views": set()}
            self.statistics(p, deletion=True)
        complete = len(p["seen_views"]) == m.view_count
        # Promote compact counts before arithmetic/comparisons: configuration
        # thresholds need not fit in a byte even for a small camera set.
        mean = (self.Double(p["gain_sum"]) / self.Double(dr.maximum(self.UInt(p["observed_views"]), 1))
                if m.count else dr.empty(self.Double, 0))
        observed = self.UInt(p["observed_views"]) > 0 if m.count else dr.empty(self.Bool, 0)
        supported = self.UInt(p["support_views"]) >= cfg.deletion_rgb_min_support_views if m.count else observed
        unopposed = self.UInt(p["conflict_views"]) < cfg.deletion_rgb_max_conflict_views if m.count else observed
        valid = (self.UInt(p["invalid_views"]) == 0) & dr.isfinite(mean) if m.count else observed
        beneficial, mature = (mean > cfg.deletion_rgb_min_mean_gain if m.count else observed), self.mature(boundary)
        if region_eligible is not None:
            mature &= region_eligible
        qualified = observed & supported & unopposed & valid & beneficial & mature & complete if m.count else observed
        selected, detail = self.confirmation(qualified, boundary)
        detail.update(status="complete" if complete else "incomplete_views",
            evidence_version=RGB_DELETION_EVIDENCE_VERSION, normalization="observed",
            views=len(p["seen_views"]), required_views=m.view_count, tested_voxels=m.count,
            eligible_voxels=self.count(mature), protected_voxels=self.count(~mature),
            support_votes=self.sum(p["support_views"]), conflict_votes=self.sum(p["conflict_views"]),
            observed_views=self.sum(p["observed_views"]), views_with_unknown=self.sum(p["unknown_views"]),
            invalid_votes=self.sum(p["invalid_views"]), voxels_with_observations=self.count(observed),
            voxels_with_unknown_views=self.count(self.UInt(p["unknown_views"]) > 0) if m.count else 0,
            observed_rays=self.sum(p["observed_rays"]), unknown_rays=self.sum(p["unknown_rays"]),
            mean_gain_sum=self.sum(dr.select(dr.isfinite(mean), mean, 0)) if m.count else 0,
            rejected_no_observation=self.count(~observed), rejected_support=self.count(~supported),
            rejected_conflict=self.count(~unopposed), rejected_invalid=self.count(~valid),
            rejected_mean_gain=self.count(~beneficial), rgb_qualified=self.count(qualified))
        event["deletion_rgb"] = detail
        m.pending_rgb_deletion = None
        return selected

    def evaluate_opacity(self, model, boundary, event):
        m, dr, cfg = self.manager, self.dr, self.manager.settings
        if not m._low_opacity_deletion_due(boundary):
            return None
        coefficients = self.Float(model.snapshot()["opacity_sh"])
        if int(model.opacity_sh_count) != 1 or len(coefficients) != m.count:
            raise ValueError("Low-opacity deletion requires isotropic parameter dimensions")
        if m.count and not bool(dr.all(dr.isfinite(coefficients))[0]):
            raise ValueError("Low-opacity deletion encountered nonfinite opacity coefficients")
        threshold = float(cfg.deletion_opacity_threshold)
        coefficient_threshold = math.log(threshold / (1. - threshold)) * (2. * math.sqrt(math.pi))
        low = self.Double(coefficients) < coefficient_threshold if m.count else dr.empty(self.Bool, 0)
        mature = self.mature(boundary)
        if m.scene_mapping == "projective_shell" and not m._background_deletion_allowed():
            from reconstruction_projective import core_mask_device
            mature &= core_mask_device(self.UInt(model.snapshot()["cell_indices"]),
                                       m.resolution, m.core_resolution, dr)
        qualified = low & mature
        selected, detail = self.confirmation(qualified, boundary, opacity=True)
        detail.update(status="complete", threshold=threshold, tested_voxels=m.count,
            low_opacity_voxels=self.count(low), eligible_voxels=self.count(mature),
            protected_voxels=self.count(~mature), qualified_voxels=self.count(qualified))
        event["deletion_opacity"] = detail
        return selected

    def select_growth(self, checkpoint, boundary, event):
        m, dr, cfg, p = self.manager, self.dr, self.manager.settings, self.manager.pending_growth
        if (p is None or p.get("mode") != "rgb" or p["boundary"] != boundary or p["resolution"] != m.resolution):
            event["growth_rgb"] = {"status": "no_prepared_candidates", "normalization": cfg.growth_rgb_normalization}
            return None
        count, seen = len(p["cells"]), len(p["seen_views"])
        normalization = m._rgb_normalization(p)
        # NumPy divides float32 gains by uint32 observation counts in float64.
        # full_image uses float32 scalar division; preserve both legacy units.
        mean = ((self.Double(p["gain_sum"]) / self.Double(dr.maximum(p["observed_views"], 1))
                if normalization == "observed" else p["gain_sum"] / m.view_count)
                if count else dr.empty(self.Double, 0))
        observed = p["observed_views"] > 0 if count else dr.empty(self.Bool, 0)
        supported = p["support_views"] >= cfg.growth_rgb_min_support_views if count else observed
        unopposed = p["conflict_views"] < cfg.growth_rgb_max_conflict_views if count else observed
        valid = p["invalid_views"] == 0 if count else observed
        beneficial = dr.isfinite(mean) & (mean > cfg.growth_rgb_min_mean_gain) if count else observed
        detail = {"status": "no_candidates" if not count else "complete" if seen == m.view_count else "incomplete_views",
            "normalization": normalization, "evidence_version": RGB_GROWTH_EVIDENCE_VERSION,
            "geometric_candidates": p["all_candidate_count"], "tested_candidates": count,
            "candidate_cells": p["candidate_cell_count"], "views": seen,
            "support_votes": self.sum(p["support_views"]), "conflict_votes": self.sum(p["conflict_views"]),
            "observed_views": self.sum(p["observed_views"]), "views_with_unknown": self.sum(p["unknown_views"]),
            "invalid_votes": self.sum(p["invalid_views"]), "candidates_with_observations": self.count(observed),
            "candidates_with_unknown_views": self.count(p["unknown_views"] > 0) if count else 0,
            "observed_rays": self.sum(p["observed_rays"]), "unknown_rays": self.sum(p["unknown_rays"]),
            "mean_gain_sum": self.sum(mean), "rgb_qualified": 0, "rgb_qualified_cells": 0,
            "rejected_no_observation": self.count(~observed), "rejected_support": self.count(~supported),
            "rejected_conflict": self.count(~unopposed), "rejected_invalid": self.count(~valid),
            "rejected_mean_gain": self.count(~beneficial), "geometry_revalidated": 0,
            "geometry_rejected": 0, "geometry_revalidated_cells": 0, "same_cell_rejected": 0,
            "capacity_rejected": 0, "selected_candidates": 0}
        event["growth_rgb"] = detail
        if seen != m.view_count or not count:
            return None
        ids = dr.compress(observed & supported & unopposed & valid & beneficial)
        detail["rgb_qualified"] = len(ids)
        targets = take_rows(p["cells"], ids)
        detail["rgb_qualified_cells"] = len(unique_first(targets))
        if not len(ids):
            return None
        from reconstruction_growth_candidates_gpu import generate_growth_gpu
        current = generate_growth_gpu(m, checkpoint, boundary, target_cells=targets,
            eligible=self.eligible(boundary), cooldown_cells=self.forbidden(boundary))
        # Sorted (cell,parent-cell) keys let us map frozen proposals to current
        # parent row IDs without constructing any Python tuples or host arrays.
        U64 = dr.uint64_array_t(self.UInt)
        n = len(current["cells"])
        if not n:
            detail["geometry_rejected"] = detail["rgb_qualified"]
            return None
        keys = (U64(current["cells"]) << 32) | U64(current["parent_cells"])
        query = (U64(targets) << 32) | U64(take_rows(p["parent_cells"], ids))
        lo, hi = dr.zeros(self.UInt, len(ids)), dr.full(self.UInt, n, len(ids))
        for _ in range(n.bit_length()):
            mid = (lo + hi) // 2
            less = (mid < n) & (dr.gather(U64, keys, mid, mid < n) < query)
            lo, hi = dr.select(less, mid + 1, lo), dr.select(less, hi, mid)
        matches = (lo < n) & (dr.gather(U64, keys, lo, lo < n) == query)
        positions = dr.compress(matches)
        ids = take_rows(ids, positions)
        detail["geometry_revalidated"] = len(ids)
        detail["geometry_rejected"] = detail["rgb_qualified"] - len(ids)
        if not len(ids):
            return None
        parent_ids = dr.gather(self.UInt, current["parent_ids"], take_rows(lo, positions))
        gains, cells = take_rows(mean, ids), take_rows(p["cells"], ids)
        # Stable lexsort: initial rows already (cell,parent-cell) ordered.
        order = dr.argsort(gains, descending=True)
        order = take_rows(order, dr.argsort(take_rows(cells, order)))
        per_cell = take_rows(order, unique_first(take_rows(cells, order)))
        detail["geometry_revalidated_cells"] = len(per_cell)
        detail["same_cell_rejected"] = len(ids) - len(per_cell)
        ids, gains, cells, parent_ids = [take_rows(v, per_cell) for v in (ids, gains, cells, parent_ids)]
        ranking = dr.argsort(gains, descending=True)  # equal gain keeps cell order
        capacity = max(0, cfg.max_active_voxels - m.count)
        winner_positions = take_rows(ranking, dr.arange(self.UInt, min(len(ranking), capacity)))
        # The original commit path applies np.unique(cells) before appending.
        winner_positions = take_rows(winner_positions, dr.argsort(take_rows(cells, winner_positions)))
        ids, parent_ids = take_rows(ids, winner_positions), take_rows(parent_ids, winner_positions)
        detail["capacity_rejected"] = len(ranking) - len(ids)
        detail["selected_candidates"] = len(ids)
        widths = dict(ROW_WIDTHS, radiance_sh=checkpoint["model_metadata"]["sh_count"] * 3,
                      opacity_sh=checkpoint["model_metadata"]["opacity_sh_count"])
        selected = {name: take_rows(p[name], ids, width) for name, width in widths.items() if name in p}
        selected["parent_ids"] = parent_ids
        # Detach the compact winners from the complete proposal tables before
        # model concatenation and Adam remapping allocate their replacements.
        dr.eval(*selected.values())
        return selected

    def remap(self, survivors, new_count, boundary):
        dr, kept = self.dr, len(survivors)
        if not new_count:
            self.fields = {name: dr.empty(type(value), 0) for name, value in self.fields.items()}
            self.candidates = dr.empty(self.Bool, 0)
            self.manager.count = 0
            return
        index = dr.arange(self.UInt, new_count)
        source = dr.gather(self.UInt, survivors, index, index < kept)
        remapped = {}
        for name, old in self.fields.items():
            newborn = (boundary + 1 if name in ("growth_birth", "growth_wait_start") else
                       boundary + self.manager.settings.growth_protection_epochs if name == "deletion_eligible" else 0)
            # Birth and wait histories often share storage and newborn values.
            # Preserve that sharing across separate evaluations, without
            # retaining any old arrays or comparing full device buffers.
            key = (type(old), old.index, newborn)
            if key in remapped:
                self.fields[name] = type(old)(remapped[key])
            else:
                self.fields[name] = dr.select(index < kept, dr.gather(type(old), old, source, index < kept), newborn)
                dr.eval(self.fields[name])
                remapped[key] = self.fields[name]
            old = None
        self.candidates = dr.gather(self.Bool, self.candidates, source, index < kept)
        self.manager.count = new_count
        dr.eval(*self.fields.values(), self.candidates)

    def refine(self, model, optimizer, boundary, event):
        """Replace fitted children without copying the parent model to NumPy."""
        from reconstruction_refinement_gpu import refine_state_gpu
        from reconstruction_topology import model_checkpoint
        m, dr = self.manager, self.dr
        started = time.perf_counter()
        replacement = refine_state_gpu(device_checkpoint(model), m.schedule, m.settings.max_active_voxels)
        dr.sync_thread()
        fitted = time.perf_counter()
        if m.schedule.save_each_level:
            # This opt-in API explicitly requests a serializable old-level
            # checkpoint. Normal training disables it and incurs no readback.
            m.before_refine_checkpoint = {
                **model_checkpoint(model), "optimizer_state": optimizer.state_dict(),
                "topology_state": m.state_dict()}
            event["pre_refine_checkpoint_available"] = True
        checkpointed = time.perf_counter()
        m.resolution = replacement["model_metadata"]["resolution"]
        m.core_resolution = replacement["model_metadata"].get("core_resolution", m.resolution)
        m.level_start_epoch = boundary
        # Host lifecycle arrays remain empty placeholders until serialization.
        # Allocate actual new-row fields directly on the GPU; old row histories
        # are discarded at refinement just as in the NumPy implementation.
        m._initialize_lifecycle(0)
        m.count = len(replacement["model_state"]["cell_indices"])
        self.fields = {name: dr.zeros(type(value), m.count) for name, value in self.fields.items()}
        self.candidates = dr.zeros(self.Bool, m.count)
        self.cooldown_cells = dr.empty(self.UInt, 0)
        self.cooldown_expiry = dr.empty(self.Int64, 0)
        m.completed_windows = 0
        m.cooldown = {}
        m.pending_growth = None
        m._growth_world_cache = m._growth_device = m._rgb_growth_device = None
        m._gpu_lifecycle = self
        optimizer.remap(dr.empty(self.UInt, 0), m.count)
        dr.eval(*self.fields.values(), self.candidates)
        event.update(refined=True, resolution=m.resolution, active_voxel_count=m.count,
                     refinement_backend="cuda", refinement_stage_seconds={
                         "fit": fitted - started, "pre_refine_checkpoint": checkpointed - fitted,
                         "reset": time.perf_counter() - checkpointed})
        if m.core_resolution != m.resolution:
            event["core_resolution"] = m.core_resolution
        return replacement, event

    def finish(self, model, optimizer, boundary):
        m, dr, cfg = self.manager, self.dr, self.manager.settings
        m.before_refine_checkpoint = None
        model.clear_growth_candidates()
        event = {"completed_epochs": boundary, "resolution": m.resolution, "deleted": 0, "grown": 0,
            "refined": False, "finished": False, "level_budget": m.level_budget,
            "growth_frozen": m.growth_frozen(boundary), "growth_freeze_last_epochs": cfg.growth_freeze_last_epochs,
            "growth_mode": "rgb_online", "growth_level_epoch": boundary - m.level_start_epoch,
            "growth_rgb_normalization": cfg.growth_rgb_normalization, "growth_warmup_epochs": m.growth_warmup,
            "growth_warmup_active": boundary - m.level_start_epoch <= m.growth_warmup,
            "growth_evidence_requested": m._growth_due(boundary),
            "deletion_mode": "rgb_online+low_opacity" if cfg.use_low_opacity_deletion else "rgb_online",
            "deletion_level_epoch": boundary - m.level_start_epoch,
            "deletion_warmup_epochs": cfg.deletion_warmup_epochs,
            "deletion_warmup_active": boundary - m.level_start_epoch <= cfg.deletion_warmup_epochs,
            "deletion_frozen": m.deletion_frozen(boundary),
            "deletion_freeze_last_epochs": cfg.deletion_freeze_last_epochs,
            "deletion_evidence_requested": m._rgb_deletion_due(boundary), "topology_backend": "cuda"}
        if m.core_resolution != m.resolution:
            event["core_resolution"] = m.core_resolution
        if cfg.use_low_opacity_deletion:
            event["deletion_opacity_requested"] = m._low_opacity_deletion_due(boundary)
        if (m.schedule.enabled and m.core_resolution < m.schedule.target_resolution and
                boundary - m.level_start_epoch >= m.level_budget):
            return self.refine(model, optimizer, boundary, event)
        region_eligible = None
        if m.scene_mapping == "projective_shell" and m._rgb_deletion_due(boundary):
            from reconstruction_projective import core_mask_device
            region_eligible = core_mask_device(self.UInt(model.snapshot()["cell_indices"]),
                                               m.resolution, m.core_resolution, dr)
        rgb = self.evaluate_rgb_deletion(boundary, event, region_eligible=region_eligible)
        opacity = self.evaluate_opacity(model, boundary, event)
        event["evidence_window_evaluated"] = rgb is not None
        masks = [mask for mask in (rgb, opacity) if mask is not None]
        checkpoint = None
        if masks:
            # Compact decisions and scalar diagnostics must not keep the full
            # deletion evidence window alive throughout model/Adam remapping.
            dr.eval(*masks, *self.fields.values())
            self.reduce_event(event)
            delete = masks[0]
            for mask in masks[1:]:
                delete = delete | mask
            survivors = dr.compress(~delete)
            if len(survivors) != m.count:
                checkpoint = device_checkpoint(model)
                removed = dr.compress(delete)
                self.cooldown_cells = concat_rows(self.cooldown_cells, take_rows(checkpoint["model_state"]["cell_indices"], removed))
                self.cooldown_expiry = concat_rows(self.cooldown_expiry,
                    dr.full(self.Int64, boundary + cfg.deletion_cooldown_epochs, len(removed)))
                old_count = m.count
                checkpoint["model_state"] = {name: take_rows(value, survivors, len(value) // old_count)
                                              for name, value in checkpoint["model_state"].items()}
                event["deleted"] = old_count - len(survivors)
                if rgb is not None:
                    event["deletion_rgb"]["deleted"] = self.count(rgb)
                if opacity is not None:
                    event["deletion_opacity"]["deleted"] = self.count(opacity)
                optimizer.remap(survivors, len(survivors))
                self.remap(survivors, len(survivors), boundary)
                self.candidates = dr.zeros(self.Bool, m.count)
                m.completed_windows = 0
        if m._growth_due(boundary):
            if checkpoint is None:
                checkpoint = device_checkpoint(model)
            selected = self.select_growth(checkpoint, boundary, event)
            self.reduce_event(event)
            m.pending_growth = None
            m._growth_world_cache = m._growth_device = m._rgb_growth_device = None
            if selected is not None and len(selected["cells"]):
                added = len(selected["cells"])
                old_count, new_count = m.count, m.count + added
                survivors = dr.arange(self.UInt, old_count)
                # Retire the old Adam tables before allocating the appended
                # model. Otherwise its large radiance array competes with
                # both the old and new radiance moments during remapping.
                optimizer.remap(survivors, new_count)
                self.remap(survivors, new_count, boundary)
                for name, value in checkpoint["model_state"].items():
                    addition = selected["cells" if name == "cell_indices" else name]
                    checkpoint["model_state"][name] = concat_rows(value, type(value)(addition))
                # Lazy appends own only the materialized winner arrays now.
                # Other selected fields and loop aliases have no later consumer.
                selected = value = addition = None
                event["grown"] = added
            elif not event["deleted"]:
                checkpoint = None
        m.pending_growth, m.pending_rgb_deletion = None, None
        m._growth_world_cache = m._growth_device = m._rgb_growth_device = m._rgb_deletion_device = None
        if len(self.cooldown_cells):
            keep = dr.compress(self.cooldown_expiry > boundary)
            self.cooldown_cells, self.cooldown_expiry = take_rows(self.cooldown_cells, keep), take_rows(self.cooldown_expiry, keep)
        final = (boundary - m.level_start_epoch >= m.level_budget if m.schedule.enabled
                 else boundary >= m.schedule.total_epochs)
        event.update(finished=final, active_voxel_count=m.count)
        self.reduce_event(event)
        return checkpoint, event

    def materialize(self):
        """Serialize once at explicit checkpoints; never called per view."""
        m, np, dr = self.manager, self.manager.np, self.dr
        dr.eval(*self.fields.values(), self.candidates, self.cooldown_cells, self.cooldown_expiry)
        for name, value in self.fields.items():
            # Keep the existing NumPy/checkpoint schema independent of the
            # narrower resident representation chosen for ordinary schedules.
            setattr(m, name, np.array(value, dtype=np.int64 if name in EPOCH_FIELDS else None, copy=True))
        m.candidates = np.array(self.candidates, copy=True)
        words = (m.view_count + 31) // 32
        m._foreground_masks = np.zeros((m.count, words), dtype=np.uint32)
        m._background_masks = np.zeros((m.count, words), dtype=np.uint32)
        m.qualified = np.zeros(m.count, dtype=bool)
        m.cooldown = dict(zip(map(int, np.array(self.cooldown_cells)), map(int, np.array(self.cooldown_expiry))))
        for key in ("pending_growth", "pending_rgb_deletion"):
            p = getattr(m, key)
            if p is None:
                continue
            for name, value in list(p.items()):
                if dr.is_array_v(type(value)):
                    dtype = np.uint32 if key == "pending_rgb_deletion" and name in DELETION_VIEW_FIELDS else None
                    array = np.array(value, dtype=dtype, copy=True)
                    width = ROW_WIDTHS.get(name, 1)
                    if name == "radiance_sh":
                        array = array.reshape(len(p["cells"]), -1, 3) if len(p["cells"]) else array.reshape(0, p.get("sh_count", 0), 3)
                    elif name == "opacity_sh":
                        array = array.reshape(len(p["cells"]), -1) if len(p["cells"]) else array.reshape(0, p.get("opacity_sh_count", 1))
                    elif width > 1:
                        array = array.reshape(-1, width)
                    p[name] = array
                elif name == "candidate_state":
                    p[name] = {field: np.array(buf, copy=True) for field, buf in value.items()}


def get_gpu_lifecycle(manager, model):
    """Use the resident path only for the RGB algorithm, preserving legacy CPU semantics."""
    cfg = manager.settings
    if (not getattr(cfg, "gpu_topology", True) or not cfg.use_rgb_growth or not cfg.use_rgb_deletion
            or type(model).__module__.split(".")[0] != "mitsuba"):
        return None
    device = getattr(manager, "_gpu_lifecycle", None)
    if device is not None:
        return device
    import drjit as dr
    value = model.snapshot()["cell_indices"]
    if dr.backend_v(type(value)) != dr.JitBackend.CUDA:
        return None
    device = manager._gpu_lifecycle = GPUTopologyLifecycle(manager)
    return device
