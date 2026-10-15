"""Falcor topology lifecycle and coarse-to-fine transfer on NumPy arrays.

Continuous parameters/Adam live on the GPU. Host snapshots are taken only for
scheduled topology work and checkpoints. Native model replacement is left to
the training driver, so allocation failure cannot partially replace its model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass
class TopologySettings:
    enable_deletion: bool = True
    enable_growth: bool = True
    use_growth_evidence: bool = True
    growth_use_26_neighbors: bool = True
    growth_evidence_penetration: float = .05
    growth_min_supporting_parents: int = 2
    growth_min_foreground_views: int = 3
    growth_background_veto_views: int = 2
    growth_min_alpha_deficit: float = .05
    growth_face_penetration: float = 1.
    growth_interval: int = 10
    growth_shrink: float = .7
    growth_contact_offset: float = .2
    growth_initial_opacity: float = .5
    growth_wait_epochs: int = 5
    growth_protection_epochs: int = 5
    deletion_cooldown_epochs: int = 5
    foreground_alpha_min: float = .95
    background_alpha_max: float = 1e-4
    min_deletion_conflict_views: int = 5
    deletion_conflict_support_ratio: int = 2
    evidence_interval: int = 5
    deletion_interval: int = 10
    max_active_voxels: int = 67108864


@dataclass
class CoarseToFineConfig:
    enabled: bool = True
    start_resolution: int = 256
    target_resolution: int = 512
    total_epochs: int = 1000
    level_weights: tuple[int, int, int, int] = (19, 12, 14, 25)
    parent_opacity_threshold: float = 0.
    opacity_optical_depth_scale: float = .65
    child_face_overlap: float = .1
    enable_coarse_growth: bool = True
    coarsest_growth_face_penetration: float = 1.4
    growth_finer_level_multiplier: float = 2.5
    coarse_growth_interval: int = 10
    save_each_level: bool = True


def level_budgets(start: int, target: int, total: int, weights=(19, 12, 14, 25)) -> dict[int, int]:
    """Integer interpolation from CoarseToFine.cpp, including one epoch per level."""
    if (start < 1 or target < start or target > 1024 or
            start & (start - 1) or target & (target - 1)):
        raise ValueError("Coarse-to-fine requires power-of-two start <= target <= 1024")
    resolutions = [start]
    while resolutions[-1] < target:
        resolutions.append(resolutions[-1] * 2)
    count = len(resolutions)
    if total < count or total > 1000000 or len(weights) != 4 or any(w <= 0 for w in weights):
        raise ValueError("Invalid coarse-to-fine total or four positive level weights")
    if count == 1:
        return {start: total}
    values = []
    for level in range(count):
        segment, fraction = divmod(3 * level, count - 1)
        values.append(weights[segment] * (count - 1 - fraction) + weights[min(segment + 1, 3)] * fraction)
    denominator, cumulative = sum(values), 0
    budgets = {}
    for resolution, weight in zip(resolutions, values):
        previous = cumulative
        cumulative += weight
        budgets[resolution] = 1 + (total - count) * cumulative // denominator - (total - count) * previous // denominator
    return budgets


def model_checkpoint(model) -> dict:
    import numpy as np
    import drjit as dr
    snapshot = model.snapshot()
    dr.eval(*snapshot.values())
    dr.sync_thread()
    return {
        "model_state": {name: np.array(value, copy=True) for name, value in snapshot.items()},
        "model_metadata": {"resolution": int(model.resolution), "grid_min": list(model.grid_min),
                           "grid_max": list(model.grid_max), "sh_count": int(model.sh_count),
                           "opacity_sh_count": int(model.opacity_sh_count)},
    }


def _coordinates(cells, resolution, np):
    cells = np.asarray(cells, dtype=np.int64)
    return np.column_stack((cells % resolution, cells // resolution % resolution, cells // (resolution * resolution)))


def _addresses(cells, resolution):
    return cells[:, 0] + resolution * (cells[:, 1] + resolution * cells[:, 2])


def _matching_growth_pairs(current, pending, np):
    """Exact membership of two uint32 cell addresses without Python tuples."""
    # Cell addresses come from the native model's uint32 cell_indices. Packing
    # both into uint64 is bijective, including addresses above signed int32.
    def keys(batch):
        return (batch["parent_cells"].astype(np.uint64) << np.uint64(32)) | batch["cells"].astype(np.uint64)

    allowed, proposed = keys(current), keys(pending)
    allowed.sort()
    positions = np.searchsorted(allowed, proposed)
    valid = positions < len(allowed)
    matches = np.zeros(len(proposed), dtype=bool)
    matches[valid] = allowed[positions[valid]] == proposed[valid]
    return matches


def _quaternion_matrix(quaternions, np):
    q = np.asarray(quaternions, dtype=np.float64).reshape(-1, 4).copy()
    norm = np.sum(q * q, axis=1)
    valid = np.isfinite(norm) & (norm > 1e-20)
    q[~valid] = (1., 0., 0., 0.)
    q[valid] /= np.sqrt(norm[valid, None])
    w, x, y, z = q.T
    result = np.empty((len(q), 3, 3), dtype=np.float64)
    result[:, 0, 0], result[:, 0, 1], result[:, 0, 2] = 1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)
    result[:, 1, 0], result[:, 1, 1], result[:, 1, 2] = 2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)
    result[:, 2, 0], result[:, 2, 1], result[:, 2, 2] = 2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)
    return result, q


def _geometry(state, metadata, np):
    center = np.asarray(state["centers"], dtype=np.float64).reshape(-1, 3)
    axes = np.exp(np.clip(np.asarray(state["log_scales"], dtype=np.float64).reshape(-1, 3), -30., 30.))
    rotation, quaternion = _quaternion_matrix(state["rotations"], np)
    size = (np.asarray(metadata["grid_max"]) - metadata["grid_min"]) / metadata["resolution"]
    transform = rotation / axes[:, None, :]
    metric = np.einsum("nik,njk->nij", transform, transform) * size[None, :, None] * size[None, None, :]
    extent = np.sqrt(np.sum((rotation * axes[:, None, :]) ** 2, axis=2)) / size
    return center, axes, rotation, quaternion, size, metric, extent


def _contact(center, metric, offset, depth, np):
    """Exact finite face/edge/corner contact, vectorized over parents."""
    crossed = np.flatnonzero(offset)
    contact = center.copy()
    for axis in crossed:
        contact[:, axis] = 1. + depth if offset[axis] > 0 else -depth
    if len(crossed) == 1:
        n = int(crossed[0])
        u, v = (n + 1) % 3, (n + 2) % 3
        qn = contact[:, n] - center[:, n]
        determinant = metric[:, u, u] * metric[:, v, v] - metric[:, u, v] ** 2
        best = np.full(len(center), np.inf)
        best_contact = contact.copy()
        for candidate in range(5):
            point = contact.copy()
            valid = np.ones(len(center), dtype=bool)
            if candidate == 0:
                valid = determinant > 1e-30
                denominator = np.where(valid, determinant, 1.)
                point[:, u] = center[:, u] - qn * (metric[:, v, v] * metric[:, u, n] - metric[:, u, v] * metric[:, v, n]) / denominator
                point[:, v] = center[:, v] - qn * (metric[:, u, u] * metric[:, v, n] - metric[:, u, v] * metric[:, u, n]) / denominator
                valid &= (point[:, u] >= 0) & (point[:, u] <= 1) & (point[:, v] >= 0) & (point[:, v] <= 1)
            else:
                fixed, free = (u, v) if candidate <= 2 else (v, u)
                point[:, fixed] = (candidate - 1) & 1
                point[:, free] = np.clip(center[:, free] - (
                    metric[:, free, n] * qn + metric[:, free, fixed] * (point[:, fixed] - center[:, fixed])) /
                    np.maximum(metric[:, free, free], 1e-30), 0., 1.)
            q = point - center
            value = np.einsum("ni,nij,nj->n", q, metric, q)
            better = valid & (value < best)
            best[better], best_contact[better] = value[better], point[better]
        return best < 1 - 1e-5, best_contact
    if len(crossed) == 2:
        free = int(np.flatnonzero(offset == 0)[0])
        coupling = np.sum(metric[:, free] * (contact - center), axis=1)
        contact[:, free] = np.clip(center[:, free] - coupling / np.maximum(metric[:, free, free], 1e-30), 0., 1.)
    q = contact - center
    return np.einsum("ni,nij,nj->n", q, metric, q) < 1 - 1e-5, contact


def _neighbor_offsets(use_26, np):
    if use_26:
        return [np.array((x, y, z)) for z in (-1, 0, 1) for y in (-1, 0, 1)
                for x in (-1, 0, 1) if (x, y, z) != (0, 0, 0)]
    return [np.eye(3, dtype=np.int64)[axis] * sign for axis in range(3) for sign in (-1, 1)]


def _closest_box(metric, lower, upper, np, inverse=None):
    """Exact constrained minimum of q^T metric q over a box (batched)."""
    best = np.clip(np.zeros_like(lower), lower, upper)
    minimum = np.einsum("ni,nij,nj->n", best, metric, best)
    if inverse is None:
        inverse = np.linalg.inv(metric)
    for normal in range(3):
        for side in (0, 1):
            bound = (lower if side == 0 else upper)[:, normal]
            point = inverse[:, :, normal] * (bound / inverse[:, normal, normal])[:, None]
            point[:, normal] = bound
            valid = np.all(point >= lower, axis=1) & np.all(point <= upper, axis=1)
            value = np.einsum("ni,nij,nj->n", point, metric, point)
            better = valid & (value < minimum)
            minimum[better], best[better] = value[better], point[better]
    for free in range(3):
        a, b = (free + 1) % 3, (free + 2) % 3
        for side in range(4):
            point = np.zeros_like(lower)
            point[:, a] = (upper if side & 1 else lower)[:, a]
            point[:, b] = (upper if side & 2 else lower)[:, b]
            point[:, free] = np.clip(-(metric[:, free, a] * point[:, a] + metric[:, free, b] * point[:, b]) /
                                     metric[:, free, free], lower[:, free], upper[:, free])
            value = np.einsum("ni,nij,nj->n", point, metric, point)
            better = value < minimum
            minimum[better], best[better] = value[better], point[better]
    return minimum, best


def refine_state(checkpoint: dict, config: CoarseToFineConfig, max_active_voxels=67108864) -> dict:
    """Port of Shader/RefineGrid.cs.slang, retaining only fitted child volumes."""
    import numpy as np
    state, metadata = checkpoint["model_state"], dict(checkpoint["model_metadata"])
    n = len(state["cell_indices"])
    if n == 0:
        metadata["resolution"] *= 2
        return {"model_metadata": metadata,
                "model_state": {key: np.array(value, copy=True) for key, value in state.items()}}
    center, axes, rotation, quaternion, size, metric, extent = _geometry(state, metadata, np)
    inverse_metric = np.linalg.inv(metric)
    old_cells = _coordinates(state["cell_indices"], metadata["resolution"], np)
    radiance = np.asarray(state["radiance_sh"]).reshape(n, metadata["sh_count"], 3)
    opacity = np.asarray(state["opacity_sh"]).reshape(n, metadata["opacity_sh_count"])
    retain = np.ones(n, dtype=bool)
    y0 = 1 / (2 * math.sqrt(math.pi))
    if config.parent_opacity_threshold > 0:
        upper = opacity[:, 0] * y0
        for k in range(1, metadata["opacity_sh_count"]):
            degree = int(math.sqrt(k))
            upper += np.abs(opacity[:, k]) * math.sqrt((2 * degree + 1) / (4 * math.pi))
        retain = upper >= math.log(config.parent_opacity_threshold / (1 - config.parent_opacity_threshold))
    fine_size = size * .5
    minimum_ratio = np.maximum(fine_size * 1e-4, 1e-8) / axes
    base_lo = np.max(minimum_ratio, axis=1)
    overlap = .5 * np.clip(config.child_face_overlap, 0., .2)
    base_hi = np.minimum(1., np.min((.25 + overlap) / extent, axis=1))
    blocks = []
    for child in range(8):
        bits = np.array((child & 1, (child >> 1) & 1, (child >> 2) & 1))
        lower = .5 * bits - center
        upper = lower + .5
        middle = lower + .25
        interior = np.ones(n, dtype=bool)
        for corner in range(8):
            point = lower + .5 * np.array((corner & 1, (corner >> 1) & 1, (corner >> 2) & 1))
            interior &= np.einsum("ni,nij,nj->n", point, metric, point) <= 1.
        lo, hi = base_lo.copy(), base_hi.copy()
        inset = np.maximum(.005, lo[:, None] * extent - overlap)
        fit = retain & (lo <= hi) & np.all(lower + inset <= upper - inset, axis=1)
        value, best = _closest_box(metric, lower + inset, upper - inset, np, inverse_metric)
        fit &= np.sqrt(np.maximum(value, 0)) + lo <= 1 - 1e-5
        for _ in range(18):
            candidate = (lo + hi) * .5
            inset = np.maximum(.005, candidate[:, None] * extent - overlap)
            value, point = _closest_box(metric, lower + inset, upper - inset, np, inverse_metric)
            valid = np.all(lower + inset <= upper - inset, axis=1) & (np.sqrt(np.maximum(value, 0)) + candidate <= 1 - 1e-5)
            lo = np.where(valid, candidate, lo)
            hi = np.where(valid, hi, candidate)
            best[valid] = point[valid]
        principal = np.einsum("nji,nj->ni", rotation, middle * size)
        normal = np.einsum("nij,nj->ni", rotation, principal / (axes * axes))
        normal_length = np.linalg.norm(normal, axis=1)
        preferred = np.clip(middle - (.1 * np.min(fine_size) / np.maximum(normal_length, 1e-30))[:, None] * normal / size,
                            lower + .005, upper - .005)
        unit_length = np.sqrt(np.maximum(np.einsum("ni,nij,nj->n", preferred, metric, preferred), 0))
        factor = np.minimum(1., 1 - 1e-5 - unit_length)
        face_limit = (np.minimum(preferred - lower, upper - preferred) + overlap) / extent
        factor = np.minimum(factor, np.min(face_limit, axis=1))
        use_preferred = (normal_length > 1e-20) & (factor >= base_lo) & (factor >= .9 * hi)
        lo[use_preferred], best[use_preferred] = factor[use_preferred], preferred[use_preferred]
        child_center = 2 * (center + best) - bits
        interior_factor = np.minimum(base_hi, 1 - 1e-5 - np.sqrt(np.maximum(
            np.einsum("ni,nij,nj->n", middle, metric, middle), 0)))
        lo[interior] = interior_factor[interior]
        child_center[interior] = .5
        fit[interior] = retain[interior] & (interior_factor[interior] >= base_lo[interior])
        ids = np.flatnonzero(fit)
        if len(ids):
            blocks.append((ids, _addresses(old_cells[ids] * 2 + bits, metadata["resolution"] * 2),
                           child_center[ids], np.log(axes[ids] * lo[ids, None])))
    count = sum(len(block[0]) for block in blocks)
    if count == 0:
        raise RuntimeError("Refinement produced no representable children; preserve the old grid")
    if count > max_active_voxels:
        raise RuntimeError("Refinement exceeds max_active_voxels")
    parent_ids = np.concatenate([block[0] for block in blocks])
    new_radiance = np.zeros((count, metadata["sh_count"], 3), dtype=np.float32)
    new_radiance[:, 0] = radiance[parent_ids, 0]
    new_opacity = np.zeros((count, metadata["opacity_sh_count"]), dtype=np.float32)
    parent_logit = opacity[parent_ids, 0].astype(np.float64) * y0
    attenuation = float(np.clip(config.opacity_optical_depth_scale, .01, 1.))
    if attenuation >= 1:
        new_opacity[:, 0] = opacity[parent_ids, 0]
    else:
        tau = np.where(parent_logit > 15, parent_logit, np.logaddexp(0., parent_logit)) * attenuation
        # log(expm1(tau)) without overflow or zero underflow.
        small = np.maximum(tau, 1e-30)
        child_logit = np.where(tau > 15, tau, np.where(tau < 1e-4, np.log(small), np.log(np.expm1(np.minimum(tau, 15)))))
        new_opacity[:, 0] = np.minimum(child_logit, parent_logit) / y0
    metadata["resolution"] *= 2
    return {"model_metadata": metadata, "model_state": {
        "cell_indices": np.concatenate([block[1] for block in blocks]).astype(np.uint32),
        "centers": np.concatenate([block[2] for block in blocks]).astype(np.float32).ravel(),
        "log_scales": np.concatenate([block[3] for block in blocks]).astype(np.float32).ravel(),
        "rotations": quaternion[parent_ids].astype(np.float32).ravel(),
        "radiance_sh": new_radiance.ravel(), "opacity_sh": new_opacity.ravel(),
    }}


class TopologyManager:
    def __init__(self, model, view_count: int, settings: TopologySettings | None = None,
                 schedule: CoarseToFineConfig | None = None, opacity_warmup=20, opacity_ramp=30):
        import numpy as np
        self.np = np
        self.settings, self.schedule = settings or TopologySettings(), schedule or CoarseToFineConfig()
        self.view_count = int(view_count)
        self.opacity_warmup, self.opacity_ramp = int(opacity_warmup), int(opacity_ramp)
        self.resolution = int(model.resolution)
        self.level_start_epoch = 0
        self.schedule_start_resolution = self.resolution
        self.budgets = (level_budgets(self.resolution, self.schedule.target_resolution, self.schedule.total_epochs,
                                     self.schedule.level_weights) if self.schedule.enabled else {self.resolution: self.schedule.total_epochs})
        self.cooldown = {}
        self.pending_growth = None
        self._growth_world_cache = None
        self.before_refine_checkpoint = None
        self.completed_windows = 0
        self._validate()
        self._initialize_lifecycle(int(model.active_voxel_count))

    def _validate(self):
        cfg, schedule = self.settings, self.schedule
        if not 1 <= self.view_count <= 32767:
            raise ValueError("Training requires 1..32767 distinct cameras")
        for name, value in asdict(cfg).items():
            if isinstance(value, bool):
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"topology.{name} must be finite and nonnegative")
        if any(getattr(cfg, name) < 1 for name in ("growth_interval", "evidence_interval", "deletion_interval",
                                                  "min_deletion_conflict_views", "max_active_voxels")):
            raise ValueError("Topology intervals, conflict count and capacity must be positive")
        if not 0 <= schedule.parent_opacity_threshold <= .99 or not 0 < schedule.opacity_optical_depth_scale <= 1:
            raise ValueError("Invalid refinement opacity configuration")
        if not 0 <= schedule.child_face_overlap <= .2 or schedule.coarse_growth_interval < 1:
            raise ValueError("Invalid child overlap or coarse growth interval")
        if not 0 <= cfg.background_alpha_max <= cfg.foreground_alpha_min <= 1:
            raise ValueError("Invalid reference alpha classification thresholds")

    def _initialize_lifecycle(self, count):
        np = self.np
        self.count = count
        words = (self.view_count + 31) // 32
        self.foreground_masks = np.zeros((count, words), dtype=np.uint32)
        self.background_masks = np.zeros((count, words), dtype=np.uint32)
        self.qualified = np.zeros(count, dtype=bool)
        self.candidates = np.zeros(count, dtype=bool)
        self.deletion_eligible = np.zeros(count, dtype=np.int64)
        self.growth_birth = np.zeros(count, dtype=np.int64)
        self.growth_wait_start = np.zeros(count, dtype=np.int64)

    @property
    def level_budget(self):
        return self.budgets[self.resolution]

    @property
    def evidence_start(self):
        return self.level_start_epoch if self.schedule.enabled else self.opacity_warmup + self.opacity_ramp

    def collect_deletion(self, epoch):
        return (self.settings.enable_deletion and epoch >= self.evidence_start and
                (not self.schedule.enabled or self.resolution == self.schedule.target_resolution))

    def _growth_due(self, boundary):
        schedule, cfg = self.schedule, self.settings
        enabled = schedule.enable_coarse_growth if schedule.enabled else cfg.enable_growth
        interval = schedule.coarse_growth_interval if schedule.enabled else cfg.growth_interval
        final = (boundary - self.level_start_epoch >= self.level_budget if schedule.enabled else boundary >= schedule.total_epochs)
        return enabled and not final and boundary >= self.evidence_start and (boundary - self.evidence_start) % interval == 0

    def needs_rendered_alpha(self, epoch):
        """Only growth-evidence views need a rendered alpha image on the CPU."""
        return self.settings.use_growth_evidence and self._growth_due(epoch + 1)

    def _generate_growth(self, checkpoint, boundary):
        np, cfg = self.np, self.settings
        state, metadata = checkpoint["model_state"], checkpoint["model_metadata"]
        n, resolution = len(state["cell_indices"]), metadata["resolution"]
        center, axes, rotation, quaternion, size, metric, extent = _geometry(state, metadata, np)
        cells = _coordinates(state["cell_indices"], resolution, np)
        eligible = ~self.candidates
        eligible &= (self.growth_wait_start == 0) | ((boundary >= self.growth_wait_start - 1) &
                                                    (boundary - (self.growth_wait_start - 1) >= cfg.growth_wait_epochs))
        occupied = np.asarray(state["cell_indices"], dtype=np.int64)
        evidence = cfg.use_growth_evidence
        if evidence:
            penetration = float(np.clip(cfg.growth_evidence_penetration, 0., 4.))
        elif self.schedule.enabled:
            penetration = self.schedule.coarsest_growth_face_penetration * (1 if resolution == self.schedule.start_resolution
                                                                           else self.schedule.growth_finer_level_multiplier)
            penetration = float(np.clip(penetration, 0., 4.))
        else:
            penetration = float(np.clip(cfg.growth_face_penetration, 0., 4.))
        pieces = []
        forbidden_cooldown = np.asarray([cell for cell, expiry in self.cooldown.items() if boundary < expiry], dtype=np.int64)
        for offset in _neighbor_offsets(evidence and cfg.growth_use_26_neighbors, np):
            neighbors = cells + offset
            targets = _addresses(neighbors, resolution)
            possible = eligible & np.all(neighbors >= 0, axis=1) & np.all(neighbors < resolution, axis=1)
            possible &= ~np.isin(targets, occupied) & ~np.isin(targets, forbidden_cooldown)
            ids = np.flatnonzero(possible)
            if not len(ids):
                continue
            touched, contact = _contact(center[ids], metric[ids], offset, 0., np)
            if penetration > 0:
                deep, _ = _contact(center[ids], metric[ids], offset, penetration, np)
                touched &= deep
            child_center = np.clip(contact, .05, .95)
            for axis in np.flatnonzero(offset):
                child_center[:, axis] = np.clip(cfg.growth_contact_offset, .01, .49) if offset[axis] > 0 else 1 - np.clip(cfg.growth_contact_offset, .01, .49)
            margin = np.minimum(child_center, 1 - child_center)
            for axis in np.flatnonzero(offset):
                margin[:, axis] = 1 - child_center[:, axis] if offset[axis] > 0 else child_center[:, axis]
            factor = np.minimum(np.clip(cfg.growth_shrink, .01, .99), np.min(np.maximum(margin - .005, 0) / np.maximum(extent[ids], 1e-20), axis=1))
            touched &= (factor > 0) & np.all(axes[ids] * factor[:, None] >= size * 1e-4, axis=1)
            selected = ids[touched]
            if len(selected):
                pieces.append({"parent_ids": selected, "parent_cells": occupied[selected], "cells": targets[selected],
                               "centers": child_center[touched], "log_scales": np.log(axes[selected] * factor[touched, None]),
                               "rotations": quaternion[selected]})
        if pieces:
            result = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
        else:
            # Freeze even an empty first-view batch for the whole epoch.
            result = {"parent_ids": np.empty(0, dtype=np.int64), "parent_cells": np.empty(0, dtype=np.int64),
                      "cells": np.empty(0, dtype=np.int64), "centers": np.empty((0, 3)),
                      "log_scales": np.empty((0, 3)), "rotations": np.empty((0, 4))}
        if evidence:
            _, inverse, counts = np.unique(result["cells"], return_inverse=True, return_counts=True)
            supported = counts[inverse] >= max(1, min(26, cfg.growth_min_supporting_parents))
            result = {key: value[supported] for key, value in result.items()}
        count = len(result["cells"])
        result.update(foreground=np.zeros(count, dtype=np.uint32), background=np.zeros(count, dtype=np.uint32),
                      seen_views=set(), boundary=boundary, resolution=resolution)
        return result

    def _growth_world_points(self, pending, metadata):
        """Reuse the seven frozen proposal samples across views, never evidence.

        This derived cache is separate from pending_growth/checkpoints. The
        live model is still rechecked by _generate_growth at the epoch boundary.
        """
        np = self.np
        key = (metadata["resolution"], tuple(metadata["grid_min"]), tuple(metadata["grid_max"]))
        cached = self._growth_world_cache
        if cached is not None and cached[0] is pending and cached[1] == key:
            return cached[2]
        center = pending["centers"]
        rotation, _ = _quaternion_matrix(pending["rotations"], np)
        axes = np.exp(np.clip(pending["log_scales"], -30., 30.))
        size = (np.asarray(metadata["grid_max"]) - metadata["grid_min"]) / metadata["resolution"]
        cells = _coordinates(pending["cells"], metadata["resolution"], np)
        points = []
        for sample in range(7):
            delta = np.zeros_like(center)
            if sample:
                axis = (sample - 1) // 2
                delta = rotation[:, :, axis] * ((-.5 if sample & 1 else .5) * axes[:, axis])[:, None] / size
            fraction = np.ones(len(center))
            for axis in range(3):
                positive, negative = delta[:, axis] > 0, delta[:, axis] < 0
                fraction[positive] = np.minimum(fraction[positive], (.999 - center[positive, axis]) / delta[positive, axis])
                fraction[negative] = np.minimum(fraction[negative], (.001 - center[negative, axis]) / delta[negative, axis])
            local = center + np.clip(fraction, 0, 1)[:, None] * delta
            world = np.asarray(metadata["grid_min"]) + (cells + local) * size
            world.setflags(write=False)
            points.append(world)
        self._growth_world_cache = (pending, key, tuple(points))
        return self._growth_world_cache[2]

    def _observe_growth(self, camera, reference_alpha, rendered_alpha, view_id, metadata):
        np, cfg, pending = self.np, self.settings, self.pending_growth
        if pending is None or view_id in pending["seen_views"]:
            return
        reference, rendered = np.asarray(reference_alpha), np.asarray(rendered_alpha)
        if reference.shape != rendered.shape or reference.ndim != 2:
            raise ValueError("Growth evidence needs matching HxW reference/rendered alpha")
        height, width = reference.shape
        position, right, up, forward = [np.asarray(camera[key]) for key in ("position", "right", "up", "forward")]
        tan_x = math.tan(camera["camera_angle_x"] * .5)
        tan_y = tan_x * height / width
        near, far = camera.get("near_clip", .1), camera.get("far_clip", 1000.)
        count = len(pending["centers"])
        fg, bg = np.zeros(count, dtype=bool), np.zeros(count, dtype=bool)
        for world in self._growth_world_points(pending, metadata):
            relative = world - position
            depth = relative @ forward
            u = .5 + .5 * (relative @ right) / np.maximum(depth * tan_x, 1e-30)
            v = .5 - .5 * (relative @ up) / np.maximum(depth * tan_y, 1e-30)
            visible = (depth >= near) & (depth <= far) & (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
            ids = np.flatnonzero(visible)
            x, y = (u[ids] * width).astype(np.int64), (v[ids] * height).astype(np.int64)
            ref, pred = reference[y, x], rendered[y, x]
            bg[ids] |= ref <= cfg.background_alpha_max
            fg[ids] |= (ref >= cfg.foreground_alpha_min) & (ref - pred >= np.clip(cfg.growth_min_alpha_deficit, .001, 1.))
        pending["background"] += bg
        pending["foreground"] += fg & ~bg
        pending["seen_views"].add(int(view_id))

    def observe_view(self, model, out: dict, *, view_id: int, epoch: int, camera: dict,
                     reference_alpha, rendered_alpha):
        np = self.np
        if not 0 <= view_id < self.view_count or int(model.active_voxel_count) != self.count:
            raise ValueError("Unexpected view ID or model topology")
        if self.collect_deletion(epoch):
            eligible = epoch >= self.deletion_eligible
            bit, word = np.uint32(1 << (view_id % 32)), view_id // 32
            for key, masks in (("foreground_evidence", self.foreground_masks), ("background_evidence", self.background_masks)):
                flags = np.asarray(out[key]).reshape(-1) != 0
                if len(flags) != self.count:
                    raise ValueError("Deletion evidence count differs from active voxel count")
                masks[flags & eligible, word] |= bit
        boundary = epoch + 1
        if self.needs_rendered_alpha(epoch):
            if rendered_alpha is None:
                raise ValueError("Growth evidence requires this view's rendered alpha before its update")
            if self.pending_growth is None or self.pending_growth["boundary"] != boundary:
                checkpoint = model_checkpoint(model)
                self.pending_growth = self._generate_growth(checkpoint, boundary)
            metadata = {"resolution": self.resolution, "grid_min": list(model.grid_min), "grid_max": list(model.grid_max)}
            self._observe_growth(camera, reference_alpha, rendered_alpha, view_id, metadata)

    def _evaluate_deletion(self, boundary):
        np, cfg = self.np, self.settings
        if not self.collect_deletion(boundary) or boundary <= self.evidence_start or (boundary - self.evidence_start) % cfg.evidence_interval:
            return False
        lut = np.asarray([int(value).bit_count() for value in range(256)], dtype=np.uint8)
        fg = lut[self.foreground_masks.view(np.uint8)].sum(axis=1)
        bg = lut[self.background_masks.view(np.uint8)].sum(axis=1)
        ratio = max(1, min(1000, cfg.deletion_conflict_support_ratio))
        dominates = bg >= ratio * fg
        reliable = max(2, (cfg.min_deletion_conflict_views + ratio - 1) // ratio)
        rejected = (fg >= reliable) & ~dominates
        qualified = (bg >= cfg.min_deletion_conflict_views) & dominates
        self.candidates = ~rejected & (self.candidates | (qualified & self.qualified))
        self.qualified = ~rejected & (self.qualified | qualified | self.candidates)
        protected = boundary < self.deletion_eligible
        self.candidates[protected] = False
        self.qualified[protected] = False
        self.foreground_masks.fill(0)
        self.background_masks.fill(0)
        self.completed_windows += 1
        return True

    def _take_rows(self, checkpoint, ids):
        np = self.np
        old_count = len(checkpoint["model_state"]["cell_indices"])
        return {"model_metadata": dict(checkpoint["model_metadata"]), "model_state": {
            key: np.asarray(value).reshape(old_count, -1)[ids].reshape(-1).copy()
            for key, value in checkpoint["model_state"].items()}}

    def _remap_lifecycle(self, survivors, new_count, boundary):
        np = self.np
        kept = len(survivors)
        for name in ("foreground_masks", "background_masks", "qualified", "candidates",
                     "deletion_eligible", "growth_birth", "growth_wait_start"):
            old = getattr(self, name)
            resized = np.zeros((new_count,) + old.shape[1:], dtype=old.dtype)
            resized[:kept] = old[survivors]
            setattr(self, name, resized)
        if new_count > kept:
            self.deletion_eligible[kept:] = boundary + self.settings.growth_protection_epochs
            self.growth_birth[kept:] = boundary + 1
            self.growth_wait_start[kept:] = boundary + 1
        self.count = new_count

    def finish_epoch(self, model, optimizer, completed_epochs: int):
        """Called after every view update. Returns (replacement checkpoint or None, event)."""
        np, cfg, schedule = self.np, self.settings, self.schedule
        boundary = int(completed_epochs)
        self.before_refine_checkpoint = None
        event = {"completed_epochs": boundary, "resolution": self.resolution, "deleted": 0, "grown": 0,
                 "refined": False, "finished": False, "level_budget": self.level_budget}
        checkpoint = None
        level_done = boundary - self.level_start_epoch >= self.level_budget
        if schedule.enabled and self.resolution < schedule.target_resolution and level_done:
            previous = model_checkpoint(model)
            replacement = refine_state(previous, schedule, cfg.max_active_voxels)
            if schedule.save_each_level:
                self.before_refine_checkpoint = {
                    **previous, "optimizer_state": optimizer.state_dict(), "topology_state": self.state_dict()}
                event["pre_refine_checkpoint_available"] = True
            self.resolution = replacement["model_metadata"]["resolution"]
            self.level_start_epoch = boundary
            self._initialize_lifecycle(len(replacement["model_state"]["cell_indices"]))
            self.completed_windows = 0
            self.cooldown = {}
            self.pending_growth = None
            self._growth_world_cache = None
            optimizer.remap(np.empty(0, dtype=np.uint32), self.count)
            event.update(refined=True, resolution=self.resolution, active_voxel_count=self.count)
            return replacement, event
        event["evidence_window_evaluated"] = self._evaluate_deletion(boundary)
        final = level_done if schedule.enabled else boundary >= schedule.total_epochs
        first = self.evidence_start + 2 * cfg.evidence_interval
        periodic = boundary >= first and (boundary - first) % cfg.deletion_interval == 0
        if np.any(self.candidates) and (periodic or final) and self.collect_deletion(boundary):
            checkpoint = model_checkpoint(model)
            old_cells = checkpoint["model_state"]["cell_indices"]
            for cell in old_cells[self.candidates]:
                self.cooldown[int(cell)] = boundary + cfg.deletion_cooldown_epochs
            survivors = np.flatnonzero(~self.candidates).astype(np.uint32)
            event["deleted"] = self.count - len(survivors)
            checkpoint = self._take_rows(checkpoint, survivors)
            optimizer.remap(survivors, len(survivors), reset_radiance=True)
            self._remap_lifecycle(survivors, len(survivors), boundary)
            self.foreground_masks.fill(0)
            self.background_masks.fill(0)
            self.qualified.fill(False)
            self.candidates.fill(False)
            self.completed_windows = 0
        if self._growth_due(boundary):
            if checkpoint is None:
                checkpoint = model_checkpoint(model)
            current = self._generate_growth(checkpoint, boundary)
            selected = current
            if cfg.use_growth_evidence:
                pending = self.pending_growth
                selected = None
                if (pending is not None and pending["boundary"] == boundary and
                        pending["resolution"] == self.resolution and len(pending["seen_views"]) == self.view_count and current is not None):
                    keep = ((pending["foreground"] >= max(2, cfg.growth_min_foreground_views)) &
                            (pending["background"] < max(1, cfg.growth_background_veto_views)))
                    keep &= _matching_growth_pairs(current, pending, np)
                    selected = {key: value[keep] for key, value in pending.items() if isinstance(value, np.ndarray)}
                    mapping = {int(cell): index for index, cell in enumerate(checkpoint["model_state"]["cell_indices"])}
                    selected["parent_ids"] = np.asarray([mapping[int(cell)] for cell in selected["parent_cells"]], dtype=np.int64)
            if selected is not None and len(selected["cells"]):
                # Falcor's atomic winner is nondeterministic. Use the first valid
                # proposal per cell deterministically without changing eligibility.
                _, winners = np.unique(selected["cells"], return_index=True)
                selected = {key: value[winners] for key, value in selected.items() if isinstance(value, np.ndarray)}
                added = len(winners)
                if self.count + added > cfg.max_active_voxels:
                    raise RuntimeError("Growth exceeds max_active_voxels")
                state, metadata = checkpoint["model_state"], checkpoint["model_metadata"]
                parents = selected["parent_ids"]
                radiance = np.zeros((added, metadata["sh_count"], 3), dtype=np.float32)
                radiance[:, 0] = state["radiance_sh"].reshape(self.count, metadata["sh_count"], 3)[parents, 0]
                opacity = np.zeros((added, metadata["opacity_sh_count"]), dtype=np.float32)
                alpha = float(np.clip(cfg.growth_initial_opacity, .01, .99))
                opacity[:, 0] = math.log(alpha / (1 - alpha)) * (2 * math.sqrt(math.pi))
                new = {"cell_indices": selected["cells"].astype(np.uint32), "centers": selected["centers"],
                       "log_scales": selected["log_scales"], "rotations": selected["rotations"],
                       "radiance_sh": radiance, "opacity_sh": opacity}
                for name, values in new.items():
                    state[name] = np.concatenate((state[name], values.astype(state[name].dtype).reshape(-1)))
                survivors = np.arange(self.count, dtype=np.uint32)
                optimizer.remap(survivors, self.count + added)
                self._remap_lifecycle(survivors, self.count + added, boundary)
                event["grown"] = added
            elif not event["deleted"]:
                checkpoint = None
        self.pending_growth = None
        self._growth_world_cache = None
        self.cooldown = {cell: expiry for cell, expiry in self.cooldown.items() if expiry > boundary}
        event.update(finished=final, active_voxel_count=self.count)
        return checkpoint, event

    def state_dict(self):
        import copy
        state = {"settings": asdict(self.settings), "schedule": asdict(self.schedule), "view_count": self.view_count,
                 "opacity_warmup": self.opacity_warmup, "opacity_ramp": self.opacity_ramp,
                 "resolution": self.resolution, "schedule_start_resolution": self.schedule_start_resolution,
                 "level_start_epoch": self.level_start_epoch, "budgets": {str(key): value for key, value in self.budgets.items()},
                 "cooldown": {str(key): value for key, value in self.cooldown.items()}, "completed_windows": self.completed_windows,
                 "count": self.count, "pending_growth": self.pending_growth}
        for name in ("foreground_masks", "background_masks", "qualified", "candidates",
                     "deletion_eligible", "growth_birth", "growth_wait_start"):
            state[name] = getattr(self, name)
        state = copy.deepcopy(state)
        if state["pending_growth"] is not None:
            state["pending_growth"]["seen_views"] = sorted(state["pending_growth"]["seen_views"])
        return state

    def load_state_dict(self, state):
        import copy
        self._growth_world_cache = None
        state = copy.deepcopy(state)
        if state["count"] != self.count or state["view_count"] != self.view_count:
            raise ValueError("Topology checkpoint dimensions do not match model or dataset")
        self.settings = TopologySettings(**state.pop("settings"))
        self.schedule = CoarseToFineConfig(**state.pop("schedule"))
        for name, value in state.items():
            setattr(self, name, value)
        self.budgets = {int(key): value for key, value in self.budgets.items()}
        self.cooldown = {int(key): value for key, value in self.cooldown.items()}
        if self.pending_growth is not None:
            self.pending_growth["seen_views"] = set(self.pending_growth["seen_views"])
        self._validate()
