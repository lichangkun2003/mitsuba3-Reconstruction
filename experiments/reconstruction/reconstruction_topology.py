"""Falcor topology lifecycle, GPU evidence, and coarse-to-fine transfer.

Continuous parameters/Adam and per-view evidence live on the GPU. Host snapshots
are taken only for scheduled topology work and checkpoints. Native replacement is left to
the training driver, so allocation failure cannot partially replace its model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

from reconstruction_growth_rgb import RGB_GROWTH_EVIDENCE_VERSION
from reconstruction_deletion_rgb import RGB_DELETION_EVIDENCE_VERSION


@dataclass
class TopologySettings:
    enable_deletion: bool = True
    # Optional RGB-only removal evidence. Historical checkpoints retain the
    # foreground/background deletion rule unless this field is explicitly set.
    use_rgb_deletion: bool = False
    # Independent cleanup of persistently transparent, isotropic voxels. The
    # default preserves the behavior of checkpoints predating this lifecycle.
    use_low_opacity_deletion: bool = False
    deletion_opacity_threshold: float = .005
    deletion_opacity_confirmation_windows: int = 2
    deletion_warmup_epochs: int = 20
    deletion_protection_epochs: int = 20
    deletion_freeze_last_epochs: int = 0
    deletion_rgb_min_support_views: int = 3
    deletion_rgb_max_conflict_views: int = 1
    deletion_rgb_gain_epsilon: float = 1e-7
    deletion_rgb_min_mean_gain: float = 1e-7
    deletion_rgb_confirmation_windows: int = 2
    enable_growth: bool = True
    use_growth_evidence: bool = True
    # RGB candidates are inserted virtually during the existing training trace.
    # Legacy alpha evidence remains readable for old experiment checkpoints.
    use_rgb_growth: bool = False
    growth_warmup_epochs: int = 20
    growth_fine_warmup_epochs: int = 20
    # Global training tail; zero preserves historical checkpoint behavior.
    growth_freeze_last_epochs: int = 0
    # Missing fields in historical checkpoints retain their old loss units.
    # The current training CONFIG explicitly selects "observed".
    growth_rgb_normalization: str = "full_image"
    growth_rgb_min_support_views: int = 3
    growth_rgb_max_conflict_views: int = 2
    growth_rgb_gain_epsilon: float = 1e-7
    growth_rgb_min_mean_gain: float = 1e-7
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
    # Fixed birth opacity for legacy runs; a birth cap when inheritance is on.
    growth_initial_opacity: float = .5
    growth_inherit_opacity: bool = False
    growth_opacity_scale: float = .8
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
    # CUDA RGB lifecycle; False retains the NumPy reference implementation.
    gpu_topology: bool = True


@dataclass
class CoarseToFineConfig:
    enabled: bool = True
    start_resolution: int = 256
    target_resolution: int = 512
    total_epochs: int = 1000
    level_weights: tuple[int, int] = (6, 4)  # Coarse and fine epoch weights.
    parent_opacity_threshold: float = 0.
    opacity_optical_depth_scale: float = .65
    child_face_overlap: float = .1
    enable_coarse_growth: bool = True
    coarsest_growth_face_penetration: float = 1.4
    growth_finer_level_multiplier: float = 2.5
    coarse_growth_interval: int = 10
    save_each_level: bool = True


def validate_growth_opacity(settings, opacity_sh_count):
    """Validate parent-opacity initialization in both CLI and topology paths."""
    if not isinstance(settings.growth_inherit_opacity, bool):
        raise ValueError("topology.growth_inherit_opacity must be a boolean")
    scale = settings.growth_opacity_scale
    if (isinstance(scale, bool) or not isinstance(scale, (int, float))
            or not math.isfinite(scale) or not 0 < scale <= 1):
        raise ValueError("topology.growth_opacity_scale must satisfy 0 < scale <= 1")
    if settings.growth_inherit_opacity:
        if opacity_sh_count != 1:
            raise ValueError("Parent-opacity growth requires isotropic opacity_sh_count == 1")
        cap = settings.growth_initial_opacity
        if (isinstance(cap, bool) or not isinstance(cap, (int, float))
                or not math.isfinite(cap) or not 0 < cap < 1):
            raise ValueError("topology.growth_initial_opacity must satisfy 0 < birth cap < 1")


def _growth_opacity_coefficients(state, metadata, parent_ids, settings, np):
    """Freeze candidate opacity in probability space, then encode its DC logit."""
    sh_count = metadata["opacity_sh_count"]
    validate_growth_opacity(settings, sh_count)
    opacity = np.zeros((len(parent_ids), sh_count), dtype=np.float32)
    inverse_y0 = 2 * math.sqrt(math.pi)
    if settings.growth_inherit_opacity:
        parent_b0 = np.asarray(state["opacity_sh"]).reshape(-1, sh_count)[parent_ids, 0]
        if not np.all(np.isfinite(parent_b0)):
            raise ValueError("Growth parent opacity coefficients must be finite")
        parent_logit = parent_b0.astype(np.float64) / inverse_y0
        # log(sigmoid(z)) avoids underflow for very transparent parents. Scaling
        # b0 itself would pull alpha toward 0.5 and brighten transparent parents.
        log_alpha = np.minimum(
            math.log(settings.growth_initial_opacity),
            math.log(settings.growth_opacity_scale) - np.logaddexp(0., -parent_logit))
        child_logit = log_alpha - np.log(-np.expm1(log_alpha))
        opacity[:, 0] = child_logit * inverse_y0
    else:
        # Preserve historical fixed-opacity and clipping behavior exactly.
        alpha = float(np.clip(settings.growth_initial_opacity, .01, .99))
        opacity[:, 0] = math.log(alpha / (1 - alpha)) * inverse_y0
    return opacity


def level_budgets(start: int, target: int, total: int, weights=(6, 4)) -> dict[int, int]:
    """Allocate coarse/fine epochs; retain four-weight legacy checkpoint budgets."""
    if (start < 1 or target < start or target > 1024 or
            start & (start - 1) or target & (target - 1)):
        raise ValueError("Coarse-to-fine requires power-of-two start <= target <= 1024")
    resolutions = [start]
    while resolutions[-1] < target:
        resolutions.append(resolutions[-1] * 2)
    count = len(resolutions)
    if (total < count or total > 1000000 or len(weights) not in (2, 4)
            or any(type(w) is not int or w <= 0 for w in weights)):
        raise ValueError("Invalid coarse-to-fine total or level weights: use two positive integers")
    if count == 1:
        return {start: total}
    if len(weights) == 2:
        if count != 2:
            raise ValueError("Two level weights require exactly two resolutions (target = 2 * start)")
        # Round the coarse share to the nearest epoch (ties round up), give
        # the remainder to the fine level, and keep at least one epoch each.
        denominator = sum(weights)
        coarse = (2 * total * weights[0] + denominator) // (2 * denominator)
        coarse = max(1, min(total - 1, coarse))
        return {start: coarse, target: total - coarse}
    # Older checkpoints/configs used four interpolated weights and reserved
    # one epoch per level first. Preserve their original transition epochs.
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


def model_checkpoint(model, fields=None) -> dict:
    import numpy as np
    import drjit as dr
    snapshot = model.snapshot()
    if fields is not None:
        snapshot = {name: snapshot[name] for name in fields}
    dr.eval(*snapshot.values())
    dr.sync_thread()
    metadata = {"resolution": int(model.resolution), "grid_min": list(model.grid_min),
                "grid_max": list(model.grid_max), "sh_count": int(model.sh_count),
                "opacity_sh_count": int(model.opacity_sh_count)}
    core_resolution = int(getattr(model, "core_resolution", model.resolution))
    if core_resolution != metadata["resolution"]:
        metadata["core_resolution"] = core_resolution
    if getattr(model, "scene_mapping", "") == "radial_seed_shell":
        metadata["scene_mapping"] = "radial_seed_shell"
    return {
        "model_state": {name: np.array(value, copy=True) for name, value in snapshot.items()},
        "model_metadata": metadata,
    }


class _GrowthBlockedCells:
    """Exact occupied/cooldown membership, shared by all neighbor directions.

    This index lives for one candidate-generation call, so no topology or
    cooldown changes can leave it stale. Up to 512 cubed the dense table
    occupies at most 128 MiB; larger grids reuse a sorted index instead.
    Neither representation changes the order or number of proposals.
    """

    def __init__(self, occupied, cooldown, resolution, np, *,
                 max_dense_bytes=128 * 1024 * 1024):
        self.np = np
        self.cell_count = int(resolution) ** 3
        self.dense = None
        self.sorted_cells = None
        blocked = np.concatenate((np.asarray(occupied, dtype=np.int64).reshape(-1),
                                  np.asarray(cooldown, dtype=np.int64).reshape(-1)))
        blocked = blocked[(blocked >= 0) & (blocked < self.cell_count)]
        if self.cell_count * np.dtype(np.bool_).itemsize <= max_dense_bytes:
            try:
                self.dense = np.zeros(self.cell_count, dtype=np.bool_)
            except MemoryError:
                # A low-memory host can use the same exact sorted lookup.
                pass
        if self.dense is not None:
            self.dense[blocked] = True
        else:
            self.sorted_cells = np.unique(blocked)

    def contains(self, addresses):
        np = self.np
        addresses = np.asarray(addresses, dtype=np.int64)
        valid = (addresses >= 0) & (addresses < self.cell_count)
        result = np.zeros(addresses.shape, dtype=np.bool_)
        query = addresses[valid]
        if self.dense is not None:
            result[valid] = self.dense[query]
        elif len(self.sorted_cells):
            positions = np.searchsorted(self.sorted_cells, query)
            result[valid] = ((positions < len(self.sorted_cells)) &
                             (self.sorted_cells[np.minimum(positions, len(self.sorted_cells) - 1)] == query))
        return result


def _coordinates(cells, resolution, np):
    cells = np.asarray(cells, dtype=np.int64)
    return np.column_stack((cells % resolution, cells // resolution % resolution, cells // (resolution * resolution)))


def _addresses(cells, resolution):
    return cells[:, 0] + resolution * (cells[:, 1] + resolution * cells[:, 2])


def _growth_core_radius_squared(metadata):
    """Twice-cell-unit core radius squared, or None for unrestricted growth.

    A radial far shell starts filled and may be pruned. Only cells whose entire
    AABB lies inside the unit core sphere may supply parents or receive growth.
    Integer coordinates make CPU/GPU classification exact, including boundary
    cells. The explicit tag leaves historical point-only shells unchanged.
    """
    if metadata.get("scene_mapping") != "radial_seed_shell":
        return None
    resolution = int(metadata["resolution"])
    core = int(metadata.get("core_resolution", resolution))
    if not 0 < core < resolution or (resolution - core) % 2:
        raise ValueError("Radial shell growth requires a centered integer core")
    return core * core


def _growth_coordinates_allowed(coordinates, metadata, np):
    resolution = int(metadata["resolution"])
    allowed = np.all((coordinates >= 0) & (coordinates < resolution), axis=1)
    radius_squared = _growth_core_radius_squared(metadata)
    if radius_squared is not None:
        farthest = np.abs(2 * coordinates + 1 - resolution) + 1
        allowed &= np.sum(farthest * farthest, axis=1) <= radius_squared
    return allowed


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
        if "core_resolution" in metadata:
            metadata["core_resolution"] *= 2
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
    if "core_resolution" in metadata:
        metadata["core_resolution"] *= 2
    return {"model_metadata": metadata, "model_state": {
        "cell_indices": np.concatenate([block[1] for block in blocks]).astype(np.uint32),
        "centers": np.concatenate([block[2] for block in blocks]).astype(np.float32).ravel(),
        "log_scales": np.concatenate([block[3] for block in blocks]).astype(np.float32).ravel(),
        "rotations": quaternion[parent_ids].astype(np.float32).ravel(),
        "radiance_sh": new_radiance.ravel(), "opacity_sh": new_opacity.ravel(),
    }}


class _DeletionEvidenceDevice:
    """Distinct-view bitsets: GPU updates, counts only at window boundaries.

    A separate array for each 32-view word avoids a scatter and keeps the two
    per-view updates proportional to the active voxel count. No host reads are
    needed to deduplicate repeated observations of the same camera.
    """

    def __init__(self, array_type, foreground, background, eligible, dr, np):
        self.dr, self.np, self.UInt = dr, np, array_type
        self.count = len(eligible)
        self.foreground = [array_type(foreground[:, word].copy()) for word in range(foreground.shape[1])]
        self.background = [array_type(background[:, word].copy()) for word in range(background.shape[1])]
        self.eligible = dr.int64_array_t(array_type)(eligible)
        self.is_cleared = False
        dr.eval(*self.foreground, *self.background, self.eligible)

    def observe(self, foreground, background, view_id, epoch):
        dr, UInt = self.dr, self.UInt
        if dr.width(foreground) != self.count or dr.width(background) != self.count:
            raise ValueError("Deletion evidence count differs from active voxel count")
        if not self.count:
            return
        self.is_cleared = False
        word = view_id // 32
        bit = dr.opaque(UInt, 1 << (view_id % 32))
        eligible = dr.opaque(type(self.eligible), epoch) >= self.eligible
        self.foreground[word] = self.foreground[word] | dr.select((foreground != 0) & eligible, bit, 0)
        self.background[word] = self.background[word] | dr.select((background != 0) & eligible, bit, 0)
        # Bound the JIT graph without synchronizing or copying flags to the CPU.
        dr.eval(self.foreground[word], self.background[word])

    def counts(self):
        dr, UInt = self.dr, self.UInt
        foreground, background = dr.zeros(UInt, self.count), dr.zeros(UInt, self.count)
        for fg, bg in zip(self.foreground, self.background):
            foreground += dr.popcnt(fg)
            background += dr.popcnt(bg)
        dr.eval(foreground, background)
        return self.np.array(foreground, copy=True), self.np.array(background, copy=True)

    def materialize(self):
        return (self.np.column_stack([self.np.array(value, copy=True) for value in arrays])
                for arrays in (self.foreground, self.background))

    def clear(self):
        self.foreground = [self.dr.zeros(self.UInt, self.count) for _ in self.foreground]
        self.background = [self.dr.zeros(self.UInt, self.count) for _ in self.background]
        self.is_cleared = True


def _cuda_array(value):
    """Keep NumPy-only topology tests independent of a CUDA runtime."""
    if type(value).__module__.split(".")[0] != "drjit":
        return None
    import drjit as dr
    return dr if dr.backend_v(type(value)) == dr.JitBackend.CUDA else None


class TopologyManager:
    def __init__(self, model, view_count: int, settings: TopologySettings | None = None,
                 schedule: CoarseToFineConfig | None = None, opacity_warmup=20, opacity_ramp=30):
        import numpy as np
        self.np = np
        self.settings, self.schedule = settings or TopologySettings(), schedule or CoarseToFineConfig()
        self.view_count = int(view_count)
        self.opacity_warmup, self.opacity_ramp = int(opacity_warmup), int(opacity_ramp)
        self.resolution = int(model.resolution)
        # The enclosing grid includes any contracted shell. Scheduling refers
        # to the core diameter, while every address and geometric operation
        # continues to use the actual enclosing grid resolution.
        self.core_resolution = int(getattr(model, "core_resolution", model.resolution))
        self._model_opacity_sh_count = getattr(model, "opacity_sh_count", None)
        self.level_start_epoch = 0
        self.schedule_start_resolution = self.core_resolution
        self.budgets = (level_budgets(self.core_resolution, self.schedule.target_resolution, self.schedule.total_epochs,
                                     self.schedule.level_weights) if self.schedule.enabled else {self.core_resolution: self.schedule.total_epochs})
        self.cooldown = {}
        self.pending_growth = None
        # Diagnostic wall times only: never serialized as algorithm state.
        self.growth_prepare_stage_seconds = {}
        self._growth_world_cache = None
        self._growth_device = None
        self._rgb_growth_device = None
        self._deletion_device = None
        self._device_statistics = {"deletion_device_views": 0, "growth_device_views": 0,
                                   "deletion_cpu_views": 0, "growth_cpu_views": 0,
                                   "deletion_count_readback_bytes": 0,
                                   "deletion_mask_readback_bytes": 0,
                                   "growth_count_readback_bytes": 0}
        self._device_statistics.update(rgb_growth_device_views=0, rgb_growth_cpu_views=0,
                                       rgb_growth_readback_bytes=0)
        self._device_statistics.update(rgb_deletion_device_views=0, rgb_deletion_cpu_views=0,
                                       rgb_deletion_readback_bytes=0)
        self._device_statistics["deletion_opacity_readback_bytes"] = 0
        self.before_refine_checkpoint = None
        self.completed_windows = 0
        self._validate()
        self._initialize_lifecycle(int(model.active_voxel_count))

    def _validate(self):
        cfg, schedule = self.settings, self.schedule
        if not 1 <= self.core_resolution <= self.resolution <= 1024:
            raise ValueError("Topology requires 1 <= core resolution <= grid resolution <= 1024")
        if self.core_resolution not in self.budgets:
            raise ValueError("Topology epoch budgets do not contain the current core resolution")
        if (schedule.enabled and
                self.resolution * schedule.target_resolution > 1024 * self.core_resolution):
            raise ValueError("Coarse-to-fine enclosing grid resolution would exceed 1024")
        if not 1 <= self.view_count <= 32767:
            raise ValueError("Training requires 1..32767 distinct cameras")
        if not isinstance(cfg.use_rgb_growth, bool):
            raise ValueError("topology.use_rgb_growth must be a boolean")
        if not isinstance(cfg.use_rgb_deletion, bool):
            raise ValueError("topology.use_rgb_deletion must be a boolean")
        if not isinstance(cfg.use_low_opacity_deletion, bool):
            raise ValueError("topology.use_low_opacity_deletion must be a boolean")
        validate_growth_opacity(cfg, self._model_opacity_sh_count)
        if (cfg.enable_deletion and cfg.use_low_opacity_deletion and
                self._model_opacity_sh_count != 1):
            raise ValueError("Low-opacity deletion requires isotropic opacity_sh_count == 1")
        for name in ("deletion_warmup_epochs", "deletion_protection_epochs",
                     "deletion_freeze_last_epochs", "deletion_rgb_min_support_views",
                     "deletion_rgb_max_conflict_views", "deletion_rgb_confirmation_windows",
                     "deletion_opacity_confirmation_windows"):
            value = getattr(cfg, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"topology.{name} must be a nonnegative integer")
        if any(getattr(cfg, name) < 1 for name in (
                "deletion_rgb_min_support_views", "deletion_rgb_max_conflict_views",
                "deletion_rgb_confirmation_windows", "deletion_opacity_confirmation_windows")):
            raise ValueError("Deletion vote and confirmation thresholds must be positive")
        if (isinstance(cfg.deletion_opacity_threshold, bool) or
                not isinstance(cfg.deletion_opacity_threshold, (int, float)) or
                not 0 < cfg.deletion_opacity_threshold < 1):
            raise ValueError("topology.deletion_opacity_threshold must satisfy 0 < threshold < 1")
        if ((cfg.use_rgb_deletion or cfg.use_low_opacity_deletion) and
                (not isinstance(cfg.deletion_interval, int) or
                 isinstance(cfg.deletion_interval, bool))):
            raise ValueError("Deletion interval must be an integer")
        if cfg.growth_rgb_normalization not in ("observed", "full_image"):
            raise ValueError("topology.growth_rgb_normalization must be 'observed' or 'full_image'")
        if (not isinstance(cfg.growth_freeze_last_epochs, int) or
                isinstance(cfg.growth_freeze_last_epochs, bool) or cfg.growth_freeze_last_epochs < 0):
            raise ValueError("topology.growth_freeze_last_epochs must be a nonnegative integer")
        for name, value in asdict(cfg).items():
            if isinstance(value, bool) or name == "growth_rgb_normalization":
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"topology.{name} must be finite and nonnegative")
        if any(getattr(cfg, name) < 1 for name in ("growth_interval", "evidence_interval", "deletion_interval",
                                                  "min_deletion_conflict_views", "max_active_voxels")):
            raise ValueError("Topology intervals, conflict count and capacity must be positive")
        rgb_integer_fields = ("growth_warmup_epochs", "growth_fine_warmup_epochs", "growth_rgb_min_support_views",
                              "growth_rgb_max_conflict_views")
        if any(not isinstance(getattr(cfg, name), int) or isinstance(getattr(cfg, name), bool)
               for name in rgb_integer_fields):
            raise ValueError("RGB growth warmups and vote thresholds must be integers")
        if any(getattr(cfg, name) < 1 for name in rgb_integer_fields[2:]):
            raise ValueError("RGB growth vote thresholds must be positive")
        if not 0 <= schedule.parent_opacity_threshold <= .99 or not 0 < schedule.opacity_optical_depth_scale <= 1:
            raise ValueError("Invalid refinement opacity configuration")
        if not 0 <= schedule.child_face_overlap <= .2 or schedule.coarse_growth_interval < 1:
            raise ValueError("Invalid child overlap or coarse growth interval")
        if not 0 <= cfg.background_alpha_max <= cfg.foreground_alpha_min <= 1:
            raise ValueError("Invalid reference alpha classification thresholds")

    def _initialize_lifecycle(self, count):
        np = self.np
        self._deletion_device = None
        self.count = count
        words = (self.view_count + 31) // 32
        self._foreground_masks = np.zeros((count, words), dtype=np.uint32)
        self._background_masks = np.zeros((count, words), dtype=np.uint32)
        self.qualified = np.zeros(count, dtype=bool)
        self.candidates = np.zeros(count, dtype=bool)
        self._deletion_eligible = np.zeros(count, dtype=np.int64)
        self.growth_birth = np.zeros(count, dtype=np.int64)
        self.growth_wait_start = np.zeros(count, dtype=np.int64)
        self.pending_rgb_deletion = None
        self._rgb_deletion_device = None
        self.rgb_deletion_streak = np.zeros(count, dtype=np.uint32)
        self.rgb_deletion_last_boundary = None
        self.low_opacity_deletion_streak = np.zeros(count, dtype=np.uint32)
        self.low_opacity_deletion_last_boundary = None

    def _materialize_rgb_deletion(self):
        if self._rgb_deletion_device is not None:
            pending, device = self._rgb_deletion_device
            if pending is self.pending_rgb_deletion:
                arrays = device.materialize()
                self._device_statistics["rgb_deletion_readback_bytes"] += sum(
                    value.nbytes for value in arrays.values())
            self._rgb_deletion_device = None

    def _materialize_deletion_masks(self):
        if self._deletion_device is not None:
            if self._deletion_device.is_cleared:
                self._foreground_masks.fill(0)
                self._background_masks.fill(0)
            else:
                self._foreground_masks, self._background_masks = self._deletion_device.materialize()
                self._device_statistics["deletion_mask_readback_bytes"] += (
                    self._foreground_masks.nbytes + self._background_masks.nbytes)
            self._deletion_device = None

    def statistics(self):
        """Cumulative transfer/path counters; reading these never synchronizes."""
        return dict(self._device_statistics)

    def _materialize_growth(self):
        if self._rgb_growth_device is not None:
            pending, device = self._rgb_growth_device
            if pending is self.pending_growth:
                arrays = device.materialize()
                self._device_statistics["rgb_growth_readback_bytes"] += sum(value.nbytes for value in arrays.values())
            self._rgb_growth_device = None
        if self._growth_device is not None:
            pending, device = self._growth_device
            if pending is self.pending_growth:
                arrays = device.materialize()
                self._device_statistics["growth_count_readback_bytes"] += sum(value.nbytes for value in arrays.values())
            self._growth_device = None

    @property
    def foreground_masks(self):
        self._materialize_deletion_masks()
        return self._foreground_masks

    @foreground_masks.setter
    def foreground_masks(self, value):
        self._materialize_deletion_masks()
        self._foreground_masks = value

    @property
    def background_masks(self):
        self._materialize_deletion_masks()
        return self._background_masks

    @background_masks.setter
    def background_masks(self, value):
        self._materialize_deletion_masks()
        self._background_masks = value

    @property
    def deletion_eligible(self):
        # The public NumPy state remains mutable for legacy checkpoint callers.
        self._materialize_deletion_masks()
        return self._deletion_eligible

    @deletion_eligible.setter
    def deletion_eligible(self, value):
        self._materialize_deletion_masks()
        self._deletion_eligible = value

    @property
    def level_budget(self):
        return self.budgets[self.core_resolution]

    @property
    def evidence_start(self):
        return self.level_start_epoch if self.schedule.enabled else self.opacity_warmup + self.opacity_ramp

    def collect_deletion(self, epoch):
        if self.settings.use_rgb_deletion:
            return self._rgb_deletion_due(int(epoch) + 1)
        return (self.settings.enable_deletion and epoch >= self.evidence_start and
                (not self.schedule.enabled or self.core_resolution == self.schedule.target_resolution))

    def deletion_frozen(self, boundary):
        count = self.settings.deletion_freeze_last_epochs
        return count > 0 and boundary > max(0, self.schedule.total_epochs - count)

    def _rgb_deletion_due(self, boundary):
        return self.settings.use_rgb_deletion and self._scheduled_deletion_due(boundary)

    def _low_opacity_deletion_due(self, boundary):
        return self.settings.use_low_opacity_deletion and self._scheduled_deletion_due(boundary)

    def _scheduled_deletion_due(self, boundary):
        """Common epoch-end schedule, independent of per-ray RGB evidence."""
        cfg = self.settings
        if not cfg.enable_deletion or self.deletion_frozen(boundary):
            return False
        local_epoch = boundary - self.level_start_epoch
        # Refinement replaces all rows and resets evidence. Avoid collecting
        # statistics that would immediately be discarded at that boundary.
        if (self.schedule.enabled and self.core_resolution < self.schedule.target_resolution and
                local_epoch >= self.level_budget):
            return False
        return (1 <= boundary <= self.schedule.total_epochs and
                local_epoch > cfg.deletion_warmup_epochs and
                (local_epoch - cfg.deletion_warmup_epochs) % cfg.deletion_interval == 0)

    def growth_frozen(self, boundary):
        """Whether a one-based epoch lies in the global growth-free training tail.

        This uses the complete training schedule, not the current level budget
        or a temporary stop-after limit. Refinement and deletion remain separate.
        """
        count = self.settings.growth_freeze_last_epochs
        return count > 0 and boundary > max(0, self.schedule.total_epochs - count)

    def _growth_due(self, boundary):
        if self.growth_frozen(boundary):
            return False
        schedule, cfg = self.schedule, self.settings
        if cfg.use_rgb_growth:
            local_epoch = boundary - self.level_start_epoch
            enabled = cfg.enable_growth and (not schedule.enabled or schedule.enable_coarse_growth)
            interval = schedule.coarse_growth_interval if schedule.enabled else cfg.growth_interval
            final = local_epoch >= self.level_budget
            warmup = self.growth_warmup
            return enabled and not final and local_epoch > warmup and (local_epoch - warmup) % interval == 0
        enabled = schedule.enable_coarse_growth if schedule.enabled else cfg.enable_growth
        interval = schedule.coarse_growth_interval if schedule.enabled else cfg.growth_interval
        final = (boundary - self.level_start_epoch >= self.level_budget if schedule.enabled else boundary >= schedule.total_epochs)
        return enabled and not final and boundary >= self.evidence_start and (boundary - self.evidence_start) % interval == 0

    def needs_rendered_alpha(self, epoch):
        """Only growth-evidence views need the rendered alpha image."""
        return (not self.settings.use_rgb_growth and self.settings.use_growth_evidence and
                self._growth_due(epoch + 1))

    @property
    def growth_warmup(self):
        """Growth-only warmup; deletion and continuous optimization are unchanged."""
        return (self.settings.growth_warmup_epochs if self.core_resolution == self.schedule_start_resolution
                else self.settings.growth_fine_warmup_epochs)

    def _generate_growth(self, checkpoint, boundary):
        np, cfg = self.np, self.settings
        state, metadata = checkpoint["model_state"], checkpoint["model_metadata"]
        n, resolution = len(state["cell_indices"]), metadata["resolution"]
        center, axes, rotation, quaternion, size, metric, extent = _geometry(state, metadata, np)
        cells = _coordinates(state["cell_indices"], resolution, np)
        eligible = ~self.candidates
        eligible &= _growth_coordinates_allowed(cells, metadata, np)
        eligible &= (self.growth_wait_start == 0) | ((boundary >= self.growth_wait_start - 1) &
                                                    (boundary - (self.growth_wait_start - 1) >= cfg.growth_wait_epochs))
        occupied = np.asarray(state["cell_indices"], dtype=np.int64)
        evidence = cfg.use_growth_evidence or cfg.use_rgb_growth
        if evidence:
            penetration = float(np.clip(cfg.growth_evidence_penetration, 0., 4.))
        elif self.schedule.enabled:
            penetration = self.schedule.coarsest_growth_face_penetration * (1 if metadata.get("core_resolution", resolution) == self.schedule.start_resolution
                                                                           else self.schedule.growth_finer_level_multiplier)
            penetration = float(np.clip(penetration, 0., 4.))
        else:
            penetration = float(np.clip(cfg.growth_face_penetration, 0., 4.))
        pieces = []
        forbidden_cooldown = np.asarray([cell for cell, expiry in self.cooldown.items() if boundary < expiry], dtype=np.int64)
        blocked = _GrowthBlockedCells(occupied, forbidden_cooldown, resolution, np)
        for offset in _neighbor_offsets(evidence and cfg.growth_use_26_neighbors, np):
            neighbors = cells + offset
            targets = _addresses(neighbors, resolution)
            possible = eligible & _growth_coordinates_allowed(neighbors, metadata, np)
            ids = np.flatnonzero(possible)
            # Filter in the original ascending parent order, querying only
            # valid neighbor cells. Contact geometry below is unchanged.
            ids = ids[~blocked.contains(targets[ids])]
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

    def prepare_rgb_growth(self, model, epoch):
        """Freeze/upload legal candidates once, before this epoch's first trace.

        ``epoch`` is zero-based. The first coarse and fine events are local
        epoch 30 with the default 20-epoch warmup and 10-epoch interval.
        The payload in pending_growth['candidate_state'] can be broadcast to
        worker models; none of these arrays are committed model parameters.
        """
        from reconstruction_topology_gpu import get_gpu_lifecycle
        device = get_gpu_lifecycle(self, model)
        if device is not None:
            return device.prepare(model, epoch)
        timings = self.growth_prepare_stage_seconds = {}
        cfg, np, boundary = self.settings, self.np, int(epoch) + 1
        if not cfg.use_rgb_growth:
            return False
        if not self._growth_due(boundary):
            started = time.perf_counter()
            model.clear_growth_candidates()
            self.pending_growth = None
            self._rgb_growth_device = None
            timings["candidate_clear"] = time.perf_counter() - started
            return False
        pending = self.pending_growth
        if (pending is None or pending.get("mode") != "rgb" or
                pending["boundary"] != boundary or pending["resolution"] != self.resolution):
            from reconstruction_growth_rgb import initialize_statistics
            started = time.perf_counter()
            checkpoint = model_checkpoint(model)
            timings["snapshot"] = time.perf_counter() - started
            started = time.perf_counter()
            proposals = self._generate_growth(checkpoint, boundary)
            timings["candidate_generation"] = time.perf_counter() - started
            started = time.perf_counter()
            # A cell can have different shape/color proposals from several
            # parents. Keep them independent until after evidence and geometry
            # revalidation: the first legal parent need not be beneficial.
            order = np.lexsort((proposals["parent_cells"], proposals["cells"]))
            pending = {name: values[order].copy() for name, values in proposals.items()
                       if isinstance(values, np.ndarray)}
            cells, inverse, counts = np.unique(pending["cells"], return_inverse=True, return_counts=True)
            count = len(order)
            state, metadata = checkpoint["model_state"], checkpoint["model_metadata"]
            radiance = np.zeros((count, metadata["sh_count"], 3), dtype=np.float32)
            radiance[:, 0] = state["radiance_sh"].reshape(self.count, metadata["sh_count"], 3)[pending["parent_ids"], 0]
            opacity = _growth_opacity_coefficients(
                state, metadata, pending["parent_ids"], cfg, np)
            pending.update(radiance_sh=radiance, opacity_sh=opacity,
                           geometric_support=counts[inverse].astype(np.uint32),
                           mode="rgb", normalization=cfg.growth_rgb_normalization,
                           evidence_version=RGB_GROWTH_EVIDENCE_VERSION,
                           boundary=boundary, resolution=self.resolution,
                           all_candidate_count=count, candidate_cell_count=len(cells), seen_views=set())
            for name in ("centers", "log_scales", "rotations"):
                pending[name] = pending[name].astype(np.float32)
            pending["candidate_state"] = {
                "cell_indices": pending["cells"].astype(np.uint32),
                **{name: pending[name].reshape(-1).copy() for name in
                   ("centers", "log_scales", "rotations", "radiance_sh", "opacity_sh")}}
            initialize_statistics(pending, np)
            self.pending_growth = pending
            self._growth_world_cache = None
            self._growth_device = None
            self._rgb_growth_device = None
            timings["proposal_packing"] = time.perf_counter() - started
        self._rgb_normalization(pending)
        if not len(pending["cells"]):
            started = time.perf_counter()
            model.clear_growth_candidates()
            timings["candidate_clear"] = time.perf_counter() - started
            return False
        from reconstruction_growth_rgb import upload_rgb_growth_candidates
        started = time.perf_counter()
        upload_rgb_growth_candidates(model, pending["candidate_state"])
        timings["candidate_install"] = time.perf_counter() - started
        return True

    def observe_rgb_growth(self, model, out, view_id, epoch, pixel_count):
        """Consume ONE full-view native statistic after chunk/rank reduction.

        Unknown rays abstain individually. A view can vote from its reliable
        positive, negative and zero-gain hits, even if other hits are unknown.
        This is evidence conditional on reliable observations, not a claim
        about missing rays. Unknown-only and invisible views cannot vote.
        """
        from reconstruction_topology_gpu import get_gpu_lifecycle
        device = get_gpu_lifecycle(self, model)
        if device is not None:
            return device.observe(model, out, view_id, epoch, pixel_count)
        cfg, pending, np = self.settings, self.pending_growth, self.np
        if not cfg.use_rgb_growth or not self._growth_due(int(epoch) + 1):
            return
        if (pending is None or pending.get("mode") != "rgb" or
                pending["boundary"] != int(epoch) + 1 or pending["resolution"] != self.resolution):
            raise ValueError("RGB growth must be prepared before tracing the evidence epoch")
        if not 0 <= view_id < self.view_count or int(model.active_voxel_count) != self.count:
            raise ValueError("Unexpected RGB growth view ID or model topology")
        if not isinstance(pixel_count, int) or pixel_count < 1:
            raise ValueError("RGB growth requires a positive full-image pixel count")
        normalization = self._rgb_normalization(pending)
        if view_id in pending["seen_views"]:
            return
        if len(pending["cells"]):
            dr = _cuda_array(out["growth_gain"])
            if dr is not None:
                from reconstruction_growth_rgb import RGBGrowthEvidenceDevice
                if self._rgb_growth_device is None:
                    self._rgb_growth_device = (pending, RGBGrowthEvidenceDevice(
                        pending, type(out["growth_gain"]), type(out["growth_observed"]), dr, np))
                self._rgb_growth_device[1].observe(out, pixel_count, cfg.growth_rgb_gain_epsilon,
                                                   normalization=normalization)
                self._device_statistics["rgb_growth_device_views"] += 1
            else:
                from reconstruction_growth_rgb import observe_numpy
                self._materialize_growth()
                observe_numpy(pending, out, pixel_count, cfg.growth_rgb_gain_epsilon, np,
                              normalization=normalization)
                self._device_statistics["rgb_growth_cpu_views"] += 1
        pending["seen_views"].add(int(view_id))

    def _rgb_normalization(self, pending):
        """Never reinterpret statistics already accumulated by an older run."""
        if pending.get("evidence_version") != RGB_GROWTH_EVIDENCE_VERSION:
            raise ValueError("RGB growth evidence version differs; start a new run "
                             "or resume with the original code")
        normalization = pending.get("normalization", "full_image")
        if normalization != self.settings.growth_rgb_normalization:
            raise ValueError("RGB growth pending evidence uses a different normalization; "
                             "resume with its original topology configuration")
        return normalization

    def observe_rgb_deletion(self, model, out, view_id, epoch, pixel_count):
        """Consume full-view removal sums after ray-chunk/rank SUM reduction.

        Statistics stay on the GPU until a topology boundary/checkpoint.
        Unknown rays do not veto other reliable rays; no missing observation
        provides a positive removal vote.
        """
        from reconstruction_topology_gpu import get_gpu_lifecycle
        device = get_gpu_lifecycle(self, model)
        if device is not None:
            return device.observe(model, out, view_id, epoch, pixel_count, deletion=True)
        if not self.collect_deletion(epoch) or not self.settings.use_rgb_deletion:
            return
        np, cfg, boundary = self.np, self.settings, int(epoch) + 1
        if not 0 <= view_id < self.view_count or int(model.active_voxel_count) != self.count:
            raise ValueError("Unexpected RGB deletion view ID or model topology")
        if not isinstance(pixel_count, int) or pixel_count < 1:
            raise ValueError("RGB deletion requires a positive full-image pixel count")
        pending = self.pending_rgb_deletion
        if (pending is None or pending["boundary"] != boundary or
                pending["resolution"] != self.resolution):
            from reconstruction_deletion_rgb import initialize_pending
            self._rgb_deletion_device = None
            pending = initialize_pending(self.count, boundary, self.resolution, np)
            self.pending_rgb_deletion = pending
        if view_id in pending["seen_views"]:
            return
        if self.count:
            dr = _cuda_array(out["deletion_gain"])
            if dr is not None:
                from reconstruction_deletion_rgb import RGBDeletionEvidenceDevice
                if self._rgb_deletion_device is None:
                    self._rgb_deletion_device = (pending, RGBDeletionEvidenceDevice(
                        pending, type(out["deletion_gain"]), type(out["deletion_observed"]), dr, np))
                self._rgb_deletion_device[1].observe(
                    out, pixel_count, cfg.deletion_rgb_gain_epsilon)
                self._device_statistics["rgb_deletion_device_views"] += 1
            else:
                from reconstruction_deletion_rgb import observe_numpy
                self._materialize_rgb_deletion()
                observe_numpy(pending, out, pixel_count, cfg.deletion_rgb_gain_epsilon, np)
                self._device_statistics["rgb_deletion_cpu_views"] += 1
        pending["seen_views"].add(int(view_id))

    def _evaluate_rgb_deletion(self, boundary, event):
        """Confirm removals using consecutive complete, reliable view windows."""
        np, cfg, pending = self.np, self.settings, self.pending_rgb_deletion
        if not self._rgb_deletion_due(boundary):
            return None
        self._materialize_rgb_deletion()
        matching = (pending is not None and pending["boundary"] == boundary and
                    pending["resolution"] == self.resolution)
        if not matching:
            from reconstruction_deletion_rgb import initialize_pending
            pending = initialize_pending(self.count, boundary, self.resolution, np)
        from reconstruction_deletion_rgb import validate_pending
        validate_pending(pending, self.count, self.resolution, self.view_count, np)
        views = len(pending["seen_views"])
        complete = views == self.view_count
        mean_gain = pending["gain_sum"] / np.maximum(pending["observed_views"], 1)
        observed = pending["observed_views"] > 0
        supported = pending["support_views"] >= cfg.deletion_rgb_min_support_views
        unopposed = pending["conflict_views"] < cfg.deletion_rgb_max_conflict_views
        valid = (pending["invalid_views"] == 0) & np.isfinite(mean_gain)
        beneficial = mean_gain > cfg.deletion_rgb_min_mean_gain
        # A voxel born after boundary b is protected for complete epochs
        # b+1 .. b+protection; it becomes eligible at b+protection+1.
        mature = ((self.growth_birth == 0) |
                  (boundary - self.growth_birth >= cfg.deletion_protection_epochs))
        qualified = observed & supported & unopposed & valid & beneficial & mature
        if not complete:
            qualified.fill(False)
        previous = self.rgb_deletion_streak.copy()
        consecutive = self.rgb_deletion_last_boundary == boundary - cfg.deletion_interval
        if not consecutive:
            self.rgb_deletion_streak.fill(0)
        self.rgb_deletion_streak = np.where(
            qualified, np.minimum(self.rgb_deletion_streak.astype(np.uint64) + 1,
                                  cfg.deletion_rgb_confirmation_windows), 0).astype(np.uint32)
        self.rgb_deletion_last_boundary = boundary
        selected = self.rgb_deletion_streak >= cfg.deletion_rgb_confirmation_windows
        known_rays = int(pending["observed_rays"].sum())
        unknown_rays = int(pending["unknown_rays"].sum())
        total_rays = known_rays + unknown_rays
        event["deletion_rgb"] = {
            "status": "complete" if complete else "incomplete_views",
            "evidence_version": RGB_DELETION_EVIDENCE_VERSION,
            "normalization": "observed", "views": views, "required_views": self.view_count,
            "tested_voxels": self.count, "eligible_voxels": int(np.count_nonzero(mature)),
            "protected_voxels": int(np.count_nonzero(~mature)),
            "support_votes": int(pending["support_views"].sum()),
            "conflict_votes": int(pending["conflict_views"].sum()),
            "observed_views": int(pending["observed_views"].sum()),
            "views_with_unknown": int(pending["unknown_views"].sum()),
            "invalid_votes": int(pending["invalid_views"].sum()),
            "voxels_with_observations": int(np.count_nonzero(observed)),
            "voxels_with_unknown_views": int(np.count_nonzero(pending["unknown_views"])),
            "observed_rays": known_rays, "unknown_rays": unknown_rays,
            "encountered_known_fraction": known_rays / total_rays if total_rays else None,
            "mean_gain_sum": float(np.where(np.isfinite(mean_gain), mean_gain, 0).sum()),
            # These gates overlap; they are not a partition of all voxels.
            "rejected_no_observation": int(np.count_nonzero(~observed)),
            "rejected_support": int(np.count_nonzero(~supported)),
            "rejected_conflict": int(np.count_nonzero(~unopposed)),
            "rejected_invalid": int(np.count_nonzero(~valid)),
            "rejected_mean_gain": int(np.count_nonzero(~beneficial)),
            "rgb_qualified": int(np.count_nonzero(qualified)),
            "confirmation_windows": cfg.deletion_rgb_confirmation_windows,
            "previous_window_consecutive": consecutive,
            "streak_reset": int(np.count_nonzero((previous > 0) &
                                               (~qualified | (not consecutive)))),
            "awaiting_confirmation": int(np.count_nonzero(qualified & ~selected)),
            "confirmed": int(np.count_nonzero(selected)),
            "deleted": 0,
            "max_streak": int(self.rgb_deletion_streak.max()) if self.count else 0,
        }
        self.pending_rgb_deletion = None
        self._rgb_deletion_device = None
        return selected

    def _evaluate_low_opacity_deletion(self, model, boundary, event):
        """Inspect only the optimized DC opacity at a scheduled epoch boundary.

        This is a parameter-based cleanup rule, not a visibility or RGB vote:
        unvisited voxels can qualify only when their own learned opacity stays
        low. No ray tracing, per-view collection, or full host snapshot is needed.
        """
        if not self._low_opacity_deletion_due(boundary):
            return None
        np, cfg = self.np, self.settings
        if int(model.opacity_sh_count) != 1:
            raise ValueError("Low-opacity deletion requires isotropic opacity_sh_count == 1")
        # snapshot() returns existing device buffers; copy just this field.
        value = model.snapshot()["opacity_sh"]
        dr = _cuda_array(value)
        if dr is not None:
            dr.eval(value)
            dr.sync_thread()
        coefficients = np.array(value, copy=True).reshape(-1)
        self._device_statistics["deletion_opacity_readback_bytes"] += coefficients.nbytes
        if coefficients.shape != (self.count,):
            raise ValueError("Low-opacity deletion parameter dimensions differ")
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("Low-opacity deletion encountered nonfinite opacity coefficients")
        # sigmoid(Y00*b0) < threshold, compared in logit space to avoid
        # overflow/underflow for nearly transparent or nearly opaque voxels.
        threshold = float(cfg.deletion_opacity_threshold)
        coefficient_threshold = math.log(threshold / (1. - threshold)) * (2. * math.sqrt(math.pi))
        low = coefficients.astype(np.float64) < coefficient_threshold
        mature = ((self.growth_birth == 0) |
                  (boundary - self.growth_birth >= cfg.deletion_protection_epochs))
        qualified = low & mature
        previous = self.low_opacity_deletion_streak.copy()
        consecutive = self.low_opacity_deletion_last_boundary == boundary - cfg.deletion_interval
        if not consecutive:
            self.low_opacity_deletion_streak.fill(0)
        self.low_opacity_deletion_streak = np.where(
            qualified, np.minimum(self.low_opacity_deletion_streak.astype(np.uint64) + 1,
                                  cfg.deletion_opacity_confirmation_windows), 0).astype(np.uint32)
        self.low_opacity_deletion_last_boundary = boundary
        selected = self.low_opacity_deletion_streak >= cfg.deletion_opacity_confirmation_windows
        event["deletion_opacity"] = {
            "status": "complete", "threshold": threshold,
            "tested_voxels": self.count, "low_opacity_voxels": int(np.count_nonzero(low)),
            "eligible_voxels": int(np.count_nonzero(mature)),
            "protected_voxels": int(np.count_nonzero(~mature)),
            "qualified_voxels": int(np.count_nonzero(qualified)),
            "confirmation_windows": cfg.deletion_opacity_confirmation_windows,
            "previous_window_consecutive": consecutive,
            "streak_reset": int(np.count_nonzero((previous > 0) &
                                               (~qualified | (not consecutive)))),
            "awaiting_confirmation": int(np.count_nonzero(qualified & ~selected)),
            "confirmed": int(np.count_nonzero(selected)), "deleted": 0,
            "max_streak": int(self.low_opacity_deletion_streak.max()) if self.count else 0,
        }
        return selected

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
        self._materialize_growth()
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
                     reference_alpha, rendered_alpha, image_shape=None):
        np = self.np
        if not 0 <= view_id < self.view_count or int(model.active_voxel_count) != self.count:
            raise ValueError("Unexpected view ID or model topology")
        if not self.settings.use_rgb_deletion and self.collect_deletion(epoch):
            dr = _cuda_array(out["foreground_evidence"])
            if dr is not None and _cuda_array(out["background_evidence"]) is not None:
                if self._deletion_device is None:
                    self._deletion_device = _DeletionEvidenceDevice(
                        type(out["foreground_evidence"]), self._foreground_masks,
                        self._background_masks, self._deletion_eligible, dr, np)
                self._deletion_device.observe(out["foreground_evidence"], out["background_evidence"], view_id, epoch)
                self._device_statistics["deletion_device_views"] += 1
            else:
                self._materialize_deletion_masks()
                eligible = epoch >= self._deletion_eligible
                bit, word = np.uint32(1 << (view_id % 32)), view_id // 32
                for key, masks in (("foreground_evidence", self._foreground_masks), ("background_evidence", self._background_masks)):
                    flags = np.asarray(out[key]).reshape(-1) != 0
                    if len(flags) != self.count:
                        raise ValueError("Deletion evidence count differs from active voxel count")
                    masks[flags & eligible, word] |= bit
                self._device_statistics["deletion_cpu_views"] += 1
        boundary = epoch + 1
        if self.needs_rendered_alpha(epoch):
            if rendered_alpha is None:
                raise ValueError("Growth evidence requires this view's rendered alpha before its update")
            if self.pending_growth is None or self.pending_growth["boundary"] != boundary:
                self._growth_device = None
                checkpoint = model_checkpoint(model, ("cell_indices", "centers", "log_scales", "rotations"))
                self.pending_growth = self._generate_growth(checkpoint, boundary)
            metadata = {"resolution": self.resolution, "grid_min": list(model.grid_min), "grid_max": list(model.grid_max)}
            dr = _cuda_array(rendered_alpha)
            if dr is not None:
                if image_shape is None or len(image_shape) != 2:
                    raise ValueError("GPU growth evidence requires an (height, width) image_shape")
                import mitsuba as mi
                from reconstruction_topology_device import GrowthEvidenceDevice
                if self._growth_device is None or self._growth_device[0] is not self.pending_growth:
                    points = self._growth_world_points(self.pending_growth, metadata)
                    self._growth_device = (self.pending_growth, GrowthEvidenceDevice(points, self.pending_growth, mi, dr, np))
                height, width = image_shape
                self._growth_device[1].observe(camera, reference_alpha, rendered_alpha,
                                               width, height, view_id, self.settings)
                self._device_statistics["growth_device_views"] += 1
            else:
                self._observe_growth(camera, reference_alpha, rendered_alpha, view_id, metadata)
                self._device_statistics["growth_cpu_views"] += 1

    def _evaluate_deletion(self, boundary):
        np, cfg = self.np, self.settings
        if cfg.use_rgb_deletion:
            return False
        if not self.collect_deletion(boundary) or boundary <= self.evidence_start or (boundary - self.evidence_start) % cfg.evidence_interval:
            return False
        if self._deletion_device is not None:
            fg, bg = self._deletion_device.counts()
            self._device_statistics["deletion_count_readback_bytes"] += fg.nbytes + bg.nbytes
            self._deletion_device.clear()
        else:
            lut = np.asarray([int(value).bit_count() for value in range(256)], dtype=np.uint8)
            fg = lut[self._foreground_masks.view(np.uint8)].sum(axis=1)
            bg = lut[self._background_masks.view(np.uint8)].sum(axis=1)
        ratio = max(1, min(1000, cfg.deletion_conflict_support_ratio))
        dominates = bg >= ratio * fg
        reliable = max(2, (cfg.min_deletion_conflict_views + ratio - 1) // ratio)
        rejected = (fg >= reliable) & ~dominates
        qualified = (bg >= cfg.min_deletion_conflict_views) & dominates
        self.candidates = ~rejected & (self.candidates | (qualified & self.qualified))
        self.qualified = ~rejected & (self.qualified | qualified | self.candidates)
        protected = boundary < self._deletion_eligible
        self.candidates[protected] = False
        self.qualified[protected] = False
        self._foreground_masks.fill(0)
        self._background_masks.fill(0)
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
        self._materialize_deletion_masks()
        kept = len(survivors)
        for name in ("foreground_masks", "background_masks", "qualified", "candidates",
                     "deletion_eligible", "growth_birth", "growth_wait_start",
                     "rgb_deletion_streak", "low_opacity_deletion_streak"):
            old = getattr(self, name)
            resized = np.zeros((new_count,) + old.shape[1:], dtype=old.dtype)
            resized[:kept] = old[survivors]
            setattr(self, name, resized)
        if new_count > kept:
            self.deletion_eligible[kept:] = boundary + self.settings.growth_protection_epochs
            self.growth_birth[kept:] = boundary + 1
            self.growth_wait_start[kept:] = boundary + 1
        self.count = new_count

    def _select_rgb_growth(self, model, checkpoint, boundary, event):
        """Read epoch statistics, recheck geometry, keep frozen parameters."""
        np, cfg, pending = self.np, self.settings, self.pending_growth
        selected = None
        if (pending is None or pending.get("mode") != "rgb" or
                pending["boundary"] != boundary or pending["resolution"] != self.resolution):
            event["growth_rgb"] = {"status": "no_prepared_candidates",
                                   "normalization": cfg.growth_rgb_normalization}
            return checkpoint, selected
        count = len(pending["cells"])
        seen = len(pending["seen_views"])
        normalization = self._rgb_normalization(pending)
        denominator = (np.maximum(pending["observed_views"], 1) if normalization == "observed"
                       else self.view_count)
        mean_gain = pending["gain_sum"] / denominator
        observed = pending["observed_views"] > 0
        supported = pending["support_views"] >= cfg.growth_rgb_min_support_views
        unopposed = pending["conflict_views"] < cfg.growth_rgb_max_conflict_views
        valid = pending["invalid_views"] == 0
        beneficial = np.isfinite(mean_gain) & (mean_gain > cfg.growth_rgb_min_mean_gain)
        known_rays = int(pending["observed_rays"].sum())
        unknown_rays = int(pending["unknown_rays"].sum())
        encountered_rays = known_rays + unknown_rays
        status = "no_candidates" if not count else ("complete" if seen == self.view_count else "incomplete_views")
        detail = {"status": status, "normalization": normalization,
                  "evidence_version": RGB_GROWTH_EVIDENCE_VERSION,
                  "geometric_candidates": pending["all_candidate_count"], "tested_candidates": count,
                  "candidate_cells": pending["candidate_cell_count"],
                  "views": seen, "support_votes": int(pending["support_views"].sum()),
                  "conflict_votes": int(pending["conflict_views"].sum()),
                  "observed_views": int(pending["observed_views"].sum()),
                  "views_with_unknown": int(pending["unknown_views"].sum()),
                  "invalid_votes": int(pending["invalid_views"].sum()),
                  "candidates_with_observations": int(np.count_nonzero(observed)),
                  "candidates_with_unknown_views": int(np.count_nonzero(pending["unknown_views"])),
                  "observed_rays": known_rays,
                  "unknown_rays": unknown_rays,
                  # Candidate-hit observations encountered before DDA stopped;
                  # unvisited regions are not covered by this diagnostic.
                  "encountered_known_fraction": (known_rays / encountered_rays
                                                 if encountered_rays else None),
                  "mean_gain_sum": float(mean_gain.sum()), "rgb_qualified": 0,
                  "rgb_qualified_cells": 0,
                  # Rejection gates are independent counts and may overlap.
                  "rejected_no_observation": int(np.count_nonzero(~observed)),
                  "rejected_support": int(np.count_nonzero(~supported)),
                  "rejected_conflict": int(np.count_nonzero(~unopposed)),
                  "rejected_invalid": int(np.count_nonzero(~valid)),
                  "rejected_mean_gain": int(np.count_nonzero(~beneficial)),
                  "geometry_revalidated": 0, "geometry_rejected": 0,
                  "geometry_revalidated_cells": 0, "same_cell_rejected": 0,
                  "capacity_rejected": 0,
                  "selected_candidates": 0}
        event["growth_rgb"] = detail
        if seen != self.view_count or not count:
            return checkpoint, selected
        keep = observed & supported & unopposed & valid & beneficial
        detail["rgb_qualified"] = int(np.count_nonzero(keep))
        detail["rgb_qualified_cells"] = len(np.unique(pending["cells"][keep]))
        if not np.any(keep):
            return checkpoint, selected
        from reconstruction_growth_filter import generate_growth_for_targets, match_growth_parents
        voted = {key: value[keep] for key, value in pending.items() if isinstance(value, np.ndarray)}
        if checkpoint is None:
            checkpoint = model_checkpoint(model)
        current = generate_growth_for_targets(self, checkpoint, boundary, voted["cells"])
        matches, parent_ids = match_growth_parents(current, voted)
        selected = {key: value[matches] for key, value in voted.items()}
        selected["parent_ids"] = parent_ids
        detail["geometry_revalidated"] = len(parent_ids)
        detail["geometry_rejected"] = detail["rgb_qualified"] - len(parent_ids)
        # Select once per cell only AFTER each parent proposal has independently
        # passed all view gates and current-geometry checks. No votes are pooled
        # between siblings. Equal local mean gains use parent-cell order.
        gains = mean_gain[keep][matches]
        proposal_order = np.lexsort((selected["parent_cells"], -gains, selected["cells"]))
        _, first = np.unique(selected["cells"][proposal_order], return_index=True)
        per_cell = proposal_order[first]
        detail["geometry_revalidated_cells"] = len(per_cell)
        detail["same_cell_rejected"] = len(parent_ids) - len(per_cell)
        selected = {key: value[per_cell] for key, value in selected.items()}
        gains = gains[per_cell]
        capacity = max(0, cfg.max_active_voxels - self.count)
        # Coverage must not bias a local-mean ranking when capacity is scarce.
        ranking = np.lexsort((selected["cells"], -gains))
        winners = ranking[:capacity]
        selected = {key: value[winners] for key, value in selected.items()}
        detail["capacity_rejected"] = len(ranking) - len(winners)
        detail["selected_candidates"] = len(winners)
        return checkpoint, selected

    def finish_epoch(self, model, optimizer, completed_epochs: int):
        """Called after every view update. Returns (replacement checkpoint or None, event)."""
        from reconstruction_topology_gpu import get_gpu_lifecycle
        device = get_gpu_lifecycle(self, model)
        if device is not None:
            return device.finish(model, optimizer, int(completed_epochs))
        np, cfg, schedule = self.np, self.settings, self.schedule
        boundary = int(completed_epochs)
        self._materialize_growth()
        self.before_refine_checkpoint = None
        event = {"completed_epochs": boundary, "resolution": self.resolution, "deleted": 0, "grown": 0,
                 "refined": False, "finished": False, "level_budget": self.level_budget,
                 "growth_frozen": self.growth_frozen(boundary),
                 "growth_freeze_last_epochs": cfg.growth_freeze_last_epochs}
        if self.core_resolution != self.resolution:
            event["core_resolution"] = self.core_resolution
        if cfg.use_rgb_growth:
            model.clear_growth_candidates()
            event.update(growth_mode="rgb_online", growth_level_epoch=boundary - self.level_start_epoch,
                         growth_rgb_normalization=cfg.growth_rgb_normalization,
                         growth_warmup_epochs=self.growth_warmup,
                         growth_warmup_active=boundary - self.level_start_epoch <= self.growth_warmup,
                         growth_evidence_requested=self._growth_due(boundary))
        if cfg.use_rgb_deletion or cfg.use_low_opacity_deletion:
            mode = ("rgb_online+low_opacity" if cfg.use_rgb_deletion else "legacy+low_opacity"
                    ) if cfg.use_low_opacity_deletion else "rgb_online"
            event.update(deletion_mode=mode,
                         deletion_level_epoch=boundary - self.level_start_epoch,
                         deletion_warmup_epochs=cfg.deletion_warmup_epochs,
                         deletion_warmup_active=boundary - self.level_start_epoch <= cfg.deletion_warmup_epochs,
                         deletion_frozen=self.deletion_frozen(boundary),
                         deletion_freeze_last_epochs=cfg.deletion_freeze_last_epochs,
                         deletion_evidence_requested=self._rgb_deletion_due(boundary))
        if cfg.use_low_opacity_deletion:
            event["deletion_opacity_requested"] = self._low_opacity_deletion_due(boundary)
        checkpoint = None
        level_done = boundary - self.level_start_epoch >= self.level_budget
        if schedule.enabled and self.core_resolution < schedule.target_resolution and level_done:
            previous = model_checkpoint(model)
            replacement = refine_state(previous, schedule, cfg.max_active_voxels)
            if schedule.save_each_level:
                self.before_refine_checkpoint = {
                    **previous, "optimizer_state": optimizer.state_dict(), "topology_state": self.state_dict()}
                event["pre_refine_checkpoint_available"] = True
            self.resolution = replacement["model_metadata"]["resolution"]
            self.core_resolution = replacement["model_metadata"].get("core_resolution", self.resolution)
            self.level_start_epoch = boundary
            self._initialize_lifecycle(len(replacement["model_state"]["cell_indices"]))
            self.completed_windows = 0
            self.cooldown = {}
            self.pending_growth = None
            self._growth_world_cache = None
            self._growth_device = None
            self._rgb_growth_device = None
            optimizer.remap(np.empty(0, dtype=np.uint32), self.count)
            event.update(refined=True, resolution=self.resolution, active_voxel_count=self.count)
            if self.core_resolution != self.resolution:
                event["core_resolution"] = self.core_resolution
            return replacement, event
        rgb_delete = self._evaluate_rgb_deletion(boundary, event) if cfg.use_rgb_deletion else None
        opacity_delete = self._evaluate_low_opacity_deletion(model, boundary, event)
        event["evidence_window_evaluated"] = (
            rgb_delete is not None if cfg.use_rgb_deletion else self._evaluate_deletion(boundary))
        final = level_done if schedule.enabled else boundary >= schedule.total_epochs
        first = self.evidence_start + 2 * cfg.evidence_interval
        periodic = boundary >= first and (boundary - first) % cfg.deletion_interval == 0
        legacy_delete = (self.candidates if not cfg.use_rgb_deletion and
                         (periodic or final) and self.collect_deletion(boundary) else None)
        masks = [mask for mask in (rgb_delete, legacy_delete, opacity_delete) if mask is not None]
        delete = np.logical_or.reduce(masks) if masks else None
        if delete is not None and np.any(delete):
            checkpoint = model_checkpoint(model)
            old_cells = checkpoint["model_state"]["cell_indices"]
            for cell in old_cells[delete]:
                self.cooldown[int(cell)] = boundary + cfg.deletion_cooldown_epochs
            survivors = np.flatnonzero(~delete).astype(np.uint32)
            event["deleted"] = self.count - len(survivors)
            checkpoint = self._take_rows(checkpoint, survivors)
            if rgb_delete is not None:
                event["deletion_rgb"]["deleted"] = int(np.count_nonzero(rgb_delete))
            if opacity_delete is not None:
                event["deletion_opacity"]["deleted"] = int(np.count_nonzero(opacity_delete))
            # Preserve every surviving Adam group for RGB/opacity cleanup.
            # Only actual legacy evidence removals retain Falcor's historical
            # radiance reset; a mere enabled legacy mode must not trigger it.
            if legacy_delete is not None and np.any(legacy_delete):
                optimizer.remap(survivors, len(survivors), reset_radiance=True)
            else:
                optimizer.remap(survivors, len(survivors))
            self._remap_lifecycle(survivors, len(survivors), boundary)
            self.foreground_masks.fill(0)
            self.background_masks.fill(0)
            self.qualified.fill(False)
            self.candidates.fill(False)
            self.completed_windows = 0
        if self._growth_due(boundary):
            selected = None
            if cfg.use_rgb_growth:
                checkpoint, selected = self._select_rgb_growth(model, checkpoint, boundary, event)
            elif cfg.use_growth_evidence:
                pending = self.pending_growth
                if (pending is not None and pending["boundary"] == boundary and
                        pending["resolution"] == self.resolution and len(pending["seen_views"]) == self.view_count):
                    keep = ((pending["foreground"] >= max(2, cfg.growth_min_foreground_views)) &
                            (pending["background"] < max(1, cfg.growth_background_veto_views)))
                    if np.any(keep):
                        from reconstruction_growth_filter import generate_growth_for_targets, match_growth_parents
                        voted = {key: value[keep] for key, value in pending.items() if isinstance(value, np.ndarray)}
                        if checkpoint is None:
                            checkpoint = model_checkpoint(model)
                        # Validate current geometry and *all* supporting parents,
                        # but only for cells whose frozen proposals passed votes.
                        current = generate_growth_for_targets(self, checkpoint, boundary, voted["cells"])
                        matches, parent_ids = match_growth_parents(current, voted)
                        selected = {key: value[matches] for key, value in voted.items()}
                        selected["parent_ids"] = parent_ids
            else:
                if checkpoint is None:
                    checkpoint = model_checkpoint(model)
                selected = self._generate_growth(checkpoint, boundary)
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
                if cfg.use_rgb_growth:
                    # Commit exactly what the training rays evaluated, even
                    # though the parent's appearance changed during the epoch.
                    radiance, opacity = selected["radiance_sh"], selected["opacity_sh"]
                else:
                    radiance = np.zeros((added, metadata["sh_count"], 3), dtype=np.float32)
                    radiance[:, 0] = state["radiance_sh"].reshape(self.count, metadata["sh_count"], 3)[parents, 0]
                    opacity = _growth_opacity_coefficients(
                        state, metadata, parents, cfg, np)
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
        self._growth_device = None
        self._rgb_growth_device = None
        self.pending_rgb_deletion = None
        self._rgb_deletion_device = None
        self.cooldown = {cell: expiry for cell, expiry in self.cooldown.items() if expiry > boundary}
        event.update(finished=final, active_voxel_count=self.count)
        return checkpoint, event

    def state_dict(self):
        import copy
        device = getattr(self, "_gpu_lifecycle", None)
        if device is not None:
            device.materialize()
            self._gpu_lifecycle = None
        self._materialize_deletion_masks()
        self._materialize_growth()
        self._materialize_rgb_deletion()
        state = {"settings": asdict(self.settings), "schedule": asdict(self.schedule), "view_count": self.view_count,
                 "rgb_growth_evidence_version": RGB_GROWTH_EVIDENCE_VERSION,
                 "rgb_deletion_evidence_version": RGB_DELETION_EVIDENCE_VERSION,
                 "opacity_warmup": self.opacity_warmup, "opacity_ramp": self.opacity_ramp,
                 "resolution": self.resolution, "schedule_start_resolution": self.schedule_start_resolution,
                 "level_start_epoch": self.level_start_epoch, "budgets": {str(key): value for key, value in self.budgets.items()},
                 "cooldown": {str(key): value for key, value in self.cooldown.items()}, "completed_windows": self.completed_windows,
                 "count": self.count, "pending_growth": self.pending_growth,
                 "pending_rgb_deletion": self.pending_rgb_deletion,
                 "rgb_deletion_streak": self.rgb_deletion_streak,
                 "rgb_deletion_last_boundary": self.rgb_deletion_last_boundary,
                 "low_opacity_deletion_streak": self.low_opacity_deletion_streak,
                 "low_opacity_deletion_last_boundary": self.low_opacity_deletion_last_boundary}
        if self.core_resolution != self.resolution:
            state["core_resolution"] = self.core_resolution
        for name in ("foreground_masks", "background_masks", "qualified", "candidates",
                     "deletion_eligible", "growth_birth", "growth_wait_start"):
            state[name] = getattr(self, name)
        state = copy.deepcopy(state)
        if state["pending_growth"] is not None:
            state["pending_growth"]["seen_views"] = sorted(state["pending_growth"]["seen_views"])
        if state["pending_rgb_deletion"] is not None:
            state["pending_rgb_deletion"]["seen_views"] = sorted(state["pending_rgb_deletion"]["seen_views"])
        return state

    def load_state_dict(self, state):
        import copy
        self._gpu_lifecycle = None
        gpu_topology = self.settings.gpu_topology
        if (state["resolution"] != self.resolution or
                state.get("core_resolution", state["resolution"]) != self.core_resolution):
            raise ValueError("Topology checkpoint grid/core resolution does not match the model")
        if (state["settings"].get("use_rgb_growth", False) and
                state.get("rgb_growth_evidence_version") != RGB_GROWTH_EVIDENCE_VERSION):
            raise ValueError("RGB growth evidence version differs; start a new run "
                             "or resume with the original code")
        if state["settings"].get("use_rgb_deletion", False):
            if state.get("rgb_deletion_evidence_version") != RGB_DELETION_EVIDENCE_VERSION:
                raise ValueError("RGB deletion evidence version differs; start a new run "
                                 "or resume with the original code")
            if any(name not in state for name in (
                    "pending_rgb_deletion", "rgb_deletion_streak", "rgb_deletion_last_boundary")):
                raise ValueError("RGB deletion lifecycle state is incomplete")
        if state["settings"].get("use_low_opacity_deletion", False):
            if any(name not in state for name in (
                    "low_opacity_deletion_streak", "low_opacity_deletion_last_boundary")):
                raise ValueError("Low-opacity deletion lifecycle state is incomplete")
        self._growth_world_cache = None
        self._growth_device = None
        self._rgb_growth_device = None
        self._deletion_device = None
        self._rgb_deletion_device = None
        self.pending_rgb_deletion = None
        self.rgb_deletion_streak = self.np.zeros(self.count, dtype=self.np.uint32)
        self.rgb_deletion_last_boundary = None
        self.low_opacity_deletion_streak = self.np.zeros(self.count, dtype=self.np.uint32)
        self.low_opacity_deletion_last_boundary = None
        state = copy.deepcopy(state)
        state.pop("rgb_growth_evidence_version", None)
        state.pop("rgb_deletion_evidence_version", None)
        state.setdefault("core_resolution", state["resolution"])
        if state["count"] != self.count or state["view_count"] != self.view_count:
            raise ValueError("Topology checkpoint dimensions do not match model or dataset")
        self.settings = TopologySettings(**state.pop("settings"))
        # Backend selection changes storage/implementation, not lifecycle
        # decisions, so a checkpoint can move between CPU reference and CUDA.
        self.settings.gpu_topology = gpu_topology
        self.schedule = CoarseToFineConfig(**state.pop("schedule"))
        for name, value in state.items():
            setattr(self, name, value)
        self.budgets = {int(key): value for key, value in self.budgets.items()}
        self.cooldown = {int(key): value for key, value in self.cooldown.items()}
        if self.pending_growth is not None:
            self.pending_growth["seen_views"] = set(self.pending_growth["seen_views"])
        if self.pending_rgb_deletion is not None:
            self.pending_rgb_deletion["seen_views"] = set(self.pending_rgb_deletion["seen_views"])
        self._validate()
        if self.pending_growth is not None and self.pending_growth.get("mode") == "rgb":
            self._rgb_normalization(self.pending_growth)
        streak = self.np.asarray(self.rgb_deletion_streak)
        if (streak.shape != (self.count,) or
                not self.np.issubdtype(streak.dtype, self.np.integer) or
                self.np.any(streak < 0)):
            raise ValueError("RGB deletion confirmation dimensions or values differ")
        if (self.rgb_deletion_last_boundary is not None and
                (not isinstance(self.rgb_deletion_last_boundary, int) or
                 isinstance(self.rgb_deletion_last_boundary, bool) or
                 self.rgb_deletion_last_boundary < 1)):
            raise ValueError("RGB deletion previous boundary must be a positive integer")
        opacity_streak = self.np.asarray(self.low_opacity_deletion_streak)
        if (opacity_streak.shape != (self.count,) or
                not self.np.issubdtype(opacity_streak.dtype, self.np.integer) or
                self.np.any(opacity_streak < 0) or
                self.np.any(opacity_streak > self.settings.deletion_opacity_confirmation_windows)):
            raise ValueError("Low-opacity deletion confirmation dimensions or values differ")
        self.low_opacity_deletion_streak = opacity_streak.astype(self.np.uint32, copy=True)
        last = self.low_opacity_deletion_last_boundary
        if last is not None:
            local_epoch = last - self.level_start_epoch if isinstance(last, int) else -1
            if (not isinstance(last, int) or isinstance(last, bool) or
                    not 1 <= last <= self.schedule.total_epochs or
                    local_epoch <= self.settings.deletion_warmup_epochs or
                    (local_epoch - self.settings.deletion_warmup_epochs) % self.settings.deletion_interval or
                    self.deletion_frozen(last) or
                    (self.schedule.enabled and self.core_resolution < self.schedule.target_resolution and
                     local_epoch >= self.level_budget)):
                raise ValueError("Low-opacity deletion previous boundary is not a scheduled window")
        elif self.np.any(self.low_opacity_deletion_streak):
            raise ValueError("Low-opacity deletion confirmation requires a previous boundary")
        if self.pending_rgb_deletion is not None:
            from reconstruction_deletion_rgb import validate_pending
            validate_pending(self.pending_rgb_deletion, self.count, self.resolution,
                             self.view_count, self.np)
