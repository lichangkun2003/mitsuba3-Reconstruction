"""Online RGB growth votes from the native training trace, without rerendering.

Input is reduced across all chunks/ranks of ONE training view before observing.
Only epoch boundaries/checkpoints materialize device statistics on the host.
"""

from __future__ import annotations


RGB_GROWTH_EVIDENCE_VERSION = 2

RGB_STAT_FIELDS = ("gain_sum", "raw_gain_sum", "support_views", "conflict_views",
                   "observed_views", "unknown_views", "invalid_views",
                   "observed_rays", "unknown_rays")


def upload_rgb_growth_candidates(model, state):
    """Upload portable checkpoint/broadcast arrays to the current native variant."""
    if type(model).__module__.split(".")[0] == "mitsuba":
        import mitsuba as mi
        state = {name: (mi.UInt32(value) if name == "cell_indices" else mi.Float(value))
                 for name, value in state.items()}
    model.set_growth_candidates(state)


def initialize_statistics(pending, np):
    count = len(pending["cells"])
    for name in RGB_STAT_FIELDS:
        pending[name] = np.zeros(count, dtype=np.float32 if name.endswith("gain_sum") else np.uint32)


def observe_numpy(pending, out, pixel_count, epsilon, np, normalization="full_image"):
    """CPU reference/fallback matching device accumulation and abstention."""
    count = len(pending["cells"])
    gain = np.asarray(out["growth_gain"], dtype=np.float32).reshape(-1)
    observed = np.asarray(out["growth_observed"]).reshape(-1)
    unknown = np.asarray(out["growth_unknown"]).reshape(-1)
    if any(len(value) != count for value in (gain, observed, unknown)):
        raise ValueError("RGB growth evidence count differs from frozen candidate count")
    if np.any(observed < 0) or np.any(unknown < 0):
        raise ValueError("RGB growth ray counts must be nonnegative")
    if normalization not in ("observed", "full_image"):
        raise ValueError("RGB growth normalization must be 'observed' or 'full_image'")
    finite = np.isfinite(gain)
    # Native counts include every reliable candidate hit, including negative
    # and zero gains. An unseen candidate has no local mean and cannot vote.
    denominator = (np.maximum(observed, 1).astype(np.float32) if normalization == "observed"
                   else np.float32(pixel_count))
    normalized = np.where(finite, gain, 0) / denominator
    # Unknown rays abstain individually, without discarding reliable signed
    # evidence from other rays in the same candidate/view. unknown_views is
    # an overlapping diagnostic, not a count of discarded view votes.
    eligible = finite & (observed > 0)
    pending["raw_gain_sum"] += normalized
    pending["gain_sum"] += np.where(eligible, normalized, 0)
    pending["support_views"] += (eligible & (normalized > epsilon)).astype(np.uint32)
    pending["conflict_views"] += (eligible & (normalized < -epsilon)).astype(np.uint32)
    pending["observed_views"] += eligible.astype(np.uint32)
    pending["unknown_views"] += (unknown > 0).astype(np.uint32)
    pending["invalid_views"] += (~finite).astype(np.uint32)
    pending["observed_rays"] += observed.astype(np.uint32)
    pending["unknown_rays"] += unknown.astype(np.uint32)


class RGBGrowthEvidenceDevice:
    def __init__(self, pending, float_type, uint_type, dr, np):
        self.pending, self.dr, self.np = pending, dr, np
        self.Float, self.UInt = float_type, uint_type
        self.count = len(pending["cells"])
        self.arrays = {name: (float_type if name.endswith("gain_sum") else uint_type)(pending[name])
                       for name in RGB_STAT_FIELDS}
        dr.eval(*self.arrays.values())

    def observe(self, out, pixel_count, epsilon, normalization="full_image"):
        dr, Float, UInt = self.dr, self.Float, self.UInt
        gain, observed, unknown = (out[name] for name in
                                   ("growth_gain", "growth_observed", "growth_unknown"))
        if any(dr.width(value) != self.count for value in (gain, observed, unknown)):
            raise ValueError("RGB growth evidence count differs from frozen candidate count")
        if normalization not in ("observed", "full_image"):
            raise ValueError("RGB growth normalization must be 'observed' or 'full_image'")
        finite = dr.isfinite(gain)
        safe_gain = dr.select(finite, gain, 0)
        normalized = (safe_gain / Float(dr.maximum(observed, 1)) if normalization == "observed"
                      else safe_gain * dr.opaque(Float, 1. / pixel_count))
        eligible = finite & (observed > 0)
        threshold = dr.opaque(Float, epsilon)
        increments = {
            "raw_gain_sum": normalized,
            "gain_sum": dr.select(eligible, normalized, 0),
            "support_views": UInt(eligible & (normalized > threshold)),
            "conflict_views": UInt(eligible & (normalized < -threshold)),
            "observed_views": UInt(eligible), "unknown_views": UInt(unknown > 0),
            "invalid_views": UInt(~finite), "observed_rays": observed, "unknown_rays": unknown,
        }
        self.arrays = {name: self.arrays[name] + increments[name] for name in RGB_STAT_FIELDS}
        # Launch only; do not synchronize or read any per-view statistic back.
        dr.eval(*self.arrays.values())

    def materialize(self):
        result = {name: self.np.array(value, copy=True) for name, value in self.arrays.items()}
        self.pending.update(result)
        return result
