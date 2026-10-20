"""Per-view RGB removal evidence, reduced on the GPU without extra tracing.

Native signed gains are positive when removing one existing voxel improves the
sample's RGB loss. Unknown rays abstain individually. As for RGB growth, a view
vote uses all reliable hits (including negative and zero gains); it is not a
foreground/background vote and not a guarantee about unobserved rays.
"""

from __future__ import annotations

from reconstruction_growth_rgb import (
    RGB_STAT_FIELDS, RGBGrowthEvidenceDevice, initialize_statistics,
    observe_numpy as _observe_growth_numpy)


RGB_DELETION_EVIDENCE_VERSION = 1


def initialize_pending(count, boundary, resolution, np):
    # Dense row IDs only describe the evidence layout; occupied cell addresses
    # remain owned by the live model. Topology is immutable within an epoch.
    pending = {
        "cells": np.arange(count, dtype=np.uint32), "boundary": int(boundary),
        "resolution": int(resolution), "seen_views": set(),
        "evidence_version": RGB_DELETION_EVIDENCE_VERSION,
    }
    initialize_statistics(pending, np)
    return pending


def _as_growth_statistics(out):
    """Reuse identical signed reliable-hit means without copying GPU arrays."""
    return {f"growth_{name}": out[f"deletion_{name}"]
            for name in ("gain", "observed", "unknown")}


def observe_numpy(pending, out, pixel_count, epsilon, np):
    _observe_growth_numpy(
        pending, _as_growth_statistics(out), pixel_count, epsilon, np,
        normalization="observed")


class RGBDeletionEvidenceDevice(RGBGrowthEvidenceDevice):
    def observe(self, out, pixel_count, epsilon):
        super().observe(
            _as_growth_statistics(out), pixel_count, epsilon,
            normalization="observed")


def validate_pending(pending, count, resolution, view_count, np):
    if pending.get("evidence_version") != RGB_DELETION_EVIDENCE_VERSION:
        raise ValueError("RGB deletion pending evidence version differs")
    if (pending.get("resolution") != resolution or
            len(pending.get("cells", ())) != count):
        raise ValueError("RGB deletion pending evidence dimensions differ")
    if (not isinstance(pending.get("boundary"), int) or
            pending["boundary"] < 1):
        raise ValueError("RGB deletion pending evidence requires an epoch boundary")
    if any(not isinstance(index, int) or not 0 <= index < view_count
           for index in pending["seen_views"]):
        raise ValueError("RGB deletion pending evidence has invalid view IDs")
    for name in RGB_STAT_FIELDS:
        if name not in pending or np.asarray(pending[name]).shape != (count,):
            raise ValueError(f"RGB deletion pending field {name} has invalid dimensions")
