"""Bounded replay of the native CUDA loss, with explicit topology invalidation.

Only GPU work is recorded. The model and all buffers are PyTree arguments, so
parameter replacements made by Adam are inputs on the next call, not captures.
Synchronized phase diagnostics deliberately use the original implementation.
"""

from __future__ import annotations

from collections import OrderedDict
from preview import native_traversal_settings


class NativeLossReplay:
    """Cache native executions without changing per-view optimization semantics.

    Call ``clear()`` when rebuilding topology, before discarding the old model.
    A changed model object also invalidates recordings, even when its voxel count
    is unchanged. At most ``max_entries * 2`` Dr.Jit recordings remain cached;
    dimensions/options have their own bounded LRU, and Dr.Jit handles additional
    input layout or execution-flag changes inside each entry.
    """

    def __init__(self, dr, enabled=True, max_entries=8):
        if not isinstance(enabled, bool):
            raise ValueError("native_replay must be a boolean")
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("Native replay max_entries must be a positive integer")
        if enabled and not hasattr(dr, "freeze"):
            raise RuntimeError("Native replay requires a source Dr.Jit build with dr.freeze")
        self.dr, self.enabled, self.max_entries = dr, enabled, max_entries
        self._entries = OrderedDict()
        self._model = None
        self._topology_revision = None
        self._calls = self._recordings = self._replays = self._bypasses = 0
        self._clears = self._model_changes = self._evictions = 0
        self._empty_model_bypasses = 0
        self._empty_growth_bypasses = 0
        self._topology_changes = 0
        self._gradient_count_readbacks = self._gradient_overflow_retries = 0
        self._gradient_capacity_hint = 0
        self._gradient_capacity_revision = None
        # The training driver may reject an overflow expansion that exceeds
        # its memory budget. Keep this policy across recording invalidations.
        self.gradient_capacity_guard = None

        # The only closure capture is the Dr.Jit module. Never capture a model,
        # options dictionary, camera, reference, or previous result here.
        def native(model, inputs, key):
            result = model.loss_and_grad(*inputs, dict(key[-1]))
            # Explicitly include nested gradient leaves; no host reads or sync.
            dr.eval(*result["gradients"].values(),
                    *(value for name, value in result.items() if name != "gradients"))
            return result

        self._native = native

    def clear(self):
        """Release recordings and their retained buffers, preserving counters."""
        self._entries.clear()
        self._model = None
        self._topology_revision = None
        self._gradient_capacity_hint = 0
        self._gradient_capacity_revision = None
        self._clears += 1

    def __call__(self, model, ray, rgb, alpha, background, inverse_pdf, options):
        options = native_traversal_settings(model, options)
        self._calls += 1
        # Keep no obsolete topology alive, including during diagnostic calls.
        if self._model is not None and self._model is not model:
            self.clear()
            self._model_changes += 1
        revision = int(getattr(model, "topology_revision", 0))
        if self._model is model and self._topology_revision != revision:
            # Same object/count can still have different candidate cells and
            # occupancy buffers. Never replay a capture of the obsolete tree.
            self.clear()
            self._topology_changes += 1
        if not self.enabled or bool(options.get("profile_stages", False)):
            self._bypasses += 1
            return model.loss_and_grad(ray, rgb, alpha, background, inverse_pdf, options)
        if self.dr.backend_v(type(rgb)) != self.dr.JitBackend.CUDA:
            raise ValueError("Native replay requires the cuda_ad_rgb CUDA backend")
        if int(model.active_voxel_count) == 0:
            # This Dr.Jit revision cannot record a model containing zero-length
            # traversed buffers (jit_var(r0): unknown variable). Empty topology
            # already has a trivial native fast path. Choose it explicitly,
            # before recording, rather than trying to catch a backend failure.
            self._bypasses += 1
            self._empty_model_bypasses += 1
            return model.loss_and_grad(ray, rgb, alpha, background, inverse_pdf, options)
        if bool(options.get("collect_growth_evidence", False)) and not int(model.growth_candidate_count):
            # Explicit requests with no proposals return zero-length evidence
            # arrays, which this Dr.Jit revision cannot record. The training
            # driver only enables collection for nonempty proposal batches.
            self._bypasses += 1
            self._empty_growth_bypasses += 1
            return model.loss_and_grad(ray, rgb, alpha, background, inverse_pdf, options)
        self._model = model
        self._topology_revision = revision
        # Every native scalar and literal buffer stride is included. Exact input
        # widths matter: native code embeds ray_count in hit/candidate offsets.
        key = (
            revision,
            getattr(model, "scene_mapping", "bounded"),
            int(getattr(model, "core_resolution", model.resolution)),
            int(model.resolution), int(model.active_voxel_count),
            int(getattr(model, "growth_candidate_count", 0)),
            int(model.sh_count), int(model.opacity_sh_count),
            tuple(float(x) for x in model.grid_min),
            tuple(float(x) for x in model.grid_max),
            tuple(float(x) for x in model.voxel_size),
            tuple(self.dr.width(x) for x in
                  (ray.o, ray.d, ray.maxt, rgb, alpha, background, inverse_pdf)),
            tuple(sorted((name, float(value)) for name, value in options.items())),
        )
        frozen = self._entries.get(key)
        if frozen is None:
            if len(self._entries) == self.max_entries:
                self._entries.popitem(last=False)
                self._evictions += 1
            frozen = self.dr.freeze(self._native, limit=2, warn_after=8,
                                    auto_opaque=False)
            self._entries[key] = frozen
        else:
            self._entries.move_to_end(key)
        before = frozen.n_recordings
        try:
            result = frozen(model, (ray, rgb, alpha, background, inverse_pdf), key)
        finally:
            # A failed recording also advances Dr.Jit's counter. Preserve the
            # real diagnostic count while letting the original error propagate.
            recorded = frozen.n_recordings - before
            self._recordings += recorded
        self._replays += int(recorded == 0)
        return result

    def statistics(self):
        """JSON-compatible counts; recordings come from Dr.Jit's real counter."""
        return {"enabled": self.enabled, "calls": self._calls,
                "recordings": self._recordings, "replays": self._replays,
                "bypasses": self._bypasses, "entries": len(self._entries),
                "empty_model_bypasses": self._empty_model_bypasses,
                "empty_growth_bypasses": self._empty_growth_bypasses,
                "cached_recordings": sum(f.n_cached_recordings for f in self._entries.values()),
                "max_entries": self.max_entries, "clears": self._clears,
                "gradient_count_readbacks": self._gradient_count_readbacks,
                "gradient_overflow_retries": self._gradient_overflow_retries,
                "gradient_capacity_rows": self._gradient_capacity_hint,
                "model_changes": self._model_changes, "topology_changes": self._topology_changes,
                "evictions": self._evictions}
