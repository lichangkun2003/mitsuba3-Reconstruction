"""GPU-resident Falcor-compatible parameter updates for voxel reconstruction.

Gradients are per-view sums over pixels, already carrying sampling-PDF and
geometry-proxy corrections. Only backward-sample averaging is applied here.
Rotation gradients/moments are local tangent vectors; parameters remain wxyz.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
import math


@dataclass
class OptimizerConfig:
    lr_radiance: float = 1e-3
    lr_opacity: float = 5e-4
    lr_center: float = 1e-3
    lr_log_scale: float = 2e-3
    lr_rotation: float = 5e-4
    beta1: float = .9
    beta2: float = .999
    epsilon: float = 1e-8
    grad_scale: float = 1.
    gradient_clip: float = 10.
    geometry_direction_guard: bool = True
    background_carve_multiplier: float = 5.
    opacity_warmup_epochs: int = 20
    opacity_ramp_epochs: int = 30
    radiance_step_limit: float = .05
    opacity_step_limit: float = .01
    center_step_limit: float = .02
    log_scale_step_limit: float = .02
    rotation_step_limit: float = .02
    max_scale_voxels: float = 4.

    def validate(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool):
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"optimizer.{name} must be finite and nonnegative")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1 or self.epsilon <= 0:
            raise ValueError("Adam requires beta in [0,1) and epsilon > 0")
        if self.max_scale_voxels <= 0 or self.gradient_clip <= 0:
            raise ValueError("Scale and gradient bounds must be positive")
        for value in (self.opacity_warmup_epochs, self.opacity_ramp_epochs):
            if not isinstance(value, int):
                raise ValueError("Opacity schedule lengths must be integers")


class FalcorOptimizer:
    """Five groups, independent step counts for every voxel and parameter group.

    Radiance moments are allocated for every active voxel; this avoids Falcor's
    D3D page-exhaustion fallback to SGD while retaining the ordinary Adam rule.
    """

    def __init__(self, model, config: OptimizerConfig | None = None, *, sparse_updates=True,
                 replay_updates=True):
        import drjit as dr
        import mitsuba as mi
        self.dr, self.mi = dr, mi
        self.config = config or OptimizerConfig()
        self.config.validate()
        # Dense execution is retained as a numerical/performance reference.
        # It does not alter the optimizer/checkpoint configuration or algorithm.
        self.sparse_updates = sparse_updates
        self.replay_updates = replay_updates
        self._update_cache = OrderedDict()
        self._update_model = self._update_key = None
        self._update_calls = self._update_recordings = self._update_replays = 0
        self._update_bypasses = self._update_clears = 0
        self._update_evictions = 0
        # All mutable buffers are explicit PyTree arguments. The closure must
        # not retain a model, its snapshot, or old optimizer state.
        def native_update(model, gradients, appearance, geometry, carving,
                          means, squares, steps, settings, allowed, check_moments):
            accepted = model.optimizer_step_device(gradients, appearance, geometry, carving,
                means, squares, steps, settings, allowed, check_moments)
            dr.eval(accepted, *means.values(), *squares.values(), *steps.values())
            return accepted
        self._native_update = native_update
        self.reset(model)

    def clear_replay(self):
        """Release recorded buffers before replacing topology or optimizer state."""
        self._update_cache.clear()
        self._update_model = self._update_key = None
        self._update_clears += 1

    def statistics(self):
        return {"execution": "device_sparse" if self.sparse_updates else "dense_reference",
                "replay_enabled": self.replay_updates, "calls": self._update_calls,
                "recordings": self._update_recordings, "replays": self._update_replays,
                "bypasses": self._update_bypasses, "clears": self._update_clears,
                "cached_batch_variants": len(self._update_cache),
                "batch_variant_evictions": self._update_evictions,
                "host_length_readbacks": 0, "capacity_rows": self.count}

    @property
    def state(self):
        """Legacy writable state access enables conservative finite validation.

        A caller can retain and mutate this dictionary/its arrays at any later
        time, even without changing a Dr.Jit index. Keep checking full moments
        until reset/load replaces the storage. state_dict() returns independent
        host copies and does not disable the usual compact validation path.
        """
        self._state_exposed = True
        return self._state

    @state.setter
    def state(self, value):
        self._state = value
        self._state_exposed = True

    def reset(self, model):
        self.clear_replay()
        self.update_healthy = self.mi.Bool(True)
        self.count = int(model.active_voxel_count)
        self.dimensions = {
            "radiance_sh": 3 * int(model.sh_count), "opacity_sh": int(model.opacity_sh_count),
            "centers": 3, "log_scales": 3, "rotation_tangent": 3,
        }
        self._state = {
            key: {"mean": self.dr.zeros(self.mi.Float, self.count * size),
                  "squared_mean": self.dr.zeros(self.mi.Float, self.count * size),
                  "steps": self.dr.zeros(self.mi.UInt32, self.count)}
            for key, size in self.dimensions.items()
        }
        self._state_exposed = False
        self._remember_state()

    def _state_signature(self):
        # Do not retain array references here: that would force copy-on-write
        # of the complete moment table on every sparse scatter.
        return tuple((id(value), value.index) for state in self._state.values()
                     for value in state.values())

    def _remember_state(self):
        self._known_state = self._state_signature()

    def _check_external_state(self):
        """Validate changed state, or state that a caller can still mutate."""
        dr, mi = self.dr, self.mi
        valid = mi.Bool(True)
        if self._state_exposed or self._state_signature() != self._known_state:
            for name, state in self._state.items():
                for key, value in state.items():
                    expected = self.count if key == "steps" else self.count * self.dimensions[name]
                    if dr.width(value) != expected:
                        raise ValueError(f"Invalid optimizer state size: {name}.{key}")
                    if key != "steps" and expected:
                        valid &= dr.all(dr.isfinite(value))
        return valid

    def effective_opacity_lr(self, epoch: int, coarse_to_fine: bool) -> float:
        cfg = self.config
        if cfg.lr_opacity <= 0 or coarse_to_fine:
            return cfg.lr_opacity
        if epoch < cfg.opacity_warmup_epochs:
            return 0.
        if cfg.opacity_ramp_epochs == 0:
            return cfg.lr_opacity
        return cfg.lr_opacity * min(1., (epoch - cfg.opacity_warmup_epochs) / cfg.opacity_ramp_epochs)

    def _finite(self, value):
        return self.dr.select(self.dr.isfinite(value), value, 0.)

    def _expand(self, value, size: int):
        index = self.dr.arange(self.mi.UInt32, self.dr.width(value) * size) // size
        return self.dr.gather(type(value), value, index)

    def _adam(self, name, gradient, valid, lr, lr_scale=1., *, state=None, count=None):
        dr, mi, cfg = self.dr, self.mi, self.config
        size = self.dimensions[name]
        count = self.count if count is None else count
        if dr.width(gradient) != count * size:
            raise ValueError(f"Gradient {name} has {dr.width(gradient)} entries; expected {count * size}")
        state = self._state[name] if state is None else state
        active = valid & (lr > 0)
        mask = self._expand(active, size)
        original = self._finite(gradient)
        gradient = dr.clip(original, -cfg.gradient_clip, cfg.gradient_clip)
        steps = state["steps"] + mi.UInt32(active)
        mean = cfg.beta1 * state["mean"] + (1 - cfg.beta1) * gradient
        square = cfg.beta2 * state["squared_mean"] + (1 - cfg.beta2) * dr.square(gradient)
        exponent = mi.Float(self._expand(steps, size))
        bias1 = dr.maximum(1 - dr.power(cfg.beta1, exponent), 1e-8)
        bias2 = dr.maximum(1 - dr.power(cfg.beta2, exponent), 1e-8)
        step = lr * (mean / bias1) / (dr.sqrt(dr.maximum(square / bias2, 0.)) + cfg.epsilon)
        step = self._finite(step)
        if name in ("centers", "log_scales", "rotation_tangent") and cfg.geometry_direction_guard:
            blocked = (original == 0) | (mean * original < 0)
            mean = dr.select(blocked, 0., mean)
            step = dr.select(blocked, 0., step)
        state["mean"] = dr.select(mask, mean, state["mean"])
        state["squared_mean"] = dr.select(mask, square, state["squared_mean"])
        state["steps"] = steps
        scale = self._expand(lr_scale, size) if dr.is_array_v(type(lr_scale)) else lr_scale
        return dr.select(mask, step * scale, 0.), mask

    def _update_rotation(self, rotation, step, valid):
        dr, mi = self.dr, self.mi
        count = dr.width(rotation) // 4
        index = dr.arange(mi.UInt32, count)
        delta = mi.Vector3f(*[-dr.gather(mi.Float, step, index * 3 + axis) for axis in range(3)])
        length = dr.norm(delta)
        factor = dr.minimum(1., self.config.rotation_step_limit / dr.maximum(length, 1e-30))
        delta *= factor
        length *= factor
        scale = dr.select(length > 1e-8, dr.sin(length * .5) / dr.maximum(length, 1e-30), 0.)
        vector = delta * scale
        scalar = dr.select(length > 1e-8, dr.cos(length * .5), 1.)
        q = [dr.gather(mi.Float, rotation, index * 4 + axis) for axis in range(4)]
        old_v = mi.Vector3f(q[1], q[2], q[3])
        new_w = q[0] * scalar - dr.dot(old_v, vector)
        new_v = q[0] * vector + scalar * old_v + dr.cross(old_v, vector)
        norm_squared = new_w * new_w + dr.dot(new_v, new_v)
        safe = dr.isfinite(norm_squared) & (norm_squared > 1e-20)
        inv_norm = dr.rsqrt(dr.select(safe, norm_squared, 1.))
        components = [dr.select(safe, new_w * inv_norm, 1.)]
        components += [dr.select(safe, new_v[axis] * inv_norm, 0.) for axis in range(3)]
        result = dr.zeros(mi.Float, count * 4)
        for axis in range(4):
            dr.scatter(result, dr.select(valid, components[axis], q[axis]), index * 4 + axis)
        return result

    def step(self, model, out: dict, *, sample_count: int = 1, epoch: int = 0,
             coarse_to_fine: bool = True, update_allowed=None) -> dict:
        implementation = self._step_sparse if self.sparse_updates else self._step_dense
        return implementation(model, out, sample_count=sample_count, epoch=epoch,
                              coarse_to_fine=coarse_to_fine, update_allowed=update_allowed)

    def _step_sparse(self, model, out, *, sample_count, epoch, coarse_to_fine, update_allowed):
        """Device-counted compact rows; no per-view host length readbacks."""
        dr, mi, cfg = self.dr, self.mi, self.config
        if not callable(getattr(model, "optimizer_step_device", None)):
            raise RuntimeError("Device sparse Adam requires optimizer_step_device; rebuild this source checkout")
        if int(model.active_voxel_count) != self.count:
            raise ValueError("Topology changed: call optimizer.remap() or reset() before step")
        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        if self._state_exposed:
            # Different array wrappers may share storage (the native scope
            # separates that storage). Reusing one mutable wrapper for two
            # state slots cannot represent independent Adam groups.
            owners = [id(value) for state in self._state.values() for value in state.values()]
            if len(set(owners)) != len(owners):
                raise ValueError("Optimizer state slots must use distinct array objects")
        guarded = update_allowed is not None
        allowed = self.update_healthy & update_allowed if guarded else mi.Bool(True)
        if guarded:
            allowed &= self._check_external_state()
        opacity_lr = self.effective_opacity_lr(epoch, coarse_to_fine)
        counts = tuple(mi.UInt32(out[name]) for name in
                       ("appearance_valid", "geometry_valid", "background_carve_valid"))
        if any(dr.width(value) != self.count for value in counts):
            raise ValueError("Validity counts must have one entry per voxel")
        gradients = {name: mi.Float(out["gradients"][name]) for name in self.dimensions}
        for name, value in gradients.items():
            if dr.width(value) != self.count * self.dimensions[name]:
                raise ValueError(f"Gradient {name} has the wrong size")
        self._update_calls += 1
        if self.count == 0:
            self._update_bypasses += 1
            if guarded:
                self.update_healthy = allowed
                dr.eval(allowed)
            return {"effective_opacity_lr": opacity_lr, "backward_samples": sample_count,
                    "accepted": self.update_healthy}
        settings = {name: float(value) for name, value in asdict(cfg).items()}
        settings.update(lr_opacity=float(opacity_lr), gradient_factor=cfg.grad_scale / sample_count)
        means = {name: state["mean"] for name, state in self._state.items()}
        squares = {name: state["squared_mean"] for name, state in self._state.items()}
        steps = {name: state["steps"] for name, state in self._state.items()}
        key = (self.count, tuple(self.dimensions.items()), tuple(float(v) for v in model.voxel_size),
               tuple(sorted((name, value) for name, value in settings.items()
                            if name != "gradient_factor")), guarded)
        if self._update_model is not model or self._update_key != key:
            self.clear_replay()
            self._update_model, self._update_key = model, key
        arguments = (model, gradients, *counts, means, squares, steps, settings, allowed, guarded)
        # Separate genuine external aliases before recording/replaying writes,
        # then stash the unique-owner reference counts across freeze's internal
        # temporary references. Without this scope freeze records a full-table
        # copy for every parameter and moment, including on later replays.
        scope = model.optimizer_update_scope_begin(means, squares, steps)
        try:
            if self.replay_updates and dr.backend_v(mi.Float) == dr.JitBackend.CUDA:
                # Full batches and the final partial batch have distinct exact
                # gradient factors. Keep both recordings across epoch boundaries
                # instead of invalidating each other twice per epoch. All other
                # settings, topology/model changes and state replacement still
                # invalidate the complete cache. The bounded LRU also avoids
                # retaining arbitrary historical factors for external callers.
                factor = settings["gradient_factor"]
                frozen = self._update_cache.pop(factor, None)
                if frozen is None:
                    if len(self._update_cache) == 2:
                        self._update_cache.popitem(last=False)
                        self._update_evictions += 1
                    frozen = dr.freeze(self._native_update, limit=2, warn_after=8,
                                       auto_opaque=False)
                self._update_cache[factor] = frozen
                before = frozen.n_recordings
                try:
                    accepted = frozen(*arguments)
                finally:
                    recorded = frozen.n_recordings - before
                    self._update_recordings += recorded
                self._update_replays += int(recorded == 0)
            else:
                self._update_bypasses += 1
                accepted = self._native_update(*arguments)
        finally:
            mi._reconstruction_optimizer_update_scope_end(scope)
        if guarded:
            self.update_healthy = accepted
        elif not bool(accepted[0]):
            raise RuntimeError("Nonfinite or invalid sparse parameter update")
        self._remember_state()
        return {"effective_opacity_lr": opacity_lr, "backward_samples": sample_count,
                "accepted": accepted}

    def _step_compact_reference(self, model, out, *, sample_count, epoch, coarse_to_fine, update_allowed):
        """Previous host-sized sparse path, retained only for diagnostics."""
        dr, mi, cfg = self.dr, self.mi, self.config
        if not callable(getattr(model, "set_parameters_sparse", None)):
            raise RuntimeError("Sparse Adam requires set_parameters_sparse; rebuild this source checkout")
        if int(model.active_voxel_count) != self.count:
            raise ValueError("Topology changed: call optimizer.remap() or reset() before step")
        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        guarded = update_allowed is not None
        allowed = self.update_healthy & update_allowed if guarded else mi.Bool(True)
        if guarded:
            allowed &= self._check_external_state()
        opacity_lr = self.effective_opacity_lr(epoch, coarse_to_fine)
        rates = {"radiance_sh": cfg.lr_radiance, "opacity_sh": opacity_lr,
                 "centers": cfg.lr_center, "log_scales": cfg.lr_log_scale,
                 "rotation_tangent": cfg.lr_rotation}
        limits = {"radiance_sh": cfg.radiance_step_limit, "opacity_sh": cfg.opacity_step_limit,
                  "centers": cfg.center_step_limit, "log_scales": cfg.log_scale_step_limit}
        if self.count == 0:
            if guarded:
                self.update_healthy = allowed
                dr.eval(self.update_healthy)
            return {"effective_opacity_lr": opacity_lr, "backward_samples": sample_count,
                    "accepted": self.update_healthy}
        appearance = mi.UInt32(out["appearance_valid"])
        geometry = mi.UInt32(out["geometry_valid"])
        carving = mi.UInt32(out["background_carve_valid"])
        if any(dr.width(counts) != self.count for counts in (appearance, geometry, carving)):
            raise ValueError("Validity counts must have one entry per voxel")
        # Valid zero gradients must still advance Adam. Never compact by g != 0.
        # compress transfers only the list lengths, but DOES synchronize CUDA;
        # performance comparisons must include this cost and finish the queue.
        appearance_ids = dr.compress((appearance > 0) & allowed) if (
            cfg.lr_radiance > 0 or opacity_lr > 0) else mi.UInt32([])
        geometry_ids = dr.compress((geometry > 0) & allowed) if (
            cfg.lr_center > 0 or cfg.lr_log_scale > 0 or cfg.lr_rotation > 0) else mi.UInt32([])
        geometry_scale = 1.
        if dr.width(geometry_ids):
            geometry_scale = 1 + (max(cfg.background_carve_multiplier, 1.) - 1) * dr.clip(
                mi.Float(dr.gather(mi.UInt32, carving, geometry_ids)) /
                dr.maximum(mi.Float(dr.gather(mi.UInt32, geometry, geometry_ids)), 1.), 0., 1.)
        snapshot = model.snapshot()
        parameters, proposals, indices = {}, {}, {}
        for name, lr in rates.items():
            if lr <= 0:
                continue
            ids = geometry_ids if name in ("centers", "log_scales", "rotation_tangent") else appearance_ids
            count, size = dr.width(ids), self.dimensions[name]
            gradient = mi.Float(out["gradients"][name])
            if dr.width(gradient) != self.count * size:
                raise ValueError(f"Gradient {name} has {dr.width(gradient)} entries; expected {self.count * size}")
            if count == 0:
                continue
            lane = dr.arange(mi.UInt32, count * size)
            index = dr.gather(mi.UInt32, ids, lane // size) * size + lane % size
            indices[name] = index
            compact = {key: dr.gather(type(value), value, ids if key == "steps" else index)
                       for key, value in self._state[name].items()}
            gradient = dr.gather(mi.Float, gradient, index) * (cfg.grad_scale / sample_count)
            step, _ = self._adam(name, gradient, dr.full(mi.Bool, True, count), lr,
                                geometry_scale if ids is geometry_ids else 1.,
                                state=compact, count=count)
            proposals[name] = compact
            if name == "rotation_tangent":
                lane4 = dr.arange(mi.UInt32, count * 4)
                old = dr.gather(mi.Float, snapshot["rotations"],
                                dr.gather(mi.UInt32, ids, lane4 // 4) * 4 + lane4 % 4)
                parameters["rotations"] = self._update_rotation(old, step, dr.full(mi.Bool, True, count))
                continue
            old = dr.gather(mi.Float, snapshot[name], index)
            updated = old - dr.clip(step, -limits[name], limits[name])
            if name in ("radiance_sh", "opacity_sh"):
                updated = self._finite(updated)
            elif name == "centers":
                updated = dr.clip(updated, .01, .99)
            elif name == "log_scales":
                voxel_size = dr.gather(mi.Float, mi.Float(list(model.voxel_size)), lane % 3)
                lower = dr.maximum(voxel_size * 1e-4, 1e-8)
                upper = dr.maximum(voxel_size * cfg.max_scale_voxels, lower)
                updated = dr.clip(self._finite(updated), dr.log(lower), dr.log(upper))
            parameters[name] = updated
        if guarded:
            for state in proposals.values():
                allowed &= dr.all(dr.isfinite(state["mean"])) & dr.all(dr.isfinite(state["squared_mean"]))
        # Materialize every proposal before writing either parameters or state.
        # Drop internal snapshot references so sparse scatter need not copy the
        # entire parameter table. Genuine external snapshots still retain COW.
        dr.eval(allowed, *parameters.values(),
                *[value for state in proposals.values() for value in state.values()])
        del snapshot
        if proposals:
            del old
        accepted = model.set_parameters_sparse(appearance_ids, geometry_ids, parameters, allowed)
        if not guarded and not bool(accepted[0]):
            raise RuntimeError("Nonfinite or invalid sparse parameter update")
        for name, compact in proposals.items():
            ids = geometry_ids if name in ("centers", "log_scales", "rotation_tangent") else appearance_ids
            for key, value in compact.items():
                dr.scatter(self._state[name][key], value, ids if key == "steps" else indices[name], accepted)
        if guarded:
            self.update_healthy = accepted
        dr.eval(self.update_healthy, *[value for state in self._state.values() for value in state.values()])
        self._remember_state()
        return {"effective_opacity_lr": opacity_lr, "backward_samples": sample_count,
                "accepted": self.update_healthy}

    def _step_dense(self, model, out: dict, *, sample_count: int = 1, epoch: int = 0,
                    coarse_to_fine: bool = True, update_allowed=None) -> dict:
        dr, mi, cfg = self.dr, self.mi, self.config
        guarded = update_allowed is not None
        if guarded and not callable(getattr(model, "set_parameters_guarded", None)):
            raise RuntimeError("Deferred checks require set_parameters_guarded; rebuild this source checkout")
        allowed = self.update_healthy & update_allowed if guarded else mi.Bool(True)
        # _adam constructs the next state. Keep the old arrays until the whole
        # transaction, including candidate parameters, has passed device checks.
        previous_state = {name: dict(state) for name, state in self._state.items()} if guarded else None
        if int(model.active_voxel_count) != self.count:
            raise ValueError("Topology changed: call optimizer.remap() or reset() before step")
        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        if self.count == 0:
            if guarded:
                self.update_healthy = allowed
                dr.eval(self.update_healthy)
            return {"effective_opacity_lr": self.effective_opacity_lr(epoch, coarse_to_fine),
                    "backward_samples": sample_count, "accepted": self.update_healthy}
        gradients = out["gradients"]
        appearance = mi.UInt32(out["appearance_valid"])
        geometry = mi.UInt32(out["geometry_valid"])
        carving = mi.UInt32(out["background_carve_valid"])
        for counts in (appearance, geometry, carving):
            if dr.width(counts) != self.count:
                raise ValueError("Validity counts must have one entry per voxel")
        geometry_lr_scale = 1 + (max(cfg.background_carve_multiplier, 1.) - 1) * dr.clip(
            mi.Float(carving) / dr.maximum(mi.Float(geometry), 1.), 0., 1.)
        snapshot = model.snapshot()
        parameters = {}
        rates = {"radiance_sh": cfg.lr_radiance,
                 "opacity_sh": self.effective_opacity_lr(epoch, coarse_to_fine),
                 "centers": cfg.lr_center, "log_scales": cfg.lr_log_scale,
                 "rotation_tangent": cfg.lr_rotation}
        limits = {"radiance_sh": cfg.radiance_step_limit, "opacity_sh": cfg.opacity_step_limit,
                  "centers": cfg.center_step_limit, "log_scales": cfg.log_scale_step_limit}
        for name, lr in rates.items():
            if lr <= 0:
                continue
            is_geometry = name in ("centers", "log_scales", "rotation_tangent")
            valid = ((geometry if is_geometry else appearance) > 0) & allowed
            gradient = mi.Float(gradients[name]) * (cfg.grad_scale / sample_count)
            step, mask = self._adam(name, gradient, valid, lr, geometry_lr_scale if is_geometry else 1.)
            if name == "rotation_tangent":
                parameters["rotations"] = self._update_rotation(snapshot["rotations"], step, valid)
                continue
            old = snapshot[name]
            updated = old - dr.clip(step, -limits[name], limits[name])
            if name in ("radiance_sh", "opacity_sh"):
                updated = self._finite(updated)
            elif name == "centers":
                updated = dr.clip(updated, .01, .99)
            elif name == "log_scales":
                size = mi.Float(list(model.voxel_size))
                axis = dr.arange(mi.UInt32, self.count * 3) % 3
                voxel_size = dr.gather(mi.Float, size, axis)
                lower = dr.maximum(voxel_size * 1e-4, 1e-8)
                upper = dr.maximum(voxel_size * cfg.max_scale_voxels, lower)
                updated = dr.clip(self._finite(updated), dr.log(lower), dr.log(upper))
            parameters[name] = dr.select(mask, updated, old)
        if parameters:
            if guarded:
                for state in self._state.values():
                    allowed &= dr.all(dr.isfinite(state["mean"])) & dr.all(dr.isfinite(state["squared_mean"]))
                accepted = model.set_parameters_guarded(parameters, allowed)
                for name, state in self._state.items():
                    for key, value in state.items():
                        state[key] = dr.select(accepted, value, previous_state[name][key])
                self.update_healthy = accepted
                dr.eval(self.update_healthy, *parameters.values(),
                        *[value for state in self._state.values() for value in state.values()])
            else:
                dr.eval(*parameters.values(), *[value for state in self._state.values() for value in state.values()])
                model.set_parameters(parameters)
        elif guarded:
            self.update_healthy = allowed
            dr.eval(self.update_healthy)
        return {"effective_opacity_lr": rates["opacity_sh"], "backward_samples": sample_count,
                "accepted": self.update_healthy}

    def remap(self, survivors, new_count: int, *, reset_radiance: bool = False):
        """Preserve surviving moments, append zero newborn state; all indices are old IDs."""
        self.clear_replay()
        dr, mi = self.dr, self.mi
        source = mi.UInt32(survivors)
        kept = dr.width(source)
        if new_count < kept:
            raise ValueError("new_count cannot be smaller than survivor count")
        if kept == 0:
            for name, state in self._state.items():
                state["mean"] = dr.zeros(mi.Float, new_count * self.dimensions[name])
                state["squared_mean"] = dr.zeros(mi.Float, new_count * self.dimensions[name])
                state["steps"] = dr.zeros(mi.UInt32, new_count)
            self.count = new_count
            self._remember_state()
            return
        destination = dr.arange(mi.UInt32, new_count)
        valid = destination < kept
        source_ids = dr.gather(mi.UInt32, source, destination, valid)
        for name, state in self._state.items():
            width = self.dimensions[name]
            index = dr.arange(mi.UInt32, new_count * width)
            exists = index // width < kept
            old = dr.gather(mi.UInt32, source, index // width, exists) * width + index % width
            if name == "radiance_sh" and reset_radiance:
                state["mean"] = dr.zeros(mi.Float, new_count * width)
                state["squared_mean"] = dr.zeros(mi.Float, new_count * width)
                state["steps"] = dr.zeros(mi.UInt32, new_count)
            else:
                state["mean"] = dr.gather(mi.Float, state["mean"], old, exists)
                state["squared_mean"] = dr.gather(mi.Float, state["squared_mean"], old, exists)
                state["steps"] = dr.gather(mi.UInt32, state["steps"], source_ids, valid)
        self.count = new_count
        dr.eval(*[value for state in self._state.values() for value in state.values()])
        self._remember_state()

    def state_dict(self) -> dict:
        import numpy as np
        self.dr.eval(*[value for state in self._state.values() for value in state.values()])
        self.dr.sync_thread()
        return {"config": asdict(self.config), "count": self.count, "dimensions": dict(self.dimensions),
                "groups": {name: {key: np.array(value, copy=True) for key, value in state.items()}
                           for name, state in self._state.items()}}

    def device_state_dict(self) -> dict:
        """Borrow device state at a topology/replica boundary, without readback.

        The receiver must finish transferring and release these references before
        the next update. Persistent checkpoints still use state_dict().
        """
        return {"config": asdict(self.config), "count": self.count,
                "dimensions": dict(self.dimensions),
                "groups": {name: dict(state) for name, state in self._state.items()}}

    def load_state_dict(self, data: dict):
        if int(data["count"]) != self.count or data["dimensions"] != self.dimensions:
            raise ValueError("Checkpoint optimizer dimensions do not match model")
        config = OptimizerConfig(**data["config"])
        config.validate()
        restored = {}
        valid = self.mi.Bool(True)
        for name, state in self._state.items():
            restored[name] = {}
            for key in state:
                expected = self.count if key == "steps" else self.count * self.dimensions[name]
                value = data["groups"][name][key]
                if len(value) != expected:
                    raise ValueError(f"Invalid optimizer state size: {name}.{key}")
                restored[name][key] = self.mi.UInt32(value) if key == "steps" else self.mi.Float(value)
                if key != "steps":
                    valid &= self.dr.all(self.dr.isfinite(restored[name][key]))
        self.dr.eval(valid, *[value for state in restored.values() for value in state.values()])
        if not bool(valid[0]):
            raise ValueError("Checkpoint optimizer moments must be finite")
        self.config, self._state = config, restored
        self.clear_replay()
        self._state_exposed = False
        self._remember_state()
