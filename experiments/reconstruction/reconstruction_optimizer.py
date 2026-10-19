"""GPU-resident Falcor-compatible parameter updates for voxel reconstruction.

Gradients are per-view sums over pixels, already carrying sampling-PDF and
geometry-proxy corrections. Only backward-sample averaging is applied here.
Rotation gradients/moments are local tangent vectors; parameters remain wxyz.
"""

from __future__ import annotations

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

    def __init__(self, model, config: OptimizerConfig | None = None):
        import drjit as dr
        import mitsuba as mi
        self.dr, self.mi = dr, mi
        self.config = config or OptimizerConfig()
        self.config.validate()
        self.reset(model)

    def reset(self, model):
        self.count = int(model.active_voxel_count)
        self.dimensions = {
            "radiance_sh": 3 * int(model.sh_count), "opacity_sh": int(model.opacity_sh_count),
            "centers": 3, "log_scales": 3, "rotation_tangent": 3,
        }
        self.state = {
            key: {"mean": self.dr.zeros(self.mi.Float, self.count * size),
                  "squared_mean": self.dr.zeros(self.mi.Float, self.count * size),
                  "steps": self.dr.zeros(self.mi.UInt32, self.count)}
            for key, size in self.dimensions.items()
        }

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
        index = self.dr.arange(self.mi.UInt32, self.count * size) // size
        return self.dr.gather(type(value), value, index)

    def _adam(self, name, gradient, valid, lr, lr_scale=1.):
        dr, mi, cfg = self.dr, self.mi, self.config
        size = self.dimensions[name]
        if dr.width(gradient) != self.count * size:
            raise ValueError(f"Gradient {name} has {dr.width(gradient)} entries; expected {self.count * size}")
        state = self.state[name]
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
        index = dr.arange(mi.UInt32, self.count)
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
        result = dr.zeros(mi.Float, self.count * 4)
        for axis in range(4):
            dr.scatter(result, dr.select(valid, components[axis], q[axis]), index * 4 + axis)
        return result

    def step(self, model, out: dict, *, sample_count: int = 1, epoch: int = 0,
             coarse_to_fine: bool = True) -> dict:
        dr, mi, cfg = self.dr, self.mi, self.config
        if int(model.active_voxel_count) != self.count:
            raise ValueError("Topology changed: call optimizer.remap() or reset() before step")
        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        if self.count == 0:
            return {"effective_opacity_lr": self.effective_opacity_lr(epoch, coarse_to_fine),
                    "backward_samples": sample_count}
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
            valid = (geometry if is_geometry else appearance) > 0
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
            dr.eval(*parameters.values(), *[value for state in self.state.values() for value in state.values()])
            model.set_parameters(parameters)
        return {"effective_opacity_lr": rates["opacity_sh"], "backward_samples": sample_count}

    def remap(self, survivors, new_count: int, *, reset_radiance: bool = False):
        """Preserve surviving moments, append zero newborn state; all indices are old IDs."""
        dr, mi = self.dr, self.mi
        source = mi.UInt32(survivors)
        kept = dr.width(source)
        if new_count < kept:
            raise ValueError("new_count cannot be smaller than survivor count")
        if kept == 0:
            for name, state in self.state.items():
                state["mean"] = dr.zeros(mi.Float, new_count * self.dimensions[name])
                state["squared_mean"] = dr.zeros(mi.Float, new_count * self.dimensions[name])
                state["steps"] = dr.zeros(mi.UInt32, new_count)
            self.count = new_count
            return
        destination = dr.arange(mi.UInt32, new_count)
        valid = destination < kept
        source_ids = dr.gather(mi.UInt32, source, destination, valid)
        for name, state in self.state.items():
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
        dr.eval(*[value for state in self.state.values() for value in state.values()])

    def state_dict(self) -> dict:
        import numpy as np
        self.dr.eval(*[value for state in self.state.values() for value in state.values()])
        self.dr.sync_thread()
        return {"config": asdict(self.config), "count": self.count, "dimensions": dict(self.dimensions),
                "groups": {name: {key: np.array(value, copy=True) for key, value in state.items()}
                           for name, state in self.state.items()}}

    def load_state_dict(self, data: dict):
        if int(data["count"]) != self.count or data["dimensions"] != self.dimensions:
            raise ValueError("Checkpoint optimizer dimensions do not match model")
        self.config = OptimizerConfig(**data["config"])
        self.config.validate()
        for name, state in self.state.items():
            for key in state:
                expected = self.count if key == "steps" else self.count * self.dimensions[name]
                value = data["groups"][name][key]
                if len(value) != expected:
                    raise ValueError(f"Invalid optimizer state size: {name}.{key}")
                state[key] = self.mi.UInt32(value) if key == "steps" else self.mi.Float(value)
