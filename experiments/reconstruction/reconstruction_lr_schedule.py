"""Pure global-epoch learning rates for a frozen-topology optimization tail.

The native pack contains four radiance slots, opacity degrees 0..2,
center, log-scale and rotation. Legacy radiance uses degrees 0..3; octahedral
radiance uses base/map in slots 0/1 and leaves 2/3 zero. Values are runtime
inputs to replay, independent of storage capacity and Adam clocks.
"""

from __future__ import annotations

import math

from reconstruction_sh_schedule import (active_opacity_sh_count_for_epoch,
                                        active_sh_count_for_epoch,
                                        radiance_model_layout,
                                        validate_sh_schedule)


RATE_NAMES = ("radiance_degree_0", "radiance_degree_1", "radiance_degree_2",
              "radiance_degree_3", "opacity_degree_0", "opacity_degree_1",
              "opacity_degree_2", "center", "log_scale", "rotation")


def validate_learning_rate_schedule(config):
    config.optimizer.validate()
    total = config.training.num_epochs
    if type(total) is not int or total < 1:
        raise ValueError("training.num_epochs must be a positive integer")
    fraction = config.topology.freeze_last_fraction
    if (type(fraction) not in (int, float) or not math.isfinite(fraction)
            or not 0 <= fraction < 1):
        raise ValueError("topology.freeze_last_fraction must be in [0, 1)")
    validate_sh_schedule(config)
    if config.optimizer.lr_schedule_enabled and not 1 <= config.model.opacity_sh_count <= 9:
        raise ValueError("Learning-rate packs support up to 9 opacity coefficients")


def topology_freeze_start_epoch(config):
    """First frozen zero-based epoch, matching topology_frozen(epoch + 1).

    TopologyManager uses boundary > floor(total * (1 - fraction)). A zero
    fraction has no frozen tail, represented here by an index past training.
    """
    total, fraction = config.training.num_epochs, config.topology.freeze_last_fraction
    return math.floor(total * (1. - fraction)) if fraction > 0 else total


def _activation_epoch(config, degree, *, opacity):
    stored = config.model.opacity_sh_count if opacity else config.model.sh_count
    if not opacity and radiance_model_layout(config.model)[0] == "octahedral_scale" and degree > 1:
        return None  # Slots 2/3 stay unused; slot 1 controls every map texel.
    if degree * degree >= stored:
        return None
    active = active_opacity_sh_count_for_epoch if opacity else active_sh_count_for_epoch
    # Read the existing SH rule rather than constructing a second activation
    # policy. Monotonic counts permit a logarithmic search of the global epochs.
    low, high = 0, config.training.num_epochs
    while low < high:
        middle = (low + high) // 2
        if active(config, middle) > degree * degree:
            high = middle
        else:
            low = middle + 1
    return low if low < config.training.num_epochs else None


def _exponential_factor(epoch, start, last, final):
    if start > last or epoch < start:
        return 1.
    # A one-epoch tail cannot include distinct initial and final rates. Prefer
    # the stated terminal rate; longer tails begin exactly at factor one.
    if start == last:
        return float(final)
    progress = min(1., max(0., (epoch - start) / (last - start)))
    return float(final) ** progress


def _band_schedule(config, epoch, degree, freeze_start, base_factor, *, opacity):
    cfg = config.optimizer
    total, last = config.training.num_epochs, config.training.num_epochs - 1
    activation = _activation_epoch(config, degree, opacity=opacity)
    base = cfg.lr_opacity if opacity else cfg.lr_radiance
    high = cfg.lr_opacity_sh if opacity else cfg.lr_radiance_sh
    peak = base if degree == 0 or high is None else high
    age = None if activation is None or epoch < activation else epoch - activation
    warmup = 0
    decay_start = freeze_start
    if degree > 0 and activation is not None:
        warmup = cfg.lr_sh_warmup_epochs
        if freeze_start < total:
            # Very short verification runs still reach the same terminal rate.
            warmup = min(warmup, max(total - activation - 1, 0))
        decay_start = max(freeze_start, activation + max(warmup - 1, 0))
    if age is None:
        factor = 0.
    elif degree == 0:
        factor = base_factor
    elif warmup > 0 and age < warmup:
        factor = (age + 1) / warmup
    else:
        factor = _exponential_factor(epoch, decay_start, last, cfg.lr_final_factor)
    return dict(degree=degree, activation_epoch=activation, age=age,
                warmup_epochs=warmup, decay_start_epoch=decay_start,
                peak_rate=float(peak), factor=float(factor), rate=float(peak * factor))


def learning_rate_schedule(config, epoch):
    """Return effective rates and explainable metadata at a zero-based epoch.

    Recomputing this function from a restored configuration and global epoch
    reproduces the same rates; there is no mutable scheduler checkpoint state.
    """
    validate_learning_rate_schedule(config)
    if type(epoch) is not int or not 0 <= epoch < config.training.num_epochs:
        raise ValueError("Learning-rate epoch must be within the training budget")
    cfg = config.optimizer
    freeze_start = topology_freeze_start_epoch(config)
    result = dict(enabled=cfg.lr_schedule_enabled, epoch=epoch,
                  freeze_start_epoch=freeze_start, rates=None)
    if not cfg.lr_schedule_enabled:
        return result
    base_factor = _exponential_factor(epoch, freeze_start, config.training.num_epochs - 1,
                                      cfg.lr_final_factor)
    radiance = [_band_schedule(config, epoch, degree, freeze_start, base_factor, opacity=False)
                for degree in range(4)]
    opacity = [_band_schedule(config, epoch, degree, freeze_start, base_factor, opacity=True)
               for degree in range(3)]
    rates = [band["rate"] for band in radiance + opacity]
    rates += [float(value * base_factor)
              for value in (cfg.lr_center, cfg.lr_log_scale, cfg.lr_rotation)]
    result.update(rates=rates, base_factor=base_factor,
                  radiance_bands=radiance, opacity_bands=opacity)
    if radiance_model_layout(config.model)[0] == "octahedral_scale":
        result["radiance_representation"] = "octahedral_scale"
        for band, name in zip(radiance, ("base_rgb", "direction_log_scale", "unused", "unused")):
            band["parameter_group"] = name
    return result


def learning_rate_schedule_protocol(config):
    validate_learning_rate_schedule(config)
    cfg = config.optimizer
    freeze_start = topology_freeze_start_epoch(config)
    octahedral = radiance_model_layout(config.model)[0] == "octahedral_scale"
    names = (("radiance_base_rgb", "radiance_direction_log_scale", "unused", "unused") + RATE_NAMES[4:]
             if octahedral else RATE_NAMES)
    return dict(enabled=cfg.lr_schedule_enabled, version=2 if octahedral else 1,
                rate_names=list(names), progress="zero-based global training epoch",
                total_epochs=config.training.num_epochs, freeze_start_epoch=freeze_start,
                final_factor=cfg.lr_final_factor, sh_warmup_epochs=cfg.lr_sh_warmup_epochs,
                radiance_high_order_peak=(cfg.lr_radiance if cfg.lr_radiance_sh is None
                                         else cfg.lr_radiance_sh),
                opacity_high_order_peak=(cfg.lr_opacity if cfg.lr_opacity_sh is None
                                        else cfg.lr_opacity_sh),
                base_policy="constant before freeze; exponential in frozen tail",
                high_order_policy="local band warmup; independent exponential after warmup and freeze",
                without_frozen_tail="base constant; high orders warm up then remain at peak",
                short_tail_policy="shorten warmup to reserve terminal epoch; one-epoch tail uses final rate",
                radiance_activation_epochs=[_activation_epoch(config, degree, opacity=False)
                                            for degree in range(4)],
                opacity_activation_epochs=[_activation_epoch(config, degree, opacity=True)
                                           for degree in range(3)])
