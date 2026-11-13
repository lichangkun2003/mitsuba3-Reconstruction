"""Global-epoch radiance activation and independent opacity SH capacities."""

from __future__ import annotations


def radiance_model_layout(model):
    """Validate a config/metadata object without importing a device runtime.

    ``sh_count`` retains its buffer-stride meaning for old call sites. In the
    octahedral mode it counts one RGB base plus N*N RGB log-scale texels.
    """
    value = model.get if isinstance(model, dict) else lambda name, default=None: getattr(model, name, default)
    representation = value("radiance_representation", "sh")
    resolution = value("radiance_resolution", 5)
    stored = value("sh_count")
    if representation not in ("sh", "octahedral_scale"):
        raise ValueError("radiance_representation must be sh or octahedral_scale")
    if type(resolution) is not int or not 2 <= resolution <= 64:
        raise ValueError("radiance_resolution must be an integer in [2, 64]")
    if type(stored) is not int:
        raise ValueError("sh_count must be an integer radiance storage stride")
    if representation == "sh":
        if not 1 <= stored <= 16:
            raise ValueError("SH radiance supports 1 through 16 stored coefficients")
    elif stored != 1 + resolution * resolution:
        raise ValueError("octahedral_scale requires sh_count = 1 + radiance_resolution**2")
    return representation, resolution, stored


def validate_sh_schedule(config):
    radiance_model_layout(config.model)
    training = config.training
    if type(training.sh_degree_progression) is not bool:
        raise ValueError("training.sh_degree_progression must be a boolean")
    if type(getattr(training, "opacity_sh_degree_progression", False)) is not bool:
        raise ValueError("training.opacity_sh_degree_progression must be a boolean")
    if (getattr(training, "opacity_sh_degree_progression", False)
            and int(config.model.opacity_sh_count) > 9):
        raise ValueError("Progressive opacity SH supports at most 9 stored coefficients (degree 2)")
    weights = training.sh_degree_weights
    if (not isinstance(weights, (list, tuple)) or len(weights) != 3
            or any(type(value) is not int or value < 0 for value in weights)
            or weights[0] == 0 or weights[2] == 0):
        raise ValueError("training.sh_degree_weights must contain three integers: "
                         "positive first/last weights and a nonnegative middle weight")


def active_sh_count_for_epoch(config, completed_epochs, sh_count=None):
    """Return the active coefficient count for the next epoch (zero-based index)."""
    stored = int(config.model.sh_count if sh_count is None else sh_count)
    representation, _, capacity = radiance_model_layout(config.model)
    if representation == "octahedral_scale" and stored != capacity:
        raise ValueError("Octahedral model storage differs from configured radiance capacity")
    if not config.training.sh_degree_progression:
        return stored
    weights = config.training.sh_degree_weights
    total = config.training.num_epochs
    first = total * weights[0] // sum(weights)
    second = total * (weights[0] + weights[1]) // sum(weights)
    if representation == "octahedral_scale":
        # The middle SH band has no analogue: the entire direction map is
        # activated together at the last window, after base-only fitting.
        return 1 if completed_epochs < second else stored
    # Three time windows do not limit storage to three SH bands. The last
    # window enables every stored radiance coefficient, including degree 3.
    # A zero middle weight enables all higher bands at the same global epoch.
    count = 1 if completed_epochs < first else 4 if completed_epochs < second else stored
    return min(stored, count)


def active_opacity_sh_count_for_epoch(config, completed_epochs, sh_count=None):
    """Opacity has its own activation flag and shares the global band boundaries."""
    stored = int(config.model.opacity_sh_count if sh_count is None else sh_count)
    if not getattr(config.training, "opacity_sh_degree_progression", False):
        return stored
    weights, total = config.training.sh_degree_weights, config.training.num_epochs
    first = total * weights[0] // sum(weights)
    second = total * (weights[0] + weights[1]) // sum(weights)
    return min(stored, 1 if completed_epochs < first else 4 if completed_epochs < second else 9)


def sh_schedule_protocol(config):
    representation, resolution, stored = radiance_model_layout(config.model)
    weights = config.training.sh_degree_weights
    total = config.training.num_epochs
    result = dict(enabled=config.training.sh_degree_progression,
                weights=list(weights), progress="total completed training epochs",
                transition_completed_epochs=[total * weights[0] // sum(weights),
                    total * (weights[0] + weights[1]) // sum(weights)],
                active_coefficient_counts=([1, 1, stored] if representation == "octahedral_scale" else
                    [min(stored, count) for count in (1, 4, stored)]),
                stored_coefficients=config.model.sh_count,
                opacity_enabled=getattr(config.training, "opacity_sh_degree_progression", False),
                opacity_stored_coefficients=getattr(config.model, "opacity_sh_count", 1),
                opacity_active_coefficient_counts=[min(getattr(config.model, "opacity_sh_count", 1), count)
                                                   for count in (1, 4, 9)])
    if representation == "octahedral_scale":
        result.update(radiance_representation=representation, radiance_resolution=resolution,
                      radiance_storage="RGB base then row-major RGB log-scale texels",
                      direction_activation="all texels together at the second transition")
    return result


def apply_sh_schedule(model, config, completed_epochs):
    count = active_sh_count_for_epoch(config, completed_epochs, int(model.sh_count))
    setter = getattr(model, "set_active_sh_count", None)
    if config.training.sh_degree_progression and not callable(setter):
        raise RuntimeError("Progressive SH requires a rebuilt native set_active_sh_count binding")
    if callable(setter):
        setter(count)
    model.sh_degree_progression = config.training.sh_degree_progression
    opacity_count = active_opacity_sh_count_for_epoch(config, completed_epochs, int(model.opacity_sh_count))
    opacity_setter = getattr(model, "set_active_opacity_sh_count", None)
    opacity_progressive = getattr(config.training, "opacity_sh_degree_progression", False)
    if opacity_progressive and not callable(opacity_setter):
        raise RuntimeError("Progressive opacity SH requires a rebuilt native set_active_opacity_sh_count binding")
    if callable(opacity_setter):
        opacity_setter(opacity_count)
    model.opacity_sh_degree_progression = opacity_progressive
    return count


def sh_model_metadata(model):
    result = {}
    representation, resolution, _ = radiance_model_layout(model)
    if representation == "octahedral_scale":
        result.update(radiance_representation=representation, radiance_resolution=resolution,
                      octahedral_radiance_version=int(getattr(model, "octahedral_radiance_version", 1)))
    for stored_name, active_name, flag in (
            ("sh_count", "active_sh_count", "sh_degree_progression"),
            ("opacity_sh_count", "active_opacity_sh_count", "opacity_sh_degree_progression")):
        stored = int(getattr(model, stored_name))
        active = int(getattr(model, active_name, stored))
        if getattr(model, flag, False) or active != stored:
            result[active_name] = active
    return result


def restore_sh_state(model, metadata):
    representation, _, stored_radiance = radiance_model_layout(model)
    for stored_name, active_name, flag in (
            ("sh_count", "active_sh_count", "sh_degree_progression"),
            ("opacity_sh_count", "active_opacity_sh_count", "opacity_sh_degree_progression")):
        if active_name not in metadata:
            continue
        active = metadata[active_name]
        if type(active) is not int or not 1 <= active <= int(getattr(model, stored_name)):
            raise ValueError(f"Model {active_name} must be an integer within stored SH capacity")
        if (representation == "octahedral_scale" and active_name == "active_sh_count"
                and active not in (1, stored_radiance)):
            raise ValueError("Octahedral active_sh_count must be 1 or the full radiance storage stride")
        setter = getattr(model, "set_" + active_name, None)
        if not callable(setter):
            raise RuntimeError(f"Progressive SH checkpoint requires a rebuilt native set_{active_name} binding")
        setter(active)
        setattr(model, flag, True)


def resolve_resume_sh_schedule(config, checkpoint, overrides):
    """Old checkpoints keep full-SH training; explicit schedule changes are rejected."""
    previous_model = checkpoint["config"].get("model", {})
    requested_model = overrides.get("model", {})
    for name, default in (("radiance_representation", "sh"), ("radiance_resolution", 5)):
        saved = previous_model.get(name, default)
        if requested_model.get(name, saved) != saved:
            raise ValueError(f"Cannot change model.{name} while resuming; start a new experiment")
        setattr(config.model, name, saved)
    previous = checkpoint["config"].get("training", {})
    requested = overrides.get("training", {})
    for name, default in (("sh_degree_progression", False), ("opacity_sh_degree_progression", False),
                          ("sh_degree_weights", (4, 3, 3))):
        saved = previous.get(name, default)
        value = requested.get(name, saved)
        same = (tuple(value) == tuple(saved) if name == "sh_degree_weights"
                and isinstance(value, (list, tuple)) and isinstance(saved, (list, tuple)) else value == saved)
        if not same:
            raise ValueError(f"Cannot change training.{name} while resuming; start a new experiment")
        setattr(config.training, name, saved)
    if (previous.get("sh_degree_progression", False) or previous.get("opacity_sh_degree_progression", False)) and config.training.num_epochs != previous["num_epochs"]:
        raise ValueError("Resume must preserve the total epoch budget of progressive SH")
