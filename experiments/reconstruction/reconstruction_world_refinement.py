"""Refine core ellipsoids and optionally subdivide world-space cube leaves.

Background rows lead the replacement. Legacy fixed shells retain their Adam
history; uniformly subdivided shells receive fresh state like core children.
"""

import math

from reconstruction_world_shell import (validate_world_shell_layout, world_shell_metadata_layout,
                                        world_shell_metadata_geometry)


def _metadata(metadata):
    mode, subdivision = world_shell_metadata_layout(metadata)
    validate_world_shell_layout(metadata["resolution"], metadata["background_shell_levels"],
                                metadata["grid_min"], metadata["grid_max"], subdivision)
    if mode == "uniform" and metadata["opacity_sh_count"] != 1:
        raise ValueError("Uniform background refinement requires opacity_sh_count=1 (isotropic opacity)")
    result = dict(metadata)
    result["resolution"] *= 2
    result["core_resolution"] = result["resolution"]
    if mode == "uniform":
        result["background_subdivision_level"] = subdivision + 1
    validate_world_shell_layout(result["resolution"], result["background_shell_levels"],
        result["grid_min"], result["grid_max"], result.get("background_subdivision_level", 0))
    return result


def _core_metadata(metadata):
    return dict({key: value for key, value in metadata.items()
                 if not key.startswith("background_") and key != "world_shell_version"},
                scene_mapping="bounded")


def _shell_grid_metadata(metadata, shell):
    """Represent one physical shell as a sparse, bounded regular grid."""
    subdivision = metadata.get("background_subdivision_level", 0)
    resolution = 4 << subdivision
    center = [(lo + hi) * .5 for lo, hi in zip(metadata["grid_min"], metadata["grid_max"])]
    half = (metadata["grid_max"][0] - metadata["grid_min"][0]) * 2 ** shell
    return dict(_core_metadata(metadata), resolution=resolution, core_resolution=resolution,
                grid_min=[value - half for value in center], grid_max=[value + half for value in center])


def _shell_coordinate_tables():
    xyz = [(x, y, z) for z in range(4) for y in range(4) for x in range(4)
           if not (1 <= x <= 2 and 1 <= y <= 2 and 1 <= z <= 2)]
    ordinal = [0] * 64
    for index, (x, y, z) in enumerate(xyz):
        ordinal[x + 4 * y + 16 * z] = index
    return xyz, ordinal


def _shell_grid_cells_numpy(local, subdivision, np):
    xyz, _ = _shell_coordinate_tables()
    local = np.asarray(local, np.uint64)
    coarse, morton = (local >> (3 * subdivision)) % 56, local % (8 ** subdivision)
    coordinates = np.asarray(xyz, np.uint64)[coarse] * (2 ** subdivision)
    for bit in range(subdivision):
        for axis in range(3):
            coordinates[:, axis] += ((morton >> (3 * bit + axis)) & 1) << bit
    resolution = 4 << subdivision
    return coordinates @ np.asarray([1, resolution, resolution ** 2], np.uint64)


def _shell_world_cells_numpy(cells, subdivision, shell, core_resolution, np):
    _, ordinal = _shell_coordinate_tables()
    cells = np.asarray(cells, np.uint64)
    resolution = 4 << subdivision
    coordinates = np.stack((cells % resolution, cells // resolution % resolution,
                            cells // (resolution ** 2)), axis=1)
    coarse = coordinates >> subdivision
    ordinal = np.asarray(ordinal, np.uint64)[coarse @ np.asarray([1, 4, 16], np.uint64)]
    morton = np.zeros(len(cells), np.uint64)
    for bit in range(subdivision):
        for axis in range(3):
            morton |= ((coordinates[:, axis] >> bit) & 1) << (3 * bit + axis)
    return core_resolution ** 3 + (shell * 56 + ordinal) * 8 ** subdivision + morton


def _shell_grid_cells_device(local, subdivision, dr):
    U, C = dr.cuda.UInt32, dr.cuda.UInt64
    xyz, _ = _shell_coordinate_tables()
    coarse = U((local >> (3 * subdivision)) % 56)
    coordinates = [dr.gather(U, U([value[axis] for value in xyz]), coarse) * (2 ** subdivision)
                   for axis in range(3)]
    for bit in range(subdivision):
        for axis in range(3):
            coordinates[axis] += U((local >> (3 * bit + axis)) & 1) << bit
    resolution = 4 << subdivision
    return C(coordinates[0]) + resolution * (C(coordinates[1]) + resolution * C(coordinates[2]))


def _shell_world_cells_device(cells, subdivision, shell, core_resolution, dr):
    U, C = dr.cuda.UInt32, dr.cuda.UInt64
    _, ordinal = _shell_coordinate_tables()
    resolution = 4 << subdivision
    # These local-grid coordinates are at most 1023; only world addresses need 64 bits.
    coordinates = [U(cells % resolution), U(cells // resolution % resolution), U(cells // (resolution ** 2))]
    coarse = [value >> subdivision for value in coordinates]
    ordinal = dr.gather(U, U(ordinal), coarse[0] + 4 * coarse[1] + 16 * coarse[2])
    morton = dr.zeros(C, len(cells))
    for bit in range(subdivision):
        for axis in range(3):
            morton |= C((coordinates[axis] >> bit) & 1) << (3 * bit + axis)
    return C(core_resolution ** 3) + (shell * 56 + C(ordinal)) * 8 ** subdivision + morton


def _fit_shells_numpy(state, metadata, next_metadata, background_rows, widths,
                     config, max_active_voxels, fit_core, np):
    from reconstruction_topology import NoRepresentableChildren
    old_count = len(state["cell_indices"])
    subdivision = metadata.get("background_subdivision_level", 0)
    local = np.asarray(state["cell_indices"])[background_rows].astype(np.uint64) - metadata["resolution"] ** 3
    shells = (local >> (3 * subdivision)) // 56
    pieces, count = [], 0
    for shell in range(metadata["background_shell_levels"]):
        selected = shells == shell
        rows = background_rows[selected]
        if not len(rows):
            continue
        shell_state = {name: np.asarray(value).reshape(old_count, widths[name])[rows].ravel()
                       for name, value in state.items()}
        shell_state["cell_indices"] = _shell_grid_cells_numpy(local[selected], subdivision, np)
        try:
            fitted = fit_core({"model_metadata": _shell_grid_metadata(metadata, shell),
                               "model_state": shell_state}, config, max_active_voxels - count)["model_state"]
        except NoRepresentableChildren:
            # An empty shell cannot invalidate other shells or the foreground.
            # Capacity and data errors remain fatal and are never caught here.
            continue
        fitted["cell_indices"] = _shell_world_cells_numpy(fitted["cell_indices"], subdivision + 1,
                                                          shell, next_metadata["resolution"], np)
        count += len(fitted["cell_indices"])
        pieces.append(fitted)
    return {name: np.concatenate([piece[name] for piece in pieces]) if pieces
            else np.empty(0, dtype=np.asarray(value).dtype) for name, value in state.items()}


def _fit_shells_device(state, metadata, next_metadata, background_rows, widths,
                      config, max_active_voxels, fit_core, dr, before_allocate):
    from reconstruction_topology_gpu import take_rows
    from reconstruction_topology import NoRepresentableChildren
    U, C, F = dr.cuda.UInt32, dr.cuda.UInt64, dr.cuda.Float32
    old_count = len(state["cell_indices"])
    subdivision = metadata.get("background_subdivision_level", 0)
    local = dr.gather(C, state["cell_indices"], background_rows) - metadata["resolution"] ** 3
    shells = (local >> (3 * subdivision)) // 56
    pieces, count = [], 0
    for shell in range(metadata["background_shell_levels"]):
        selected = dr.compress(shells == shell)
        rows = dr.gather(U, background_rows, selected)
        if not len(rows):
            continue
        shell_state = {name: take_rows(value, rows, widths[name]) for name, value in state.items()}
        shell_state["cell_indices"] = _shell_grid_cells_device(dr.gather(C, local, selected), subdivision, dr)

        def guard(request):
            before_allocate(dict(request, stage="world_background_appearance", exact_count=False,
                active_voxels=count + request["active_voxels"], old_voxels=old_count,
                existing_background_child_count=count, background_parent_count=len(rows),
                background_child_count=count + request["active_voxels"], model_metadata=next_metadata))

        try:
            fitted = fit_core({"model_metadata": _shell_grid_metadata(metadata, shell), "model_state": shell_state},
                config, max_active_voxels - count, dr,
                **({"before_allocate": guard} if before_allocate else {}))["model_state"]
        except NoRepresentableChildren:
            del shell_state, rows, selected
            continue
        fitted["cell_indices"] = _shell_world_cells_device(fitted["cell_indices"], subdivision + 1,
                                                           shell, next_metadata["resolution"], dr)
        dr.eval(*fitted.values())
        count += len(fitted["cell_indices"])
        pieces.append(fitted)
        del shell_state, fitted, rows, selected
    result = {name: dr.concat([piece[name] for piece in pieces]) if pieces
              else dr.empty(C if name == "cell_indices" else F, 0) for name in state}
    dr.eval(*result.values())
    return result


def _half_opacity_numpy(values, np):
    """Halve sigmoid optical depth without forming alpha close to zero/one."""
    y0 = 1 / (2 * math.sqrt(math.pi))
    logit = values.astype(np.float64) * y0
    softplus = lambda x: np.maximum(x, 0.) + np.log1p(np.exp(-np.abs(x)))
    # log(expm1(softplus(logit) / 2)) has the equivalent stable form below.
    return ((logit - softplus(.5 * softplus(logit))) / y0).astype(values.dtype)


def _half_opacity_device(values, dr):
    y0 = 1 / (2 * math.sqrt(math.pi))
    logit = dr.cuda.Float64(values) * y0
    softplus = lambda x: dr.maximum(x, 0.) + dr.log1p(dr.exp(-dr.abs(x)))
    return dr.cuda.Float32((logit - softplus(.5 * softplus(logit))) / y0)


def refine_world_numpy(checkpoint, config, max_active_voxels, fit_core, np):
    metadata, state = checkpoint["model_metadata"], checkpoint["model_state"]
    # Validate both layouts before gathering parameters or fitting core children.
    next_metadata = _metadata(metadata)
    old_resolution = int(metadata["resolution"])
    count = len(state["cell_indices"])
    cells = np.asarray(state["cell_indices"], np.uint64)
    state = dict(state, cell_indices=cells)
    core_rows = np.flatnonzero(cells < old_resolution ** 3)
    background_rows = np.flatnonzero(cells >= old_resolution ** 3).astype(np.uint32)
    uniform = metadata.get("background_refinement", "fixed") == "uniform"
    ellipsoids = uniform and world_shell_metadata_geometry(metadata) == "ellipsoid"
    widths = {"cell_indices": 1, "centers": 3, "log_scales": 3, "rotations": 4,
              "radiance_sh": 3 * metadata["sh_count"], "opacity_sh": metadata["opacity_sh_count"]}
    background_state = (_fit_shells_numpy(state, metadata, next_metadata, background_rows, widths,
        config, max_active_voxels, fit_core, np) if ellipsoids else None)
    background_count = (len(background_state["cell_indices"]) if ellipsoids else
                        len(background_rows) * (8 if uniform else 1))
    if background_count > max_active_voxels:
        raise RuntimeError("Refinement exceeds max_active_voxels")
    core_metadata = _core_metadata(metadata)
    core_state = {name: np.asarray(value).reshape(count, widths[name])[core_rows].ravel()
                  for name, value in state.items()}
    core_result = fit_core({"model_metadata": core_metadata, "model_state": core_state},
                          config, max_active_voxels - background_count)["model_state"]
    if ellipsoids and count and not background_count and not len(core_result["cell_indices"]):
        from reconstruction_topology import NoRepresentableChildren
        raise NoRepresentableChildren("Refinement produced no representable children; preserve the old grid")
    if len(core_result["cell_indices"]) + background_count > max_active_voxels:
        raise RuntimeError("Refinement exceeds max_active_voxels")
    result = {}
    for name, value in state.items():
        if ellipsoids:
            result[name] = np.concatenate((background_state.pop(name), core_result.pop(name)))
            continue
        background = np.asarray(value).reshape(count, widths[name])[background_rows].copy()
        if uniform:
            background = np.repeat(background, 8, axis=0)
        if name == "cell_indices":
            if metadata.get("background_refinement") != "adaptive":
                local = background.astype(np.uint64).ravel() - old_resolution ** 3
                if uniform:
                    local = local * 8 + np.arange(background_count, dtype=np.uint64) % 8
                background = local + next_metadata["resolution"] ** 3
        elif uniform and name == "centers":
            background.fill(.5)
        elif uniform and name == "log_scales":
            background -= math.log(2.)
        elif uniform and name == "opacity_sh":
            background[:, 0] = _half_opacity_numpy(background[:, 0], np)
        result[name] = np.concatenate((background.ravel(), core_result.pop(name)))
    return {"model_metadata": next_metadata, "model_state": result,
            "refinement_background_rows": np.empty(0, np.uint32) if uniform else background_rows}


def refine_world_device(checkpoint, config, max_active_voxels, fit_core, dr, *, before_allocate=None):
    metadata = dict(checkpoint["model_metadata"])
    next_metadata = _metadata(metadata)
    from reconstruction_topology_gpu import take_rows
    U, C, F = dr.cuda.UInt32, dr.cuda.UInt64, dr.cuda.Float32
    state = {name: (C if name == "cell_indices" else F)(value)
             for name, value in checkpoint["model_state"].items()}
    old_resolution = int(metadata["resolution"])
    count = len(state["cell_indices"])
    core_mask = state["cell_indices"] < old_resolution ** 3
    core_rows, background_rows = dr.compress(core_mask), dr.compress(~core_mask)
    uniform = metadata.get("background_refinement", "fixed") == "uniform"
    ellipsoids = uniform and world_shell_metadata_geometry(metadata) == "ellipsoid"
    widths = {"cell_indices": 1, "centers": 3, "log_scales": 3, "rotations": 4,
              "radiance_sh": 3 * metadata["sh_count"], "opacity_sh": metadata["opacity_sh_count"]}
    background_state = (_fit_shells_device(state, metadata, next_metadata, background_rows, widths,
        config, max_active_voxels, fit_core, dr, before_allocate) if ellipsoids else None)
    background_count = (len(background_state["cell_indices"]) if ellipsoids else
                        len(background_rows) * (8 if uniform else 1))
    if background_count > max_active_voxels:
        raise RuntimeError("Refinement exceeds max_active_voxels")
    core_metadata = _core_metadata(metadata)
    core_state = {name: take_rows(value, core_rows, widths[name]) for name, value in state.items()}

    def guard(request):
        before_allocate(dict(request, stage="world_core_appearance",
            active_voxels=request["active_voxels"] + background_count,
            old_voxels=count, core_parent_count=len(core_rows),
            background_child_count=background_count, model_metadata=next_metadata,
            **({"existing_background_child_count": background_count} if ellipsoids else {})))

    core_result = fit_core({"model_metadata": core_metadata, "model_state": core_state},
        config, max_active_voxels - background_count, dr,
        **({"before_allocate": guard} if before_allocate else {}))["model_state"]
    del core_state, core_mask, core_rows
    core_count = len(core_result["cell_indices"])
    if ellipsoids and count and not background_count and not core_count:
        from reconstruction_topology import NoRepresentableChildren
        raise NoRepresentableChildren("Refinement produced no representable children; preserve the old grid")
    if core_count + background_count > max_active_voxels:
        raise RuntimeError("Refinement exceeds max_active_voxels")
    if before_allocate:
        before_allocate({"stage": "world_model", "active_voxels": core_count + background_count,
            "exact_count": True, "allocation_voxels": core_count + background_count,
            "old_voxels": count, "existing_child_count": core_count,
            "background_child_count": background_count, "model_metadata": next_metadata,
            **({"existing_background_child_count": background_count} if ellipsoids else {})})
    child_rows = dr.arange(U, background_count) if uniform and not ellipsoids else None
    source_rows = dr.gather(U, background_rows, child_rows // 8) if uniform and not ellipsoids else background_rows
    result = {}
    for name, value in state.items():
        background = background_state.pop(name) if ellipsoids else take_rows(value, source_rows, widths[name])
        if ellipsoids:
            pass  # Fitted geometry and appearance already use the new world layout.
        elif name == "cell_indices":
            if metadata.get("background_refinement") != "adaptive":
                background = background - old_resolution ** 3
                if uniform:
                    background = background * 8 + child_rows % 8
                background += next_metadata["resolution"] ** 3
        elif uniform and name == "centers":
            background = dr.full(F, .5, background_count * 3)
        elif uniform and name == "log_scales":
            background -= math.log(2.)
        elif uniform and name == "opacity_sh":
            dc = dr.gather(F, background, child_rows * widths[name])
            dr.scatter(background, _half_opacity_device(dc, dr), child_rows * widths[name])
        core_value = core_result.pop(name)
        target = dr.concat([background, core_value])
        dr.eval(target)
        result[name] = target
        del background, core_value, target
    return {"model_metadata": next_metadata, "model_state": result,
            "refinement_background_rows": dr.empty(U, 0) if uniform else background_rows}
