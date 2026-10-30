"""Coordinate and cell-region rules for the six projective background charts.

World coordinates are normalized to the core cube [-1, 1]^3. Chart ties use
the first dominant axis (X, then Y, then Z), exactly as native traversal does.
"""

from __future__ import annotations


def projective_points(points, shell_width, np):
    """Map normalized world points into the core and six face slabs."""
    if not 0 < shell_width < float("inf"):
        raise ValueError("Projective shell width must be positive and finite")
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("Projective points must be a finite Nx3 array")
    q = p.copy()
    axis = np.argmax(np.abs(p), axis=1)
    dominant = np.take_along_axis(p, axis[:, None], axis=1)[:, 0]
    h = np.abs(dominant)
    ids = np.flatnonzero(h > 1.)
    q[ids] /= h[ids, None]
    q[ids, axis[ids]] = np.sign(dominant[ids]) * (1. + shell_width - shell_width / h[ids])
    # Face and core boundary ownership is half-open in the cell grid. Keep
    # tangent coordinates on their own chart when a world point lies on a tie.
    q = q.astype(np.float32)
    inner = np.nextafter(np.float32(1.), np.float32(0.))
    q[h <= 1.] = np.clip(q[h <= 1.], -1., inner)
    for j in range(3):
        tangent = ids[axis[ids] != j]
        q[tangent, j] = np.clip(q[tangent, j], -inner, inner)
    negative = ids[dominant[ids] < 0]
    q[negative, axis[negative]] = np.minimum(q[negative, axis[negative]],
                                            np.nextafter(np.float32(-1.), np.float32(-np.inf)))
    outer = np.nextafter(np.float32(1. + shell_width), np.float32(0.))
    np.clip(q, -outer, outer, out=q)
    return np.ascontiguousarray(q)


def core_mask_numpy(cells, resolution, core_resolution, np):
    """Cell membership uses immutable addresses, not ellipsoid centers."""
    pad = (int(resolution) - int(core_resolution)) // 2
    cells = np.asarray(cells, dtype=np.int64)
    xyz = (cells % resolution, cells // resolution % resolution,
           cells // (resolution * resolution))
    return ((xyz[0] >= pad) & (xyz[0] < pad + core_resolution) &
            (xyz[1] >= pad) & (xyz[1] < pad + core_resolution) &
            (xyz[2] >= pad) & (xyz[2] < pad + core_resolution))


def core_mask_device(cells, resolution, core_resolution, dr):
    if not len(cells):
        return dr.empty(dr.mask_t(type(cells)), 0)
    pad = (int(resolution) - int(core_resolution)) // 2
    xyz = (cells % resolution, cells // resolution % resolution,
           cells // (resolution * resolution))
    return ((xyz[0] >= pad) & (xyz[0] < pad + core_resolution) &
            (xyz[1] >= pad) & (xyz[1] < pad + core_resolution) &
            (xyz[2] >= pad) & (xyz[2] < pad + core_resolution))


def face_cells_device(core, layers, UInt, dr):
    """Enumerate 6*core^2*layers cells without a cubic dense scratch table."""
    resolution = core + 2 * layers
    index = dr.arange(UInt, core * core * layers)
    tangent0, tangent1 = index % core + layers, index // core % core + layers
    radial = index // (core * core)
    faces = []
    for axis in range(3):
        for positive in (False, True):
            xyz = [None, None, None]
            xyz[axis] = radial + (layers + core if positive else 0)
            others = [j for j in range(3) if j != axis]
            xyz[others[0]], xyz[others[1]] = tangent0, tangent1
            faces.append(xyz[0] + resolution * (xyz[1] + resolution * xyz[2]))
    return dr.concat(faces)


def refine_projective_numpy(checkpoint, config, max_active_voxels, fit_core, np):
    """Fit core children; split every existing background cell into eight.

    Background geometry is inherited in physical storage coordinates, clipped
    by the child cell during traversal. It is not refitted or moved across
    charts. Its per-hit opacity uses the ordinary optical-depth attenuation.
    """
    metadata, state = dict(checkpoint["model_metadata"]), checkpoint["model_state"]
    old_resolution = metadata["resolution"]
    count = len(state["cell_indices"])
    inside = core_mask_numpy(state["cell_indices"], old_resolution, metadata["core_resolution"], np)
    background = np.flatnonzero(~inside)
    if 8 * len(background) > max_active_voxels:
        raise RuntimeError("Refinement exceeds max_active_voxels")
    core_state = {name: np.asarray(value).reshape(count, -1)[inside].ravel()
                  for name, value in state.items()} if count else {
                      name: np.array(value, copy=True) for name, value in state.items()}
    core_metadata = dict(metadata, scene_mapping="bounded")
    core_result = fit_core({"model_metadata": core_metadata, "model_state": core_state},
                          config, max_active_voxels - 8 * len(background))
    metadata["resolution"] *= 2
    metadata["core_resolution"] *= 2
    metadata["background_layers"] = (metadata["resolution"] - metadata["core_resolution"]) // 2
    if not len(background):
        return {"model_metadata": metadata, "model_state": core_result["model_state"]}
    core_result = core_result["model_state"]
    parents = np.tile(background, 8)
    old_cells = np.asarray(state["cell_indices"], dtype=np.int64)[parents]
    xyz = np.column_stack((old_cells % old_resolution, old_cells // old_resolution % old_resolution,
                           old_cells // (old_resolution * old_resolution)))
    child = np.repeat(np.arange(8), len(background))
    bits = np.column_stack((child & 1, child >> 1 & 1, child >> 2 & 1))
    xyz = xyz * 2 + bits
    new_resolution = metadata["resolution"]
    shell = {name: np.asarray(value).reshape(count, -1)[parents].copy()
             for name, value in state.items()}
    shell["cell_indices"] = (xyz[:, 0] + new_resolution * (xyz[:, 1] + new_resolution * xyz[:, 2])).astype(np.uint32)
    shell["centers"] = (shell["centers"] * 2 - bits).astype(np.float32)
    # All SH coefficients are inherited. Isotropic opacity is the supported
    # background parameterization; retaining higher coefficients is defensive.
    y0 = 1 / (2 * np.sqrt(np.pi))
    attenuation = float(np.clip(config.opacity_optical_depth_scale, .01, 1.))
    if attenuation < 1:
        logit = shell["opacity_sh"][:, 0].astype(np.float64) * y0
        tau = np.logaddexp(0., logit) * attenuation
        child_logit = np.where(tau > 15., tau, np.log(np.maximum(np.expm1(np.minimum(tau, 15.)), 1e-30)))
        shell["opacity_sh"][:, 0] = np.minimum(child_logit, logit) / y0
    return {"model_metadata": metadata, "model_state": {
        name: np.concatenate((core_result[name], np.asarray(value).ravel()))
        for name, value in shell.items()}}


def refine_projective_device(checkpoint, config, max_active_voxels, fit_core, dr, *, before_allocate=None):
    """CUDA counterpart: fitting workspace scales with core rows only."""
    from reconstruction_topology_gpu import take_rows
    import math
    U, F, D = dr.cuda.UInt32, dr.cuda.Float32, dr.cuda.Float64
    metadata = dict(checkpoint["model_metadata"])
    state = {name: (U if name == "cell_indices" else F)(value)
             for name, value in checkpoint["model_state"].items()}
    count, old_resolution = len(state["cell_indices"]), metadata["resolution"]
    inside = core_mask_device(state["cell_indices"], old_resolution, metadata["core_resolution"], dr)
    core_ids, bg_ids = dr.compress(inside), dr.compress(~inside)
    bg_count = len(bg_ids)
    if 8 * bg_count > max_active_voxels:
        raise RuntimeError("Refinement exceeds max_active_voxels")
    next_metadata = dict(metadata, resolution=metadata["resolution"] * 2,
                         core_resolution=metadata["core_resolution"] * 2)
    next_metadata["background_layers"] = (next_metadata["resolution"] - next_metadata["core_resolution"]) // 2
    if before_allocate is not None:
        # Existing projective cells always produce eight background children.
        # This is a lower bound for the complete model, not an eightfold guess
        # for the fitted core. Refuse only when even this lower bound cannot fit.
        before_allocate({"stage": "projective_forecast", "active_voxels": 8 * bg_count,
            "exact_count": False, "allocation_voxels": 8 * bg_count,
            "old_voxels": count, "existing_child_count": 0,
            "core_parent_count": len(core_ids), "background_child_count": 8 * bg_count,
            "model_metadata": dict(next_metadata)})

    def core_allocation_guard(request):
        # The exact core fit is now known, before its large appearance table.
        # Expose the final whole-scene count to the training budget callback.
        before_allocate(dict(request, stage="projective_core_appearance",
            active_voxels=request["active_voxels"] + 8 * bg_count,
            old_voxels=count, core_parent_count=len(core_ids),
            background_child_count=8 * bg_count, model_metadata=dict(next_metadata)))

    core_state = {name: take_rows(value, core_ids, len(value)//count) for name, value in state.items()} if count else state
    options = {} if before_allocate is None else {"before_allocate": core_allocation_guard}
    core_result = fit_core({"model_metadata": dict(metadata, scene_mapping="bounded"), "model_state": core_state},
                          config, max_active_voxels - 8 * bg_count, dr, **options)["model_state"]
    del core_state, core_ids, inside
    metadata = next_metadata
    if before_allocate is not None:
        core_count = len(core_result["cell_indices"])
        before_allocate({"stage": "projective_model", "active_voxels": core_count + 8 * bg_count,
            "exact_count": True, "allocation_voxels": core_count + 8 * bg_count if bg_count else 0,
            "old_voxels": count, "existing_child_count": core_count,
            "background_child_count": 8 * bg_count, "model_metadata": dict(metadata)})
    if not bg_count:
        return {"model_metadata": metadata, "model_state": core_result}
    rows = dr.arange(U, 8 * bg_count)
    parents = dr.gather(U, bg_ids, rows % bg_count)
    child = rows // bg_count
    dr.eval(parents)
    del bg_ids

    def child_component(name, value, width, component):
        # This scope releases coordinate/FP64 opacity scratch after its scatter
        # is evaluated. Never materialize a full background SH table.
        if name == "cell_indices":
            old = dr.gather(U, value, parents)
            xyz = [old % old_resolution, old // old_resolution % old_resolution,
                   old // (old_resolution * old_resolution)]
            xyz = [xyz[j] * 2 + ((child >> j) & 1) for j in range(3)]
            return xyz[0] + metadata["resolution"] * (xyz[1] + metadata["resolution"] * xyz[2])
        values = dr.gather(F, value, parents * width + component)
        if name == "centers":
            bits = (child >> component) & 1
            values = values * 2 - F(bits)
        if name == "opacity_sh" and component == 0 and config.opacity_optical_depth_scale < 1:
            attenuation = min(max(float(config.opacity_optical_depth_scale), .01), 1.)
            y0 = 1 / (2 * math.sqrt(math.pi))
            logit = D(values) * y0
            tau = (dr.maximum(logit, 0.) + dr.log1p(dr.exp(-dr.abs(logit)))) * attenuation
            child_logit = dr.select(tau > 15., tau, dr.log(dr.maximum(dr.expm1(dr.minimum(tau, 15.)), 1e-30)))
            values = F(dr.minimum(child_logit, logit) / y0)
        return values

    result = {}
    for name in ("cell_indices", *(key for key in state if key != "cell_indices")):
        value = state[name]
        width = len(value) // count
        core_value = core_result.pop(name)
        core_size = len(core_value)
        target = dr.empty(U if name == "cell_indices" else F, core_size + 8 * bg_count * width)
        if core_size:
            dr.scatter(target, core_value, dr.arange(U, core_size))
            dr.eval(target)
        del core_value
        # Write directly into the only final buffer. Evaluating each component
        # bounds scratch by background rows, not rows * SH width, and releases
        # the current component before the next one. No host synchronization.
        for component in range(width):
            values = child_component(name, value, width, component)
            dr.scatter(target, values, core_size + rows * width + component)
            dr.eval(target)
            del values
        result[name] = target
        del target
    return {"model_metadata": metadata, "model_state": result}
