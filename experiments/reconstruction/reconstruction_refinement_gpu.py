"""Device coarse-to-fine transfer, with the NumPy fitter as its oracle.

Geometry is evaluated in FP64. Children retain the reference's child-major,
then parent-major order. Only compacted counts and a singular-matrix flag cross
the host boundary; radiance is allocated after fitting, never at eight times the
parent SH table size. Fully enclosed children use the reference's final analytic
override directly, so the discarded 18-step fit is not computed for them.
"""

from __future__ import annotations

import math

from reconstruction_growth_candidates_gpu import geometry_gpu, _quadratic, _pack


def _gather(value, ids, dr):
    if isinstance(value, list):
        return [_gather(v, ids, dr) for v in value]
    return dr.gather(type(value), value, ids)


def _inverse_3x3(metric, dr):
    """Pivoted FP64 elimination, avoiding an ill-conditioned cofactor formula."""
    D, B = dr.cuda.Float64, dr.cuda.Bool
    count = len(metric[0][0])
    rows = [list(row) + [dr.full(D, float(i == j), count) for j in range(3)]
            for i, row in enumerate(metric)]
    singular = dr.zeros(B, count)
    for pivot in range(3):
        for row in range(pivot + 1, 3):
            swap = dr.abs(rows[row][pivot]) > dr.abs(rows[pivot][pivot])
            first, second = rows[pivot], rows[row]
            rows[pivot] = [dr.select(swap, b, a) for a, b in zip(first, second)]
            rows[row] = [dr.select(swap, a, b) for a, b in zip(first, second)]
        divisor = rows[pivot][pivot]
        singular |= divisor == 0
        divisor = dr.select(divisor == 0, 1., divisor)
        rows[pivot] = [value / divisor for value in rows[pivot]]
        for row in range(3):
            if row != pivot:
                multiplier = rows[row][pivot]
                rows[row] = [a - multiplier * b for a, b in zip(rows[row], rows[pivot])]
    if bool(dr.any(singular)[0]):
        # Match np.linalg.inv's failure instead of silently dropping parents.
        from numpy.linalg import LinAlgError
        raise LinAlgError("Singular matrix")
    inverse = [row[3:] for row in rows]
    dr.eval(inverse)
    return inverse


def _closest_box(metric, lower, upper, inverse, dr):
    """Same ordered interior/face/edge constrained minimization as NumPy."""
    best = [dr.clip(dr.zeros(type(v), len(v)), a, b) for v, a, b in zip(lower, lower, upper)]
    minimum = _quadratic(best, metric)
    for normal in range(3):
        for side in (0, 1):
            bound = (lower if side == 0 else upper)[normal]
            point = [inverse[i][normal] * (bound / inverse[normal][normal]) for i in range(3)]
            point[normal] = bound
            valid = (point[0] >= lower[0]) & (point[0] <= upper[0])
            for i in (1, 2):
                valid &= (point[i] >= lower[i]) & (point[i] <= upper[i])
            value = _quadratic(point, metric)
            better = valid & (value < minimum)
            minimum = dr.select(better, value, minimum)
            best = [dr.select(better, a, b) for a, b in zip(point, best)]
    for free in range(3):
        a, b = (free + 1) % 3, (free + 2) % 3
        for side in range(4):
            point = list(lower)
            point[a] = (upper if side & 1 else lower)[a]
            point[b] = (upper if side & 2 else lower)[b]
            point[free] = dr.clip(-(metric[free][a] * point[a] + metric[free][b] * point[b]) /
                                  metric[free][free], lower[free], upper[free])
            value = _quadratic(point, metric)
            better = value < minimum
            minimum = dr.select(better, value, minimum)
            best = [dr.select(better, a, b) for a, b in zip(point, best)]
    return minimum, best


def refine_state_gpu(checkpoint, config, max_active_voxels=67108864, *, before_allocate=None):
    """Fit children on CUDA and return a device-backed replacement checkpoint."""
    import drjit as dr
    with dr.scoped_set_flag(dr.JitFlag.FastMath, False):
        return _refine(checkpoint, config, max_active_voxels, dr, before_allocate=before_allocate)


def _refine(checkpoint, config, max_active_voxels, dr, *, before_allocate=None):
    U, D, F, B = dr.cuda.UInt32, dr.cuda.Float64, dr.cuda.Float32, dr.cuda.Bool
    if checkpoint["model_metadata"].get("scene_mapping") == "projective_shell":
        from reconstruction_projective import refine_projective_device
        return refine_projective_device(checkpoint, config, max_active_voxels, _refine, dr,
                                        before_allocate=before_allocate)
    metadata = dict(checkpoint["model_metadata"])
    state = {name: (U if name == "cell_indices" else F)(value)
             for name, value in checkpoint["model_state"].items()}
    n = len(state["cell_indices"])
    fixed_shell = metadata.get("background_refinement") == "fixed_layers"
    if fixed_shell:
        from reconstruction_shell_refinement import fixed_shell_metadata, remap_children_gpu
        metadata = fixed_shell_metadata(metadata)
    else:
        metadata["resolution"] *= 2
        if "core_resolution" in metadata:
            metadata["core_resolution"] *= 2
    if not n:
        return {"model_metadata": metadata, "model_state": state}
    result, parents = _fit_children(state, checkpoint["model_metadata"], config,
                                   max_active_voxels, fixed_shell, dr)
    total = len(parents)
    if fixed_shell:
        result = remap_children_gpu(
            checkpoint["model_metadata"], result["cell_indices"], result["centers"],
            result["log_scales"], result["rotations"], parents, dr)
        parents = result.pop("parents")
        inherit_shell = result.pop("inherit_shell")
        total = len(parents)
        if total > max_active_voxels:
            raise RuntimeError("Refinement exceeds max_active_voxels")
    # No child SH table is allocated until radial compression and deduplication
    # have completed. This bounds the largest appearance allocation by the
    # replacement model size, rather than all eight children of every parent.
    if before_allocate is not None:
        before_allocate({"stage": "core_appearance", "active_voxels": total,
            "exact_count": True, "allocation_voxels": total, "old_voxels": n,
            "existing_child_count": total, "background_child_count": 0,
            "model_metadata": dict(metadata)})
    y0 = 1 / (2 * math.sqrt(math.pi))
    opacity_count = metadata["opacity_sh_count"]
    rows = dr.arange(U, total)
    sh_count = metadata["sh_count"]
    result["radiance_sh"] = dr.zeros(F, total * sh_count * 3)
    # Progressive SH is global across spatial levels. Keep all stored parent
    # coefficients so refinement does not erase already trained higher degrees.
    radiance_components = sh_count * 3 if "active_sh_count" in metadata else 3
    for component in range(radiance_components):
        value = dr.gather(F, state["radiance_sh"], parents * sh_count * 3 + component)
        dr.scatter(result["radiance_sh"], value, rows * sh_count * 3 + component)
    if fixed_shell and radiance_components == 3:
        for coefficient in range(3, sh_count * 3):
            value = dr.gather(F, state["radiance_sh"], parents * sh_count * 3 + coefficient)
            dr.scatter(result["radiance_sh"], value, rows * sh_count * 3 + coefficient, inherit_shell)
    result["opacity_sh"] = dr.zeros(F, total * opacity_count)
    dc = dr.gather(F, state["opacity_sh"], parents * opacity_count)
    parent_dc = dc
    attenuation = min(max(float(config.opacity_optical_depth_scale), .01), 1.)
    if attenuation < 1:
        logit = D(dc) * y0
        tau = dr.select(logit > 15, logit, dr.maximum(logit, 0.) + dr.log1p(dr.exp(-dr.abs(logit)))) * attenuation
        child_logit = dr.select(tau > 15, tau, dr.select(tau < 1e-4, dr.log(dr.maximum(tau, 1e-30)),
                                                     dr.log(dr.expm1(dr.minimum(tau, 15.)))))
        dc = F(dr.minimum(child_logit, logit) / y0)
    if fixed_shell:
        dc = dr.select(inherit_shell, parent_dc, dc)
        for coefficient in range(1, opacity_count):
            value = dr.gather(F, state["opacity_sh"], parents * opacity_count + coefficient)
            dr.scatter(result["opacity_sh"], value, rows * opacity_count + coefficient, inherit_shell)
    dr.scatter(result["opacity_sh"], dc, rows * opacity_count)
    dr.eval(*result.values())
    return {"model_metadata": metadata, "model_state": result}


def _fit_children(state, metadata, config, max_active_voxels, fixed_shell, dr):
    """Finish geometry before allocating or remapping any appearance tables.

    Keeping this stage in its own scope releases parent-sized FP64 fitting
    temporaries before the shell's sort/reduction workspace is allocated.
    """
    U, D, F, B = dr.cuda.UInt32, dr.cuda.Float64, dr.cuda.Float32, dr.cuda.Bool
    n = len(state["cell_indices"])
    resolution = int(metadata["resolution"])
    center, axes, quaternion, size, metric, extent = geometry_gpu(
        state, metadata, dr)
    inverse = _inverse_3x3(metric, dr)
    index = dr.arange(U, n)
    y0 = 1 / (2 * math.sqrt(math.pi))
    retain = dr.full(B, True, n)
    opacity_count = metadata["opacity_sh_count"]
    if config.parent_opacity_threshold > 0:
        # The reference evaluates this upper bound in the stored float32
        # coefficient dtype, unlike the remaining double-precision geometry.
        upper_opacity = dr.gather(F, state["opacity_sh"], index * opacity_count) * F(y0)
        for k in range(1, opacity_count):
            bound = math.sqrt((2 * int(math.sqrt(k)) + 1) / (4 * math.pi))
            upper_opacity += dr.abs(dr.gather(F, state["opacity_sh"], index * opacity_count + k)) * F(bound)
        retain = upper_opacity >= F(math.log(config.parent_opacity_threshold / (1 - config.parent_opacity_threshold)))
    fine_size = [v * .5 for v in size]
    ratios = [max(v * 1e-4, 1e-8) / axis for v, axis in zip(fine_size, axes)]
    base_lo = dr.maximum(dr.maximum(ratios[0], ratios[1]), ratios[2])
    overlap = .5 * min(max(config.child_face_overlap, 0.), .2)
    base_hi = dr.minimum(1., dr.minimum(dr.minimum((.25 + overlap) / extent[0],
                                                  (.25 + overlap) / extent[1]), (.25 + overlap) / extent[2]))
    w, x, y, z = quaternion
    rotation = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    dr.eval(retain, base_lo, base_hi, rotation)
    pieces, total = [], 0
    for child in range(8):
        bits = [child & 1, (child >> 1) & 1, (child >> 2) & 1]
        lower = [.5 * bit - c for bit, c in zip(bits, center)]
        upper = [v + .5 for v in lower]
        middle = [v + .25 for v in lower]
        interior = dr.full(B, True, n)
        for corner in range(8):
            point = [lower[i] + .5 * ((corner >> i) & 1) for i in range(3)]
            interior &= _quadratic(point, metric) <= 1.
        interior_factor = dr.minimum(base_hi, 1 - 1e-5 - dr.sqrt(dr.maximum(_quadratic(middle, metric), 0.)))
        fit = retain & interior & (interior_factor >= base_lo)
        factors = dr.select(fit, interior_factor, 0.)
        centers = [dr.full(D, .5, n) for _ in range(3)]
        # Only boundary children can use the constrained fit. Enclosed children
        # already have exactly the analytic final override from the CPU code.
        surface = dr.compress(retain & ~interior & (base_lo <= base_hi))
        if len(surface):
            m, inv = _gather(metric, surface, dr), _gather(inverse, surface, dr)
            low, high, e = _gather(lower, surface, dr), _gather(upper, surface, dr), _gather(extent, surface, dr)
            lo, hi = _gather(base_lo, surface, dr), _gather(base_hi, surface, dr)
            inset = [dr.maximum(.005, lo * value - overlap) for value in e]
            a, b = [v + d for v, d in zip(low, inset)], [v - d for v, d in zip(high, inset)]
            value, best = _closest_box(m, a, b, inv, dr)
            possible = (a[0] <= b[0]) & (a[1] <= b[1]) & (a[2] <= b[2])
            possible &= dr.sqrt(dr.maximum(value, 0.)) + lo <= 1 - 1e-5
            chosen = dr.compress(possible)
            if len(chosen):
                surface = _gather(surface, chosen, dr)
                m, inv, low, high, e = [_gather(v, chosen, dr) for v in (m, inv, low, high, e)]
                lo, hi, best = [_gather(v, chosen, dr) for v in (lo, hi, best)]

                def refine_step(iteration, lo, hi, best):
                    candidate = (lo + hi) * .5
                    inset = [dr.maximum(.005, candidate * value - overlap) for value in e]
                    a, b = [v + d for v, d in zip(low, inset)], [v - d for v, d in zip(high, inset)]
                    value, point = _closest_box(m, a, b, inv, dr)
                    valid = (a[0] <= b[0]) & (a[1] <= b[1]) & (a[2] <= b[2])
                    valid &= dr.sqrt(dr.maximum(value, 0.)) + candidate <= 1 - 1e-5
                    return (iteration + 1, dr.select(valid, candidate, lo), dr.select(valid, hi, candidate),
                            [dr.select(valid, a, b) for a, b in zip(point, best)])

                _, lo, hi, best = dr.while_loop(
                    (U(0), lo, hi, best), lambda iteration, lo, hi, best: iteration < 18,
                    refine_step, mode="symbolic", label="Refine boundary children")
                r, ax = _gather(rotation, surface, dr), _gather(axes, surface, dr)
                mid = _gather(middle, surface, dr)
                principal = [sum(r[j][i] * mid[j] * size[j] for j in range(3)) for i in range(3)]
                normal = [sum(r[i][j] * principal[j] / (ax[j] * ax[j]) for j in range(3)) for i in range(3)]
                normal_length = dr.sqrt(sum(v * v for v in normal))
                move = .1 * min(fine_size) / dr.maximum(normal_length, 1e-30)
                preferred = [dr.clip(mid[i] - move * normal[i] / size[i], low[i] + .005, high[i] - .005)
                             for i in range(3)]
                factor = dr.minimum(1., 1 - 1e-5 - dr.sqrt(dr.maximum(_quadratic(preferred, m), 0.)))
                for i in range(3):
                    factor = dr.minimum(factor, (dr.minimum(preferred[i] - low[i], high[i] - preferred[i]) + overlap) / e[i])
                use_preferred = (normal_length > 1e-20) & (factor >= _gather(base_lo, surface, dr)) & (factor >= .9 * hi)
                lo = dr.select(use_preferred, factor, lo)
                best = [dr.select(use_preferred, a, b) for a, b in zip(preferred, best)]
                child_center = [2 * (_gather(center[i], surface, dr) + best[i]) - bits[i] for i in range(3)]
                dr.scatter(fit, True, surface)
                dr.scatter(factors, lo, surface)
                for target, values in zip(centers, child_center):
                    dr.scatter(target, values, surface)
        ids = dr.compress(fit)
        if not len(ids):
            continue
        total += len(ids)
        if not fixed_shell and total > max_active_voxels:
            raise RuntimeError("Refinement exceeds max_active_voxels")
        cells = _gather(state["cell_indices"], ids, dr)
        old = [cells % resolution, cells // resolution % resolution, cells // (resolution * resolution)]
        xyz = [old[i] * 2 + bits[i] for i in range(3)]
        cells = xyz[0] + (resolution * 2) * (xyz[1] + (resolution * 2) * xyz[2])
        scale = _gather(factors, ids, dr)
        pieces.append({"parents": ids, "cell_indices": cells,
                       "centers": _pack(_gather(centers, ids, dr), F, dr),
                       "log_scales": _pack([dr.log(_gather(axis, ids, dr) * scale) for axis in axes], F, dr)})
        dr.eval(*pieces[-1].values())
    if not total:
        raise RuntimeError("Refinement produced no representable children; preserve the old grid")
    result = {name: dr.concat([piece[name] for piece in pieces])
              for name in ("cell_indices", "centers", "log_scales")}
    parents = dr.concat([piece["parents"] for piece in pieces])
    result["rotations"] = _pack(_gather(quaternion, parents, dr), F, dr)
    dr.eval(*result.values(), parents)
    return result, parents
