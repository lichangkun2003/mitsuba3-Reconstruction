"""Remap fitted fine children into a fixed-width radial shell.

Only existing children provide support. This is a geometric initialization for
the next optimization level, not a ray-preserving coordinate transformation.
All appearance/opacity transfer is left to the caller via representative parent
IDs; no eight-times-larger appearance table is constructed here.
"""

from __future__ import annotations

import math


def fixed_shell_metadata(old_metadata):
    """Return the next core/grid layout, retaining the radial cell count."""
    if old_metadata.get("background_refinement") != "fixed_layers":
        raise ValueError("Fixed shell remapping requires background_refinement='fixed_layers'")
    resolution = old_metadata.get("resolution")
    core = old_metadata.get("core_resolution")
    if (type(resolution) is not int or type(core) is not int or core < 1
            or resolution <= core or (resolution - core) % 2):
        raise ValueError("Invalid fixed-shell core/storage resolution")
    layers = (resolution - core) // 2
    if old_metadata.get("background_layers", layers) != layers:
        raise ValueError("Fixed-shell layer metadata disagrees with storage padding")
    new_core, new_resolution = core * 2, core * 2 + layers * 2
    if new_core > 1024 or new_resolution > 1040:
        raise ValueError("Fixed-shell refinement exceeds core 1024 or storage 1040")
    extent = resolution / core
    for key, sign in (("grid_min", -1.), ("grid_max", 1.)):
        bound = old_metadata.get(key, ())
        if len(bound) != 3 or any(not math.isfinite(float(v)) or not math.isclose(
                float(v), sign * extent, rel_tol=0., abs_tol=1e-7) for v in bound):
            raise ValueError("Fixed-shell bounds must preserve a unit-radius core")
    result = dict(old_metadata)
    extent = new_resolution / new_core
    result.update(resolution=new_resolution, core_resolution=new_core,
                  grid_min=[-extent] * 3, grid_max=[extent] * 3,
                  background_layers=layers, background_refinement="fixed_layers")
    return result


def _rotation(q):
    w, x, y, z = q
    return [[1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]]


def remap_children_numpy(old_metadata, cells, centers, log_scales, rotations, parents):
    import numpy as np

    metadata = fixed_shell_metadata(old_metadata)
    cells, parents = np.asarray(cells, np.uint32).reshape(-1), np.asarray(parents, np.uint32).reshape(-1)
    count = len(cells)
    center_input = np.asarray(centers, np.float32).reshape(-1, 3)
    scale_input = np.asarray(log_scales, np.float32).reshape(-1, 3)
    rotation_input = np.asarray(rotations, np.float32).reshape(-1, 4)
    if any(len(value) != count for value in (parents, center_input, scale_input, rotation_input)):
        raise ValueError("Fixed-shell child arrays have incompatible lengths")
    if not count:
        return dict(cell_indices=cells.copy(), centers=center_input.ravel().copy(),
                    log_scales=scale_input.ravel().copy(), rotations=rotation_input.ravel().copy(),
                    parents=parents.copy(), inherit_shell=np.empty(0, bool))
    fine_resolution = old_metadata["resolution"] * 2
    if (np.any(cells >= fine_resolution**3) or
            not all(np.isfinite(value).all() for value in (center_input, scale_input, rotation_input)) or
            np.any(center_input < 0) or np.any(center_input > 1)):
        raise ValueError("Fixed-shell children must have finite, in-cell geometry")
    xyz = np.column_stack((cells % fine_resolution, cells // fine_resolution % fine_resolution,
                           cells // (fine_resolution**2))).astype(np.int64)
    h = 1. / old_metadata["core_resolution"]
    old_extent = old_metadata["resolution"] / old_metadata["core_resolution"]
    world = (xyz + center_input.astype(np.float64)) * h - old_extent
    radius = np.sqrt(np.sum(world * world, axis=1))
    outer = radius > 1.
    unit = world / np.maximum(radius[:, None], 1e-30)
    tangent = np.where(outer, .5 + .5 / np.maximum(radius, 1e-30), 1.)
    radial_delta = np.where(outer, .5 - tangent, 0.)
    mapped = world * tangent[:, None]
    new_resolution = metadata["resolution"]
    new_extent = new_resolution / metadata["core_resolution"]
    target_xyz = np.floor((mapped + new_extent) / h).astype(np.int64)
    # Exact integer address transfer for unchanged inner children avoids an
    # otherwise unnecessary world-to-grid round trip at cell boundaries.
    target_xyz[~outer] = xyz[~outer] - metadata["background_layers"]
    if np.any(target_xyz < 0) or np.any(target_xyz >= new_resolution):
        raise ValueError("Fixed-shell mapping placed a child outside the new grid")
    target = (target_xyz[:, 0] + new_resolution *
              (target_xyz[:, 1] + new_resolution * target_xyz[:, 2])).astype(np.uint32)
    q = rotation_input.astype(np.float64)
    norm = np.sum(q*q, axis=1)
    if np.any(norm <= 1e-20):
        raise ValueError("Fixed-shell children require nondegenerate rotations")
    q /= np.sqrt(norm)[:, None]
    child_rotation = np.asarray(_rotation(q.T))  # [world axis, local axis, child]
    clipped_scale = np.clip(scale_input.astype(np.float64), -30., 30.)
    axes = np.exp(clipped_scale)
    log_volume = np.sum(clipped_scale, axis=1) + np.where(outer, math.log(.5) + 2*np.log(tangent), 0.)
    # A float32 quantized score makes ties independent of FP64 library last
    # bits. The geometric weights still use the unquantized mapped volume.
    score = np.floor(log_volume.astype(np.float32) * np.float32(4096.)).astype(np.int32)
    order = np.argsort(target, kind="stable")
    ordered = target[order]
    first = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    groups = np.empty(count, np.uint32)
    groups[order] = np.repeat(np.arange(len(first), dtype=np.uint32),
                              np.diff(np.r_[first, count]))
    winners = np.minimum.reduceat(np.where(
        score[order] == np.maximum.reduceat(score[order], first)[groups[order]],
        order, np.iinfo(np.uint32).max), first).astype(np.uint32)
    maximum_volume = np.maximum.reduceat(log_volume[order], first)
    weight = np.exp(log_volume - maximum_volume[groups])
    sum_weight = np.add.reduceat(weight[order], first)
    merged = np.add.reduceat((mapped * weight[:, None])[order], first, axis=0) / sum_weight[:, None]
    representative_rotation = np.asarray(_rotation(q[winners].T))
    merged_axes = []
    for axis in range(3):
        direction = representative_rotation[:, axis, :][:, groups].T
        adjusted = tangent[:, None] * direction + radial_delta[:, None] * unit * np.sum(unit*direction, axis=1)[:, None]
        projected = np.sum(child_rotation.transpose(2, 1, 0) * adjusted[:, None, :], axis=2)
        support = np.sqrt(np.sum((axes * projected)**2, axis=1))
        support += np.abs(np.sum((mapped - merged[groups]) * direction, axis=1))
        merged_axes.append(np.maximum.reduceat(support[order], first))
    unique_cells = ordered[first]
    unique_xyz = target_xyz[winners]
    local_center = ((merged + new_extent) / h - unique_xyz).astype(np.float32)
    local_center = np.clip(local_center, np.float32(0.), np.nextafter(np.float32(1.), np.float32(0.)))
    output_scales = np.log(np.asarray(merged_axes).T).astype(np.float32)
    output_rotation = q[winners].astype(np.float32)
    inherit_shell = np.maximum.reduceat(outer[order], first)
    single_core = (np.diff(np.r_[first, count]) == 1) & ~inherit_shell
    local_center[single_core] = center_input[winners[single_core]]
    output_scales[single_core] = scale_input[winners[single_core]]
    output_rotation[single_core] = rotation_input[winners[single_core]]
    if not all(np.isfinite(value).all() for value in (local_center, output_scales, output_rotation)):
        raise ValueError("Fixed-shell merged geometry is nonfinite")
    return dict(cell_indices=unique_cells.copy(), centers=local_center.ravel(),
                log_scales=output_scales.ravel(), rotations=output_rotation.ravel(),
                parents=parents[winners], inherit_shell=inherit_shell)


def remap_children_gpu(old_metadata, cells, centers, log_scales, rotations, parents, dr):
    """Same mapping and group reductions as NumPy; only lengths/flags sync."""
    from reconstruction_growth_candidates_gpu import _fields, _pack

    metadata = fixed_shell_metadata(old_metadata)
    U, I, F, D, B = dr.cuda.UInt32, dr.cuda.Int32, dr.cuda.Float32, dr.cuda.Float64, dr.cuda.Bool
    cells, parents = U(cells), U(parents)
    centers, log_scales, rotations = F(centers), F(log_scales), F(rotations)
    count = len(cells)
    if len(parents) != count or len(centers) != 3*count or len(log_scales) != 3*count or len(rotations) != 4*count:
        raise ValueError("Fixed-shell child arrays have incompatible lengths")
    if not count:
        return dict(cell_indices=cells, centers=centers, log_scales=log_scales,
                    rotations=rotations, parents=parents, inherit_shell=dr.empty(B, 0))
    fine_resolution = old_metadata["resolution"] * 2
    valid = dr.all(cells < fine_resolution**3) & dr.all(dr.isfinite(centers)) & \
            dr.all(dr.isfinite(log_scales)) & dr.all(dr.isfinite(rotations)) & \
            dr.all((centers >= 0) & (centers <= 1))
    if not bool(valid[0]):
        raise ValueError("Fixed-shell children must have finite, in-cell geometry")
    with dr.scoped_set_flag(dr.JitFlag.FastMath, False):
        row = dr.arange(U, count)
        xyz = [cells % fine_resolution, cells // fine_resolution % fine_resolution,
               cells // (fine_resolution * fine_resolution)]
        local = _fields(centers, count, 3, D, dr)
        scales = _fields(log_scales, count, 3, D, dr)
        quaternion = _fields(rotations, count, 4, D, dr)
        norm = sum(value*value for value in quaternion)
        if not bool(dr.all(norm > 1e-20)[0]):
            raise ValueError("Fixed-shell children require nondegenerate rotations")
        quaternion = [value / dr.sqrt(norm) for value in quaternion]
        child_rotation = _rotation(quaternion)
        h = 1. / old_metadata["core_resolution"]
        old_extent = old_metadata["resolution"] / old_metadata["core_resolution"]
        world = [(D(x) + c)*h - old_extent for x, c in zip(xyz, local)]
        radius = dr.sqrt(sum(value*value for value in world))
        outer = radius > 1.
        unit = [value / dr.maximum(radius, 1e-30) for value in world]
        tangent = dr.select(outer, .5 + .5/dr.maximum(radius, 1e-30), 1.)
        radial_delta = dr.select(outer, .5 - tangent, 0.)
        mapped = [value*tangent for value in world]
        new_resolution = metadata["resolution"]
        new_extent = new_resolution / metadata["core_resolution"]
        target_xyz = [dr.select(outer, I(dr.floor((value + new_extent)/h)), I(x)-metadata["background_layers"])
                      for value, x in zip(mapped, xyz)]
        valid = B(True)
        for value in target_xyz:
            valid &= dr.all((value >= 0) & (value < new_resolution))
        if not bool(valid[0]):
            raise ValueError("Fixed-shell mapping placed a child outside the new grid")
        target = U(target_xyz[0] + new_resolution*(target_xyz[1] + new_resolution*target_xyz[2]))
        clipped_scale = [dr.clip(value, -30., 30.) for value in scales]
        axes = [dr.exp(value) for value in clipped_scale]
        log_volume = sum(clipped_scale) + dr.select(outer, math.log(.5) + 2*dr.log(tangent), 0.)
        score = I(dr.floor(F(log_volume)*F(4096.)))
        order = dr.argsort(target)
        ordered = dr.gather(U, target, order)
        preceding = dr.gather(U, ordered, row-1, row > 0)
        starts = (row == 0) | (ordered != preceding)
        first = dr.compress(starts)
        size = len(first)
        group_sorted = dr.cumsum(U(starts))-1
        groups = dr.empty(U, count)
        dr.scatter(groups, group_sorted, order)
        maximum_score = dr.full(I, -2147483648, size)
        maximum_volume = dr.full(D, -float("inf"), size)
        dr.scatter_reduce(dr.ReduceOp.Max, maximum_score, score, groups)
        dr.scatter_reduce(dr.ReduceOp.Max, maximum_volume, log_volume, groups)
        dr.eval(groups, maximum_score, maximum_volume)
        winners = dr.full(U, 0xffffffff, size)
        dr.scatter_reduce(dr.ReduceOp.Min, winners, row, groups,
                          score == dr.gather(I, maximum_score, groups))
        weight = dr.exp(log_volume - dr.gather(D, maximum_volume, groups))
        sum_weight = dr.zeros(D, size)
        merged = [dr.zeros(D, size) for _ in range(3)]
        group_count, group_outer = dr.zeros(U, size), dr.zeros(U, size)
        dr.scatter_reduce(dr.ReduceOp.Add, sum_weight, weight, groups)
        dr.scatter_reduce(dr.ReduceOp.Add, group_count, U(1), groups)
        dr.scatter_reduce(dr.ReduceOp.Or, group_outer, U(outer), groups)
        for output, value in zip(merged, mapped):
            dr.scatter_reduce(dr.ReduceOp.Add, output, value*weight, groups)
        unique_cells = dr.gather(U, ordered, first)
        dr.eval(unique_cells, winners, sum_weight, merged, group_count, group_outer)
        # At 1024, each materialized child-sized sort buffer can occupy hundreds
        # of MiB. Retire them before fitting the three merged axis supports.
        del order, ordered, preceding, starts, first, group_sorted
        del maximum_score, maximum_volume, score, weight, log_volume, target, row
        merged = [value/sum_weight for value in merged]
        dr.eval(merged)
        del sum_weight
        output_quaternion = [dr.gather(D, value, winners) for value in quaternion]
        representative_rotation = _rotation(output_quaternion)
        merged_axes = []
        for axis in range(3):
            direction = [dr.gather(D, representative_rotation[i][axis], groups) for i in range(3)]
            dot = sum(unit[i]*direction[i] for i in range(3))
            adjusted = [tangent*direction[i] + radial_delta*unit[i]*dot for i in range(3)]
            projected = [sum(child_rotation[j][i]*adjusted[j] for j in range(3)) for i in range(3)]
            support = dr.sqrt(sum(dr.square(axes[i]*projected[i]) for i in range(3)))
            support += dr.abs(sum((mapped[i]-dr.gather(D, merged[i], groups))*direction[i] for i in range(3)))
            output = dr.zeros(D, size)
            dr.scatter_reduce(dr.ReduceOp.Max, output, support, groups)
            dr.eval(output)
            merged_axes.append(output)
            del direction, dot, adjusted, projected, support
        unique_xyz = [dr.gather(I, value, winners) for value in target_xyz]
        single_core = (group_count == 1) & (group_outer == 0)
        output_center = [dr.select(single_core, dr.gather(F, F(local[i]), winners),
            dr.clip(F((merged[i]+new_extent)/h-D(unique_xyz[i])), 0., float.fromhex("0x1.fffffep-1")))
            for i in range(3)]
        output_scale = [dr.select(single_core, dr.gather(F, F(scales[i]), winners),
                                  F(dr.log(merged_axes[i]))) for i in range(3)]
        original_quaternion = _fields(rotations, count, 4, F, dr)
        output_rotation = [dr.select(single_core, dr.gather(F, original_quaternion[i], winners),
                                      F(output_quaternion[i])) for i in range(4)]
        valid = B(True)
        for value in output_center + output_scale + output_rotation:
            valid &= dr.all(dr.isfinite(value))
        if not bool(valid[0]):
            raise ValueError("Fixed-shell merged geometry is nonfinite")
        result = dict(cell_indices=unique_cells, centers=_pack(output_center, F, dr),
                      log_scales=_pack(output_scale, F, dr), rotations=_pack(output_rotation, F, dr),
                      parents=dr.gather(U, parents, winners), inherit_shell=group_outer != 0)
        dr.eval(*result.values())
        return result
