"""Dense initial background for projective charts or saved radial proxies.

New projective scenes populate six face slabs and select native segmented ray
queries. Legacy radial scenes preserve their spherical shell and straight-ray
behavior. Region membership always follows the immutable cell address.
"""

from __future__ import annotations

import math


def initialize_background_model(model, mi, dr, np, opacity=.5, *, refinement=None, layers=None,
                                mapping="radial_seed_shell"):
    """Keep core seeds and fill the far shell with covering, black ellipsoids.

    Existing far-cell radiance is retained, but its initial geometry and opacity
    use the same complete-coverage policy as empty far cells. All parameter
    assembly occurs on the selected device. Deletion may subsequently create
    holes: initialization is not an opaque boundary condition.
    """
    if not math.isfinite(opacity) or not 0 < opacity < 1:
        raise ValueError("Background opacity must be finite and between zero and one")
    if not hasattr(model, "__dict__"):
        raise RuntimeError("Rebuild this checkout: background initialization requires Python model metadata support")
    resolution, core = int(model.resolution), int(model.core_resolution)
    if not 0 < core < resolution or (resolution - core) % 2:
        raise ValueError("A background shell requires a positive, symmetric grid padding")
    if refinement not in (None, "fixed_layers", "preserve_physical_width"):
        raise ValueError("Unknown background refinement policy")
    if mapping not in ("radial_seed_shell", "projective_shell"):
        raise ValueError("Unknown background mapping")
    if mapping == "projective_shell" and refinement != "preserve_physical_width":
        raise ValueError("Projective charts require fixed physical shell width")
    if refinement is not None and (type(layers) is not int or layers < 1
                                  or resolution != core + 2 * layers):
        raise ValueError("Background layer metadata must match initial grid padding")
    extent = resolution / core
    if (not np.allclose(model.grid_min, [-extent]*3, rtol=0, atol=1e-7)
            or not np.allclose(model.grid_max, [extent]*3, rtol=0, atol=1e-7)):
        raise ValueError("Background shell bounds must preserve the unit-radius core")

    UInt, Int, Float = mi.UInt32, mi.Int32, mi.Float
    seed = model.snapshot()
    missing = 0xffffffff
    if mapping == "projective_shell":
        from reconstruction_projective import core_mask_device, face_cells_device
        # Enumerate the six slabs directly. The sparse sorted lookup has one
        # entry per seed; neither occupancy nor selection allocates G^3 rows.
        core_seed = dr.compress(core_mask_device(seed["cell_indices"], resolution, core, dr))
        selected = face_cells_device(core, (resolution-core)//2, UInt, dr)
        if len(core_seed):
            selected = dr.concat([dr.gather(UInt, seed["cell_indices"], core_seed), selected])
        selected = dr.gather(UInt, selected, dr.argsort(selected))
        is_far = ~core_mask_device(selected, resolution, core, dr)
        seed_count = int(model.active_voxel_count)
        if seed_count:
            order = dr.argsort(seed["cell_indices"])
            ordered = dr.gather(UInt, seed["cell_indices"], order)
            found = dr.binary_search(0, seed_count, lambda i: dr.gather(UInt, ordered, i) < selected)
            valid = (found < seed_count) & (dr.gather(UInt, ordered, dr.minimum(found, seed_count-1)) == selected)
            rows = dr.select(valid, dr.gather(UInt, order, dr.minimum(found, seed_count-1)), missing)
        else:
            rows = dr.full(UInt, missing, len(selected))
    else:
        index = dr.arange(UInt, resolution**3)
        q = (index % resolution, (index // resolution) % resolution,
             index // (resolution * resolution))
        d = [dr.abs(2*Int(axis) + 1 - resolution) for axis in q]
        far = (sum(dr.square(axis + 1) for axis in d) > core*core) & (
            sum(dr.square(dr.maximum(axis - 1, 0)) for axis in d) < resolution*resolution)
        seed_row = dr.full(UInt, missing, resolution**3)
        dr.scatter(seed_row, dr.arange(UInt, int(model.active_voxel_count)), seed["cell_indices"])
        selected = dr.compress(far | (seed_row != missing))
        is_far = dr.gather(type(far), far, selected)
        rows = dr.gather(UInt, seed_row, selected)
    has_seed = rows != missing
    count = len(selected)
    sh, opacity_sh = int(model.sh_count), int(model.opacity_sh_count)
    widths = {"centers": 3, "log_scales": 3, "rotations": 4,
              "radiance_sh": 3*sh, "opacity_sh": opacity_sh}
    y00 = .28209479177387814
    covering_log_scale = math.log((2/core) * math.sqrt(3)/2 * (1 + 1e-6))
    initial_opacity = math.log(opacity / (1-opacity)) / y00
    state = {"cell_indices": selected}
    for key, width in widths.items():
        flat = dr.arange(UInt, count*width)
        parent = dr.gather(UInt, rows, flat // width)
        present = dr.gather(type(has_seed), has_seed, flat // width)
        value = (dr.gather(Float, seed[key], parent*width + flat % width, present)
                 if len(seed[key]) else dr.zeros(Float, count*width))
        if key != "radiance_sh":
            if key == "centers":
                initial = Float(.5)
            elif key == "log_scales":
                initial = Float(covering_log_scale)
            elif key == "rotations":
                initial = Float(flat % width == 0)
            else:
                initial = dr.select(flat % width == 0, initial_opacity, 0.)
            value = dr.select(dr.gather(type(is_far), is_far, flat // width), initial, value)
        state[key] = value
    dr.eval(*state.values())
    result = mi.VoxelReconstruction.from_state(
        resolution, model.grid_min, model.grid_max, state, sh, opacity_sh,
        core_resolution=core)
    result.scene_mapping = mapping
    if mapping == "projective_shell":
        result.set_scene_mapping(mapping)
        result.background_refinement = "preserve_physical_width"
        result.background_layers = (resolution-core)//2
    elif refinement == "fixed_layers":
        result.background_refinement = refinement
        result.background_layers = layers
    result.background_initialization = dict(
        region="six_projective_face_slabs" if mapping == "projective_shell" else "cell_AABB_intersects_radial_shell", opacity=opacity,
        initial_background_layers=(resolution-core)//2,
        background_refinement=refinement or "preserve_physical_width",
        color="point_seed_radiance_else_black", seed_voxels=int(model.active_voxel_count),
        initialized_voxels=count, straight_ray_forward=mapping != "projective_shell")
    return result
