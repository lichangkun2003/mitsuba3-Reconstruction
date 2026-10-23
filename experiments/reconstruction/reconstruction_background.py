"""Dense initial far-field shell; subsequent rendering remains straight-ray.

Only cells intersecting the radial contraction shell are populated. The static
cell footprint also defines the topology region, independently of the optimized
ellipsoid center. This module does not alter ray generation or intersection.
"""

from __future__ import annotations

import math


def initialize_background_model(model, mi, dr, np, opacity=.5):
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
    extent = resolution / core
    if (not np.allclose(model.grid_min, [-extent]*3, rtol=0, atol=1e-7)
            or not np.allclose(model.grid_max, [extent]*3, rtol=0, atol=1e-7)):
        raise ValueError("Background shell bounds must preserve the unit-radius core")

    UInt, Int, Float = mi.UInt32, mi.Int32, mi.Float
    seed = model.snapshot()
    index = dr.arange(UInt, resolution**3)
    q = (index % resolution, (index // resolution) % resolution,
         index // (resolution * resolution))
    d = [dr.abs(2*Int(axis) + 1 - resolution) for axis in q]
    # Exact integer bounds in half-cell units; matches the growth exclusion.
    far = (sum(dr.square(axis + 1) for axis in d) > core*core) & (
        sum(dr.square(dr.maximum(axis - 1, 0)) for axis in d) < resolution*resolution)
    missing = 0xffffffff
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
        value = dr.gather(Float, seed[key], parent*width + flat % width, present)
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
    result.scene_mapping = "radial_seed_shell"
    result.background_initialization = dict(
        region="cell_AABB_intersects_radial_shell", opacity=opacity,
        color="point_seed_radiance_else_black", seed_voxels=int(model.active_voxel_count),
        initialized_voxels=count, straight_ray_forward=True)
    return result
