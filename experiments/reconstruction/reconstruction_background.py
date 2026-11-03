"""Initialize coarse background leaves in physical world-space shells."""

from __future__ import annotations

import math

from reconstruction_world_shell import (
    background_cells_numpy, configure_world_shell_model, leaf_boxes_numpy,
    require_world_shell_native, shell_coordinates_numpy, validate_world_shell_layout,
    validate_adaptive_cells_numpy, WORLD_SHELL_VERSION,
)


def _concat_nonempty(values, dr):
    """Dr.Jit concat does not accept a zero-width input beside live lanes."""
    live = [value for value in values if len(value)]
    return dr.concat(live) if len(live) > 1 else live[0] if live else values[0]


def background_point_colors(positions, colors, grid_min, grid_max, shell_levels, np):
    """Average point colors in all coarse shell leaves, with black for empty leaves.

    Positions are world coordinates; source colors are encoded sRGB in [0, 1].
    Decode each point before averaging, matching native foreground initialization.
    Every leaf is half open: a point on its minimum face belongs to the leaf, and
    a point on its maximum face does not. Core and outside-root points are ignored.
    """
    from reconstruction_color import srgb_to_linear

    validate_world_shell_layout(2, shell_levels, grid_min, grid_max)
    positions, colors = np.asarray(positions), np.asarray(colors)
    if (positions.ndim != 2 or positions.shape[1] != 3 or positions.dtype.kind not in "iuf"
            or colors.shape != positions.shape or colors.dtype.kind not in "iuf"):
        raise ValueError("Background point positions and colors must be matching numeric (N, 3) arrays")
    lo, hi = np.asarray(grid_min, np.float64), np.asarray(grid_max, np.float64)
    center, radius = (lo + hi) * .5, (hi[0] - lo[0]) * .5
    count = 56 * shell_levels
    sums, counts = np.zeros((count, 3), np.float64), np.zeros(count, np.int64)
    coordinates = shell_coordinates_numpy(np)
    ordinal = np.full(64, -1, np.int32)
    ordinal[coordinates @ np.array([1, 4, 16])] = np.arange(56)
    # Keep point-selection and color-decoding temporaries bounded for dense PLYs.
    for start in range(0, len(positions), 262144):
        points, rgb = positions[start:start + 262144], colors[start:start + 262144]
        if not np.isfinite(points).all() or not np.isfinite(rgb).all():
            raise ValueError("Background point positions and colors must be finite")
        if np.any(rgb < 0) or np.any(rgb > 1):
            raise ValueError("Background source point colors must lie in [0, 1]")
        remaining = ~np.all((points >= lo) & (points < hi), axis=1)
        for shell in range(shell_levels):
            edge = radius * 2 ** shell
            outer_min, outer_max = center - 2 * edge, center + 2 * edge
            selected = remaining & np.all((points >= outer_min) & (points < outer_max), axis=1)
            if not np.any(selected):
                continue
            remaining[selected] = False
            selected_points = points[selected]
            # searchsorted preserves half-open ownership even for values one ULP
            # below an internal face, where subtract/divide can round upward.
            xyz = np.stack([np.searchsorted(center[axis] + (np.arange(5) - 2) * edge,
                                           selected_points[:, axis], side="right") - 1
                            for axis in range(3)], axis=1)
            local = ordinal[xyz @ np.array([1, 4, 16])]
            if np.any(local < 0):
                raise ValueError("Background point lies in an inconsistent core/shell partition")
            rows = shell * 56 + local
            counts += np.bincount(rows, minlength=count)
            linear = srgb_to_linear(rgb[selected], np)
            for channel in range(3):
                sums[:, channel] += np.bincount(rows, weights=linear[:, channel], minlength=count)
    means = np.zeros((count, 3), np.float32)
    occupied = counts > 0
    means[occupied] = (sums[occupied] / counts[occupied, None]).astype(np.float32)
    return means, counts


def _validated_background_colors(colors, counts, count, np, *, empty_color=0.):
    if (colors is None) != (counts is None):
        raise ValueError("background_colors and background_point_counts must be provided together")
    if colors is None:
        return np.zeros((count, 3), np.float32), np.zeros(count, np.int64)
    colors, counts = np.asarray(colors), np.asarray(counts)
    if (colors.shape != (count, 3) or colors.dtype.kind not in "iuf"
            or not np.isfinite(colors).all() or np.any(colors < 0) or np.any(colors > 1)):
        raise ValueError("background_colors must contain finite linear RGB values in [0, 1] for every shell leaf")
    if counts.shape != (count,) or counts.dtype.kind not in "iu" or np.any(counts < 0):
        raise ValueError("background_point_counts must contain one nonnegative integer per shell leaf")
    if np.any(colors[counts == 0] != empty_color):
        raise ValueError("Background leaves without source points must have "
                         + ("black initial colors" if empty_color == 0 else "neutral linear gray initial colors"))
    return colors.astype(np.float32, copy=False), counts


def _shell_bounds(grid_min, grid_max, shell, np):
    lo, hi = np.asarray(grid_min, np.float32).astype(np.float64), np.asarray(grid_max, np.float32).astype(np.float64)
    center, edge = (lo + hi) * .5, (hi[0] - lo[0]) * .5 * 2 ** shell
    return (center - 2 * edge).astype(np.float32), (center + 2 * edge).astype(np.float32)


def _compact_shell_cells(local_cells, core_resolution, shell, subdivision, np):
    """Reindex occupied cells of a shell's local 4*2^S cube; never expand its volume."""
    local_cells = np.asarray(local_cells, dtype=np.uint64)
    side = 4 << subdivision
    if np.any(local_cells >= side ** 3):
        raise ValueError("Sparse background local cell exceeds its shell cube")
    xyz = np.stack((local_cells % side, local_cells // side % side, local_cells // (side * side)), axis=1)
    coarse = xyz >> subdivision
    ordinal = np.full(64, -1, np.int64)
    ordinal[shell_coordinates_numpy(np) @ np.array([1, 4, 16])] = np.arange(56)
    flat = coarse[:, 0] + 4 * coarse[:, 1] + 16 * coarse[:, 2]
    cell = ordinal[flat.astype(np.int64)]
    if np.any(cell < 0):
        raise ValueError("Sparse background point was assigned to the interior of its shell")
    morton = np.zeros(len(xyz), np.uint64)
    for bit in range(subdivision):
        for axis in range(3):
            morton |= ((xyz[:, axis] >> bit) & np.uint64(1)) << (3 * bit + axis)
    return (np.uint64(core_resolution ** 3) +
            (np.uint64(shell * 56) + cell.astype(np.uint64)) * np.uint64(8 ** subdivision) + morton)


def prepare_sparse_background(positions, colors, core_resolution, grid_min, grid_max,
                              shell_levels, np, background_subdivision_level=0):
    """Keep only point-occupied background leaves and their CPU initialization data.

    Assignment uses the same float32 input/bounds/voxel widths and float64
    quotient as native load_points(). Means decode each sRGB point before
    averaging. Storage scales with input points and occupied cells, never the
    theoretical shell address space (which can exceed 2^32 at S=8).
    """
    from reconstruction_color import srgb_to_linear

    subdivision = background_subdivision_level
    validate_world_shell_layout(core_resolution, shell_levels, grid_min, grid_max, subdivision)
    positions, colors = np.asarray(positions), np.asarray(colors)
    if (positions.ndim != 2 or positions.shape[1] != 3 or positions.dtype.kind not in "iuf"
            or colors.shape != positions.shape or colors.dtype.kind not in "iuf"):
        raise ValueError("Background point positions and colors must be matching numeric (N, 3) arrays")
    lo, hi = np.asarray(grid_min, np.float32), np.asarray(grid_max, np.float32)
    side = 4 << subdivision
    selected_positions, selected_colors, selected_shells, selected_cells = [], [], [], []
    core_points = outside_points = 0
    for start in range(0, len(positions), 262144):
        source_points, source_rgb = positions[start:start + 262144], colors[start:start + 262144]
        if not np.isfinite(source_points).all() or not np.isfinite(source_rgb).all():
            raise ValueError("Background point positions and colors must be finite")
        if np.any(source_rgb < 0) or np.any(source_rgb > 1):
            raise ValueError("Background source point colors must lie in [0, 1]")
        points, rgb = source_points.astype(np.float32), source_rgb.astype(np.float32)
        if not np.isfinite(points).all():
            raise ValueError("Background point positions must fit float32 native coordinates")
        inside = np.all((points >= lo) & (points < hi), axis=1)
        core_points += int(np.count_nonzero(inside))
        remaining = ~inside
        for shell in range(shell_levels):
            lower, upper = _shell_bounds(lo, hi, shell, np)
            # m_size is evaluated in float32 by the native constructor.
            edge = (upper - lower) / np.float32(side)
            coordinates = (points.astype(np.float64) - lower.astype(np.float64)) / edge.astype(np.float64)
            selected = remaining & np.all((coordinates >= 0) & (coordinates < side), axis=1)
            if not np.any(selected):
                continue
            remaining[selected] = False
            xyz = np.floor(coordinates[selected]).astype(np.uint64)
            local = xyz[:, 0] + side * xyz[:, 1] + side * side * xyz[:, 2]
            selected_positions.append(points[selected])
            selected_colors.append(rgb[selected])
            selected_shells.append(np.full(int(np.count_nonzero(selected)), shell, np.uint8))
            selected_cells.append(_compact_shell_cells(local, core_resolution, shell, subdivision, np))
        outside_points += int(np.count_nonzero(remaining))
    points = np.concatenate(selected_positions) if selected_positions else np.empty((0, 3), np.float32)
    rgb = np.concatenate(selected_colors) if selected_colors else np.empty((0, 3), np.float32)
    shells = np.concatenate(selected_shells) if selected_shells else np.empty(0, np.uint8)
    point_cells = np.concatenate(selected_cells) if selected_cells else np.empty(0, np.uint64)
    cells, inverse, counts = np.unique(point_cells, return_inverse=True, return_counts=True)
    means = np.zeros((len(cells), 3), np.float32)
    linear = srgb_to_linear(rgb, np)
    for axis in range(3):
        means[:, axis] = np.bincount(inverse, weights=linear[:, axis], minlength=len(cells)) / np.maximum(counts, 1)
    return dict(positions=np.ascontiguousarray(points), colors=np.ascontiguousarray(rgb),
                shell_indices=shells, point_cell_indices=point_cells, cell_indices=cells,
                point_counts=counts.astype(np.int64, copy=False), mean_linear_colors=means,
                excluded_core_points=core_points, outside_root_points=outside_points,
                core_resolution=core_resolution, shell_levels=shell_levels,
                background_subdivision_level=subdivision, grid_min=list(grid_min), grid_max=list(grid_max))


def _sparse_background_state(points, model, mi, dr, np, opacity, shell_levels, subdivision):
    core, sh, opacity_sh = int(model.core_resolution), int(model.sh_count), int(model.opacity_sh_count)
    for key, expected in (("core_resolution", core), ("shell_levels", shell_levels),
                          ("background_subdivision_level", subdivision),
                          ("grid_min", list(model.grid_min)), ("grid_max", list(model.grid_max))):
        if points.get(key) != expected:
            raise ValueError(f"Sparse background data disagrees with model {key}")
    seed = model.snapshot()
    components = {name: [mi.UInt64(values) if name == "cell_indices" else values]
                  for name, values in seed.items()}
    actual_cells = []
    side = 4 << subdivision
    for shell in range(shell_levels):
        selected = points["shell_indices"] == shell
        if not np.any(selected):
            continue
        positions, colors = points["positions"][selected], points["colors"][selected]
        lower, upper = _shell_bounds(model.grid_min, model.grid_max, shell, np)
        if side > 1024:
            raise ValueError("Point-fitted background initialization supports subdivisions up to S=8")
        fitted = mi.VoxelReconstruction.from_points(
            np.ascontiguousarray(positions.reshape(-1), dtype=np.float32),
            np.ascontiguousarray(colors.reshape(-1), dtype=np.float32),
            side, lower.tolist(), upper.tolist(), sh, opacity_sh)
        snapshot = fitted.snapshot()
        local = np.asarray(snapshot["cell_indices"], dtype=np.uint64)
        cells = _compact_shell_cells(local, core, shell, subdivision, np)
        expected = points["cell_indices"][(points["cell_indices"] - np.uint64(core ** 3)) // np.uint64(56 * 8 ** subdivision) == shell]
        if not np.array_equal(np.sort(cells), expected):
            raise RuntimeError("Native point-fit occupancy differs from sparse background CPU preflight")
        actual_cells.append(cells)
        for name, values in snapshot.items():
            if name == "cell_indices":
                values = mi.UInt64(cells)
            elif name == "opacity_sh":
                coefficients = np.zeros((len(cells), opacity_sh), np.float32)
                coefficients[:, 0] = math.log(opacity / (1. - opacity)) / .28209479177387814
                values = mi.Float(coefficients.ravel())
            components[name].append(values)
    state = {name: _concat_nonempty(values, dr)
             for name, values in components.items()}
    actual_count = sum(len(cells) for cells in actual_cells)
    if actual_count != len(points["cell_indices"]):
        raise RuntimeError("Native background occupancy count differs from CPU preflight")
    dr.eval(*state.values())
    return state, actual_count


def initialize_background_model(model, mi, dr, np, opacity=.5, *, shell_levels=5,
                                mapping="world_shell", background_refinement="uniform",
                                background_colors=None, background_point_counts=None,
                                background_geometry="ellipsoid", background_points=None,
                                background_subdivision_level=0, adaptive_background=None):
    """Append point-fitted sparse leaves, or retain the legacy dense S=0 layout."""
    if mapping != "world_shell":
        raise ValueError("Only world_shell background initialization is supported")
    if background_geometry not in ("ellipsoid", "occupied_cell"):
        raise ValueError("Background geometry must be ellipsoid or occupied_cell")
    if isinstance(opacity, bool) or not math.isfinite(opacity) or not 0 < opacity < 1:
        raise ValueError("Background opacity must be finite and between zero and one")
    core = int(model.core_resolution)
    subdivision = background_subdivision_level
    adaptive = background_refinement == "adaptive"
    validate_world_shell_layout(core, shell_levels, model.grid_min, model.grid_max, subdivision)
    if adaptive:
        if adaptive_background is None or subdivision != 0 or background_geometry != "ellipsoid":
            raise ValueError("Adaptive background requires prepared mixed-depth leaves, S=0 and ellipsoid geometry")
        if background_points is not None or background_colors is not None or background_point_counts is not None:
            raise ValueError("Adaptive initialization cannot use legacy sparse/dense background color tables")
        for key, expected in (("core_resolution", core), ("shell_levels", shell_levels),
                              ("grid_min", list(model.grid_min)), ("grid_max", list(model.grid_max))):
            if adaptive_background.get(key) != expected:
                raise ValueError(f"Adaptive preparation disagrees with model {key}")
        cells = np.asarray(adaptive_background["cell_indices"], np.uint64)
        validate_adaptive_cells_numpy(cells, core, shell_levels, np)
        from reconstruction_world_shell import ADAPTIVE_TAG
        if not len(cells) or np.any(cells < ADAPTIVE_TAG):
            raise ValueError("Adaptive initialization requires background-only coverage leaves")
        background_colors, background_point_counts = (adaptive_background["mean_linear_colors"],
                                                       adaptive_background["point_counts"])
    elif adaptive_background is not None:
        raise ValueError("Adaptive prepared leaves require background_refinement=adaptive")
    if subdivision and background_points is None:
        raise ValueError("Subdivided background initialization requires sparse background_points")
    if background_points is not None and (background_colors is not None or background_point_counts is not None):
        raise ValueError("Sparse background_points cannot be combined with dense background color tables")
    point_colored = background_colors is not None
    colors, point_counts = _validated_background_colors(
        background_colors, background_point_counts, len(cells) if adaptive else 56 * shell_levels, np,
        empty_color=.5 if adaptive else 0.)
    require_world_shell_native(mi.VoxelReconstruction)
    if int(model.resolution) != core:
        raise ValueError("World shell storage uses the core resolution without grid padding")
    seed = model.snapshot()
    if len(seed["cell_indices"]) and bool(dr.any(seed["cell_indices"] >= core ** 3)):
        raise ValueError("Background initialization requires foreground-only point seeds")
    if background_points is not None:
        state, count = _sparse_background_state(background_points, model, mi, dr, np,
                                                 opacity, shell_levels, subdivision)
        result = mi.VoxelReconstruction.from_state(
            core, model.grid_min, model.grid_max, state, int(model.sh_count), int(model.opacity_sh_count),
            core_resolution=core, background_shell_levels=shell_levels,
            background_subdivision_level=subdivision, background_use_ellipsoid=background_geometry == "ellipsoid")
        configure_world_shell_model(result, shell_levels, background_refinement, subdivision, background_geometry)
        result.background_initialization = dict(
            region="nested_world_cube_shells", initialization="point_cloud_sparse",
            geometry_initialization="native_point_covariance_fit", background_shell_levels=shell_levels,
            background_voxels=count, background_refinement=background_refinement,
            background_subdivision_level=subdivision, world_shell_version=WORLD_SHELL_VERSION,
            cell_index_bits=64, opacity=opacity, color="point_cloud_mean_linear_rgb",
            background_colored_voxels=count, background_color_source_points=len(background_points["positions"]),
            excluded_core_points=background_points["excluded_core_points"],
            outside_root_points=background_points["outside_root_points"],
            seed_voxels=int(model.active_voxel_count), initialized_voxels=int(model.active_voxel_count) + count,
            background_geometry=background_geometry, background_sampling="leaf_traversal", straight_ray_forward=True)
        return result
    if not adaptive:
        cells = background_cells_numpy(core, shell_levels, np)
    lower, upper = leaf_boxes_numpy(cells, core, model.grid_min, model.grid_max, shell_levels, np)
    count = len(cells)
    sh, opacity_sh = int(model.sh_count), int(model.opacity_sh_count)
    background = {
        "cell_indices": cells,
        "centers": np.full((count, 3), .5, dtype=np.float32),
        # An initially enclosing sphere keeps every occupied leaf covered,
        # including its corners. Geometry can shrink and rotate during training.
        "log_scales": np.log((upper - lower) * (math.sqrt(3.) * .5 + 1e-6
                             if background_geometry == "ellipsoid" else .5)).astype(np.float32),
        "rotations": np.tile(np.asarray([1., 0., 0., 0.], dtype=np.float32), (count, 1)),
        "radiance_sh": np.zeros((count, 3 * sh), dtype=np.float32),
        "opacity_sh": np.zeros((count, opacity_sh), dtype=np.float32),
    }
    background["radiance_sh"][:, :3] = colors / .28209479177387814
    background["opacity_sh"][:, 0] = math.log(opacity / (1. - opacity)) / .28209479177387814
    state = {key: _concat_nonempty([mi.UInt64(seed[key]) if key == "cell_indices" else seed[key],
                           (mi.UInt64 if key == "cell_indices" else mi.Float)(value.ravel())], dr)
             for key, value in background.items()}
    dr.eval(*state.values())
    result = mi.VoxelReconstruction.from_state(
        core, model.grid_min, model.grid_max, state, sh, opacity_sh,
        core_resolution=core, background_shell_levels=shell_levels,
        background_subdivision_level=0, background_use_ellipsoid=background_geometry == "ellipsoid",
        **({"background_adaptive": True} if adaptive else {}))
    configure_world_shell_model(result, shell_levels, background_refinement=background_refinement,
                                background_subdivision_level=0, background_geometry=background_geometry)
    result.background_initialization = dict(
        region="nested_world_cube_shells", background_shell_levels=shell_levels,
        background_voxels=count, background_refinement=background_refinement,
        background_subdivision_level=0, world_shell_version=WORLD_SHELL_VERSION, opacity=opacity,
        color="point_cloud_mean_linear_rgb" if point_colored else "black",
        background_colored_voxels=int(np.count_nonzero(point_counts)),
        background_color_source_points=sum(int(value) for value in point_counts),
        empty_leaf_color="black", seed_voxels=int(model.active_voxel_count),
        initialized_voxels=int(model.active_voxel_count) + count,
        background_geometry=background_geometry, background_sampling="leaf_traversal",
        straight_ray_forward=True)
    if adaptive:
        result.background_max_depth = adaptive_background["max_depth"]
        result.background_initialization.update(initialization="camera_sampling_adaptive",
            geometry_initialization="cell_covering_clipped_ellipsoid",
            initial_leaf_budget=adaptive_background["initial_leaf_budget"],
            pixel_threshold=adaptive_background["pixel_threshold"],
            max_depth=adaptive_background["max_depth"], actual_max_depth=adaptive_background["actual_max_depth"],
            depth_histogram=adaptive_background["depth_histogram"],
            camera_sample_count=adaptive_background["camera_sample_count"], cell_index_bits=64,
            empty_leaf_color=[.5, .5, .5])
    return result
