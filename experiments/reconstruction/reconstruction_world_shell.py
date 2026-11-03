"""Nested world-space cube shells in one sparse voxel address space."""

from __future__ import annotations

import math

SHELL_CELLS = 56
MAX_SHELL_LEVELS = 10
WORLD_SHELL_VERSION = 5
ADAPTIVE_TAG = 1 << 63
ADAPTIVE_MAX_DEPTH = 9
REMOVED_BACKGROUND_FIELDS = frozenset({
    "background_layers", "background_final_layers", "background_far_distance",
    "background_far_percentile", "background_far_margin",
    "background_use_ellipsoid", "background_layered_sampling",
})


def reject_removed_background_fields(values):
    removed = REMOVED_BACKGROUND_FIELDS.intersection(values)
    if removed:
        raise ValueError("Compressed background settings are no longer supported: "
                         + ", ".join(sorted(removed))
                         + "; use data.background_shell_levels for world-space shells")


def validate_shell_levels(levels):
    if type(levels) is not int or not 1 <= levels <= MAX_SHELL_LEVELS:
        raise ValueError(f"background_shell_levels must be an integer in [1, {MAX_SHELL_LEVELS}]")
    return levels


def validate_world_shell_layout(core_resolution, levels, grid_min=None, grid_max=None,
                                background_subdivision_level=0):
    validate_shell_levels(levels)
    if (type(core_resolution) is not int or not 2 <= core_resolution <= 1024
            or core_resolution & (core_resolution - 1)):
        raise ValueError("world_shell core resolution must be a power of two in [2, 1024]")
    maximum = min(core_resolution.bit_length() - 2, 9)
    if (type(background_subdivision_level) is not int
            or not 0 <= background_subdivision_level <= maximum):
        raise ValueError(f"background_subdivision_level must be an integer in [0, {maximum}]")
    if core_resolution ** 3 + SHELL_CELLS * levels * 8 ** background_subdivision_level > 2 ** 64:
        raise ValueError("World shell cell addresses exceed the UInt64 address limit")
    depth = levels + core_resolution.bit_length() - 1
    if depth > 21:
        raise ValueError("World shell tree depth exceeds the 64-bit Morton address limit")
    if grid_min is not None or grid_max is not None:
        if (grid_min is None or grid_max is None or len(grid_min) != 3 or len(grid_max) != 3
                or not all(math.isfinite(float(v)) for v in (*grid_min, *grid_max))):
            raise ValueError("World shell bounds must contain three finite coordinates")
        size = [float(hi) - float(lo) for lo, hi in zip(grid_min, grid_max)]
        if min(size) <= 0 or not all(math.isclose(v, size[0], rel_tol=1e-6) for v in size):
            raise ValueError("World shell core bounds must form a cube")
        extent = size[0] * .5 * (2 ** levels)
        if not math.isfinite(extent) or extent >= 3.402823466e38:
            raise ValueError("World shell root bounds exceed float32 coordinates")
    return depth


def world_shell_metadata_layout(metadata):
    """Read old fixed-shell records without changing their refinement policy."""
    mode = metadata.get("background_refinement", "fixed")
    subdivision = metadata.get("background_subdivision_level", 0)
    version = metadata.get("world_shell_version", 1)
    if mode not in ("fixed", "uniform", "adaptive"):
        raise ValueError("background_refinement must be fixed, uniform or adaptive")
    if type(subdivision) is not int or subdivision < 0:
        raise ValueError("background_subdivision_level must be a nonnegative integer")
    if type(version) is not int or version not in (1, 2, 3, 4, WORLD_SHELL_VERSION):
        raise ValueError("Unsupported world_shell_version")
    if mode == "adaptive" and (version != 5 or subdivision != 0):
        raise ValueError("Adaptive background requires world_shell_version 5 and background_subdivision_level 0")
    if version == 1 and (mode != "fixed" or subdivision != 0):
        raise ValueError("Subdivided or uniform background metadata requires world_shell_version 2")
    world_shell_metadata_geometry(metadata)
    return mode, subdivision


def background_initial_subdivision(resolution, core_resolution):
    """First-shell edge is core_extent / resolution, with two cells at S=0."""
    if (type(resolution) is not int or resolution < 2 or resolution > min(core_resolution, 512)
            or resolution & (resolution - 1)):
        raise ValueError("background_initial_resolution must be a power of two in [2, min(initial core resolution, 512)]")
    return resolution.bit_length() - 2


def world_shell_metadata_geometry(metadata):
    """Missing geometry in old world-shell checkpoints means occupied boxes."""
    version = metadata.get("world_shell_version", 1)
    geometry = metadata.get("background_geometry")
    if geometry is None and version in (1, 2):
        return "occupied_cell"
    if geometry not in ("ellipsoid", "occupied_cell"):
        raise ValueError("world_shell_version 3 requires background_geometry=ellipsoid or occupied_cell")
    return geometry


def shell_coordinates_numpy(np):
    return np.asarray([(x, y, z) for z in range(4) for y in range(4) for x in range(4)
                       if not (1 <= x <= 2 and 1 <= y <= 2 and 1 <= z <= 2)], dtype=np.int32)


def background_cells_numpy(core_resolution, levels, np, background_subdivision_level=0):
    validate_world_shell_layout(core_resolution, levels,
                                background_subdivision_level=background_subdivision_level)
    count = SHELL_CELLS * levels * 8 ** background_subdivision_level
    return np.arange(core_resolution ** 3, core_resolution ** 3 + count,
                     dtype=np.uint64)


def adaptive_encode_numpy(coarse, depth, morton, np):
    coarse, depth, morton = np.broadcast_arrays(np.asarray(coarse), np.asarray(depth), np.asarray(morton))
    if (any(value.dtype.kind not in "iu" for value in (coarse, depth, morton))
            or np.any(coarse < 0) or np.any(coarse >= 56 * MAX_SHELL_LEVELS)
            or np.any(depth < 0) or np.any(depth > ADAPTIVE_MAX_DEPTH) or np.any(morton < 0)):
        raise ValueError("Invalid adaptive background address components")
    coarse, depth, morton = (value.astype(np.uint64) for value in (coarse, depth, morton))
    sentinel = np.left_shift(np.uint64(1), 3 * depth)
    if np.any(morton >= sentinel):
        raise ValueError("Adaptive child Morton code exceeds its depth")
    return np.uint64(ADAPTIVE_TAG) | (coarse << np.uint64(32)) | sentinel | morton


def adaptive_decode_numpy(cells, levels, np):
    validate_shell_levels(levels)
    cells = np.asarray(cells)
    if cells.dtype.kind not in "iu" or np.any(cells < 0):
        raise ValueError("Invalid adaptive background cell address")
    cells = cells.astype(np.uint64, copy=False)
    coarse = (cells & np.uint64(ADAPTIVE_TAG - 1)) >> np.uint64(32)
    path = cells & np.uint64(2**32 - 1)
    depth = np.full(cells.shape, -1, np.int32)
    for value in range(ADAPTIVE_MAX_DEPTH + 1):
        sentinel = np.uint64(1 << (3 * value))
        depth[(path >= sentinel) & (path < 2 * sentinel)] = value
    if np.any(cells < ADAPTIVE_TAG) or np.any(coarse >= 56 * levels) or np.any(depth < 0):
        raise ValueError("Invalid adaptive background cell address")
    return coarse, depth, path - np.left_shift(np.uint64(1), 3 * depth.astype(np.uint64))


def adaptive_intervals_numpy(cells, levels, np):
    """Disjoint fixed-depth intervals make duplicate/ancestor overlap exact."""
    coarse, depth, morton = adaptive_decode_numpy(cells, levels, np)
    shift = (3 * (ADAPTIVE_MAX_DEPTH - depth)).astype(np.uint64)
    start = (coarse << np.uint64(3 * ADAPTIVE_MAX_DEPTH)) | np.left_shift(morton, shift)
    return start, start + np.left_shift(np.uint64(1), shift)


def validate_adaptive_cells_numpy(cells, core_resolution, levels, np):
    cells = np.asarray(cells)
    if cells.ndim != 1 or cells.dtype.kind not in "iu" or np.any(cells < 0):
        raise ValueError("Invalid adaptive cell address")
    background = cells >= ADAPTIVE_TAG
    if np.any(cells[~background] >= core_resolution ** 3):
        raise ValueError("Adaptive models cannot contain legacy shell cell addresses")
    if len(np.unique(cells)) != len(cells):
        raise ValueError("cell_indices must be unique")
    start, end = adaptive_intervals_numpy(cells[background], levels, np)
    order = np.argsort(start)
    if np.any(start[order][1:] < end[order][:-1]):
        raise ValueError("Adaptive background leaves contain ancestor/descendant overlap")


def leaf_boxes_numpy(cells, core_resolution, grid_min, grid_max, levels, np,
                     background_subdivision_level=0):
    """Decode foreground and Morton-ordered shell children to world-space AABBs."""
    validate_world_shell_layout(core_resolution, levels, grid_min, grid_max,
                                background_subdivision_level)
    cells = np.asarray(cells)
    children = 8 ** background_subdivision_level
    limit = core_resolution ** 3 + SHELL_CELLS * levels * children
    if cells.ndim != 1 or cells.dtype.kind not in "ui" or np.any(cells < 0):
        raise ValueError("Invalid world shell cell address")
    adaptive = cells >= ADAPTIVE_TAG
    if np.any(cells[~adaptive] >= limit):
        raise ValueError("Invalid world shell cell address")
    lo, hi = np.asarray(grid_min, dtype=np.float64), np.asarray(grid_max, dtype=np.float64)
    lower = np.empty((len(cells), 3), dtype=np.float64)
    upper = np.empty_like(lower)
    foreground = cells < core_resolution ** 3
    ids = cells[foreground]
    coordinates = np.stack((ids % core_resolution, ids // core_resolution % core_resolution,
                            ids // (core_resolution ** 2)), axis=-1)
    edge = (hi - lo) / core_resolution
    lower[foreground] = lo + coordinates * edge
    upper[foreground] = lower[foreground] + edge
    legacy = ~foreground & ~adaptive
    address = cells[legacy].astype(np.int64) - core_resolution ** 3
    coarse, morton = address // children, address % children
    shell = coarse // SHELL_CELLS
    coordinate = shell_coordinates_numpy(np)[coarse % SHELL_CELLS]
    shell_edge = (hi[0] - lo[0]) * .5 * np.exp2(shell)
    child_coordinate = np.zeros((len(address), 3), dtype=np.int64)
    for bit in range(background_subdivision_level):
        for axis in range(3):
            child_coordinate[:, axis] |= ((morton >> (3 * bit + axis)) & 1) << bit
    child_edge = shell_edge / 2 ** background_subdivision_level
    lower[legacy] = ((lo + hi) * .5 + (coordinate - 2) * shell_edge[:, None]
                          + child_coordinate * child_edge[:, None])
    upper[legacy] = lower[legacy] + child_edge[:, None]
    if np.any(adaptive):
        coarse, depth, morton = adaptive_decode_numpy(cells[adaptive], levels, np)
        shell_edge = (hi[0] - lo[0]) * .5 * np.exp2(coarse // 56)
        coordinate = shell_coordinates_numpy(np)[(coarse % 56).astype(np.int64)]
        child_coordinate = np.zeros((len(coarse), 3), np.uint64)
        for bit in range(ADAPTIVE_MAX_DEPTH):
            for axis in range(3):
                child_coordinate[:, axis] |= ((morton >> np.uint64(3 * bit + axis)) & np.uint64(1)) << np.uint64(bit)
        child_edge = shell_edge / np.exp2(depth)
        lower[adaptive] = ((lo + hi) * .5 + (coordinate - 2) * shell_edge[:, None]
                          + child_coordinate * child_edge[:, None])
        upper[adaptive] = lower[adaptive] + child_edge[:, None]
    return lower, upper


def require_world_shell_native(native):
    if any(not hasattr(native, name) for name in (
            "set_world_shells", "native_background_shell_levels", "native_background_subdivision_level",
            "native_background_use_ellipsoid", "set_background_use_ellipsoid")):
        raise RuntimeError("Rebuild this checkout: world-space background ellipsoid support is missing")
    if getattr(native, "cell_index_bits", 0) != 64:
        raise RuntimeError("Rebuild this checkout: 64-bit sparse world-shell addresses are missing")


def configure_world_shell_model(model, levels, background_refinement="fixed",
                                background_subdivision_level=0, background_geometry="ellipsoid"):
    world_shell_metadata_layout(dict(background_refinement=background_refinement,
                                     background_subdivision_level=background_subdivision_level,
                                     world_shell_version=WORLD_SHELL_VERSION,
                                     background_geometry=background_geometry))
    validate_world_shell_layout(int(model.core_resolution), levels, model.grid_min, model.grid_max,
                                background_subdivision_level)
    require_world_shell_native(model)
    adaptive = background_refinement == "adaptive"
    if adaptive and (not hasattr(model, "native_background_adaptive")
                     or getattr(model, "world_shell_version", 0) < 5):
        raise RuntimeError("Rebuild this checkout: adaptive background support is missing")
    if (int(model.native_background_shell_levels()) != levels
            or int(model.native_background_subdivision_level()) != background_subdivision_level
            or (bool(model.native_background_adaptive()) if hasattr(model, "native_background_adaptive") else False) != adaptive):
        if adaptive or hasattr(model, "native_background_adaptive"):
            model.set_world_shells(levels, background_subdivision_level, adaptive)
        else:
            model.set_world_shells(levels, background_subdivision_level)
    if bool(model.native_background_use_ellipsoid()) != (background_geometry == "ellipsoid"):
        model.set_background_use_ellipsoid(background_geometry == "ellipsoid")
    model.scene_mapping = "world_shell"
    model.background_shell_levels = levels
    model.background_refinement = background_refinement
    model.background_subdivision_level = background_subdivision_level
    model.background_geometry = background_geometry
    model.world_shell_version = WORLD_SHELL_VERSION
    return model


def world_shell_ray_maxt(ray, camera, mi, dr):
    """Finite exit distance from the world root, including parallel-ray cases."""
    lower, upper = mi.Point3f(camera["world_shell_root_min"]), mi.Point3f(camera["world_shell_root_max"])
    parallel = ray.d == 0
    safe_direction = dr.select(parallel, 1., ray.d)
    first, second = (lower - ray.o) / safe_direction, (upper - ray.o) / safe_direction
    enter, leave = dr.minimum(first, second), dr.maximum(first, second)
    enter = dr.select(parallel, float("-inf"), enter)
    leave = dr.select(parallel, float("inf"), leave)
    outside = dr.any(parallel & ((ray.o < lower) | (ray.o > upper)))
    begin, end = dr.maximum(dr.max(enter), 0.), dr.min(leave)
    return dr.select(~outside & (end >= begin), dr.maximum(end, 0.), 0.)
