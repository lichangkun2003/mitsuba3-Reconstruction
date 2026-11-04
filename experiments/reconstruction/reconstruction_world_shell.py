"""Nested world-space cube shells in one sparse voxel address space."""

from __future__ import annotations

import math

SHELL_CELLS = 56
MAX_SHELL_LEVELS = 10
WORLD_SHELL_VERSION = 8
CORE_ADAPTIVE_TAG = 1 << 61
CORE_ADAPTIVE_MAX_DEPTH = 10
ADAPTIVE_TAG = 1 << 63
ADAPTIVE_EXTENDED_TAG = 1 << 62
ADAPTIVE_DEEPEST_TAG = 0xF << 60
ADAPTIVE_PATH_BITS = 52
ADAPTIVE_LEGACY_MAX_DEPTH = 9
ADAPTIVE_MAX_DEPTH = 18
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
    if type(version) is not int or version not in (1, 2, 3, 4, 5, 6, 7, WORLD_SHELL_VERSION):
        raise ValueError("Unsupported world_shell_version")
    if mode == "adaptive" and (version not in (5, 6, 7, 8) or subdivision != 0):
        raise ValueError("Adaptive background requires world_shell_version >=5 and background_subdivision_level 0")
    foreground = metadata.get("foreground_refinement", "uniform")
    if foreground not in ("uniform", "adaptive"):
        raise ValueError("foreground_refinement must be uniform or adaptive")
    if foreground == "adaptive":
        if version < 8 or mode != "adaptive":
            raise ValueError("Adaptive foreground requires world_shell_version 8 and adaptive background")
        foreground_adaptive_fields(metadata)
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


def core_adaptive_encode_numpy(depth, morton, np):
    """Self-describing core leaves; depth is relative to the complete core box."""
    depth, morton = np.broadcast_arrays(np.asarray(depth), np.asarray(morton))
    if (depth.dtype.kind not in "iu" or morton.dtype.kind not in "iu"
            or np.any(depth < 1) or np.any(depth > CORE_ADAPTIVE_MAX_DEPTH)
            or np.any(morton < 0)):
        raise ValueError("Invalid adaptive foreground address components")
    depth, morton = depth.astype(np.uint64), morton.astype(np.uint64)
    sentinel = np.left_shift(np.uint64(1), 3 * depth)
    if np.any(morton >= sentinel):
        raise ValueError("Adaptive foreground Morton code exceeds its depth")
    return np.uint64(CORE_ADAPTIVE_TAG) | sentinel | morton


def core_adaptive_decode_numpy(cells, np):
    cells = np.asarray(cells)
    if cells.dtype.kind not in "iu" or np.any(cells < 0):
        raise ValueError("Invalid adaptive foreground cell address")
    cells = cells.astype(np.uint64, copy=False)
    path = cells & np.uint64((1 << 31) - 1)
    depth = np.full(cells.shape, -1, np.int32)
    for value in range(1, CORE_ADAPTIVE_MAX_DEPTH + 1):
        sentinel = np.uint64(1 << (3 * value))
        depth[(path >= sentinel) & (path < 2 * sentinel)] = value
    if (np.any((cells & np.uint64(~((1 << 31) - 1) & ((1 << 64) - 1)))
               != np.uint64(CORE_ADAPTIVE_TAG)) or np.any(depth < 0)):
        raise ValueError("Invalid adaptive foreground cell address")
    return depth, path - np.left_shift(np.uint64(1), 3 * depth.astype(np.uint64))


def core_cells_to_adaptive_numpy(cells, resolution, np):
    """Convert legacy flat core addresses without changing their world boxes."""
    if type(resolution) is not int or not 2 <= resolution <= 1024 or resolution & (resolution - 1):
        raise ValueError("Adaptive foreground resolution must be a power of two in [2, 1024]")
    cells = np.asarray(cells)
    if cells.dtype.kind not in "iu" or np.any(cells < 0) or np.any(cells >= resolution ** 3):
        raise ValueError("Expected legacy foreground cells")
    cells = cells.astype(np.uint64, copy=False)
    xyz = (cells % resolution, cells // resolution % resolution, cells // (resolution ** 2))
    morton = np.zeros(cells.shape, np.uint64)
    depth = resolution.bit_length() - 1
    for bit in range(depth):
        for axis in range(3):
            morton |= ((xyz[axis] >> np.uint64(bit)) & np.uint64(1)) << np.uint64(3 * bit + axis)
    return core_adaptive_encode_numpy(depth, morton, np)


def core_leaf_depths_numpy(cells, core_resolution, np):
    """Decode mixed legacy/tagged core leaves to depths and Morton paths."""
    cells = np.asarray(cells, np.uint64)
    tagged = (cells & np.uint64(CORE_ADAPTIVE_TAG)) != 0
    encoded = cells.copy()
    encoded[~tagged] = core_cells_to_adaptive_numpy(cells[~tagged], core_resolution, np)
    return core_adaptive_decode_numpy(encoded, np)


def foreground_adaptive_fields(metadata):
    if metadata.get("foreground_refinement", "uniform") != "adaptive":
        return {}
    maximum = metadata.get("foreground_max_resolution", 1024)
    if type(maximum) is not int or not 2 <= maximum <= 1024 or maximum & (maximum - 1):
        raise ValueError("foreground_max_resolution must be a power of two in [2, 1024]")
    return dict(foreground_refinement="adaptive", foreground_max_resolution=maximum)


def adaptive_encode_numpy(coarse, depth, morton, np):
    # Depth 18 uses a reserved prefix and implicit shell/depth so all 54
    # Morton bits fit in UInt64 without changing any older leaf address.
    coarse, depth, morton = np.broadcast_arrays(np.asarray(coarse), np.asarray(depth), np.asarray(morton))
    if (any(value.dtype.kind not in "iu" for value in (coarse, depth, morton))
            or np.any(coarse < 0) or np.any(coarse >= 56 * MAX_SHELL_LEVELS)
            or np.any(depth < 0) or np.any(depth > ADAPTIVE_MAX_DEPTH) or np.any(morton < 0)):
        raise ValueError("Invalid adaptive background address components")
    coarse, depth, morton = (value.astype(np.uint64) for value in (coarse, depth, morton))
    sentinel = np.left_shift(np.uint64(1), 3 * depth)
    if np.any(morton >= sentinel):
        raise ValueError("Adaptive child Morton code exceeds its depth")
    deepest = depth == 18
    if np.any(deepest & (coarse < 504)):
        raise ValueError("Adaptive depth 18 is only valid in the outermost of ten shells")
    extended = depth > ADAPTIVE_LEGACY_MAX_DEPTH
    shift = np.where(extended, ADAPTIVE_PATH_BITS, 32).astype(np.uint64)
    tag = np.where(extended, np.uint64(ADAPTIVE_TAG | ADAPTIVE_EXTENDED_TAG), np.uint64(ADAPTIVE_TAG))
    result = tag | np.left_shift(coarse, shift) | sentinel | morton
    deepest_root = np.where(deepest, coarse, np.uint64(504)) - np.uint64(504)
    return np.where(deepest, np.uint64(ADAPTIVE_DEEPEST_TAG) | (deepest_root << np.uint64(54)) | morton, result)


def adaptive_decode_numpy(cells, levels, np):
    validate_shell_levels(levels)
    cells = np.asarray(cells)
    if cells.dtype.kind not in "iu" or np.any(cells < 0):
        raise ValueError("Invalid adaptive background cell address")
    cells = cells.astype(np.uint64, copy=False)
    deepest = (cells >> np.uint64(60)) == 15
    extended = (cells & np.uint64(ADAPTIVE_EXTENDED_TAG)) != 0
    coarse = np.where(extended, (cells >> np.uint64(ADAPTIVE_PATH_BITS)) & np.uint64(1023),
                      (cells & np.uint64(ADAPTIVE_TAG - 1)) >> np.uint64(32))
    path = cells & np.where(extended, np.uint64((1 << ADAPTIVE_PATH_BITS)-1), np.uint64(2**32-1))
    coarse = np.where(deepest, np.uint64(504) + ((cells >> np.uint64(54)) & np.uint64(63)), coarse)
    path = np.where(deepest, np.uint64(1 << 54) | (cells & np.uint64((1 << 54)-1)), path)
    depth = np.full(cells.shape, -1, np.int32)
    for value in range(ADAPTIVE_MAX_DEPTH + 1):
        sentinel = np.uint64(1 << (3 * value))
        depth[(path >= sentinel) & (path < 2 * sentinel)] = value
    if (np.any(cells < ADAPTIVE_TAG) or np.any(coarse >= 56 * levels) or np.any(depth < 0)
            or np.any(extended != (depth > ADAPTIVE_LEGACY_MAX_DEPTH))
            or np.any(deepest != (depth == 18))):
        raise ValueError("Invalid adaptive background cell address")
    return coarse, depth, path - np.left_shift(np.uint64(1), 3 * depth.astype(np.uint64))


def adaptive_depth_limit(metadata, shell):
    """Maximum local depth; core/core_parent align every shell in world space.

    ``shell`` is zero-based and may also be a NumPy/Dr.Jit array. Old saved
    models without a resolution mode keep their original per-shell cap.
    """
    mode = metadata.get("background_resolution_mode", "per_shell")
    if mode == "per_shell":
        depth = metadata.get("background_max_depth", ADAPTIVE_LEGACY_MAX_DEPTH)
        if type(depth) is not int or not 0 <= depth <= ADAPTIVE_LEGACY_MAX_DEPTH:
            raise ValueError("Legacy background_max_depth must be in [0, 9]")
        return depth
    if mode not in ("core_parent", "core"):
        raise ValueError("background_resolution_mode must be per_shell, core_parent or core")
    core = (metadata.get("foreground_max_resolution", 1024)
            if metadata.get("foreground_refinement") == "adaptive" else
            metadata.get("core_resolution", metadata.get("resolution")))
    maximum = metadata.get("background_max_resolution", 1024 if mode == "core" else 512)
    divisor = 1 if mode == "core" else 2
    if (type(core) is not int or not 2 * divisor <= core <= 1024 or core & (core-1)
            or type(maximum) is not int or not 2 <= maximum <= 1024 or maximum & (maximum-1)):
        raise ValueError(f"{mode} requires power-of-two core >={2 * divisor} and background_max_resolution in [2, 1024]")
    return min(maximum, core // divisor).bit_length() - 2 + shell


def adaptive_max_depth(metadata):
    return int(adaptive_depth_limit(metadata, int(metadata["background_shell_levels"]) - 1))


def adaptive_limit_fields(metadata):
    """Canonical policy metadata, recomputed after foreground refinement."""
    result = {"background_max_depth": adaptive_max_depth(metadata), **foreground_adaptive_fields(metadata)}
    mode = metadata.get("background_resolution_mode", "per_shell")
    if mode in ("core_parent", "core"):
        result.update(background_resolution_mode=mode,
                      background_max_resolution=metadata.get("background_max_resolution", 1024 if mode == "core" else 512))
    return result


def model_adaptive_limit_fields(model, metadata):
    mode = getattr(model, "background_resolution_mode", "per_shell")
    return adaptive_limit_fields(dict(metadata,
        foreground_refinement=getattr(model, "foreground_refinement", metadata.get("foreground_refinement", "uniform")),
        foreground_max_resolution=int(getattr(model, "foreground_max_resolution", metadata.get("foreground_max_resolution", 1024))),
        background_resolution_mode=mode,
        background_max_resolution=int(getattr(model, "background_max_resolution", 1024 if mode == "core" else 512)),
        background_max_depth=int(getattr(model, "background_max_depth", 9))))


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
    tagged_core = ~background & ((cells.astype(np.uint64) & np.uint64(CORE_ADAPTIVE_TAG)) != 0)
    if np.any(cells[~background & ~tagged_core] >= core_resolution ** 3):
        raise ValueError("Adaptive models cannot contain legacy shell cell addresses")
    coarse, depth, _ = adaptive_decode_numpy(cells[background], levels, np)
    if np.any(depth > 9 + coarse // 56):
        raise ValueError("Adaptive leaf is smaller than the world-tree base cell")
    if len(np.unique(cells)) != len(cells):
        raise ValueError("cell_indices must be unique")
    core_depth, core_morton = core_leaf_depths_numpy(cells[~background], core_resolution, np)
    core_shift = (3 * (CORE_ADAPTIVE_MAX_DEPTH - core_depth)).astype(np.uint64)
    core_start = np.left_shift(core_morton, core_shift)
    core_end = core_start + np.left_shift(np.uint64(1), core_shift)
    core_order = np.argsort(core_start)
    if np.any(core_start[core_order][1:] < np.maximum.accumulate(core_end[core_order])[:-1]):
        raise ValueError("Adaptive foreground leaves contain ancestor/descendant overlap")
    start, end = adaptive_intervals_numpy(cells[background], levels, np)
    order = np.argsort(start)
    if np.any(start[order][1:] < np.maximum.accumulate(end[order])[:-1]):
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
    core_adaptive = ~adaptive & ((cells.astype(np.uint64) & np.uint64(CORE_ADAPTIVE_TAG)) != 0)
    if np.any(cells[~adaptive & ~core_adaptive] >= limit):
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
    if np.any(core_adaptive):
        depth, morton = core_adaptive_decode_numpy(cells[core_adaptive], np)
        xyz = np.zeros((len(depth), 3), np.uint64)
        for bit in range(CORE_ADAPTIVE_MAX_DEPTH):
            for axis in range(3):
                xyz[:, axis] |= ((morton >> np.uint64(3 * bit + axis)) & np.uint64(1)) << np.uint64(bit)
        edge = (hi - lo) / np.exp2(depth)[:, None]
        lower[core_adaptive] = lo + xyz * edge
        upper[core_adaptive] = lower[core_adaptive] + edge
    legacy = ~foreground & ~adaptive & ~core_adaptive
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
        if np.any(depth > 9 + coarse // 56):
            raise ValueError("Adaptive leaf is smaller than the world-tree base cell")
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
