"""Bounded camera-driven preparation and selective world-shell refinement.

Preparation has a CPU oracle. Device refinement fits bounded batches with the
same FP64 core fitter and keeps all geometry, appearance and Adam row maps on GPU.
"""

from __future__ import annotations

import math

from reconstruction_world_shell import (
    ADAPTIVE_TAG, ADAPTIVE_MAX_DEPTH, ADAPTIVE_LEGACY_MAX_DEPTH,
    ADAPTIVE_EXTENDED_TAG, ADAPTIVE_DEEPEST_TAG, ADAPTIVE_PATH_BITS, WORLD_SHELL_VERSION, adaptive_decode_numpy,
    adaptive_encode_numpy, leaf_boxes_numpy, shell_coordinates_numpy,
    validate_adaptive_cells_numpy, validate_world_shell_layout, adaptive_depth_limit, adaptive_max_depth,
)


def prepare_projection_cameras(views, np, max_views=None):
    if max_views is not None and (type(max_views) is not int or max_views < 1):
        raise ValueError("camera_max_views must be a positive integer")
    cameras = [view.get("camera", view) for view in views]
    if not cameras:
        raise ValueError("Adaptive background preparation requires training cameras")
    indices = (np.arange(len(cameras)) if max_views is None else
               np.unique(np.linspace(0, len(cameras)-1, min(max_views, len(cameras))).astype(np.int64)))
    result = []
    for index in indices:
        camera = cameras[int(index)]
        width, height = int(camera["width"]), int(camera["height"])
        fx = float(camera.get("fx", width / (2 * math.tan(float(camera.get("camera_angle_x", .7)) / 2))))
        result.append(dict(position=np.asarray(camera["position"], np.float64),
            right=np.asarray(camera["right"], np.float64), up=np.asarray(camera["up"], np.float64),
            forward=np.asarray(camera["forward"], np.float64), fx=fx, fy=float(camera.get("fy", fx)),
            cx=float(camera.get("cx", width*.5)), cy=float(camera.get("cy", height*.5)),
            width=width, height=height, near_clip=max(float(camera.get("near_clip", 1e-4)), 1e-8)))
    return result


def projected_boxes_pixels(lower, upper, cameras, np, *, chunk_size=32768):
    """Maximum projected AABB edge over every supplied training camera."""
    scores = np.zeros(len(lower), np.float32)
    for start in range(0, len(lower), chunk_size):
        low, high = lower[start:start+chunk_size], upper[start:start+chunk_size]
        center, half = (low+high)*.5, (high-low)*.5
        current = np.zeros(len(low), np.float64)
        for camera in cameras:
            relative = center-camera["position"]
            x, y, z = (relative @ camera[name] for name in ("right", "up", "forward"))
            min_x = np.full(len(low), np.inf)
            min_y = min_x.copy()
            max_x, max_y = -min_x, -min_x.copy()
            min_z, max_z = min_x.copy(), -min_x.copy()
            for octant in range(8):
                offset = half * np.array([1 if octant & (1 << axis) else -1 for axis in range(3)])
                depth = z + offset @ camera["forward"]
                px = camera["cx"] + camera["fx"] * (x+offset @ camera["right"]) / np.maximum(depth, camera["near_clip"])
                py = camera["cy"] - camera["fy"] * (y+offset @ camera["up"]) / np.maximum(depth, camera["near_clip"])
                min_x, max_x = np.minimum(min_x, px), np.maximum(max_x, px)
                min_y, max_y = np.minimum(min_y, py), np.maximum(max_y, py)
                min_z, max_z = np.minimum(min_z, depth), np.maximum(max_z, depth)
            visible = ((max_z > camera["near_clip"]) & (max_x >= 0) & (min_x < camera["width"])
                       & (max_y >= 0) & (min_y < camera["height"]))
            span = np.maximum(max_x-min_x, max_y-min_y)
            # A leaf crossing the camera plane is conservatively retained;
            # finite image dimensions cap its ranking and avoid infinity.
            span = np.minimum(span, 4 * max(camera["width"], camera["height"]))
            current = np.maximum(current, np.where(visible, span, 0.))
        scores[start:start+len(low)] = current
    return scores


def projected_leaf_pixels(cells, metadata, cameras, np, *, chunk_size=32768, projector=None):
    cells = np.asarray(cells, np.uint64)
    scores = np.zeros(len(cells), np.float32)
    for start in range(0, len(cells), chunk_size):
        low, high = leaf_boxes_numpy(cells[start:start+chunk_size], metadata["resolution"], metadata["grid_min"],
            metadata["grid_max"], metadata["background_shell_levels"], np)
        scores[start:start+len(low)] = (projected_boxes_pixels(low,high,cameras,np,chunk_size=chunk_size)
                                      if projector is None else projector(low,high,cameras))
    return scores


def _projection_signature(metadata,cameras):
    # Adaptive IDs and boxes remain unchanged when only the foreground refines.
    return (tuple(metadata["grid_min"]),tuple(metadata["grid_max"]),metadata["background_shell_levels"],
        tuple(tuple(float(v) for name in ("position","right","up","forward") for v in camera[name])
              + tuple(camera[name] for name in ("fx","fy","cx","cy","width","height","near_clip"))
              for camera in cameras))


def cached_leaf_projection(cells,metadata,cameras,np,cache=None,*,projector=None):
    if cache is None:
        return projected_leaf_pixels(cells,metadata,cameras,np,projector=projector)
    signature = _projection_signature(metadata,cameras)
    if cache.get("signature") != signature:
        cache.clear()
        cache.update(signature=signature,cells=np.empty(0,np.uint64),pixels=np.empty(0,np.float32))
    cells = np.asarray(cells,np.uint64)
    keys,values = cache["cells"],cache["pixels"]
    index = np.searchsorted(keys,cells)
    present = index < len(keys)
    present[present] &= keys[index[present]] == cells[present]
    result = np.empty(len(cells),np.float32)
    result[present] = values[index[present]]
    missing, inverse = np.unique(cells[~present],return_inverse=True)
    if len(missing):
        pixels = projected_leaf_pixels(missing,metadata,cameras,np,projector=projector)
        result[~present] = pixels[inverse]
        merged = np.r_[keys,missing]
        order = np.argsort(merged)
        cache["cells"],cache["pixels"] = merged[order],np.r_[values,pixels][order]
    return result


def prune_projection_cache(cache,active_cells,np):
    if cache is None or not len(cache.get("cells",())):
        return
    active = np.sort(np.asarray(active_cells,np.uint64))
    index = np.searchsorted(active,cache["cells"])
    present = index < len(active)
    present[present] &= active[index[present]] == cache["cells"][present]
    if not np.all(present):
        cache["cells"],cache["pixels"] = cache["cells"][present],cache["pixels"][present]


def projected_ellipsoid_pixels(state,metadata,cameras,np,*,projector=None):
    from reconstruction_topology import _quaternion_matrix
    low,high = leaf_boxes_numpy(state["cell_indices"],metadata["resolution"],metadata["grid_min"],
        metadata["grid_max"],metadata["background_shell_levels"],np)
    centers = low+(high-low)*np.asarray(state["centers"]).reshape(-1,3)
    axes = np.exp(np.clip(np.asarray(state["log_scales"],np.float64).reshape(-1,3),-30.,30.))
    rotation,_ = _quaternion_matrix(state["rotations"],np)
    extent = np.sqrt(np.sum((rotation*axes[:,None,:])**2,axis=2))
    lower,upper = np.maximum(low,centers-extent),np.minimum(high,centers+extent)
    valid = np.all(upper > lower,axis=1)
    scores = np.zeros(len(low),np.float32)
    scores[valid] = (projected_boxes_pixels(lower[valid],upper[valid],cameras,np)
                     if projector is None else projector(lower[valid],upper[valid],cameras))
    return scores


def _children(cells, levels, np):
    coarse, depth, morton = adaptive_decode_numpy(cells, levels, np)
    if np.any(depth >= ADAPTIVE_MAX_DEPTH):
        raise ValueError("Adaptive background has reached the maximum supported depth")
    return adaptive_encode_numpy(np.repeat(coarse, 8), np.repeat(depth+1, 8),
        np.repeat(morton*8, 8)+np.tile(np.arange(8, dtype=np.uint64), len(cells)), np)


def _point_colors(cells, positions, colors, metadata, np):
    """Assign points to existing mixed-depth leaves; points never create leaves."""
    from reconstruction_color import srgb_to_linear
    positions, colors = np.asarray(positions), np.asarray(colors)
    if (positions.ndim != 2 or positions.shape[1:] != (3,) or colors.shape != positions.shape
            or not np.isfinite(positions).all() or not np.isfinite(colors).all()
            or np.any(colors < 0) or np.any(colors > 1)):
        raise ValueError("Adaptive source points require finite (N,3) positions and sRGB colors in [0,1]")
    coarse, depths, _ = adaptive_decode_numpy(cells, metadata["background_shell_levels"], np)
    sorted_rows = np.argsort(cells)
    sorted_cells = cells[sorted_rows]
    sums, counts = np.zeros((len(cells), 3), np.float64), np.zeros(len(cells), np.int64)
    low, high = np.asarray(metadata["grid_min"], float), np.asarray(metadata["grid_max"], float)
    center, radius = (low+high)*.5, (high[0]-low[0])*.5
    coordinates = shell_coordinates_numpy(np)
    ordinal = np.full(64, -1, np.int32)
    ordinal[coordinates @ np.array([1, 4, 16])] = np.arange(56)
    for start in range(0, len(positions), 262144):
        points, rgb = positions[start:start+262144], colors[start:start+262144]
        remaining = ~np.all((points >= low) & (points < high), axis=1)
        for shell in range(metadata["background_shell_levels"]):
            edge = radius * 2**shell
            selected = remaining & np.all((points >= center-2*edge) & (points < center+2*edge), axis=1)
            remaining[selected] = False
            if not selected.any():
                continue
            pts = points[selected]
            xyz = np.stack([np.searchsorted(center[a] + (np.arange(5)-2)*edge, pts[:, a], side="right")-1
                            for a in range(3)], axis=1)
            code = shell*56 + ordinal[xyz @ np.array([1, 4, 16])]
            fractional = (pts - (center + (xyz-2)*edge)) / edge
            fractional = np.clip(fractional, 0., np.nextafter(1., 0.))
            linear = srgb_to_linear(rgb[selected], np)
            matched = np.zeros(len(pts), bool)
            for depth in np.unique(depths[coarse//56 == shell]):
                child = np.floor(fractional*(2**int(depth))).astype(np.uint64)
                morton = np.zeros(len(pts), np.uint64)
                for bit in range(int(depth)):
                    for axis in range(3):
                        morton |= ((child[:, axis] >> np.uint64(bit)) & np.uint64(1)) << np.uint64(3*bit+axis)
                candidate = adaptive_encode_numpy(code, np.full(len(code), depth, np.int32), morton, np)
                index = np.searchsorted(sorted_cells, candidate)
                valid = (index < len(cells)) & ~matched
                valid[valid] &= sorted_cells[index[valid]] == candidate[valid]
                if not valid.any():
                    continue
                rows = sorted_rows[index[valid]]
                counts += np.bincount(rows, minlength=len(cells))
                for axis in range(3):
                    sums[:, axis] += np.bincount(rows, weights=linear[valid, axis], minlength=len(cells))
                matched |= valid
    # A strictly positive neutral seed keeps the native radiance clamp away
    # from its zero derivative on leaves without source points.
    means = np.full((len(cells), 3), .5, np.float32)
    occupied = counts > 0
    means[occupied] = sums[occupied] / counts[occupied, None]
    return means, counts


def prepare_adaptive_background(positions, colors, core_resolution, grid_min, grid_max,
                                shell_levels, views, np, *, initial_leaf_budget=200000,
                                pixel_threshold=32., max_depth=8, camera_max_views=None,
                                resolution_mode="per_shell", max_resolution=1024):
    validate_world_shell_layout(core_resolution, shell_levels, grid_min, grid_max)
    if (type(initial_leaf_budget) is not int or initial_leaf_budget < 56*shell_levels
            or type(max_depth) is not int or not 0 <= max_depth <= ADAPTIVE_MAX_DEPTH
            or not math.isfinite(pixel_threshold) or pixel_threshold <= 0):
        raise ValueError("Adaptive initial leaf budget, pixel threshold or maximum depth is invalid")
    metadata = dict(resolution=core_resolution, core_resolution=core_resolution,
        grid_min=list(grid_min), grid_max=list(grid_max), background_shell_levels=shell_levels,
        background_refinement="adaptive", background_subdivision_level=0,
        background_geometry="ellipsoid", background_max_depth=max_depth,
        world_shell_version=WORLD_SHELL_VERSION, scene_mapping="world_shell")
    if resolution_mode not in ("per_shell", "core_parent", "core"):
        raise ValueError("Unsupported adaptive background resolution mode")
    if resolution_mode != "per_shell":
        metadata.update(background_resolution_mode=resolution_mode,
                        background_max_resolution=max_resolution)
        max_depth = adaptive_max_depth(metadata)
        metadata["background_max_depth"] = max_depth
    cameras = prepare_projection_cameras(views, np, camera_max_views)
    cells = adaptive_encode_numpy(np.arange(56*shell_levels), 0, 0, np)
    for _ in range(max_depth):
        coarse, depth, _ = adaptive_decode_numpy(cells, shell_levels, np)
        scores = projected_leaf_pixels(cells, metadata, cameras, np)
        candidates = np.flatnonzero((depth < adaptive_depth_limit(metadata, coarse//56))
                                    & (scores > pixel_threshold))
        budget = (initial_leaf_budget-len(cells))//7
        if not len(candidates) or budget <= 0:
            break
        ranking = np.lexsort((cells[candidates], -scores[candidates]))
        chosen = candidates[ranking[:budget]]
        keep = np.ones(len(cells), bool)
        keep[chosen] = False
        cells = np.concatenate((cells[keep], _children(cells[chosen], shell_levels, np)))
    cells.sort()
    validate_adaptive_cells_numpy(cells, core_resolution, shell_levels, np)
    means, counts = _point_colors(cells, positions, colors, metadata, np)
    _, depths, _ = adaptive_decode_numpy(cells, shell_levels, np)
    return dict(cell_indices=cells, mean_linear_colors=means, point_counts=counts,
        core_resolution=core_resolution, grid_min=list(grid_min), grid_max=list(grid_max),
        shell_levels=shell_levels, background_refinement="adaptive", background_subdivision_level=0,
        world_shell_version=WORLD_SHELL_VERSION, initial_leaf_budget=initial_leaf_budget, pixel_threshold=float(pixel_threshold),
        background_resolution_mode=resolution_mode,
        background_max_resolution=max_resolution if resolution_mode != "per_shell" else 1 << (max_depth+1),
        max_depth=max_depth, background_max_depth=max_depth,
        actual_max_depth=int(depths.max()), camera_sample_count=len(cameras),
        depth_histogram={str(int(value)): int(np.count_nonzero(depths == value)) for value in np.unique(depths)})


def select_adaptive_background(checkpoint, cameras, signals, np, *, pixel_threshold=8.,
                                max_new_leaves=100000, max_depth=8, projection_cache=None, projector=None):
    metadata, state = checkpoint["model_metadata"], checkpoint["model_state"]
    if metadata.get("background_refinement") != "adaptive":
        raise ValueError("Adaptive selection requires adaptive model metadata")
    if max_new_leaves < 0 or not 0 <= max_depth <= ADAPTIVE_MAX_DEPTH or pixel_threshold <= 0:
        raise ValueError("Invalid adaptive refinement budget")
    cells = np.asarray(state["cell_indices"], np.uint64)
    rows = np.flatnonzero(cells >= ADAPTIVE_TAG)
    if not len(rows) or max_new_leaves < 7:
        return np.empty(0, np.uint32)
    coarse, depth, _ = adaptive_decode_numpy(cells[rows], metadata["background_shell_levels"], np)
    values = signals.get("adaptive_priority", 0.) if isinstance(signals, dict) else signals
    values = np.asarray(values, np.float64)
    if values.shape != (len(cells),) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Adaptive priorities must contain one finite nonnegative scalar per active row")
    limits = adaptive_depth_limit(metadata, coarse//56)
    if metadata.get("background_resolution_mode", "per_shell") == "per_shell":
        limits = np.minimum(limits, max_depth)
    eligible = (depth < limits) & (values[rows] > 0)
    prepared_cameras = prepare_projection_cameras(cameras,np)
    projection = np.zeros(len(rows),np.float32)
    projection[eligible] = cached_leaf_projection(cells[rows[eligible]], metadata, prepared_cameras, np,
                                                projection_cache,projector=projector)
    valid = eligible & (projection > pixel_threshold)
    candidates = np.flatnonzero(valid)
    # Optimization evidence and camera undersampling jointly rank parents.
    score = projection[candidates] * np.sqrt(values[rows[candidates]])
    order = np.lexsort((cells[rows[candidates]], -score))
    pool = candidates[order[:4*(max_new_leaves//7)]]
    if all(name in state for name in ("centers", "log_scales", "rotations")) and len(pool):
        subset = {name: np.asarray(state[name]).reshape(len(cells), width)[rows[pool]].ravel()
                  for name, width in (("cell_indices",1),("centers",3),("log_scales",3),("rotations",4))}
        coverage = adaptive_child_coverage_counts(subset, metadata, np)
        # Learned ellipsoids can be much smaller than their static leaves.
        # Recheck their clipped AABB every round; this geometry is never cached.
        actual_projection = projected_ellipsoid_pixels(subset,metadata,prepared_cameras,np,projector=projector)
        adjusted = actual_projection * np.sqrt(values[rows[pool]]) / np.maximum(coverage.astype(np.int32)-1,1)
        # Zero-coverage parents cannot yield fitted children and are retained.
        useful = (coverage > 0) & (actual_projection > pixel_threshold)
        pool, adjusted = pool[useful], adjusted[useful]
        pool = pool[np.lexsort((cells[rows[pool]], -adjusted))]
    return rows[pool[:max_new_leaves//7]].astype(np.uint32)


def adaptive_child_coverage_counts(state, metadata, np):
    """Exact ellipsoid/child-AABB contact, including face/edge/corner minima."""
    from reconstruction_topology import _quaternion_matrix, _closest_box
    cells = np.asarray(state["cell_indices"], np.uint64)
    low, high = leaf_boxes_numpy(cells, metadata["resolution"], metadata["grid_min"],
        metadata["grid_max"], metadata["background_shell_levels"], np)
    edge = high-low
    centers = np.asarray(state["centers"], np.float64).reshape(-1,3)
    axes = np.exp(np.clip(np.asarray(state["log_scales"],np.float64).reshape(-1,3),-30.,30.))
    rotation, _ = _quaternion_matrix(state["rotations"],np)
    transform = rotation/axes[:,None,:]
    metric = np.einsum("nik,njk->nij",transform,transform)*edge[:,:,None]*edge[:,None,:]
    inverse = np.einsum("nik,njk->nij",rotation*axes[:,None,:],rotation*axes[:,None,:])/edge[:,:,None]/edge[:,None,:]
    result = np.zeros(len(cells),np.uint8)
    for octant in range(8):
        lower = .5*np.array([(octant>>axis)&1 for axis in range(3)])-centers
        value, _ = _closest_box(metric,lower,lower+.5,np,inverse)
        result += value <= 1.+1e-9
    return result


def select_adaptive_background_device(checkpoint, cameras, signals, dr, *, pixel_threshold=8.,
                                       max_new_leaves=100000, max_depth=8, projection_cache=None):
    import numpy as np
    from reconstruction_adaptive_projection_gpu import make_adaptive_projector
    projector = make_adaptive_projector(dr,np)
    U, C, F = dr.cuda.UInt32, dr.cuda.UInt64, dr.cuda.Float32
    cells = checkpoint["model_state"]["cell_indices"]
    rows = dr.compress(C(cells) >= ADAPTIVE_TAG)
    selected_cells = dr.gather(C, C(cells), rows)
    source = signals.get("adaptive_priority", dr.zeros(F, len(cells))) if isinstance(signals, dict) else signals
    # Native loss statistics use mi.Float (cuda.ad.Float), while selection is
    # detached topology work. Gather requires its source to match the dtype.
    priority = dr.gather(F, F(source), rows)
    dr.eval(rows, selected_cells, priority)
    # Transfer addresses and one accumulated scalar only, never the full SH or geometry.
    light = dict(model_metadata=checkpoint["model_metadata"],
                 model_state={"cell_indices": np.asarray(selected_cells)})
    prune_projection_cache(projection_cache,light["model_state"]["cell_indices"],np)
    pool = select_adaptive_background(light, cameras, np.asarray(priority), np,
        pixel_threshold=pixel_threshold, max_new_leaves=4*max_new_leaves, max_depth=max_depth,
        projection_cache=projection_cache,projector=projector)
    parent_rows = dr.gather(U, rows, U(pool))
    geometry = {"cell_indices": np.asarray(selected_cells)[pool]}
    from reconstruction_topology_gpu import take_rows
    for name,width in (("centers",3),("log_scales",3),("rotations",4)):
        values = take_rows(checkpoint["model_state"][name],parent_rows,width)
        dr.eval(values)
        geometry[name] = np.asarray(values)
    chosen = select_adaptive_background(dict(model_metadata=checkpoint["model_metadata"],model_state=geometry),
        cameras,np.asarray(priority)[pool],np,pixel_threshold=pixel_threshold,
        max_new_leaves=max_new_leaves,max_depth=max_depth,projection_cache=projection_cache,projector=projector)
    return dr.gather(U,parent_rows,U(chosen))


def refine_adaptive_background_numpy(checkpoint, parent_rows, config, max_active_voxels, fit_core, np):
    from reconstruction_topology import NoRepresentableChildren
    from reconstruction_world_refinement import (_shell_grid_cells_numpy, _shell_grid_metadata,
                                                 _shell_world_cells_numpy)
    metadata, state = checkpoint["model_metadata"], checkpoint["model_state"]
    if metadata.get("background_refinement") != "adaptive":
        raise ValueError("Adaptive refinement requires adaptive model metadata")
    cells, rows = np.asarray(state["cell_indices"], np.uint64), np.asarray(parent_rows)
    if (rows.ndim != 1 or rows.dtype.kind not in "iu" or np.any(rows < 0)
            or np.any(rows >= len(cells)) or len(np.unique(rows)) != len(rows)):
        raise ValueError("Invalid adaptive parent row selection")
    rows = rows.astype(np.int64)
    coarse, depth, morton = adaptive_decode_numpy(cells[rows], metadata["background_shell_levels"], np)
    if np.any(depth >= np.minimum(ADAPTIVE_MAX_DEPTH, adaptive_depth_limit(metadata, coarse//56))):
        raise ValueError("Adaptive selected parent is already at maximum depth")
    widths = dict(cell_indices=1, centers=3, log_scales=3, rotations=4,
                  radiance_sh=3*metadata["sh_count"], opacity_sh=metadata["opacity_sh_count"])
    fitted_pieces, replaced = [], []
    for shell, level in sorted(set(zip((coarse//56).tolist(), depth.tolist()))):
        selected = (coarse//56 == shell) & (depth == level)
        source_rows = rows[selected]
        local = (coarse[selected] % 56) * np.uint64(8**level) + morton[selected]
        local_cells = _shell_grid_cells_numpy(local, level, np)
        parent_state = {name: np.asarray(value).reshape(len(cells), widths[name])[source_rows].ravel()
                        for name, value in state.items()}
        parent_state["cell_indices"] = local_cells
        local_metadata = _shell_grid_metadata(dict(metadata, background_subdivision_level=level), shell)
        # Preserve every SH coefficient even when this split occurs before global unfreezing.
        local_metadata.setdefault("active_sh_count", metadata["sh_count"])
        try:
            result = fit_core(dict(model_metadata=local_metadata, model_state=parent_state),
                              config, 8*len(source_rows))["model_state"]
        except NoRepresentableChildren:
            continue
        fine = np.asarray(result["cell_indices"], np.uint64)
        n = int(local_metadata["resolution"])*2
        xyz = np.stack((fine % n, fine//n % n, fine//(n*n)), axis=1)//2
        parent_cells = xyz @ np.asarray([1, n//2, (n//2)**2], np.uint64)
        order = np.argsort(local_cells)
        indices = np.searchsorted(local_cells[order], parent_cells)
        if np.any(indices >= len(local_cells)) or np.any(local_cells[order[indices]] != parent_cells):
            raise ValueError("Fitted background child does not belong to a selected parent")
        # Parents producing no representable children retain both parameters and Adam.
        replaced.extend(source_rows[np.unique(order[indices])].tolist())
        compact = _shell_world_cells_numpy(fine, level+1, shell, 0, np)
        result["cell_indices"] = adaptive_encode_numpy(compact//np.uint64(8**(level+1)),
            np.full(len(compact), level+1, np.int32), compact % np.uint64(8**(level+1)), np)
        fitted_pieces.append(result)
    keep = np.ones(len(cells), bool)
    keep[np.asarray(replaced, np.int64)] = False
    retained = np.flatnonzero(keep)
    child_count = sum(len(piece["cell_indices"]) for piece in fitted_pieces)
    target_count = len(retained)+child_count
    if target_count > max_active_voxels:
        raise RuntimeError("Adaptive refinement exceeds max_active_voxels")
    result = {name: np.concatenate([np.asarray(value).reshape(len(cells), widths[name])[retained].ravel()]
                                  + [piece[name] for piece in fitted_pieces]) for name, value in state.items()}
    validate_adaptive_cells_numpy(result["cell_indices"], metadata["resolution"], metadata["background_shell_levels"], np)
    return dict(model_metadata=dict(metadata), model_state=result,
        source_rows=np.r_[retained, np.full(child_count, -1, np.int64)],
        selected=len(rows), replaced=len(replaced), children=child_count, added=target_count-len(cells))


_FIT_PARENT_BATCH = 4096


def refine_adaptive_background_device(checkpoint, parent_rows, config, max_active_voxels, dr, *, before_allocate=None):
    """Fit selected leaves on GPU, retaining parents that produce no children.

    Only selected addresses/row IDs reach the host for validation and grouping.
    Each fitting batch has at most 4096 parents; its FP64 scratch is released
    before the next batch. Final assembly streams one parameter field at a time.
    """
    import numpy as np
    from reconstruction_refinement_gpu import refine_state_gpu
    from reconstruction_topology import NoRepresentableChildren
    from reconstruction_topology_gpu import take_rows
    from reconstruction_world_refinement import (_shell_grid_cells_device, _shell_grid_metadata,
                                                 _shell_world_cells_device)
    U, C, F, B = dr.cuda.UInt32, dr.cuda.UInt64, dr.cuda.Float32, dr.cuda.Bool
    metadata = checkpoint["model_metadata"]
    if metadata.get("background_refinement") != "adaptive":
        raise ValueError("Adaptive refinement requires adaptive model metadata")
    state = {name:(C(value) if name == "cell_indices" else F(value))
             for name,value in checkpoint["model_state"].items()}
    count = len(state["cell_indices"])
    parent_rows = U(parent_rows)
    cpu_rows = np.asarray(parent_rows).astype(np.int64)
    if np.any(cpu_rows >= count) or len(np.unique(cpu_rows)) != len(cpu_rows):
        raise ValueError("Invalid adaptive parent row selection")
    cells = dr.gather(C,state["cell_indices"],parent_rows)
    coarse, depth, morton = adaptive_decode_numpy(np.asarray(cells),metadata["background_shell_levels"],np)
    if np.any(depth >= np.minimum(ADAPTIVE_MAX_DEPTH, adaptive_depth_limit(metadata, coarse//56))):
        raise ValueError("Adaptive selected parent is already at maximum depth")
    widths = dict(cell_indices=1, centers=3, log_scales=3, rotations=4,
                  radiance_sh=3*metadata["sh_count"], opacity_sh=metadata["opacity_sh_count"])
    if len(parent_rows) and before_allocate:
        # One conservative preflight covers every batch. A guard can flush the
        # allocator/cache and query driver memory, so invoking it per 4096 rows
        # would serialize training behind repeated device-wide synchronization.
        worst_children = 8*len(parent_rows)
        before_allocate(dict(stage="world_background_appearance",exact_count=False,
            old_voxels=count,active_voxels=count+7*len(parent_rows),
            allocation_voxels=worst_children,existing_child_count=worst_children,
            existing_background_child_count=0,background_parent_count=min(len(parent_rows),_FIT_PARENT_BATCH),
            background_child_count=worst_children,model_metadata=metadata))
    pieces, keys, child_count, group_offset = [], [], 0, 0
    replaced = dr.zeros(B,len(parent_rows))
    for shell, level in sorted(set(zip((coarse//56).tolist(),depth.tolist()))):
        group = np.flatnonzero((coarse//56 == shell) & (depth == level))
        local_metadata = _shell_grid_metadata(dict(metadata,background_subdivision_level=level),shell)
        # Active SH scheduling must never discard already stored higher bands.
        local_metadata.setdefault("active_sh_count",metadata["sh_count"])
        for start in range(0,len(group),_FIT_PARENT_BATCH):
            indices = group[start:start+_FIT_PARENT_BATCH]
            rows = dr.gather(U,parent_rows,U(indices))
            parent_state = {name:take_rows(value,rows,widths[name]) for name,value in state.items()}
            local = (coarse[indices] % 56)*np.uint64(8**level)+morton[indices]
            local_cells = _shell_grid_cells_device(C(local),level,dr)
            parent_state["cell_indices"] = local_cells

            try:
                fitted = refine_state_gpu(dict(model_metadata=local_metadata,model_state=parent_state),
                                          config,8*len(rows))["model_state"]
            except NoRepresentableChildren:
                del parent_state,rows,local_cells
                continue
            fine = C(fitted["cell_indices"])
            resolution = int(local_metadata["resolution"])*2
            # A deep shell's temporary fitting grid can exceed 1024 cubed;
            # retain UInt64 address arithmetic throughout the conversion.
            xyz = [fine % resolution,fine//resolution % resolution,fine//(resolution*resolution)]
            parent_cells = (xyz[0]//2+(resolution//2)*(xyz[1]//2)
                            +(resolution//2)**2*(xyz[2]//2))
            order = dr.argsort(local_cells)
            ordered = dr.gather(C,local_cells,order)
            row_count = dr.opaque(U,len(rows))
            found = dr.binary_search(U(0),row_count,lambda i:dr.gather(C,ordered,i)<parent_cells)
            valid = (found<row_count) & (dr.gather(C,ordered,dr.minimum(found,row_count-1))==parent_cells)
            if not bool(dr.all(valid)):
                raise ValueError("Fitted background child does not belong to a selected parent")
            parents = dr.gather(U,order,found)
            # Repeated writes have the same value. Unrepresented parents stay
            # false even when another parent in this fitting batch succeeds.
            dr.scatter(replaced,True,dr.gather(U,U(indices),parents))
            octant = (xyz[0]&1)+2*(xyz[1]&1)+4*(xyz[2]&1)
            # Changing selected counts/offsets must not specialize a new JIT
            # kernel at every adaptive boundary.
            keys.append(dr.opaque(C,group_offset)+octant*dr.opaque(C,len(group))
                        +dr.opaque(C,start)+C(parents))
            compact = _shell_world_cells_device(fine,level+1,shell,0,dr)
            child_bits = 3*(level+1)
            if level+1 == 18:
                # The tenth shell uses an implicit depth and its local coarse
                # ordinal to reserve all 54 low bits for the Morton path.
                fitted["cell_indices"] = (C(ADAPTIVE_DEEPEST_TAG)|(((compact>>54)-504)<<54)
                                          |(compact&((1<<54)-1)))
            else:
                address_shift = ADAPTIVE_PATH_BITS if level+1 > ADAPTIVE_LEGACY_MAX_DEPTH else 32
                address_tag = ADAPTIVE_TAG | (ADAPTIVE_EXTENDED_TAG if level+1 > ADAPTIVE_LEGACY_MAX_DEPTH else 0)
                fitted["cell_indices"] = (C(address_tag)|((compact>>child_bits)<<address_shift)
                                          |C(1<<child_bits)|(compact&((1<<child_bits)-1)))
            dr.eval(*fitted.values(),replaced,keys[-1])
            child_count += len(fitted["cell_indices"])
            pieces.append(fitted)
            del parent_state,rows,local_cells,fitted,fine,xyz,order,ordered,found,valid,parents,compact,row_count
        group_offset += 8*len(group)
    replaced_selected = dr.compress(replaced)
    replaced_count = len(replaced_selected)
    if not replaced_count:
        return dict(model_metadata=dict(metadata),model_state=dict(state),source_rows=dr.arange(U,count),
                    selected=len(parent_rows),replaced=0,children=0,added=0)
    target_count = count-replaced_count+child_count
    if target_count > max_active_voxels:
        raise RuntimeError("Adaptive refinement exceeds max_active_voxels")
    if before_allocate:
        before_allocate(dict(stage="adaptive_background_commit",exact_count=True,
            active_voxels=target_count,old_voxels=count,allocation_voxels=target_count,
            existing_child_count=0,existing_background_child_count=child_count,
            background_child_count=child_count,model_metadata=metadata))
    keep = dr.full(B,True,count)
    dr.scatter(keep,False,dr.gather(U,parent_rows,replaced_selected))
    retained = dr.compress(keep)
    # Restore CPU oracle order (shell/depth, octant, input parent). This makes
    # batching independent of row order without copying any full SH tables.
    key = dr.concat(keys) if len(keys)>1 else keys[0]
    child_dest = dr.empty(U,child_count)
    dr.scatter(child_dest,dr.arange(U,child_count),dr.argsort(key))
    dr.eval(retained,child_dest)
    retained_count = dr.opaque(U,len(retained))
    result = {}
    for name,value in state.items():
        width = widths[name]
        target = dr.empty(C if name=="cell_indices" else F,target_count*width)
        if len(retained):
            old = take_rows(value,retained,width)
            dr.scatter(target,old,dr.arange(U,len(old)))
            dr.eval(target)
            del old
        offset = 0
        for piece in pieces:
            values = piece.pop(name)
            flat = dr.arange(U,len(values))
            destination = ((retained_count+dr.gather(U,child_dest,dr.opaque(U,offset)+flat//width))*width+flat%width)
            dr.scatter(target,values,destination)
            dr.eval(target)
            offset += len(values)//width
            del values,destination,flat
        result[name] = target
    newborn = dr.full(U,0xffffffff,child_count)
    source_rows = dr.concat([retained,newborn]) if len(retained) else newborn
    dr.eval(source_rows)
    return dict(model_metadata=dict(metadata),model_state=result,source_rows=source_rows,
        selected=len(parent_rows),replaced=replaced_count,children=child_count,added=target_count-count)
