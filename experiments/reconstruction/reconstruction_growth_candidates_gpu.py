"""Exact RGB growth proposals using device-resident double precision geometry.

Only compaction sizes and small per-direction counts cross the host boundary.
Proposal arrays and model fields remain on the device; the NumPy generator is
retained as an independent oracle.
"""

import math


DEFAULT_PARENT_BATCH_SIZE = 65536
# Bound pre-contact proposal storage even if a caller requests larger batches.
# At most 26 directions * 65536 parents: 149.5 MiB of logical proposal fields,
# about 216 MiB after this Dr.Jit allocator's per-field power-of-two rounding.
MAX_BATCH_PAIRS = 26 * DEFAULT_PARENT_BATCH_SIZE


def gather_rows(value, ids, width, dr):
    """Gather flattened rows without materializing a host index array."""
    if not dr.width(ids):
        return dr.empty(type(value), 0)
    UInt = dr.uint32_array_t(type(ids))
    index = dr.arange(UInt, dr.width(ids) * width)
    return dr.gather(type(value), value,
                     dr.gather(type(ids), ids, index // width) * width + index % width)


def _fields(value, count, width, T, dr):
    index = dr.arange(dr.cuda.UInt32, count)
    return [T(dr.gather(type(value), value, index * width + j)) for j in range(width)]


def _pack(fields, T, dr):
    count, width = dr.width(fields[0]), len(fields)
    result = dr.empty(T, count * width)
    index = dr.arange(dr.cuda.UInt32, count)
    for j, value in enumerate(fields):
        dr.scatter(result, T(value), index * width + j)
    return result


def geometry_gpu(state, metadata, dr):
    """The same normalized quaternion, metric, and extent as _geometry()."""
    D = dr.cuda.Float64
    n = dr.width(state['cell_indices'])
    center = _fields(state['centers'], n, 3, D, dr)
    axes = [dr.exp(dr.clip(x, -30., 30.))
            for x in _fields(state['log_scales'], n, 3, D, dr)]
    q = _fields(state['rotations'], n, 4, D, dr)
    norm = sum(x * x for x in q)
    valid = dr.isfinite(norm) & (norm > 1e-20)
    denom = dr.sqrt(dr.select(valid, norm, 1.))
    q = [dr.select(valid, x / denom, float(j == 0)) for j, x in enumerate(q)]
    w, x, y, z = q
    rotation = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    size = [(float(b)-float(a))/metadata['resolution'] for a, b in
            zip(metadata['grid_min'], metadata['grid_max'])]
    transform = [[rotation[i][k] / axes[k] for k in range(3)] for i in range(3)]
    metric = [[sum(transform[i][k]*transform[j][k] for k in range(3))*size[i]*size[j]
               for j in range(3)] for i in range(3)]
    extent = [dr.sqrt(sum((rotation[i][k]*axes[k])**2 for k in range(3))) / size[i]
              for i in range(3)]
    dr.eval(center, axes, q, metric, extent)
    return center, axes, q, size, metric, extent


def _quadratic(q, metric):
    # np.einsum('ni,nij,nj->n') accumulates in i,j order.
    return sum(q[i]*metric[i][j]*q[j] for i in range(3) for j in range(3))


def contact_gpu(center, metric, offset, depth, dr):
    """Constrained finite face/edge/corner contact; no bounding-box shortcut."""
    crossed = [i for i in range(3) if offset[i]]
    contact = list(center)
    for axis in crossed:
        contact[axis] = dr.full(dr.cuda.Float64, 1.+depth if offset[axis] > 0 else -depth,
                                dr.width(center[0]))
    if len(crossed) == 1:
        n = crossed[0]
        u, v = (n+1) % 3, (n+2) % 3
        qn = contact[n]-center[n]
        determinant = metric[u][u]*metric[v][v]-metric[u][v]**2
        best, best_contact = dr.cuda.Float64(float('inf')), list(contact)
        for candidate in range(5):
            point, valid = list(contact), dr.cuda.Bool(True)
            if candidate == 0:
                valid = determinant > 1e-30
                denominator = dr.select(valid, determinant, 1.)
                point[u] = center[u]-qn*(metric[v][v]*metric[u][n]-metric[u][v]*metric[v][n])/denominator
                point[v] = center[v]-qn*(metric[u][u]*metric[v][n]-metric[u][v]*metric[u][n])/denominator
                valid &= (point[u]>=0)&(point[u]<=1)&(point[v]>=0)&(point[v]<=1)
            else:
                fixed, free = (u, v) if candidate <= 2 else (v, u)
                point[fixed] = dr.full(dr.cuda.Float64, (candidate-1)&1, dr.width(center[0]))
                point[free] = dr.clip(center[free]-(metric[free][n]*qn+
                    metric[free][fixed]*(point[fixed]-center[fixed])) /
                    dr.maximum(metric[free][free], 1e-30), 0., 1.)
            value = _quadratic([point[i]-center[i] for i in range(3)], metric)
            better = valid & (value < best)
            best = dr.select(better, value, best)
            best_contact = [dr.select(better, a, b) for a, b in zip(point, best_contact)]
        return best < 1-1e-5, best_contact
    if len(crossed) == 2:
        free = next(i for i in range(3) if not offset[i])
        coupling = sum(metric[free][i]*(contact[i]-center[i]) for i in range(3))
        contact[free] = dr.clip(center[free]-coupling/dr.maximum(metric[free][free], 1e-30), 0., 1.)
    return _quadratic([contact[i]-center[i] for i in range(3)], metric) < 1-1e-5, contact


def membership_gpu(values, queries, dr):
    """Exact sorted membership; uint32 addresses, no dense grid allocation."""
    U = dr.cuda.UInt32
    if not dr.width(values):
        return dr.zeros(dr.cuda.Bool, dr.width(queries))
    ordered = dr.sort(U(values))
    count = dr.width(ordered)
    index = dr.binary_search(0, count, lambda i: dr.gather(U, ordered, i) < queries)
    return (index < count) & (dr.gather(U, ordered, dr.minimum(index, count-1)) == queries)


def _make_bitmap(cell_count, sources, dr):
    """One exact occupancy bit per cell; duplicate writers combine with OR."""
    U = dr.cuda.UInt32
    bitmap = dr.zeros(U, (cell_count + 31) // 32)
    for source in sources:
        if dr.width(source):
            cells = U(source)
            dr.scatter_reduce(dr.ReduceOp.Or, bitmap,
                              U(1) << (cells & 31), cells >> 5)
    dr.eval(bitmap)
    return bitmap


def _bitmap_contains(bitmap, cells, active, dr):
    U = dr.cuda.UInt32
    word = dr.gather(U, bitmap, cells >> 5, active)
    return (word & (U(1) << (cells & 31))) != 0


def _reorder_fields(fields, order, dimensions, dr):
    """Materialize one reordered field before releasing its old allocation."""
    for name, width in dimensions.items():
        previous = fields.pop(name)
        fields[name] = gather_rows(previous, order, width, dr)
        dr.eval(fields[name])
        del previous


def _concatenate_fields(pieces, dimensions, dr):
    """Merge one field at a time, with data-dependent offsets kept out of PTX."""
    U, D = dr.cuda.UInt32, dr.cuda.Float64
    count = sum(dr.width(piece['cells']) for piece in pieces)
    result = {}
    for name, width in dimensions.items():
        merged = dr.empty(U if width == 1 else D, count * width)
        begin = 0
        for piece in pieces:
            source = piece.pop(name)
            positions = dr.arange(U, dr.width(source)) + dr.opaque(U, begin)
            # Ranges are disjoint and cover the complete result. Unlike
            # dr.concat's literal offsets, these reuse code when counts change.
            dr.scatter(merged, source, positions, mode=dr.ReduceMode.Permute)
            begin += dr.width(source)
        dr.eval(merged)
        result[name] = merged
        if pieces:
            del source, positions
    pieces.clear()
    return result


def _neighbor_mask(cells, offset, resolution, blocked, selected_targets,
                   region_allowed, active, dr):
    """Cheap eligibility shared by the parent prefilter and pair generation."""
    U = dr.cuda.UInt32
    neighbors = [cells[i] + offset[i] for i in range(3)]
    possible = active & region_allowed(neighbors)
    targets = U(neighbors[0] + resolution * (neighbors[1] + resolution * neighbors[2]))
    possible &= ~_bitmap_contains(blocked, targets, possible, dr)
    if selected_targets is not None:
        possible &= _bitmap_contains(selected_targets, targets, possible, dr)
    return possible


def _exposed_parent_ids(occupied, eligible, resolution, blocked, selected_targets,
                        offsets, region_allowed, dr):
    """Discard only parents that cannot propose any legal requested empty cell."""
    I, B = dr.cuda.Int32, dr.cuda.Bool
    cells = [I(occupied % resolution), I(occupied // resolution % resolution),
             I(occupied // (resolution * resolution))]
    allowed = B(eligible) & region_allowed(cells)
    exposed = dr.zeros(B, dr.width(occupied))
    for offset in offsets:
        # A single accumulated mask, not a direction-by-full-model table.
        # Once a parent has a legal target, further bitmap reads are masked.
        exposed |= _neighbor_mask(cells, offset, resolution, blocked, selected_targets,
                                  region_allowed, allowed & ~exposed, dr)
    return dr.compress(exposed)


def _batch_pairs(parent_cells, resolution, blocked, selected_targets,
                 offset_fields, region_allowed, divisor, dr):
    """Compact all directions together; returned indices remain direction-major."""
    import numpy as np
    U, I = dr.cuda.UInt32, dr.cuda.Int32
    count, directions = dr.width(parent_cells), dr.width(offset_fields[0])
    lanes = dr.arange(U, count * directions)
    addresses = dr.gather(U, parent_cells, lanes % divisor)
    cells = [I(addresses % resolution), I(addresses // resolution % resolution),
             I(addresses // (resolution * resolution))]
    offsets = [dr.gather(I, value, lanes // divisor) for value in offset_fields]
    possible = _neighbor_mask(cells, offsets, resolution, blocked, selected_targets,
                              region_allowed, dr.cuda.Bool(True), dr)
    pairs = dr.compress(possible)
    # Stable compaction groups each direction contiguously. Read one small
    # histogram (6/26 uint32 values), never one length per neighbor direction.
    counts = dr.zeros(U, directions)
    if dr.width(pairs):
        dr.scatter_reduce(dr.ReduceOp.Add, counts, U(1), pairs // divisor)
    dr.eval(counts)
    return pairs, np.array(counts).tolist()


def _write_direction(fields, accepted, pairs, begin, count, parent_ids, parent_cells,
                     geometry, offset, resolution, penetration, cfg, divisor, dr):
    """Write a direction's unfiltered proposals; compact acceptance once per batch."""
    U, D = dr.cuda.UInt32, dr.cuda.Float64
    center, axes, q, size, metric, extent = geometry
    positions = dr.arange(U, count) + dr.opaque(U, begin)
    ids = dr.gather(U, pairs, positions) % divisor

    def gather(value):
        return dr.gather(type(value), value, ids)

    c = [gather(v) for v in center]
    m = [[gather(v) for v in row] for row in metric]
    a = [gather(v) for v in axes]
    e = [gather(v) for v in extent]
    touched, contact = contact_gpu(c, m, offset, 0., dr)
    if penetration > 0:
        deep, _ = contact_gpu(c, m, offset, penetration, dr)
        touched &= deep
    child = [dr.clip(v, .05, .95) for v in contact]
    margin = [dr.minimum(v, 1-v) for v in child]
    for axis in range(3):
        if offset[axis]:
            v = min(max(cfg.growth_contact_offset, .01), .49)
            child[axis] = dr.full(D, v if offset[axis] > 0 else 1-v, count)
            margin[axis] = 1-child[axis] if offset[axis] > 0 else child[axis]
    factor = dr.full(D, min(max(cfg.growth_shrink, .01), .99), count)
    for axis in range(3):
        factor = dr.minimum(factor, dr.maximum(margin[axis]-.005, 0.)/dr.maximum(e[axis], 1e-20))
    touched &= factor > 0
    for axis in range(3):
        touched &= a[axis]*factor >= size[axis]*1e-4
    source_cells = gather(parent_cells)
    target_delta = offset[0] + resolution * (offset[1] + resolution * offset[2])
    # Pair construction already proved each target is inside the allowed grid.
    targets = source_cells + U(target_delta & 0xffffffff)
    for name, value in (('parent_ids', gather(parent_ids)), ('parent_cells', source_cells),
                        ('cells', targets)):
        dr.scatter(fields[name], value, positions, mode=dr.ReduceMode.Permute)
    for name, values in (('centers', child),
                         ('log_scales', [dr.log(dr.select(touched, v*factor, 1.)) for v in a]),
                         ('rotations', [gather(v) for v in q])):
        for lane, value in enumerate(values):
            dr.scatter(fields[name], value, positions * len(values) + lane,
                       mode=dr.ReduceMode.Permute)
    dr.scatter(accepted, touched, positions, mode=dr.ReduceMode.Permute)
    # Every row in this contiguous range is written. Unwritten later ranges are
    # not read until all directions have completed; rejected log inputs use 1.
    # Component and direction ranges never overlap. Permute avoids flushing a
    # dirty field before each component write, so they share the evaluation.
    dr.eval(accepted, *fields.values())


def _parent_batch(state, metadata, parent_ids, blocked, selected_targets,
                  offsets, offset_fields, penetration, cfg, region_allowed, batch_size, dr,
                  parameter_parent_rows=None):
    """Two compactions per bounded parent batch, with unchanged FP64 contact math."""
    U, D = dr.cuda.UInt32, dr.cuda.Float64
    resolution = metadata['resolution']
    parent_cells = dr.gather(U, state['cell_indices'], parent_ids)
    parent_count = dr.width(parent_ids)
    # Full batches retain constant (usually power-of-two) division. A changing
    # tail size must not become a new divisor literal in every contact kernel.
    divisor = parent_count if parent_count == batch_size else dr.opaque(U, parent_count)
    pairs, direction_counts = _batch_pairs(parent_cells, resolution, blocked, selected_targets,
                                          offset_fields, region_allowed, divisor, dr)
    count = dr.width(pairs)
    if not count:
        return None
    geometry_state = {'cell_indices': parent_cells}
    parameter_ids = (parent_ids if parameter_parent_rows is None else
                     dr.gather(U, parameter_parent_rows, parent_ids))
    for name, width in (('centers', 3), ('log_scales', 3), ('rotations', 4)):
        geometry_state[name] = gather_rows(state[name], parameter_ids, width, dr)
    geometry = geometry_gpu(geometry_state, metadata, dr)
    del geometry_state
    dimensions = dict(parent_ids=1, parent_cells=1, cells=1, centers=3, log_scales=3, rotations=4)
    fields = {name: dr.empty(U if width == 1 else D, count * width)
              for name, width in dimensions.items()}
    accepted = dr.empty(dr.cuda.Bool, count)
    begin = 0
    for offset, size in zip(offsets, direction_counts):
        if size:
            _write_direction(fields, accepted, pairs, begin, size, parent_ids, parent_cells,
                             geometry, offset, resolution, penetration, cfg, divisor, dr)
        begin += size
    del geometry, pairs
    keep = dr.compress(accepted)
    del accepted
    if not dr.width(keep):
        return None
    if dr.width(keep) != count:
        _reorder_fields(fields, keep, dimensions, dr)
    return fields


def generate_growth_gpu(manager, checkpoint, boundary, target_cells=None, *,
                        eligible=None, cooldown_cells=None,
                        parent_batch_size=DEFAULT_PARENT_BATCH_SIZE):
    """Generate all parent-specific proposals, ordered by (cell,parent_cell).

    `eligible` and unexpired `cooldown_cells` may be supplied by the device
    lifecycle. A target subset restricts geometry revalidation without changing
    supporting-parent counts for any retained target. `parent_batch_size` bounds
    temporary FP64 geometry; batches are also capped by `MAX_BATCH_PAIRS` before
    contact filtering. Neither bound limits parents or final candidate counts.
    """
    if (isinstance(parent_batch_size, bool) or not isinstance(parent_batch_size, int)
            or parent_batch_size < 1):
        raise ValueError("parent_batch_size must be a positive integer")
    import drjit as dr
    # Keep geometric threshold decisions in FP64 with fast math disabled,
    # instead of silently using the renderer's float32 arithmetic.
    with dr.scoped_set_flag(dr.JitFlag.FastMath, False):
        return _generate(manager, checkpoint, boundary, target_cells, eligible,
                         cooldown_cells, parent_batch_size, dr)


def _generate(manager, checkpoint, boundary, target_cells, eligible, cooldown_cells,
              parent_batch_size, dr):
    U, I, D, F, B = dr.cuda.UInt32, dr.cuda.Int32, dr.cuda.Float64, dr.cuda.Float32, dr.cuda.Bool
    state, metadata, cfg = checkpoint['model_state'], checkpoint['model_metadata'], manager.settings
    state = {k: U(v) if k == 'cell_indices' else F(v) for k, v in state.items()}
    # A memory-limited prune keeps parameter fields in the original row order
    # until the old native model can retire. Occupancy and result parent IDs
    # still use the compact surviving-row order; only parameter reads remap.
    parameter_parent_rows = checkpoint.get('parameter_parent_rows')
    if parameter_parent_rows is not None:
        parameter_parent_rows = U(parameter_parent_rows)
        if len(parameter_parent_rows) != len(state['cell_indices']):
            raise ValueError('Parameter parent map must cover every surviving cell')
    occupied, resolution = state['cell_indices'], metadata['resolution']
    def empty_result():
        result = {k: dr.empty(U if k in ('parent_ids','parent_cells','cells','geometric_support') else D, 0)
                  for k in ('parent_ids','parent_cells','cells','geometric_support','centers','log_scales','rotations')}
        result.update(boundary=boundary,resolution=resolution,seen_views=set(),foreground=U([]),background=U([]))
        result['candidate_state'] = {k: dr.empty(U if k=='cell_indices' else F, 0) for k in state}
        result.update(radiance_sh=F([]),opacity_sh=F([]))
        return result
    if not dr.width(occupied) or (target_cells is not None and not dr.width(target_cells)):
        return empty_result()
    if eligible is None:
        wait = dr.cuda.Int64(manager.growth_wait_start)
        eligible = ~B(manager.candidates) & ((wait == 0) |
            ((boundary >= wait-1) & (boundary-(wait-1) >= cfg.growth_wait_epochs)))
    if cooldown_cells is None:
        cooldown_cells = U([cell for cell, expiry in manager.cooldown.items() if boundary < expiry])
    from reconstruction_topology import _growth_core_radius_squared
    radius_squared = _growth_core_radius_squared(metadata)
    def region_allowed(coordinates):
        allowed = dr.full(B, True, dr.width(coordinates[0]))
        for c in coordinates:
            allowed &= (c >= 0) & (c < resolution)
        if metadata.get("scene_mapping") == "projective_shell":
            core = int(metadata["core_resolution"])
            pad = (resolution - core) // 2
            for c in coordinates:
                allowed &= (c >= pad) & (c < pad + core)
        if radius_squared is not None:
            farthest_squared = sum((dr.abs(2*c + 1 - resolution) + 1)**2
                                   for c in coordinates)
            allowed &= farthest_squared <= radius_squared
        return allowed

    evidence = cfg.use_growth_evidence or cfg.use_rgb_growth
    if evidence:
        penetration = min(max(float(cfg.growth_evidence_penetration), 0.), 4.)
    elif manager.schedule.enabled:
        penetration = manager.schedule.coarsest_growth_face_penetration * (1 if
            metadata.get('core_resolution', resolution) == manager.schedule.start_resolution else
            manager.schedule.growth_finer_level_multiplier)
        penetration = min(max(float(penetration), 0.), 4.)
    else:
        penetration = min(max(float(cfg.growth_face_penetration), 0.), 4.)
    # Eight times smaller than byte-per-cell Bool tables, including high-grid
    # addresses and duplicate cooldown/target entries. Masked neighbor queries
    # never address words outside the grid.
    blocked = _make_bitmap(resolution**3, (occupied, cooldown_cells), dr)
    selected_targets = (_make_bitmap(resolution**3, (target_cells,), dr)
                        if target_cells is not None else None)
    offsets = ([(x,y,z) for z in (-1,0,1) for y in (-1,0,1) for x in (-1,0,1)
                if (x,y,z)!=(0,0,0)] if evidence and cfg.growth_use_26_neighbors else
               [tuple(sign if j==axis else 0 for j in range(3))
                for axis in range(3) for sign in (-1,1)])
    # Interior parents and parents outside a requested target neighborhood
    # cannot contribute a proposal. Cull them before any FP64 geometry, keeping
    # every eligible supporting parent and its original model row ID.
    parent_ids = _exposed_parent_ids(occupied, eligible, resolution, blocked, selected_targets,
                                     offsets, region_allowed, dr)
    if not dr.width(parent_ids):
        return empty_result()
    offset_fields = [I([offset[axis] for offset in offsets]) for axis in range(3)]
    batch_size = min(parent_batch_size, max(1, MAX_BATCH_PAIRS // len(offsets)))
    pieces = []
    for begin in range(0, dr.width(parent_ids), batch_size):
        count = min(batch_size, dr.width(parent_ids)-begin)
        positions = dr.arange(U, count) + dr.opaque(U, begin)
        batch_ids = dr.gather(U, parent_ids, positions)
        piece = _parent_batch(state, metadata, batch_ids, blocked, selected_targets,
                              offsets, offset_fields, penetration, cfg, region_allowed, batch_size, dr,
                              parameter_parent_rows=parameter_parent_rows)
        if piece is not None:
            pieces.append(piece)
    del blocked, selected_targets, parent_ids, batch_ids, positions, offset_fields, piece
    dimensions = dict(parent_ids=1, parent_cells=1, cells=1, centers=3, log_scales=3, rotations=4)
    result = _concatenate_fields(pieces, dimensions, dr)
    n = dr.width(result['cells'])
    if n:
        # Parent batches change enumeration order only. Every (cell,parent-cell)
        # pair is unique; global sorting restores the original canonical order
        # before support is counted across all batches.
        key = (dr.cuda.UInt64(result['cells']) << 32) | dr.cuda.UInt64(result['parent_cells'])
        order = dr.argsort(key)
        dr.eval(order)
        del key
        _reorder_fields(result, order, dimensions, dr)
        del order
        index = dr.arange(U, n)
        first = (index == 0) | (result['cells'] != dr.gather(U, result['cells'], dr.maximum(index, 1)-1))
        groups = dr.cumsum(U(first))-1
        starts = dr.compress(first)
        counts = dr.zeros(U, dr.width(starts))
        dr.scatter_reduce(dr.ReduceOp.Add, counts, U(1), groups)
        support = dr.gather(U, counts, groups)
        keep = dr.compress(support >= max(1, min(26, cfg.growth_min_supporting_parents))) if evidence else index
        geometric_support = dr.gather(U, support, keep) if dr.width(keep) else U([])
        dr.eval(keep, geometric_support)
        del support, groups, starts, counts, first, index
        if dr.width(keep) != n:
            _reorder_fields(result, keep, dimensions, dr)
        result['geometric_support'] = geometric_support
        del keep, geometric_support
    else:
        result['geometric_support'] = U([])
    n = dr.width(result['cells'])
    result.update(boundary=boundary, resolution=resolution, seen_views=set(),
                  foreground=dr.zeros(U,n), background=dr.zeros(U,n))
    if not n:
        result['candidate_state'] = {k: dr.empty(U if k=='cell_indices' else F, 0) for k in state}
        result.update(radiance_sh=F([]),opacity_sh=F([]))
        return result
    from reconstruction_topology import validate_growth_opacity
    validate_growth_opacity(cfg, metadata['opacity_sh_count'])
    inverse_y0 = 2*math.sqrt(math.pi)
    parameter_ids = (result['parent_ids'] if parameter_parent_rows is None else
                     dr.gather(U, parameter_parent_rows, result['parent_ids']))
    if cfg.growth_inherit_opacity:
        b0 = D(dr.gather(F, state['opacity_sh'], parameter_ids*metadata['opacity_sh_count']))
        z = b0 / inverse_y0
        # Stable log(sigmoid(z)), including extremely negative parent logits.
        log_alpha = dr.minimum(math.log(cfg.growth_initial_opacity), math.log(cfg.growth_opacity_scale)-
                               (dr.maximum(-z,0.)+dr.log1p(dr.exp(-dr.abs(z)))))
        child_b0 = (log_alpha-dr.log(-dr.expm1(log_alpha)))*inverse_y0
    else:
        alpha = min(max(float(cfg.growth_initial_opacity), .01), .99)
        child_b0 = dr.full(D, math.log(alpha/(1-alpha))*inverse_y0, n)
    opacity = dr.zeros(F,n*metadata['opacity_sh_count'])
    radiance = dr.zeros(F,n*metadata['sh_count']*3)
    index = dr.arange(U,n)
    dr.scatter(opacity,F(child_b0),index*metadata['opacity_sh_count'])
    for channel in range(3):
        dr.scatter(radiance,dr.gather(F,state['radiance_sh'],
             parameter_ids*metadata['sh_count']*3+channel),index*metadata['sh_count']*3+channel)
    result['candidate_state'] = dict(cell_indices=result['cells'],
        centers=F(result['centers']),log_scales=F(result['log_scales']),rotations=F(result['rotations']),
        radiance_sh=radiance,opacity_sh=opacity)
    result.update(radiance_sh=radiance,opacity_sh=opacity)
    dr.eval(*[v for v in result.values() if dr.is_array_v(type(v))],*result['candidate_state'].values())
    return result
