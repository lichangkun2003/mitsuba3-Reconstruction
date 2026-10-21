"""Exact RGB growth proposals using device-resident double precision geometry.

Only compaction sizes cross the host boundary. Proposal arrays and model fields
remain on the device; the NumPy generator is retained as an independent oracle.
"""

import math


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


def generate_growth_gpu(manager, checkpoint, boundary, target_cells=None, *,
                        eligible=None, cooldown_cells=None):
    """Generate all parent-specific proposals, ordered by (cell,parent_cell).

    `eligible` and unexpired `cooldown_cells` may be supplied by the device
    lifecycle. A target subset restricts geometry revalidation without changing
    supporting-parent counts for any retained target.
    """
    import drjit as dr
    # Keep geometric threshold decisions in FP64 with fast math disabled,
    # instead of silently using the renderer's float32 arithmetic.
    with dr.scoped_set_flag(dr.JitFlag.FastMath, False):
        return _generate(manager, checkpoint, boundary, target_cells, eligible, cooldown_cells, dr)


def _generate(manager, checkpoint, boundary, target_cells, eligible, cooldown_cells, dr):
    U, I, D, F, B = dr.cuda.UInt32, dr.cuda.Int32, dr.cuda.Float64, dr.cuda.Float32, dr.cuda.Bool
    state, metadata, cfg = checkpoint['model_state'], checkpoint['model_metadata'], manager.settings
    state = {k: U(v) if k == 'cell_indices' else F(v) for k, v in state.items()}
    occupied, resolution = state['cell_indices'], metadata['resolution']
    count = dr.width(occupied)
    if not count:
        result = {k: dr.empty(U if k in ('parent_ids','parent_cells','cells','geometric_support') else D, 0)
                  for k in ('parent_ids','parent_cells','cells','geometric_support','centers','log_scales','rotations')}
        result.update(boundary=boundary,resolution=resolution,seen_views=set(),foreground=U([]),background=U([]))
        result['candidate_state'] = {k: dr.empty(U if k=='cell_indices' else F, 0) for k in state}
        result.update(radiance_sh=F([]),opacity_sh=F([]))
        return result
    if eligible is None:
        wait = dr.cuda.Int64(manager.growth_wait_start)
        eligible = ~B(manager.candidates) & ((wait == 0) |
            ((boundary >= wait-1) & (boundary-(wait-1) >= cfg.growth_wait_epochs)))
    if cooldown_cells is None:
        cooldown_cells = U([cell for cell, expiry in manager.cooldown.items() if boundary < expiry])
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
    # Avoid materializing 26 copies of each parent's geometry. Occupancy is
    # shared, and only legal unoccupied pairs enter the expensive contact test.
    blocked = dr.zeros(B, resolution**3)
    dr.scatter(blocked, True, occupied)
    if dr.width(cooldown_cells):
        dr.scatter(blocked, True, U(cooldown_cells))
    selected_targets = None
    if target_cells is not None:
        selected_targets = dr.zeros(B, resolution**3)
        if dr.width(target_cells):
            dr.scatter(selected_targets, True, U(target_cells))
    dr.eval(blocked, selected_targets)
    center, axes, q, size, metric, extent = geometry_gpu(state, metadata, dr)
    cells = [I(occupied % resolution), I(occupied // resolution % resolution),
             I(occupied // (resolution*resolution))]
    offsets = ([(x,y,z) for z in (-1,0,1) for y in (-1,0,1) for x in (-1,0,1)
                if (x,y,z)!=(0,0,0)] if evidence and cfg.growth_use_26_neighbors else
               [tuple(sign if j==axis else 0 for j in range(3))
                for axis in range(3) for sign in (-1,1)])
    pieces = []
    def gather(value, ids):
        return dr.gather(type(value), value, ids)
    for offset in offsets:
        neighbors = [cells[i]+offset[i] for i in range(3)]
        possible = B(eligible)
        for c in neighbors:
            possible &= (c >= 0) & (c < resolution)
        targets = U(neighbors[0]+resolution*(neighbors[1]+resolution*neighbors[2]))
        possible &= ~dr.gather(B, blocked, targets, possible)
        if selected_targets is not None:
            possible &= dr.gather(B, selected_targets, targets, possible)
        ids = dr.compress(possible)
        if not dr.width(ids):
            continue
        c = [gather(v, ids) for v in center]
        m = [[gather(v, ids) for v in row] for row in metric]
        a = [gather(v, ids) for v in axes]
        e = [gather(v, ids) for v in extent]
        touched, contact = contact_gpu(c, m, offset, 0., dr)
        if penetration > 0:
            deep, _ = contact_gpu(c, m, offset, penetration, dr)
            touched &= deep
        child = [dr.clip(v, .05, .95) for v in contact]
        margin = [dr.minimum(v, 1-v) for v in child]
        for axis in range(3):
            if offset[axis]:
                v = min(max(cfg.growth_contact_offset, .01), .49)
                child[axis] = dr.full(D, v if offset[axis] > 0 else 1-v, dr.width(ids))
                margin[axis] = 1-child[axis] if offset[axis] > 0 else child[axis]
        factor = dr.full(D, min(max(cfg.growth_shrink, .01), .99), dr.width(ids))
        for axis in range(3):
            factor = dr.minimum(factor, dr.maximum(margin[axis]-.005, 0.)/dr.maximum(e[axis], 1e-20))
        touched &= factor > 0
        for axis in range(3):
            touched &= a[axis]*factor >= size[axis]*1e-4
        chosen = dr.compress(touched)
        if not dr.width(chosen):
            continue
        parents = gather(ids, chosen)
        pieces.append(dict(parent_ids=parents, parent_cells=gather(occupied, parents),
            cells=gather(targets, parents),
            centers=_pack([gather(v, chosen) for v in child], D, dr),
            log_scales=_pack([dr.log(gather(v*factor, chosen)) for v in a], D, dr),
            rotations=_pack([gather(v, parents) for v in q], D, dr)))
        dr.eval(*pieces[-1].values())
    dimensions = dict(parent_ids=1, parent_cells=1, cells=1, centers=3, log_scales=3, rotations=4)
    result = {k: dr.concat([p[k] for p in pieces]) if pieces else
              dr.empty(U if width == 1 else D, 0) for k, width in dimensions.items()}
    n = dr.width(result['cells'])
    if n:
        key = (dr.cuda.UInt64(result['cells']) << 32) | dr.cuda.UInt64(result['parent_cells'])
        order = dr.argsort(key)
        result = {k: gather_rows(v, order, dimensions[k], dr) for k, v in result.items()}
        index = dr.arange(U, n)
        first = (index == 0) | (result['cells'] != dr.gather(U, result['cells'], dr.maximum(index, 1)-1))
        groups = dr.cumsum(U(first))-1
        starts = dr.compress(first)
        counts = dr.zeros(U, dr.width(starts))
        dr.scatter_reduce(dr.ReduceOp.Add, counts, U(1), groups)
        support = dr.gather(U, counts, groups)
        keep = dr.compress(support >= max(1, min(26, cfg.growth_min_supporting_parents))) if evidence else index
        result = {k: gather_rows(v, keep, dimensions[k], dr) for k, v in result.items()}
        result['geometric_support'] = dr.gather(U, support, keep) if dr.width(keep) else U([])
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
    if cfg.growth_inherit_opacity:
        b0 = D(dr.gather(F, state['opacity_sh'], result['parent_ids']*metadata['opacity_sh_count']))
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
            result['parent_ids']*metadata['sh_count']*3+channel),index*metadata['sh_count']*3+channel)
    result['candidate_state'] = dict(cell_indices=result['cells'],
        centers=F(result['centers']),log_scales=F(result['log_scales']),rotations=F(result['rotations']),
        radiance_sh=radiance,opacity_sh=opacity)
    result.update(radiance_sh=radiance,opacity_sh=opacity)
    dr.eval(*[v for v in result.values() if dr.is_array_v(type(v))],*result['candidate_state'].values())
    return result
