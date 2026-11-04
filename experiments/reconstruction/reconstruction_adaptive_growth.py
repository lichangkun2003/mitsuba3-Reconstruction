"""Sparse mixed-leaf neighbor growth, with bounded device-resident proposals.

The index is a sorted array of dyadic Morton intervals, never a finest-grid
bitmap. A target containing occupied descendants is split along the contacting
face; an occupied ancestor blocks it. The CPU implementation is an independent
geometry oracle and is also used by the non-CUDA topology manager.
"""
from __future__ import annotations

import math
import time

from reconstruction_world_shell import (ADAPTIVE_TAG, CORE_ADAPTIVE_TAG, ADAPTIVE_MAX_DEPTH,
    ADAPTIVE_LEGACY_MAX_DEPTH, ADAPTIVE_EXTENDED_TAG, ADAPTIVE_DEEPEST_TAG, ADAPTIVE_PATH_BITS,
    adaptive_depth_limit)

_HOLES = (21, 22, 25, 26, 37, 38, 41, 42)
_WIDTHS = dict(parent_ids=1, parent_cells=1, cells=1, centers=3, log_scales=3, rotations=4)


class _Ops:
    def __init__(self, module, gpu=False):
        self.x, self.gpu = module, gpu
        if gpu:
            self.C, self.I, self.U, self.D = (module.cuda.UInt64, module.cuda.Int32,
                                           module.cuda.UInt32, module.cuda.Float64)
            self.choose = module.select
        else:
            self.C = lambda x: module.asarray(x, dtype=module.uint64)
            self.I = lambda x: module.asarray(x, dtype=module.int32)
            self.U = lambda x: module.asarray(x, dtype=module.uint32)
            self.D = lambda x: module.asarray(x, dtype=module.float64)
            self.choose = module.where
        self.minimum, self.maximum, self.abs = module.minimum, module.maximum, module.abs


def _morton(xyz, bits, op):
    result = op.C(xyz[0]) * 0
    for bit in range(bits):
        for axis in range(3):
            result |= ((op.C(xyz[axis]) >> bit) & 1) << (3*bit+axis)
    return result


def _decode(cells, metadata, op):
    """Integer virtual-grid anchors (core 1024), plus log2 leaf edge."""
    n, h = int(metadata['resolution']), int(metadata['background_shell_levels'])
    root = 1024 << h
    core_l = 10 - (n.bit_length()-1)
    core_offset = (root-1024)//2
    cells = op.C(cells)
    bg = cells >= ADAPTIVE_TAG
    tagged_core = ~bg & ((cells & CORE_ADAPTIVE_TAG) != 0)
    core = op.C(op.choose(bg | tagged_core, 0, cells))
    xyz = [core_offset + op.I(core // (n**axis) % n) * (1 << core_l) for axis in range(3)]
    core_path = cells & ((1 << 31)-1)
    core_depth = op.I(tagged_core)*0
    for d in range(1, 11):
        core_depth = op.choose(core_path >= (1 << (3*d)), d, core_depth)
    tagged_l = 10-core_depth
    core_morton = core_path-(op.C(1) << op.U(3*core_depth))
    for axis in range(3):
        coordinate = op.I(tagged_core)*0
        for bit in range(10):
            coordinate |= op.I((core_morton >> (3*bit+axis)) & 1) << bit
        xyz[axis] = op.I(op.choose(tagged_core,
            core_offset + coordinate*(op.I(1) << op.I(tagged_l)), xyz[axis]))
    core_l = op.I(op.choose(tagged_core, tagged_l, core_l))
    # Inactive tagged core lanes must not create out-of-range shell shifts.
    cells = op.C(op.choose(bg, cells, op.C(ADAPTIVE_TAG | 1)))
    extended = (cells & ADAPTIVE_EXTENDED_TAG) != 0
    deepest = (cells & ADAPTIVE_DEEPEST_TAG) == ADAPTIVE_DEEPEST_TAG
    coarse = op.I(op.choose(extended,
        (cells >> ADAPTIVE_PATH_BITS) & 1023,
        (cells & (ADAPTIVE_TAG-1)) >> 32))
    coarse = op.I(op.choose(deepest, 504+op.I((cells >> 54) & 63), coarse))
    shell, ordinal = coarse // 56, coarse % 56
    flat = ordinal
    for hole in _HOLES:
        flat = flat + op.I(flat >= hole)
    path = op.C(op.choose(extended, cells & ((1 << ADAPTIVE_PATH_BITS)-1), cells & (2**32-1)))
    depth = op.I(bg)*0
    for d in range(1, ADAPTIVE_MAX_DEPTH+1):
        depth = op.choose(path >= (1 << (3*d)), d, depth)
    depth = op.choose(deepest, 18, depth)
    morton = op.C(op.choose(deepest, cells & ((1<<54)-1), path-(op.C(1) << op.U(3*depth))))
    coarse_l = 9 + shell
    leaf_l = coarse_l-depth
    for axis in range(3):
        child = op.I(bg)*0
        for bit in range(ADAPTIVE_MAX_DEPTH):
            child |= op.I((morton >> (3*bit+axis)) & 1) << bit
        a = root//2 + ((flat // (4**axis) % 4)-2) * (op.I(1) << op.I(coarse_l))
        a += child * (op.I(1) << op.I(op.maximum(leaf_l, 0)))
        xyz[axis] = op.I(op.choose(bg, a, xyz[axis]))
    return xyz, op.I(op.choose(bg, leaf_l, core_l))


def _interval(xyz, log_size, metadata, op):
    start = _morton(xyz, 10+int(metadata['background_shell_levels']), op)
    return start, start + (op.C(1) << op.U(3*log_size))


def _canonical(xyz, log_size, metadata, op):
    """Map aligned target cubes to real leaves, requesting subdivision if needed."""
    n, h = int(metadata['resolution']), int(metadata['background_shell_levels'])
    root, core_l = 1024 << h, 10-(n.bit_length()-1)
    edge = op.I(1) << op.I(log_size)
    valid = (xyz[0] >= 0) & (xyz[0]+edge <= root)
    radius2 = op.abs(2*xyz[0]+edge-root)
    for a in range(1, 3):
        valid &= (xyz[a] >= 0) & (xyz[a]+edge <= root)
        radius2 = op.maximum(radius2, op.abs(2*xyz[a]+edge-root))
    bg = radius2 >= 1024
    shell = op.I(radius2)*0
    for k in range(1, h):
        shell = op.choose(radius2 >= (1024 << k), k, shell)
    coarse_l = 9+shell
    # The configured finest depth bounds growth as well as refinement.
    max_depth = op.minimum(ADAPTIVE_MAX_DEPTH, adaptive_depth_limit(metadata, shell))
    unified = metadata.get('foreground_refinement') == 'adaptive'
    if unified:
        finest_l = 10-(int(metadata.get('foreground_max_resolution', 1024)).bit_length()-1)
        # Keep the foreground's initialization scale as its coarsest prior.
        # A coarse shell entering the core splits to that scale; existing fine
        # core/shell neighbors keep their finer size instead of merging back.
        core_target_l = op.minimum(op.maximum(log_size, finest_l), core_l)
        core_needs_split = log_size > core_l
    else:
        core_target_l, core_needs_split = core_l, log_size > core_l
    desired_l = op.I(op.choose(bg, op.maximum(log_size, coarse_l-max_depth), core_target_l))
    needs_split = op.choose(bg, log_size > coarse_l, core_needs_split)
    lower = [op.I(a >> op.I(desired_l)) << op.I(desired_l) for a in xyz]
    coarse_xyz = [((lower[a]-root//2) >> op.I(coarse_l))+2 for a in range(3)]
    inside_hole = (coarse_xyz[0] >= 1) & (coarse_xyz[0] <= 2)
    for a in range(1,3):
        inside_hole &= (coarse_xyz[a] >= 1) & (coarse_xyz[a] <= 2)
    needs_split |= bg & inside_hole
    # A cube awaiting subdivision must query its whole volume, not merely the
    # first future core/coarse-shell child selected by its center.
    desired_l=op.I(op.choose(needs_split,log_size,desired_l))
    lower=[op.I(op.choose(needs_split,xyz[a],lower[a])) for a in range(3)]
    flat = coarse_xyz[0]+4*coarse_xyz[1]+16*coarse_xyz[2]
    ordinal = flat
    for hole in _HOLES:
        ordinal = ordinal-op.I(flat > hole)
    depth = op.maximum(coarse_l-desired_l, 0)
    local = [((lower[a]-root//2) >> op.I(desired_l)) -
             ((coarse_xyz[a]-2) << op.I(depth)) for a in range(3)]
    extended = depth > ADAPTIVE_LEGACY_MAX_DEPTH
    coarse_address = op.C(shell*56+ordinal)
    morton = _morton(local, ADAPTIVE_MAX_DEPTH, op)
    address_bg = op.C(ADAPTIVE_TAG) | op.C(op.choose(extended,
        op.C(ADAPTIVE_EXTENDED_TAG) | (coarse_address << ADAPTIVE_PATH_BITS), coarse_address << 32)) | \
        (op.C(1) << op.U(3*depth)) | morton
    address_bg = op.C(op.choose(depth == 18,
        op.C(ADAPTIVE_DEEPEST_TAG) | ((coarse_address-504) << 54) | morton, address_bg))
    if unified:
        core_depth = op.maximum(1, 10-desired_l)
        core_coords = [(a-(root-1024)//2) >> op.I(desired_l) for a in lower]
        address_core = (op.C(CORE_ADAPTIVE_TAG) | (op.C(1) << op.U(3*core_depth))
                        | _morton(core_coords, 10, op))
    else:
        core_coords = [(a-(root-1024)//2) >> core_l for a in lower]
        address_core = op.C(core_coords[0]+n*(core_coords[1]+n*core_coords[2]))
    return op.C(op.choose(bg, address_bg, address_core)), lower, desired_l, valid, needs_split


def _index_numpy(cells, metadata, np):
    if not len(cells):
        return np.empty(0,np.uint64), np.empty(0,np.uint64)
    op = _Ops(np)
    start,end = _interval(*_decode(cells,metadata,op),metadata,op)
    order=np.argsort(start,kind='stable')
    return start[order],np.maximum.accumulate(end[order])


def _query_numpy(index, start, end, np):
    starts,ends=index
    if not len(starts):
        return np.zeros(len(start),bool),np.zeros(len(start),bool)
    p=np.searchsorted(starts,start,side='right')
    prev=ends[np.maximum(p,1)-1]
    covered=(p>0)&(prev>=end)
    overlap=((p>0)&(prev>start))|((p<len(starts))&(starts[np.minimum(p,len(starts)-1)]<end))
    return covered,overlap


def _index_gpu(cells, metadata, dr):
    if not dr.width(cells):
        return dr.cuda.UInt64([]),dr.cuda.UInt64([])
    op=_Ops(dr,True)
    start,end=_interval(*_decode(cells,metadata,op),metadata,op)
    order=dr.argsort(start)
    start=dr.gather(type(start),start,order)
    end=dr.prefix_reduce(dr.ReduceOp.Max,dr.gather(type(end),end,order),exclusive=False)
    dr.eval(start,end)
    return start,end


def _query_gpu(index,start,end,dr):
    C,U,B=dr.cuda.UInt64,dr.cuda.UInt32,dr.cuda.Bool
    starts,ends=index
    n=dr.width(starts)
    if not n:
        return dr.zeros(B,dr.width(start)),dr.zeros(B,dr.width(start))
    limit=dr.opaque(U,n)
    p=dr.binary_search(U(0),limit,lambda i:(i<limit)&(dr.gather(C,starts,i,i<limit)<=start))
    prev=dr.gather(C,ends,dr.maximum(p,1)-1)
    covered=(p>0)&(prev>=end)
    overlap=((p>0)&(prev>start))|((p<limit)&(dr.gather(C,starts,dr.minimum(p,limit-1))<end))
    return covered,overlap


def _sorted_membership_gpu(ordered,queries,dr):
    """Repeated queries against one already sorted target set."""
    C,U,B=dr.cuda.UInt64,dr.cuda.UInt32,dr.cuda.Bool
    if not dr.width(ordered):
        return dr.zeros(B,dr.width(queries))
    limit=dr.opaque(U,dr.width(ordered))
    at=dr.binary_search(U(0),limit,lambda i:(i<limit)&(dr.gather(C,ordered,i,i<limit)<queries))
    return (at<limit)&(dr.gather(C,ordered,dr.minimum(at,limit-1))==queries)


def filter_nonoverlapping_growth_numpy(cells, metadata, np):
    """Keep finest disjoint proposals; equal cells keep their first input row."""
    cells=np.asarray(cells,dtype=np.uint64)
    if not len(cells):
        return np.empty(0,np.int64)
    op=_Ops(np)
    start,end=_interval(*_decode(cells,metadata,op),metadata,op)
    order=np.lexsort((np.arange(len(cells)),end,start))
    s,e=start[order],end[order]
    # A finer dyadic interval is either at the same start or the next start.
    first=np.r_[True,s[1:]!=s[:-1]]
    next_index=np.searchsorted(s,s,side='right')
    descendant=(next_index<len(s))&(s[np.minimum(next_index,len(s)-1)]<e)
    keep=first&~descendant
    return np.sort(order[keep])


def filter_nonoverlapping_growth_gpu(cells, metadata, dr):
    C,U=dr.cuda.UInt64,dr.cuda.UInt32
    n=dr.width(cells)
    if not n:
        return dr.empty(U,0)
    op=_Ops(dr,True)
    start,end=_interval(*_decode(C(cells),metadata,op),metadata,op)
    by_end=dr.argsort(end)
    order=dr.gather(U,by_end,dr.argsort(dr.gather(C,start,by_end)))
    s,e=dr.gather(C,start,order),dr.gather(C,end,order)
    row=dr.arange(U,n)
    first=(row==0)|(s!=dr.gather(C,s,dr.maximum(row,1)-1))
    limit=dr.opaque(U,n)
    following=dr.binary_search(U(0),limit,lambda i:(i<limit)&(dr.gather(C,s,i,i<limit)<=s))
    descendant=(following<limit)&(dr.gather(C,s,dr.minimum(following,limit-1))<e)
    return dr.sort(dr.gather(U,order,dr.compress(first&~descendant)))


def _options(manager, metadata):
    cfg=manager.settings
    evidence=cfg.use_growth_evidence or cfg.use_rgb_growth
    if evidence:
        penetration=cfg.growth_evidence_penetration
    elif manager.schedule.enabled:
        penetration=manager.schedule.coarsest_growth_face_penetration*(1 if
            metadata.get('core_resolution',metadata['resolution'])==manager.schedule.start_resolution
            else manager.schedule.growth_finer_level_multiplier)
    else:
        penetration=cfg.growth_face_penetration
    offsets=([(x,y,z) for z in (-1,0,1) for y in (-1,0,1) for x in (-1,0,1)
              if (x,y,z)!=(0,0,0)] if evidence and cfg.growth_use_26_neighbors else
             [tuple(sign if a==axis else 0 for a in range(3)) for axis in range(3) for sign in (-1,1)])
    return offsets,min(max(float(penetration),0.),4.),evidence


def _empty_numpy(np):
    return dict(parent_ids=np.empty(0,np.int64),parent_cells=np.empty(0,np.uint64),
                cells=np.empty(0,np.uint64),centers=np.empty((0,3)),
                log_scales=np.empty((0,3)),rotations=np.empty((0,4)))


def _geometry_numpy(state, metadata, parents, target_xyz, target_l, offset, penetration, cfg, np):
    from reconstruction_topology import _quaternion_matrix,_contact
    from reconstruction_world_shell import leaf_boxes_numpy
    occupied=np.asarray(state['cell_indices'],dtype=np.uint64)
    lower,upper=leaf_boxes_numpy(occupied[parents],metadata['resolution'],metadata['grid_min'],
                                metadata['grid_max'],metadata['background_shell_levels'],np)
    center=np.asarray(state['centers'],dtype=np.float64).reshape(-1,3)[parents]
    axes=np.exp(np.clip(np.asarray(state['log_scales'],dtype=np.float64).reshape(-1,3)[parents],-30,30))
    rotation,q=_quaternion_matrix(np.asarray(state['rotations']).reshape(-1,4)[parents],np)
    unit=(float(metadata['grid_max'][0])-float(metadata['grid_min'][0]))/1024
    root=1024<<int(metadata['background_shell_levels'])
    root_lo=(np.asarray(metadata['grid_min'])+metadata['grid_max'])*.5-root*.5*unit
    target_lo=root_lo+np.stack(target_xyz,axis=1)*unit
    edge=np.exp2(target_l)*unit
    contact_scale=np.minimum((upper[:,0]-lower[:,0])/edge,1.)
    center=(lower+center*(upper-lower)-target_lo)/edge[:,None]
    transform=rotation/axes[:,None,:]
    metric=np.einsum('nik,njk->nij',transform,transform)*edge[:,None,None]**2
    extent=np.sqrt(np.sum((rotation*axes[:,None,:])**2,axis=2))/edge[:,None]
    opposite=-np.asarray(offset)
    touched,contact=_contact(center,metric,opposite,0.,np)
    if penetration>0:
        deep,_=_contact(center,metric,opposite,-penetration,np)
        touched&=deep
    # Promotion into a coarser neighbor must not move a tiny inherited
    # ellipsoid by 20% of that much larger leaf. Preserve the parent's world
    # contact offset; equal-size growth is exactly the existing rule.
    child=np.minimum(np.maximum(contact,.05*contact_scale[:,None]),1-.05*contact_scale[:,None])
    for a in np.flatnonzero(offset):
        gap=min(max(cfg.growth_contact_offset,.01),.49)*contact_scale
        child[:,a]=gap if offset[a]>0 else 1-gap
    margin=np.minimum(child,1-child)
    for a in np.flatnonzero(offset):
        margin[:,a]=1-child[:,a] if offset[a]>0 else child[:,a]
    factor=np.minimum(min(max(cfg.growth_shrink,.01),.99),np.min(np.maximum(margin-.005*contact_scale[:,None],0)/np.maximum(extent,1e-20),axis=1))
    touched&=(factor>0)&np.all(axes*factor[:,None]>=edge[:,None]*1e-4,axis=1)
    return touched,child,np.log(np.maximum(axes*factor[:,None],1e-300)),q


def generate_adaptive_growth_numpy(manager, checkpoint, boundary, target_cells=None):
    """CPU reference, including core/core and mixed-size background neighbors."""
    np,cfg=manager.np,manager.settings
    state,metadata=checkpoint['model_state'],checkpoint['model_metadata']
    op=_Ops(np)
    cells=np.asarray(state['cell_indices'],dtype=np.uint64)
    xyz,logs=_decode(cells,metadata,op)
    wait=np.asarray(manager.growth_wait_start)
    eligible=~np.asarray(manager.candidates)&((wait==0)|((boundary>=wait-1)&(boundary-(wait-1)>=cfg.growth_wait_epochs)))
    cooldown=np.asarray([c for c,e in manager.cooldown.items() if boundary<e],dtype=np.uint64)
    index=_index_numpy(np.concatenate((cells,cooldown)),metadata,np)
    offsets,penetration,evidence=_options(manager,metadata)
    cap=max(0,int(getattr(cfg,'background_growth_max_candidates',200000)))
    remaining=cap
    pieces=[]
    unified=metadata.get('foreground_refinement') == 'adaptive'
    # Separate core/core enumeration preserves its original unlimited capacity.
    for extra in ((True,) if unified else (False,True)):
        for offset in offsets:
            ids=np.flatnonzero(eligible)
            for begin in range(0,len(ids),2048):
                parents=ids[begin:begin+2048]
                if not extra:
                    parents=parents[cells[parents]<ADAPTIVE_TAG]
                if extra and remaining<=0:
                    break
                active=[xyz[a][parents]+offset[a]*np.left_shift(1,logs[parents]) for a in range(3)]
                level=logs[parents]
                while len(parents):
                    targets,anchor,leaf_l,valid,split=_canonical(active,level,metadata,op)
                    mixed=(cells[parents]>=ADAPTIVE_TAG)|(targets>=ADAPTIVE_TAG)
                    if not unified:
                        valid&=mixed if extra else ~mixed
                    start,end=_interval(anchor,leaf_l,metadata,op)
                    covered,overlap=_query_numpy(index,start,end,np)
                    split|=overlap&~covered
                    valid&=~covered
                    accepted=valid&~split
                    if target_cells is not None:
                        accepted&=np.isin(targets,np.asarray(target_cells,dtype=np.uint64))
                    rows=np.flatnonzero(accepted)
                    if len(rows):
                        touched,center,scale,q=_geometry_numpy(state,metadata,parents[rows],
                            [a[rows] for a in anchor],leaf_l[rows],offset,penetration,cfg,np)
                        selected=rows[touched]
                        if extra:
                            selected=selected[:remaining]
                            take=np.flatnonzero(touched)[:remaining]
                            remaining-=len(selected)
                        else:
                            take=np.flatnonzero(touched)
                        if len(selected):
                            pieces.append(dict(parent_ids=parents[selected],parent_cells=cells[parents[selected]],
                                cells=targets[selected],centers=center[take],log_scales=scale[take],rotations=q[take]))
                    if extra and remaining<=0:
                        break
                    rows=np.flatnonzero(valid&split&(level>0))
                    if not len(rows):
                        break
                    children=[(x,y,z) for z in (0,1) for y in (0,1) for x in (0,1)
                              if all(not offset[a] or (x,y,z)[a]==int(offset[a]<0) for a in range(3))]
                    parents=np.concatenate([parents[rows] for _ in children])
                    newlevel=np.concatenate([level[rows]-1 for _ in children])
                    active=[np.concatenate([active[a][rows]+child[a]*np.left_shift(1,level[rows]-1)
                                             for child in children]) for a in range(3)]
                    level=newlevel
                    if extra:
                        # Bound each frontier before geometry or child expansion.
                        limit=max(1,remaining)
                        parents,level=parents[:limit],level[:limit]
                        active=[a[:limit] for a in active]
    result={k:np.concatenate([p[k] for p in pieces]) for k in _WIDTHS} if pieces else _empty_numpy(np)
    if len(result['cells']):
        order=np.lexsort((result['parent_cells'],result['cells']))
        unique=np.r_[True,(result['cells'][order][1:]!=result['cells'][order][:-1])|
                     (result['parent_cells'][order][1:]!=result['parent_cells'][order][:-1])]
        order=order[unique]
        result={k:v[order] for k,v in result.items()}
        _,inverse,counts=np.unique(result['cells'],return_inverse=True,return_counts=True)
        keep=counts[inverse]>=max(1,min(26,cfg.growth_min_supporting_parents)) if evidence else np.ones(len(order),bool)
        result={k:v[keep] for k,v in result.items()}
        targets=np.unique(result['cells'])
        targets=targets[filter_nonoverlapping_growth_numpy(targets,metadata,np)]
        keep=np.isin(result['cells'],targets)
        result={k:v[keep] for k,v in result.items()}
    count=len(result['cells'])
    result.update(boundary=boundary,resolution=metadata['resolution'],seen_views=set(),
                  foreground=np.zeros(count,np.uint32),background=np.zeros(count,np.uint32))
    return result


def _geometry_gpu(state, metadata, parents, target_xyz, target_l, offset, penetration, cfg, dr,
                  parameter_parent_rows=None):
    from reconstruction_growth_candidates_gpu import _fields, geometry_gpu, contact_gpu, gather_rows, _pack
    U,C,D=dr.cuda.UInt32,dr.cuda.UInt64,dr.cuda.Float64
    op=_Ops(dr,True)
    parameter_ids=parents if parameter_parent_rows is None else dr.gather(U,parameter_parent_rows,parents)
    local={k:gather_rows(state[k],parameter_ids,w,dr) for k,w in
           dict(centers=3,log_scales=3,rotations=4).items()}
    local['cell_indices']=dr.gather(C,state['cell_indices'],parents)
    center,axes,q,_,metric,extent=geometry_gpu(local,metadata,dr)
    parent_xyz,parent_l=_decode(local['cell_indices'],metadata,op)
    unit=(float(metadata['grid_max'][0])-float(metadata['grid_min'][0]))/1024
    parent_edge=D(dr.cuda.Int32(1)<<dr.cuda.Int32(parent_l))*unit
    target_edge=D(dr.cuda.Int32(1)<<dr.cuda.Int32(target_l))*unit
    contact_scale=dr.minimum(parent_edge/target_edge,1.)
    core_edge=(float(metadata['grid_max'][0])-float(metadata['grid_min'][0]))/metadata['resolution']
    center=[(D(parent_xyz[a]-target_xyz[a])*unit+center[a]*parent_edge)/target_edge for a in range(3)]
    ratio=target_edge/core_edge
    metric=[[v*ratio*ratio for v in row] for row in metric]
    extent=[v/ratio for v in extent]
    opposite=tuple(-v for v in offset)
    touched,contact=contact_gpu(center,metric,opposite,0.,dr)
    if penetration>0:
        deep,_=contact_gpu(center,metric,opposite,-penetration,dr)
        touched&=deep
    child=[dr.minimum(dr.maximum(v,.05*contact_scale),1-.05*contact_scale) for v in contact]
    for a in range(3):
        if offset[a]:
            value=min(max(cfg.growth_contact_offset,.01),.49)
            gap=value*contact_scale
            child[a]=gap if offset[a]>0 else 1-gap
    margin=[dr.minimum(v,1-v) for v in child]
    for a in range(3):
        if offset[a]:
            margin[a]=1-child[a] if offset[a]>0 else child[a]
    factor=dr.full(D,min(max(cfg.growth_shrink,.01),.99),dr.width(parents))
    for a in range(3):
        factor=dr.minimum(factor,dr.maximum(margin[a]-.005*contact_scale,0)/dr.maximum(extent[a],1e-20))
    touched&=factor>0
    for a in range(3):
        touched&=axes[a]*factor>=target_edge*1e-4
    return touched,_pack(child,D,dr),_pack([dr.log(dr.maximum(v*factor,1e-300)) for v in axes],D,dr),_pack(q,D,dr)


def _concat(pieces, dr):
    """Concatenate geometry while preserving full addresses and bounded scratch."""
    U,C,D=dr.cuda.UInt32,dr.cuda.UInt64,dr.cuda.Float64
    result={}
    count=sum(dr.width(p['cells']) for p in pieces)
    for name,width in _WIDTHS.items():
        T=C if name in ('cells','parent_cells') else U if name=='parent_ids' else D
        out=dr.empty(T,count*width)
        begin=0
        for piece in pieces:
            value=piece.pop(name)
            dr.scatter(out,T(value),dr.arange(U,dr.width(value))+dr.opaque(U,begin),mode=dr.ReduceMode.Permute)
            begin+=dr.width(value)
        dr.eval(out)
        result[name]=out
        if pieces:
            del value
    pieces.clear()
    return result


class _BackgroundProposalBuffer:
    """Bounded prefix append preserving the original frontier enumeration.

    A nonempty accepted batch allocates the configured cap once. Each batch is
    scattered immediately, so thousands of tiny geometry arrays do not survive
    until the final merge. Empty/cap-zero streams allocate no proposal fields.
    """

    def __init__(self, capacity, dr):
        self.capacity, self.dr = int(capacity), dr
        self.count = 0
        self.fields = None

    def append(self, piece):
        dr = self.dr
        count = dr.width(piece['cells'])
        if not count:
            return
        if self.count+count > self.capacity:
            raise RuntimeError("Background proposal append exceeds its candidate cap")
        U,C,D = dr.cuda.UInt32,dr.cuda.UInt64,dr.cuda.Float64
        if self.fields is None:
            self.fields = {
                name: dr.empty(C if name in ('cells','parent_cells') else U if name=='parent_ids' else D,
                               self.capacity*width)
                for name,width in _WIDTHS.items()}
        for name,width in _WIDTHS.items():
            positions = dr.arange(U,count*width)+dr.opaque(U,self.count*width)
            dr.scatter(self.fields[name],piece[name],positions,mode=dr.ReduceMode.Permute)
        # Finish this batch's scatter graph before constructing the next one.
        # This retains only the fixed-capacity destination, not every source.
        dr.eval(*self.fields.values())
        self.count += count

    def finish(self):
        if self.fields is None:
            return None
        dr,U = self.dr,self.dr.cuda.UInt32
        result = {}
        for name,width in _WIDTHS.items():
            source = self.fields.pop(name)
            result[name] = (source if self.count==self.capacity else
                            dr.gather(type(source),source,dr.arange(U,self.count*width)))
            dr.eval(result[name])
        self.fields = None
        return result


def _append_appearance(result,state,metadata,cfg,dr,parameter_parent_rows=None):
    from reconstruction_topology import validate_growth_opacity
    U,C,F,D=dr.cuda.UInt32,dr.cuda.UInt64,dr.cuda.Float32,dr.cuda.Float64
    validate_growth_opacity(cfg,metadata['opacity_sh_count'],
                            active_opacity_sh_count=metadata.get('active_opacity_sh_count'))
    n=dr.width(result['cells'])
    parents=result['parent_ids']
    if parameter_parent_rows is not None:
        parents=dr.gather(U,parameter_parent_rows,parents)
    inverse_y0=2*math.sqrt(math.pi)
    if cfg.growth_inherit_opacity:
        z=D(dr.gather(F,state['opacity_sh'],parents*metadata['opacity_sh_count']))/inverse_y0
        log_alpha=dr.minimum(math.log(cfg.growth_initial_opacity),math.log(cfg.growth_opacity_scale)-
                            (dr.maximum(-z,0)+dr.log1p(dr.exp(-dr.abs(z)))))
        b0=(log_alpha-dr.log(-dr.expm1(log_alpha)))*inverse_y0
    else:
        alpha=min(max(float(cfg.growth_initial_opacity),.01),.99)
        b0=dr.full(D,math.log(alpha/(1-alpha))*inverse_y0,n)
    opacity=dr.zeros(F,n*metadata['opacity_sh_count'])
    radiance=dr.zeros(F,n*metadata['sh_count']*3)
    rows=dr.arange(U,n)
    dr.scatter(opacity,F(b0),rows*metadata['opacity_sh_count'])
    for channel in range(3):
        dr.scatter(radiance,dr.gather(F,state['radiance_sh'],parents*metadata['sh_count']*3+channel),
                   rows*metadata['sh_count']*3+channel)
    result.update(radiance_sh=radiance,opacity_sh=opacity)
    result['candidate_state']=dict(cell_indices=C(result['cells']),centers=F(result['centers']),
        log_scales=F(result['log_scales']),rotations=F(result['rotations']),radiance_sh=radiance,opacity_sh=opacity)
    dr.eval(*result['candidate_state'].values())


def _merge_proposal_rows_gpu(cells, parent_cells, metadata, minimum_support, dr):
    """Return the canonical retained rows without repeatedly moving geometry.

    Stable pair ordering preserves the first proposal's geometry when multiple
    traversal frontiers produced the same (target, parent). Support counts are
    computed after this pair deduplication, before competing coarse targets are
    removed, exactly as in the original full-field merge.
    """
    from reconstruction_growth_filter import growth_pair_order_gpu
    C,U,B=dr.cuda.UInt64,dr.cuda.UInt32,dr.cuda.Bool
    n=dr.width(cells)
    if not n:
        return dr.empty(U,0),dr.empty(U,0)
    order=growth_pair_order_gpu(cells,parent_cells,dr)
    ordered_cells=dr.gather(C,cells,order)
    ordered_parents=dr.gather(C,parent_cells,order)
    lane=dr.arange(U,n)
    prev=dr.maximum(lane,1)-1
    distinct=(lane==0)|(ordered_cells!=dr.gather(C,ordered_cells,prev))|\
             (ordered_parents!=dr.gather(C,ordered_parents,prev))
    unique=dr.compress(distinct)
    order=dr.gather(U,order,unique)
    ordered_cells=dr.gather(C,ordered_cells,unique)
    del ordered_parents,unique,distinct,prev,lane
    n=dr.width(order)
    lane=dr.arange(U,n)
    first=(lane==0)|(ordered_cells!=dr.gather(C,ordered_cells,dr.maximum(lane,1)-1))
    groups=dr.cumsum(U(first))-1
    # A row-sized counter table avoids an extra compress/count host barrier;
    # unused trailing entries are zero and never read.
    counts=dr.zeros(U,n)
    dr.scatter_reduce(dr.ReduceOp.Add,counts,U(1),groups)
    support=dr.gather(U,counts,groups)
    keep=dr.compress(support>=minimum_support) if minimum_support>1 else lane
    order=dr.gather(U,order,keep)
    ordered_cells=dr.gather(C,ordered_cells,keep)
    support=dr.gather(U,support,keep)
    del first,groups,counts,keep,lane
    n=dr.width(order)
    if not n:
        return order,support
    lane=dr.arange(U,n)
    first=(lane==0)|(ordered_cells!=dr.gather(C,ordered_cells,dr.maximum(lane,1)-1))
    target_rows=dr.compress(first)
    unique_targets=dr.gather(C,ordered_cells,target_rows)
    retained=filter_nonoverlapping_growth_gpu(unique_targets,metadata,dr)
    # Every proposal of an exact target shares its group. Broadcasting a mask
    # over those groups replaces a binary search per parent proposal.
    target_keep=dr.zeros(B,dr.width(unique_targets))
    dr.scatter(target_keep,B(True),retained)
    groups=dr.cumsum(U(first))-1
    keep=dr.compress(dr.gather(B,target_keep,groups))
    order,support=dr.gather(U,order,keep),dr.gather(U,support,keep)
    dr.eval(order,support)
    return order,support


def generate_adaptive_growth_gpu(manager, checkpoint, boundary, target_cells=None, *, eligible=None,
                                 cooldown_cells=None, parent_batch_size=65536, dr=None):
    """Generate bounded mixed-leaf proposals; retain the legacy core fast path.

    Geometry, occupancy indexes, wait masks, and candidates remain on device.
    Only compressed sizes are read by the bounded batching loops.
    """
    if dr is None:
        import drjit as dr
    from reconstruction_growth_candidates_gpu import _generate,gather_rows
    U,C,I,D,F,B=dr.cuda.UInt32,dr.cuda.UInt64,dr.cuda.Int32,dr.cuda.Float64,dr.cuda.Float32,dr.cuda.Bool
    op=_Ops(dr,True)
    timings=manager.growth_generation_stage_seconds={}
    started=time.perf_counter()
    state,metadata,cfg=checkpoint['model_state'],checkpoint['model_metadata'],manager.settings
    state={k:C(v) if k=='cell_indices' else F(v) for k,v in state.items()}
    occupied=state['cell_indices']
    unified=metadata.get('foreground_refinement') == 'adaptive'
    if not dr.width(occupied) or (target_cells is not None and not dr.width(target_cells)):
        if unified:
            result=_concat([],dr)
            result.update(boundary=boundary,resolution=metadata['resolution'],seen_views=set(),
                          foreground=U([]),background=U([]),geometric_support=U([]),
                          radiance_sh=F([]),opacity_sh=F([]))
            result['candidate_state']={k:dr.empty(C if k=='cell_indices' else F,0) for k in state}
            return result
        result=_generate(manager,checkpoint,boundary,target_cells,eligible,cooldown_cells,parent_batch_size,dr)
        result['cells'],result['parent_cells']=C(result['cells']),C(result['parent_cells'])
        return result
    if eligible is None:
        wait=dr.cuda.Int64(manager.growth_wait_start)
        eligible=~B(manager.candidates)&((wait==0)|((boundary>=wait-1)&(boundary-(wait-1)>=cfg.growth_wait_epochs)))
    if cooldown_cells is None:
        cooldown_cells=C([c for c,e in manager.cooldown.items() if boundary<e])
    parameter_rows=checkpoint.get('parameter_parent_rows')
    if parameter_rows is not None:
        parameter_rows=U(parameter_rows)
    # Do not discard a core target supported by one core and one background
    # parent before their two proposal streams have been combined.
    pieces=[]
    if not unified:
        core=_generate(manager,checkpoint,boundary,target_cells,eligible,cooldown_cells,parent_batch_size,dr,
                       minimum_support=1,geometry_only=True)
        pieces=[{k:core[k] for k in _WIDTHS}] if dr.width(core['cells']) else []
        del core
    timings['core_candidates']=time.perf_counter()-started
    started=time.perf_counter()
    offsets,penetration,evidence=_options(manager,metadata)
    remaining=max(0,int(getattr(cfg,'background_growth_max_candidates',200000)))
    background_buffer=_BackgroundProposalBuffer(remaining,dr)
    xyz,logs=_decode(occupied,metadata,op)
    # Only background parents and the one-cell core boundary can cross into a
    # background leaf. Interior foreground geometry never enters this path.
    root=1024<<int(metadata['background_shell_levels'])
    clo,chi=(root-1024)//2,(root+1024)//2
    edge=I(1)<<I(logs)
    boundary_parent=(occupied>=ADAPTIVE_TAG)
    for a in range(3):
        boundary_parent|=(xyz[a]==clo)|(xyz[a]+edge==chi)
    parent_ids=dr.compress(B(eligible) if unified else B(eligible)&boundary_parent)
    blocked=occupied
    if dr.width(cooldown_cells):
        blocked=(dr.concat([occupied,C(cooldown_cells)]) if dr.width(occupied) else C(cooldown_cells))
    index=_index_gpu(blocked,metadata,dr)
    selected_targets=dr.sort(C(target_cells)) if target_cells is not None else None
    if selected_targets is not None:
        dr.eval(selected_targets)
    timings['mixed_index']=time.perf_counter()-started
    started=time.perf_counter()
    batch=min(max(1,parent_batch_size),2048)
    for offset in offsets:
        if remaining<=0 or not dr.width(parent_ids):
            break
        # One resident address-only pass per direction removes interior
        # parents. A dense shell must not cause thousands of synchronized
        # 2048-row geometry/frontier batches that all turn out occupied.
        initial_l=dr.gather(I,logs,parent_ids)
        initial_xyz=[dr.gather(I,xyz[a],parent_ids)+offset[a]*(I(1)<<I(initial_l)) for a in range(3)]
        initial_cells,initial_xyz,initial_l,initial_valid,_=_canonical(initial_xyz,initial_l,metadata,op)
        if not unified:
            initial_valid&=(dr.gather(C,occupied,parent_ids)>=ADAPTIVE_TAG)|(initial_cells>=ADAPTIVE_TAG)
        initial_start,initial_end=_interval(initial_xyz,initial_l,metadata,op)
        initial_covered,_=_query_gpu(index,initial_start,initial_end,dr)
        exposed=dr.gather(U,parent_ids,dr.compress(initial_valid&~initial_covered))
        dr.eval(exposed)
        del initial_l,initial_xyz,initial_cells,initial_valid,initial_start,initial_end,initial_covered
        for begin in range(0,dr.width(exposed),batch):
            if remaining<=0:
                break
            parents=dr.gather(U,exposed,dr.arange(U,min(batch,dr.width(exposed)-begin))+dr.opaque(U,begin))
            level=dr.gather(I,logs,parents)
            active=[dr.gather(I,xyz[a],parents)+offset[a]*(I(1)<<I(level)) for a in range(3)]
            while dr.width(parents):
                targets,anchor,leaf_l,valid,split=_canonical(active,level,metadata,op)
                if not unified:
                    valid&=(dr.gather(C,occupied,parents)>=ADAPTIVE_TAG)|(targets>=ADAPTIVE_TAG)
                start,end=_interval(anchor,leaf_l,metadata,op)
                covered,overlap=_query_gpu(index,start,end,dr)
                split|=overlap&~covered
                valid&=~covered
                accepted=valid&~split
                if selected_targets is not None:
                    accepted&=_sorted_membership_gpu(selected_targets,targets,dr)
                rows=dr.compress(accepted)
                if dr.width(rows):
                    ids=dr.gather(U,parents,rows)
                    touched,center,scale,q=_geometry_gpu(state,metadata,ids,
                        [dr.gather(I,a,rows) for a in anchor],dr.gather(I,leaf_l,rows),offset,penetration,cfg,dr,parameter_rows)
                    take=dr.compress(touched)
                    if dr.width(take)>remaining:
                        take=dr.gather(U,take,dr.arange(U,remaining))
                    if dr.width(take):
                        ids=dr.gather(U,ids,take)
                        background_buffer.append(dict(parent_ids=ids,parent_cells=dr.gather(C,occupied,ids),
                            cells=dr.gather(C,targets,dr.gather(U,rows,take)),centers=gather_rows(center,take,3,dr),
                            log_scales=gather_rows(scale,take,3,dr),rotations=gather_rows(q,take,4,dr)))
                        remaining-=dr.width(take)
                if remaining<=0:
                    break
                rows=dr.compress(valid&split&(level>0))
                if not dr.width(rows):
                    break
                children=[(x,y,z) for z in (0,1) for y in (0,1) for x in (0,1)
                          if all(not offset[a] or (x,y,z)[a]==int(offset[a]<0) for a in range(3))]
                size=min(dr.width(rows)*len(children),remaining)
                lane=dr.arange(U,size)
                divisor=dr.opaque(U,dr.width(rows))
                source=dr.gather(U,rows,lane%divisor)
                child=lane//divisor
                newlevel=dr.gather(I,level,source)-1
                active=[dr.gather(I,active[a],source)+dr.gather(I,I([c[a] for c in children]),child)*(I(1)<<I(newlevel))
                        for a in range(3)]
                parents=dr.gather(U,parents,source)
                level=newlevel
                dr.eval(parents,level,*active)
    background=background_buffer.finish()
    if background is not None:
        pieces.append(background)
    timings['background_candidates']=time.perf_counter()-started
    started=time.perf_counter()
    result=_concat(pieces,dr)
    n=dr.width(result['cells'])
    if n:
        minimum_support=max(1,min(26,cfg.growth_min_supporting_parents)) if evidence else 1
        order,support=_merge_proposal_rows_gpu(result['cells'],result['parent_cells'],metadata,
                                              minimum_support,dr)
        # Parameters are moved only once after the final decision. One field at
        # a time also bounds replacement scratch when proposal counts are large.
        for key,width in _WIDTHS.items():
            previous=result.pop(key)
            result[key]=gather_rows(previous,order,width,dr)
            dr.eval(result[key])
            del previous
        result['geometric_support']=support
    else:
        result['geometric_support']=U([])
    n=dr.width(result['cells'])
    result.update(boundary=boundary,resolution=metadata['resolution'],seen_views=set(),
                  foreground=dr.zeros(U,n),background=dr.zeros(U,n))
    timings['merge_filter']=time.perf_counter()-started
    started=time.perf_counter()
    if n:
        _append_appearance(result,state,metadata,cfg,dr,parameter_rows)
    else:
        result.update(radiance_sh=F([]),opacity_sh=F([]))
        result['candidate_state']={k:dr.empty(C if k=='cell_indices' else F,0) for k in state}
    timings['appearance']=time.perf_counter()-started
    return result
