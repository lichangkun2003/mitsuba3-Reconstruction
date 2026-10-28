"""Independent world-space cell-preimage oracle for piecewise projective tracing.

The oracle intersects each leaf's projective preimage in world space, rather
than sharing the native region cursor or its DDA/octree stepping.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preview import RuntimeConfig
from reconstruction_common import initialize_runtime, restore_model
from reconstruction_replay import NativeLossReplay
from test_reconstruction_distortion import rotation_matrix, perturb

Y00 = 1 / (2 * math.sqrt(math.pi))


def fixture(np):
    core, layers = 4, 1
    grid = core + 2 * layers
    coords = [(x,y,z) for z in range(grid) for y in range(grid) for x in range(grid)
              if sum(not layers <= c < layers+core for c in (x,y,z)) <= 1]
    count = len(coords)
    rng = np.random.default_rng(415)
    axes = rng.normal(size=(count,3)); axes /= np.linalg.norm(axes,axis=1)[:,None]
    angle = rng.uniform(-.4,.4,size=count)
    rotations = np.column_stack((np.cos(angle/2),axes*np.sin(angle/2)[:,None]))
    return {"model_metadata": {"resolution": grid, "core_resolution": core,
        "grid_min": [-1.5]*3, "grid_max": [1.5]*3,
        "scene_mapping": "projective_shell", "background_refinement": "preserve_physical_width", "background_layers": layers,
        "sh_count": 1, "opacity_sh_count": 1},
        "model_state": {"cell_indices": np.array([x+grid*(y+grid*z) for x,y,z in coords],np.uint32),
        "centers": np.tile([.53,.46,.48],count),
        "log_scales": np.tile(np.log([.46,.35,.42]),count),
        "rotations": rotations.ravel(),
        "radiance_sh": (rng.uniform(.15,.85,size=(count,3))/Y00).ravel(),
        "opacity_sh": np.full(count,math.log(.12/.88)/Y00)}}


def oracle(payload, origin, direction, np, *, maxt=math.inf, max_hits=64, threshold=0.):
    """Intersect the linear halfspaces of every leaf preimage, then ellipsoids."""
    md, state = payload["model_metadata"], payload["model_state"]
    grid, core = md["resolution"], md["core_resolution"]
    pad = (grid-core)//2
    size = 2/core
    width = 2*pad/core
    o, d = np.asarray(origin,float), np.asarray(direction,float)
    length = np.linalg.norm(d)
    d = d/length
    hits = []
    for row, spatial in enumerate(state["cell_indices"]):
        cell = np.array([spatial%grid,spatial//grid%grid,spatial//grid**2])
        low, high = -1-pad*size+cell*size, -1-pad*size+(cell+1)*size
        outside = np.flatnonzero((cell<pad)|(cell>=pad+core))
        axis = int(outside[0]) if len(outside) else None
        sign = (-1 if cell[axis]<pad else 1) if axis is not None else 1
        bounds = [0.,maxt*length]
        valid = True
        def clip(a,b,strict=False):
            nonlocal valid
            if abs(b)<1e-14:
                valid &= a>0 if strict else a>=-1e-12
            elif b>0:
                bounds[0] = max(bounds[0],-a/b)
            else:
                bounds[1] = min(bounds[1],-a/b)
        if axis is None:
            for j in range(3):
                clip(o[j]-low[j],d[j])
                clip(high[j]-o[j],-d[j],cell[j]!=pad+core-1)
        else:
            h0, hd = sign*o[axis], sign*d[axis]
            qlow,qhigh = sorted((sign*low[axis],sign*high[axis]))
            min_h = width/(1+width-qlow)
            max_h = math.inf if abs(1+width-qhigh)<1e-12 else width/(1+width-qhigh)
            clip(h0-min_h,hd,abs(min_h-1)<1e-12)
            if math.isfinite(max_h): clip(max_h-h0,-hd,True)
            for j in range(3):
                if j==axis: continue
                clip(o[j]-low[j]*h0,d[j]-low[j]*hd)
                clip(high[j]*h0-o[j],high[j]*hd-d[j],cell[j]!=pad+core-1)
                # Same deterministic ownership on exactly coplanar chart seams.
                if j<axis:
                    clip(h0-o[j],hd-d[j],True)
                    clip(h0+o[j],hd+d[j],True)
        begin,end = bounds
        if not valid or end<=begin: continue
        def mapped(t):
            if axis is None: return o+t*d
            if not math.isfinite(t):
                q = d/(sign*d[axis]); q[axis]=sign*(1+width)
            else:
                p = o+t*d; h=sign*p[axis]
                q = p/h; q[axis]=sign*(1+width-width/h)
            return q
        p0,p1 = mapped(begin),mapped(end)
        center=low+state["centers"].reshape(-1,3)[row]*size
        scale=np.exp(state["log_scales"].reshape(-1,3)[row])
        rotation=rotation_matrix(state["rotations"].reshape(-1,4)[row],np)
        u=rotation.T@(p0-center)/scale
        v=rotation.T@(p1-p0)/scale
        a,b,c=v@v,2*(u@v),u@u-1
        disc=b*b-4*a*c
        if a<=1e-20: continue
        root=math.sqrt(max(disc,0))
        if disc<0 or max(0,(-b-root)/(2*a))>min(1,(-b+root)/(2*a)): continue
        star=float(np.clip(-(u@v)/a,0,1))
        q=p0+star*(p1-p0)
        if axis is not None and 1+width-sign*q[axis]<1e-14:
            depth=1.
        else:
            if axis is not None:
                h=width/(1+width-sign*q[axis])
                p=q*h; p[axis]=sign*h
            else: p=q
            t=float((p-o)@d)
            depth=t/(t+1)
        alpha=1/(1+math.exp(-state["opacity_sh"][row]*Y00))
        color=np.maximum(0,state["radiance_sh"].reshape(-1,3)[row]*Y00)
        hits.append((begin,row,alpha,color,depth,star))
    hits.sort(key=lambda x:(x[0],x[1]))
    color=np.zeros(3);transmittance=1.;weights=[];depths=[];used=[]
    for item in hits:
        if len(used)>=max_hits or transmittance<=threshold: break
        _,row,alpha,rgb,depth,star=item
        weight=transmittance*alpha
        color+=weight*rgb; transmittance*=1-alpha
        weights.append(weight);depths.append(depth);used.append(item)
    distortion=sum(2*weights[i]*weights[j]*abs(depths[i]-depths[j])
                   for i in range(len(used)) for j in range(i+1,len(used)))
    return dict(rgb=color,alpha=1-transmittance,hits=used,distortion=distortion)


def make_ray(mi, np, origins, directions, maxt=math.inf):
    o=np.asarray(origins,np.float32).reshape(-1,3);d=np.asarray(directions,np.float32).reshape(-1,3)
    ray=mi.Ray3f(mi.Point3f(*(mi.Float(o[:,i].copy()) for i in range(3))),
                 mi.Vector3f(*(mi.Float(d[:,i].copy()) for i in range(3))))
    ray.maxt=maxt
    return ray


def native(model, mi, dr, ray, *, use_octree=True, coefficient=.7, max_hits=64,
           threshold=0., replay=None, evidence=False, growth=False,
           reference=(.3,.4,.5), dssim_weight=0., image_size=None):
    options={"rgb_only":1.,"max_hits":float(max_hits),"max_candidates":4.,
        "use_octree":float(use_octree),"distortion_loss_weight":coefficient,
        "transmittance_threshold":threshold,"collect_rgb_deletion_evidence":float(evidence),
        "collect_growth_evidence":float(growth),"max_growth_hits":8.}
    if dssim_weight:
        options.update(dssim_loss_weight=dssim_weight,
                       image_width=float(image_size[0]), image_height=float(image_size[1]))
    args=(ray,mi.Color3f(*(float(v) for v in reference)),mi.Float(0),mi.Color3f(0),mi.Float(1),options)
    result=model.loss_and_grad(*args) if replay is None else replay(model,*args)
    dr.eval(result);dr.sync_thread()
    return result


def run_checks(mi,dr,np):
    assert int(getattr(mi.VoxelReconstruction,'background_rgb_deletion_version',0)) >= 1, \
        'Rebuild this checkout before testing projective background RGB deletion'
    payload=fixture(np);model=restore_model(payload,mi)
    rays=[((.8,0,0),(.6,0,.8)), ((3,.15,.23),(-1,.02,.04)),
          ((0,0,0),(1,1,0)),((0,0,0),(-1,-1,-1)),
          ((1,0,0),(0,1,0)),((.13,.22,0),(0,0,1)),
          ((0,0,0),(0,-1,0)),((1.8,-.7,.2),(-.2,.8,.1))]
    rng=np.random.default_rng(27)
    rays += [(tuple(rng.uniform(-2,2,3)),tuple(rng.normal(size=3))) for _ in range(20)]
    report={}
    for maxt in (math.inf,2.3):
        ray=make_ray(mi,np,[x[0] for x in rays],[x[1] for x in rays],maxt)
        expected=[oracle(payload,o,d,np,maxt=maxt) for o,d in rays]
        previous=None
        for octree in (False,True):
            result=native(model,mi,dr,ray,use_octree=octree,evidence=True)
            rgb,alpha,hits=model.trace(ray,mi.Color3f(0),0.,64,0.,octree)
            dr.eval(rgb,alpha,hits)
            np.testing.assert_array_equal(np.asarray(hits),[len(e['hits']) for e in expected])
            np.testing.assert_allclose(np.asarray(rgb).T,[e['rgb'] for e in expected],atol=3e-6,rtol=2e-5)
            np.testing.assert_allclose(np.asarray(result['raw_rgb']),np.asarray(rgb),atol=2e-6)
            np.testing.assert_array_equal(np.asarray(result['hit_count']),np.asarray(hits))
            np.testing.assert_allclose(np.asarray(result['distortion_loss']),[e['distortion'] for e in expected],atol=2e-6,rtol=2e-5)
            if previous is not None:
                for key in result['gradients']:
                    np.testing.assert_allclose(np.asarray(result['gradients'][key]),np.asarray(previous['gradients'][key]),atol=2e-5,rtol=2e-4)
            previous=result
        report[f'world_preimage_oracle_far_{maxt}']='passed'
    ray=make_ray(mi,np,[rays[0][0]],[rays[0][1]])
    for cap,threshold in ((2,0.),(64,.8)):
        expected=oracle(payload,*rays[0],np,max_hits=cap,threshold=threshold)
        result=native(model,mi,dr,ray,max_hits=cap,threshold=threshold,evidence=True)
        assert int(np.asarray(result['hit_count'])[0])==len(expected['hits'])
        np.testing.assert_allclose(np.asarray(result['raw_rgb']).ravel(),expected['rgb'],atol=2e-6)
        assert not np.asarray(result['deletion_observed']).any(), (cap,threshold,np.flatnonzero(np.asarray(result['deletion_observed'])),len(expected['hits']))
    report['global_hit_cap_and_transmittance']='passed'

    # Isolate exact Distortion derivatives by subtracting the identical RGB proxy.
    on=native(model,mi,dr,ray,coefficient=.7)
    off=native(model,mi,dr,ray,coefficient=0.)
    expected=oracle(payload,*rays[0],np)
    epsilon=2e-5
    checked=0
    for _,row,_,_,_,star in expected['hits']:
        if not .05<star<.95: continue
        for key,count in [('opacity_sh',1),('centers',3),('log_scales',3),('rotation_tangent',3)]:
            for component in range(count):
                index=row*count+component
                plus=perturb(payload,key,index,epsilon,np)
                minus=perturb(payload,key,index,-epsilon,np)
                numerical=.7*(oracle(plus,*rays[0],np)['distortion']-oracle(minus,*rays[0],np)['distortion'])/(2*epsilon)
                analytic=float(np.asarray(on['gradients'][key])[index]-np.asarray(off['gradients'][key])[index])
                np.testing.assert_allclose(analytic,numerical,atol=1e-5,rtol=3e-3,err_msg=f'{key}[{index}]')
                checked+=1
    assert checked>=20,checked
    report['world_depth_distortion_finite_differences']=checked
    replay=NativeLossReplay(dr)
    for i in (0,1,2,0):
        changing=make_ray(mi,np,[rays[i][0]],[rays[i][1]])
        reference=native(model,mi,dr,changing,evidence=True)
        actual=native(model,mi,dr,changing,replay=replay,evidence=True)
        for key in ('raw_rgb','hit_count','distortion_loss','deletion_gain',
                    'deletion_observed','deletion_unknown'):
            np.testing.assert_allclose(np.asarray(actual[key]),np.asarray(reference[key]),atol=2e-6)
    report['changing_camera_frozen_replay']='passed'
    report.update(check_appearance_and_evidence(payload,rays,mi,dr,np))
    report.update(check_background_removal(mi,dr,np))
    return report


def check_appearance_and_evidence(payload,rays,mi,dr,np):
    count=len(payload['model_state']['cell_indices'])
    directional=copy.deepcopy(payload)
    directional['model_metadata']['sh_count']=4
    coefficients=np.array([[.6/Y00,.5/Y00,.4/Y00],[.3,.2,.1],[-.2,.1,.25],[.1,-.3,.2]])
    directional['model_state']['radiance_sh']=np.tile(coefficients.ravel(),count)
    model=restore_model(directional,mi)
    selected=[rays[i] for i in (0,1,3,6)]
    ray=make_ray(mi,np,[r[0] for r in selected],[r[1] for r in selected])
    result=native(model,mi,dr,ray)
    expected=[]
    for _,d in selected:
        x,y,z=-np.asarray(d)/np.linalg.norm(d)
        expected.append(np.maximum(0,np.array([Y00,y*Y00*math.sqrt(3),z*Y00*math.sqrt(3),x*Y00*math.sqrt(3)])@coefficients))
    np.testing.assert_allclose(np.asarray(result['raw_rgb']).T,
        np.asarray(expected)*np.asarray(result['raw_alpha'])[:,None],atol=2e-6)
    target=np.array([.3,.4,.5])
    grid=payload['model_metadata']['resolution'];pad=1;core=4
    cells=payload['model_state']['cell_indices']
    xyz=np.column_stack((cells%grid,cells//grid%grid,cells//grid**2))
    core_mask=np.all((xyz>=pad)&(xyz<pad+core),axis=1)
    def subset(indices):
        result=copy.deepcopy(payload)
        result['model_state']={key:value.reshape(count,-1)[indices].ravel().copy()
                               for key,value in payload['model_state'].items()}
        return result
    for case in (0,1):
        o,d=rays[case]
        expected=oracle(payload,o,d,np)
        row=next(hit[1] for hit in expected['hits'] if core_mask[hit[1]])
        omitted=subset(np.arange(count)!=row)
        candidate=subset([row])['model_state']
        ray=make_ray(mi,np,[o],[d])
        original=restore_model(payload,mi)
        deletion=native(original,mi,dr,ray,evidence=True)
        background_hits=[hit[1] for hit in expected['hits'] if not core_mask[hit[1]]]
        assert background_hits,case
        for background_row in background_hits:
            background_omitted=subset(np.arange(count)!=background_row)
            background_without=oracle(background_omitted,o,d,np)
            background_gain=(np.sum((expected['rgb']-target)**2)-
                             np.sum((background_without['rgb']-target)**2))
            np.testing.assert_allclose(np.asarray(deletion['deletion_gain'])[background_row],
                                       background_gain,atol=3e-6)
            assert int(np.asarray(deletion['deletion_observed'])[background_row])==1
            assert int(np.asarray(deletion['deletion_unknown'])[background_row])==0
        without=oracle(omitted,o,d,np)
        gain=np.sum((expected['rgb']-target)**2)-np.sum((without['rgb']-target)**2)
        np.testing.assert_allclose(float(np.asarray(deletion['deletion_gain'])[row]),gain,atol=3e-6)
        assert int(np.asarray(deletion['deletion_observed'])[row])==1
        baseline=restore_model(omitted,mi)
        baseline.set_growth_candidates({key:(mi.UInt32(value) if key=='cell_indices' else mi.Float(value))
                                         for key,value in candidate.items()})
        insertion=native(baseline,mi,dr,ray,growth=True)
        np.testing.assert_allclose(np.asarray(insertion['growth_gain']),[-gain],atol=3e-6)
        np.testing.assert_array_equal(np.asarray(insertion['growth_observed']),[1])
        np.testing.assert_array_equal(np.asarray(insertion['growth_unknown']),[0])
        if case==0:
            truncated=native(baseline,mi,dr,ray,growth=True,max_hits=2)
            np.testing.assert_array_equal(np.asarray(truncated['growth_observed']),[0])
            np.testing.assert_array_equal(np.asarray(truncated['growth_unknown']),[1])
    return {'world_direction_spherical_harmonics':'passed',
        'core_growth_and_removal_with_cross_chart_background_suffix':'passed',
        'projective_background_single_removal_matches_world_oracle':'passed',
        'truncated_core_growth_abstention':'passed'}


def check_background_removal(mi,dr,np):
    # Two covering background cells on the +Z chart. The front white, half-
    # transparent cell contaminates the desired color of the cell behind it.
    # Neither cell is below the ordinary low-opacity deletion threshold.
    grid,core,layers=8,4,2
    payload={'model_metadata':{'resolution':grid,'core_resolution':core,
        'grid_min':[-2.]*3,'grid_max':[2.]*3,'scene_mapping':'projective_shell',
        'background_refinement':'preserve_physical_width','background_layers':layers,
        'sh_count':1,'opacity_sh_count':1},
        'model_state':{'cell_indices':np.array([4+grid*(4+grid*z) for z in (6,7)],np.uint32),
            'centers':np.full(6,.5),
            'log_scales':np.full(6,math.log(math.sqrt(3)/4*(1+1e-6))),
            'rotations':np.tile([1.,0.,0.,0.],2),
            'radiance_sh':np.array([[1.,1.,1.],[.6,.8,1.]]).ravel()/Y00,
            'opacity_sh':np.zeros(2)}}
    omitted=copy.deepcopy(payload)
    omitted['model_state']={key:value.reshape(2,-1)[1:].ravel().copy()
                            for key,value in payload['model_state'].items()}
    origin,direction=(.13,.22,0.),(0.,0.,1.)
    expected=oracle(payload,origin,direction,np)
    without=oracle(omitted,origin,direction,np)
    target=np.array([.3,.4,.5])
    assert len(expected['hits'])==2
    np.testing.assert_allclose(without['rgb'],target,atol=1e-12)
    gain=np.sum((expected['rgb']-target)**2)-np.sum((without['rgb']-target)**2)
    assert gain>.1,gain
    model=restore_model(payload,mi)
    ray=make_ray(mi,np,[origin],[direction])
    result=native(model,mi,dr,ray,evidence=True)
    np.testing.assert_allclose(np.asarray(result['deletion_gain'])[0],gain,atol=3e-6)
    np.testing.assert_array_equal(np.asarray(result['deletion_observed']),[1,1])
    np.testing.assert_array_equal(np.asarray(result['deletion_unknown']),[0,0])
    # Evidence must also protect a useful half-transparent background cell.
    perfect=native(model,mi,dr,ray,evidence=True,reference=expected['rgb'])
    expected_negative=-np.sum((without['rgb']-expected['rgb'])**2)
    np.testing.assert_allclose(np.asarray(perfect['deletion_gain'])[0],expected_negative,atol=3e-6)
    assert float(np.asarray(perfect['deletion_gain'])[0])<0
    for cap,threshold in ((1,0.),(64,.6)):
        truncated=native(model,mi,dr,ray,evidence=True,max_hits=cap,threshold=threshold)
        np.testing.assert_array_equal(np.asarray(truncated['hit_count']),[1])
        np.testing.assert_array_equal(np.asarray(truncated['deletion_gain']),[0.,0.])
        np.testing.assert_array_equal(np.asarray(truncated['deletion_observed']),[0,0])
        np.testing.assert_array_equal(np.asarray(truncated['deletion_unknown']),[1,0])

    # A complete image uses its DSSIM gradient for optimization, while the
    # deletion criterion remains exact RGB-L2 removal gain for every hit.
    origins=[(.13+.01*x,.22+.01*y,0.) for y in range(3) for x in range(4)]
    image=make_ray(mi,np,origins,[direction]*len(origins))
    off=native(model,mi,dr,image,evidence=True)
    on=native(model,mi,dr,image,evidence=True,dssim_weight=.2,image_size=(4,3))
    for key in ('deletion_gain','deletion_observed','deletion_unknown'):
        np.testing.assert_allclose(np.asarray(on[key]),np.asarray(off[key]),atol=3e-6)
    np.testing.assert_allclose(np.asarray(on['deletion_gain'])[0],len(origins)*gain,atol=1e-5)
    assert np.isfinite(np.asarray(on['dssim_loss'])).all()
    assert not np.allclose(np.asarray(on['gradients']['radiance_sh']),
                           np.asarray(off['gradients']['radiance_sh']))
    return {'half_transparent_background_positive_and_negative_rgb_removal_gain':'passed',
            'truncated_background_removal_abstains':'passed',
            'dssim_background_removal_keeps_rgb_criterion':'passed'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--gpu',type=int,default=0)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='projective-native-') as directory:
        mi,dr,np,_=initialize_runtime(RuntimeConfig(gpu=args.gpu),Path(directory),{})
        print(json.dumps(run_checks(mi,dr,np),indent=2))


if __name__=='__main__': main()
