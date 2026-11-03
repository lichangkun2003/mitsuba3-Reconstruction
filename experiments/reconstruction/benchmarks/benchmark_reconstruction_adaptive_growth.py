"""Reproduce candidate generation on one immutable real-scene snapshot.

No training updates or renders. An optional implementation file permits exact
baseline comparisons after optimization; all candidate buffers are hashed and
can be persisted for byte-for-byte/numerical comparison. Reported memory is an
allocator watermark for this diagnostic, not a complete training VRAM peak.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--snapshot',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--implementation')
    parser.add_argument('--save-candidates')
    parser.add_argument('--compare-candidates')
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--kernel-history',action='store_true')
    parser.add_argument('--derive-from',help='Create a deterministic sparse variant of this saved initial snapshot')
    parser.add_argument('--drop-background-every',type=int,default=7)
    args=parser.parse_args()
    from reconstruction_common import initialize_runtime,prepare_dataset,initialize_model,model_metadata,WORKSPACE
    from reconstruction_growth_candidates_gpu import generate_growth_gpu
    from train import TrainConfig,update,kernel_profile
    from preview import RuntimeConfig
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='running',rounds=[],snapshot=args.snapshot)
    def write():
        out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    write()
    mi,dr,np,handles=initialize_runtime(RuntimeConfig(),out.parent,report)
    config=TrainConfig()
    update(config,json.loads(Path(args.config).read_text(encoding='utf-8')))
    path=Path(args.snapshot)
    if path.exists():
        with np.load(path,allow_pickle=False) as saved:
            metadata=json.loads(str(saved['metadata']))
            arrays={key:saved[key] for key in saved.files if key!='metadata'}
    elif args.derive_from:
        with np.load(args.derive_from,allow_pickle=False) as saved:
            metadata=json.loads(str(saved['metadata']))
            arrays={key:saved[key] for key in saved.files if key!='metadata'}
        cells=arrays['cell_indices']
        keep=np.ones(len(cells),bool)
        background=np.flatnonzero(cells>=2**63)
        keep[background[::args.drop_background_every]]=False
        widths=dict(cell_indices=1,centers=3,log_scales=3,rotations=4,
                    radiance_sh=metadata['sh_count']*3,opacity_sh=metadata['opacity_sh_count'])
        arrays={key:value.reshape(-1,widths[key])[keep].ravel() for key,value in arrays.items()}
        np.savez(path,metadata=json.dumps(metadata),**arrays)
        report['synthetic_derivation']=dict(source=args.derive_from,drop_background_every=args.drop_background_every,
            note='Sparse holes for candidate-path stress; not a trained reconstruction')
    else:
        print('Preparing actual scene initializer',flush=True)
        prepared=prepare_dataset(config,np)
        model=initialize_model(config,WORKSPACE/config.data.dataset,mi,np,prepared=prepared)
        metadata=model_metadata(model)
        arrays={name:np.asarray(value) for name,value in model.snapshot().items()}
        np.savez(path,metadata=json.dumps(metadata),**arrays)
        del model,prepared
        gc.collect()
    checkpoint=dict(model_metadata=metadata,model_state={key:dr.cuda.UInt64(value) if key=='cell_indices'
        else dr.cuda.Float32(value) for key,value in arrays.items()})
    count=len(arrays['cell_indices'])
    report.update(voxels=count,background_voxels=int(np.sum(arrays['cell_indices']>=2**63)),metadata=metadata)
    del arrays
    manager=SimpleNamespace(settings=config.topology,schedule=config.schedule)
    eligible=dr.full(dr.cuda.Bool,True,count)
    cooldown=dr.cuda.UInt64([])
    if args.implementation:
        spec=importlib.util.spec_from_file_location('reconstruction_adaptive_growth',args.implementation)
        implementation=importlib.util.module_from_spec(spec)
        sys.modules[spec.name]=implementation
        spec.loader.exec_module(implementation)
        report['implementation_sha256']=hashlib.sha256(Path(args.implementation).read_bytes()).hexdigest()
    else:
        import reconstruction_adaptive_growth as implementation
        report['implementation_sha256']=hashlib.sha256(Path(implementation.__file__).read_bytes()).hexdigest()
    dr.eval(*checkpoint['model_state'].values())
    dr.sync_thread()
    dr.flush_malloc_cache()
    for repeat in range(args.repeats):
        report['current_repeat']=repeat
        write()
        started=time.perf_counter()
        with dr.kernel_history() if args.kernel_history else nullcontext() as history:
            result=generate_growth_gpu(manager,checkpoint,30,eligible=eligible,cooldown_cells=cooldown)
            dr.eval(*result['candidate_state'].values())
            dr.sync_thread()
        row=dict(repeat=repeat,seconds=time.perf_counter()-started,proposals=len(result['cells']),
                 allocator_watermark_bytes=int(dr.detail.malloc_watermark(dr.JitBackend.CUDA)),
                 stages=getattr(manager,'growth_generation_stage_seconds',{}))
        if args.kernel_history:
            row['kernels']=kernel_profile(history)
            row['kernels'].pop('jit_hashes',None)
        exported={key:np.asarray(result[key]) for key in ('cells','parent_cells','parent_ids','geometric_support',
                    'centers','log_scales','rotations')}
        exported.update({'state_'+key:np.asarray(value) for key,value in result['candidate_state'].items()})
        row['hashes']={key:hashlib.sha256(value.tobytes()).hexdigest() for key,value in exported.items()}
        row['candidate_cells']=len(np.unique(exported['cells']))
        if args.save_candidates:
            np.savez(args.save_candidates,**exported)
        if args.compare_candidates:
            with np.load(args.compare_candidates,allow_pickle=False) as baseline:
                row['comparison']={}
                for key,value in exported.items():
                    expected=baseline[key]
                    same=np.array_equal(value,expected)
                    row['comparison'][key]=dict(bitwise_equal=bool(same),shape=list(value.shape))
                    if value.dtype.kind in 'iu' or key.startswith('state_'):
                        np.testing.assert_array_equal(value,expected,err_msg=key)
                    else:
                        np.testing.assert_allclose(value,expected,rtol=3e-12,atol=3e-12,err_msg=key)
        report['rounds'].append(row)
        print(json.dumps({key:value for key,value in row.items() if key not in ('hashes','comparison')}),flush=True)
        del result,exported
        dr.sync_thread()
        dr.flush_malloc_cache()
        write()
    report['status']='passed'
    write()


if __name__=='__main__':
    main()
