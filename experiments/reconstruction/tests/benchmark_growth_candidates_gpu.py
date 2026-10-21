"""Read-only CPU/CUDA proposal comparison on a saved reconstruction model.

Run after correctness tests, with no other GPU workload. No training or files
are written. Reports warmed GPU time including proposal packing, plus the
unchanged CPU geometry generator time. The checkpoint is never modified.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from reconstruction_topology import TopologySettings,CoarseToFineConfig,TopologyManager
from reconstruction_growth_candidates_gpu import generate_growth_gpu
from reconstruction_common import initialize_runtime
from preview import RuntimeConfig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint',type=Path)
    args=parser.parse_args()
    with np.load(args.checkpoint,allow_pickle=False) as archive:
        header=json.loads(bytes(archive['metadata_json']).decode('utf-8'))
        def decode(node):
            if not isinstance(node,dict):return node
            if node['type']=='array':return archive[node['key']]
            if node['type']=='dict':return {k:decode(v) for k,v in node['items'].items()}
            if node['type']=='list':return [decode(v) for v in node['items']]
            if node['type']=='tuple':return tuple(decode(v) for v in node['items'])
            raise ValueError(node['type'])
        raw=header['payload']['items']
        checkpoint={k:decode(raw[k]) for k in ('model_state','model_metadata')}
        settings=TopologySettings(**decode(raw['config'])['topology'])
    n=len(checkpoint['model_state']['cell_indices'])
    manager=SimpleNamespace(np=np,settings=settings,schedule=CoarseToFineConfig(enabled=False),
                            candidates=np.zeros(n,dtype=bool),growth_wait_start=np.zeros(n,dtype=np.int64),cooldown={})
    start=time.perf_counter()
    reference=TopologyManager._generate_growth(manager,checkpoint,100)
    cpu_seconds=time.perf_counter()-start
    order=np.lexsort((reference['parent_cells'],reference['cells']))
    with tempfile.TemporaryDirectory(prefix='mitsuba-candidate-benchmark-') as directory:
        mi,dr,_,handles=initialize_runtime(RuntimeConfig(),Path(directory),{})
        gpu={**checkpoint,'model_state':{k:(mi.UInt32(v) if k=='cell_indices' else mi.Float(v))
                                        for k,v in checkpoint['model_state'].items()}}
        dr.eval(*gpu['model_state'].values());dr.sync_thread()
        times=[]
        for repeat in range(3):
            start=time.perf_counter()
            result=generate_growth_gpu(manager,gpu,100)
            dr.sync_thread();times.append(time.perf_counter()-start)
            for key,width in [('parent_ids',1),('parent_cells',1),('cells',1),('centers',3),('log_scales',3),('rotations',4)]:
                actual=np.array(result[key]).reshape(-1,width)
                expected=reference[key][order].reshape(-1,width)
                if width==1:np.testing.assert_array_equal(actual,expected,err_msg=key)
                else:np.testing.assert_allclose(actual,expected,rtol=3e-11,atol=3e-11,err_msg=key)
            del result
        print(json.dumps({'voxels':n,'proposals':len(order),'cpu_geometry_seconds':cpu_seconds,
            'gpu_geometry_and_packing_seconds':times,'warmed_speedup':cpu_seconds/min(times[1:]),
            'candidate_ids_order':'exact','fp64_geometry':'allclose 3e-11',
            'scope':'all parents mature, no cooldown; candidate generation only'},indent=2))


if __name__=='__main__':main()
