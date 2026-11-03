"""Focused CUDA regression: AD statistics, selective fit and preserved histories."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import copy
import sys
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))


def check(mi,dr,np):
    from reconstruction_common import restore_model
    from reconstruction_world_shell import ADAPTIVE_TAG,adaptive_encode_numpy,leaf_boxes_numpy
    from reconstruction_adaptive_background import (select_adaptive_background_device,
        refine_adaptive_background_device,refine_adaptive_background_numpy,prepare_adaptive_background)
    from reconstruction_topology import CoarseToFineConfig,refine_state,TopologyManager,TopologySettings
    from reconstruction_topology_gpu import GPUTopologyLifecycle,get_gpu_lifecycle
    from reconstruction_optimizer import FalcorOptimizer,OptimizerConfig
    from reconstruction_background import initialize_background_model
    from test_reconstruction_adaptive_background import fixture,cameras

    payload=fixture()
    md=payload["model_metadata"]
    md["background_max_depth"]=8
    cells=np.r_[np.uint64(21),adaptive_encode_numpy([45,46],0,0,np)]
    payload["model_state"]={key:values.reshape(4,-1)[:3].ravel().copy()
                            for key,values in payload["model_state"].items()}
    payload["model_state"]["cell_indices"]=cells
    lo,hi=leaf_boxes_numpy(cells,4,[-1.]*3,[1.]*3,2,np)
    payload["model_state"]["log_scales"]=np.log((hi-lo)*.9).astype(np.float32).ravel()
    model=restore_model(payload,mi)
    state=dict(model_metadata=md,model_state=model.snapshot())
    priority=mi.Float([0.,1.,0.])
    assert type(priority) is mi.Float
    selected=select_adaptive_background_device(state,cameras(),priority,dr,
        max_new_leaves=7,pixel_threshold=8,projection_cache={})
    np.testing.assert_array_equal(np.asarray(selected),[1])
    result=refine_adaptive_background_device(state,selected,CoarseToFineConfig(),20,dr)
    expected=refine_adaptive_background_numpy(payload,np.array([1],np.uint32),CoarseToFineConfig(),20,refine_state,np)
    for key,value in result["model_state"].items():
        np.testing.assert_allclose(np.asarray(value),expected["model_state"][key],rtol=1e-6,atol=1e-7)
    print("PASS actual AD priority selector and mixed survivor refinement",flush=True)
    for tiny in (False,True):
        one=copy.deepcopy(payload)
        one["model_state"]={key:value.reshape(3,-1)[1:2].ravel().copy() for key,value in payload["model_state"].items()}
        if tiny:
            one["model_state"]["log_scales"][:]=np.log(1e-8)
        source=restore_model(one,mi)
        outcome=refine_adaptive_background_device(dict(model_metadata=md,model_state=source.snapshot()),
            mi.UInt32([0]),CoarseToFineConfig(),20,dr)
        dr.eval(*outcome["model_state"].values(),outcome["source_rows"])
        assert outcome["replaced"]==(0 if tiny else 1)
        assert len(outcome["model_state"]["cell_indices"])==(1 if tiny else 8)
        print("PASS no-child preserve" if tiny else "PASS all-parent replacement empty retained",flush=True)

    # Force multiple fitting batches and shuffled inputs. Include a parent with
    # no children in a partly successful group, plus depth 8 -> 9 whose local
    # regular-grid child address exceeds UInt32 before tagged re-encoding.
    mixed=copy.deepcopy(payload)
    mixed["model_metadata"].update(background_shell_levels=10,background_max_depth=9)
    mixed_cells=np.r_[np.uint64(21),adaptive_encode_numpy([45,45,45,55,559],
        [2,2,2,8,8],[0,1,2,8**8-1,8**8-1],np)]
    lo,hi=leaf_boxes_numpy(mixed_cells,4,[-1.]*3,[1.]*3,10,np)
    rng=np.random.default_rng(194)
    mixed["model_state"]=dict(cell_indices=mixed_cells,
        centers=np.tile(np.array([.43,.53,.47],np.float32),len(mixed_cells)),
        log_scales=np.log((hi-lo)*np.array([.28,.42,.61])).astype(np.float32).ravel(),
        rotations=np.tile(np.array([.9238795,0.,0.,.3826834],np.float32),len(mixed_cells)),
        radiance_sh=rng.normal(size=len(mixed_cells)*27).astype(np.float32),
        opacity_sh=np.linspace(-2.,2.,len(mixed_cells),dtype=np.float32))
    mixed["model_state"]["log_scales"].reshape(-1,3)[3]=np.log(1e-12)
    selected_rows=np.array([5,3,2,4,1],np.uint32)
    expected=refine_adaptive_background_numpy(mixed,selected_rows,CoarseToFineConfig(),100,refine_state,np)
    native=restore_model(mixed,mi)
    live=dict(model_metadata=mixed["model_metadata"],model_state=native.snapshot())
    requests=[]
    with patch("reconstruction_adaptive_background._FIT_PARENT_BATCH",2), \
         patch("reconstruction_adaptive_background.refine_adaptive_background_numpy",
               side_effect=AssertionError("GPU fitting must not call the NumPy fitter")):
        actual=refine_adaptive_background_device(live,mi.UInt32(selected_rows),CoarseToFineConfig(),100,dr,
                                                 before_allocate=lambda request:requests.append(request))
    for name,value in actual["model_state"].items():
        if name=="cell_indices":
            np.testing.assert_array_equal(np.asarray(value),expected["model_state"][name])
        else:
            np.testing.assert_allclose(np.asarray(value),expected["model_state"][name],rtol=2e-5,atol=2e-6)
    np.testing.assert_array_equal(np.asarray(actual["source_rows"]),expected["source_rows"].astype(np.uint32))
    for name in ("selected","replaced","children","added"):
        assert actual[name]==expected[name],(name,actual[name],expected[name])
    assert len(requests)==2,requests
    assert requests[0]["allocation_voxels"]==8*len(selected_rows)
    assert requests[0]["background_parent_count"]==2
    assert requests[1]["existing_background_child_count"]==actual["children"]
    assert requests[1]["exact_count"] is True
    assert 3 in np.asarray(actual["source_rows"])
    print("PASS batched GPU fit agrees with CPU, deep UInt64 addresses, empty-parent preservation and two preflights",flush=True)

    original={name:np.asarray(value).copy() for name,value in live["model_state"].items()}
    def reject(_):
        raise RuntimeError("test memory preflight refused")
    with patch("reconstruction_refinement_gpu.refine_state_gpu",
               side_effect=AssertionError("Initial refusal must precede fitting")):
        try:
            refine_adaptive_background_device(live,mi.UInt32(selected_rows),CoarseToFineConfig(),100,dr,
                                              before_allocate=reject)
        except RuntimeError as error:
            assert str(error)=="test memory preflight refused"
        else:
            raise AssertionError("Missing memory refusal")
    for name,value in live["model_state"].items():
        np.testing.assert_array_equal(np.asarray(value),original[name])
    print("PASS memory refusal precedes fitting and preserves every parent field",flush=True)
    empty={key:(mi.UInt64([]) if key=="cell_indices" else mi.Float([])) for key in payload["model_state"]}
    seed=mi.VoxelReconstruction.from_state(4,[-1.]*3,[1.]*3,empty,9,1)
    prepared=prepare_adaptive_background(np.empty((0,3)),np.empty((0,3)),4,[-1.]*3,[1.]*3,
        1,cameras(),np,initial_leaf_budget=56,max_depth=3)
    background=initialize_background_model(seed,mi,dr,np,shell_levels=1,
        background_refinement="adaptive",adaptive_background=prepared)
    snapshot=background.snapshot()
    dr.eval(*snapshot.values())
    assert len(snapshot["cell_indices"])==56
    np.testing.assert_allclose(np.asarray(snapshot["radiance_sh"]).reshape(-1,9,3)[:,0],.5/.28209479177387814,rtol=1e-6)
    print("PASS foreground-empty initializer and gray DC",flush=True)

    lifecycle=GPUTopologyLifecycle.__new__(GPUTopologyLifecycle)
    lifecycle.manager=SimpleNamespace(np=np)
    lifecycle.dr=dr
    lifecycle.Cell,lifecycle.UInt=dr.cuda.UInt64,dr.cuda.UInt32
    lifecycle.Float,lifecycle.Double=dr.cuda.Float32,dr.cuda.Float64
    high=adaptive_encode_numpy([0,559],[0,9],[0,2**27-1],np)
    pending=dict(cells=high.copy(),parent_cells=high[::-1].copy(),parent_ids=np.array([17,3],np.uint32),
        support_views=np.array([5,7],np.uint32),candidate_state=dict(cell_indices=high.copy(),centers=np.zeros(6,np.float32)))
    lifecycle.upload_pending(pending)
    for name,wanted in (("cells",high),("parent_cells",high[::-1])):
        assert type(pending[name]) is dr.cuda.UInt64
        np.testing.assert_array_equal(np.asarray(pending[name]),wanted)
    assert type(pending["support_views"]) is dr.cuda.UInt32
    np.testing.assert_array_equal(np.asarray(pending["candidate_state"]["cell_indices"]),high)
    print("PASS pending upload preserves maximum adaptive tag and UInt32 counters",flush=True)

    model=restore_model(payload,mi)
    optimizer=FalcorOptimizer(model,OptimizerConfig(),replay_updates=False)
    manager=TopologyManager(model,2,settings=TopologySettings(enable_growth=False,enable_deletion=False,
        gpu_topology=True,use_rgb_growth=True,use_rgb_deletion=True,use_growth_evidence=False),
        schedule=CoarseToFineConfig(start_resolution=4,target_resolution=8,total_epochs=2,level_weights=(1,1),save_each_level=False))
    lifecycle=get_gpu_lifecycle(manager,model)
    assert lifecycle is not None,"Fixture must activate the CUDA RGB lifecycle"
    history={}
    for name,values in lifecycle.fields.items():
        lifecycle.fields[name]=type(values)([11,22,33])
        history[name]=np.asarray(lifecycle.fields[name]).copy()
    lifecycle.candidates=lifecycle.Bool([True,False,True])
    lifecycle.cooldown_cells=lifecycle.Cell(cells)
    lifecycle.cooldown_expiry=lifecycle.Int64([101,202,303])
    replacement,_=lifecycle.refine(model,optimizer,1,{})
    count=len(replacement["model_state"]["cell_indices"])
    for name,values in lifecycle.fields.items():
        np.testing.assert_array_equal(np.asarray(values)[:2],history[name][1:])
        np.testing.assert_array_equal(np.asarray(values)[2:],np.zeros(count-2))
    np.testing.assert_array_equal(np.asarray(lifecycle.candidates)[:2],[False,True])
    np.testing.assert_array_equal(np.asarray(lifecycle.cooldown_cells),cells[1:])
    np.testing.assert_array_equal(np.asarray(lifecycle.cooldown_expiry),[202,303])
    assert np.all(np.asarray(lifecycle.cooldown_cells)>=ADAPTIVE_TAG)
    dr.sync_thread()
    print("PASS foreground split preserves unchanged background lifecycle and cooldown",flush=True)


def main():
    from preview import RuntimeConfig
    from reconstruction_common import initialize_runtime,WORKSPACE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir",default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu",type=int,default=0)
    args=parser.parse_args()
    output=WORKSPACE/".cache"/"adaptive-focused"
    output.mkdir(parents=True,exist_ok=True)
    mi,dr,np,_=initialize_runtime(RuntimeConfig(build_dir=args.build_dir,gpu=args.gpu),output,{})
    check(mi,dr,np)


if __name__=="__main__":
    main()
