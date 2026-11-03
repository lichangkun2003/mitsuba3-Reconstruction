"""Mixed-depth growth CPU oracle; pass --gpu for resident/native parity checks."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import reconstruction_adaptive_growth as growth
from reconstruction_world_shell import (ADAPTIVE_TAG,adaptive_encode_numpy,
    adaptive_intervals_numpy,leaf_boxes_numpy,shell_coordinates_numpy,validate_adaptive_cells_numpy)
from reconstruction_growth_filter import match_growth_parents
from test_reconstruction_growth_filter import checkpoint as core_checkpoint,manager_for


def bg(shell=0,coordinate=(3,1,1),depth=0,child=(0,0,0)):
    table=shell_coordinates_numpy(np)
    ordinal=int(np.flatnonzero(np.all(table==coordinate,axis=1))[0])
    morton=sum(((child[a]>>b)&1)<<(3*b+a) for b in range(depth) for a in range(3))
    return adaptive_encode_numpy(shell*56+ordinal,depth,morton,np).item()


def fixture(cells,core=8,levels=3,fraction=.8):
    cells=np.asarray(cells,np.uint64)
    lower,upper=leaf_boxes_numpy(cells,core,[-1.]*3,[1.]*3,levels,np)
    count=len(cells)
    metadata=dict(resolution=core,core_resolution=core,grid_min=[-1.]*3,grid_max=[1.]*3,
                  background_shell_levels=levels,background_refinement='adaptive',background_geometry='ellipsoid',
                  world_shell_version=5,sh_count=4,opacity_sh_count=1)
    state=dict(cell_indices=cells,centers=np.full(count*3,.5,np.float32),
               log_scales=np.log((upper-lower)*fraction).astype(np.float32).ravel(),
               rotations=np.tile([1,0,0,0],count).astype(np.float32),
               opacity_sh=np.zeros(count,np.float32),radiance_sh=np.arange(count*12,dtype=np.float32)/100)
    return dict(model_metadata=metadata,model_state=state)


def manager(state,**options):
    defaults=dict(use_rgb_growth=True,growth_min_supporting_parents=1,
                  growth_evidence_penetration=0.,growth_use_26_neighbors=False)
    defaults.update(options)
    return manager_for(state,**defaults)


def assert_disjoint(test,state,result):
    cells=np.unique(np.r_[state['model_state']['cell_indices'],result['cells']])
    validate_adaptive_cells_numpy(cells,state['model_metadata']['resolution'],
                                  state['model_metadata']['background_shell_levels'],np)


class Cpu(unittest.TestCase):
    def test_integer_boxes_and_round_trip(self):
        rng=np.random.default_rng(29302)
        d=rng.integers(0,10,2000)
        cells=adaptive_encode_numpy(rng.integers(0,560,2000),d,
                                   (rng.random(2000)*8.**d).astype(np.uint64),np)
        for n in (2,8,256,1024):
            state=fixture(np.r_[np.array([0,n**3-1],np.uint64),cells],core=n,levels=10)
            meta=state['model_metadata']
            c=state['model_state']['cell_indices']
            xyz,l=growth._decode(c,meta,growth._Ops(np))
            got,_,_,valid,split=growth._canonical(xyz,l,meta,growth._Ops(np))
            np.testing.assert_array_equal(got,c)
            self.assertTrue(valid.all())
            self.assertFalse(split.any())
            lo,hi=leaf_boxes_numpy(c,n,[-1.]*3,[1.]*3,10,np)
            np.testing.assert_array_equal(np.stack(xyz,1)/512-1024,lo)
            np.testing.assert_array_equal(np.exp2(l)/512,hi[:,0]-lo[:,0])

    def test_full_uint64_pair_matching(self):
        cells=np.array([bg(0,depth=8,child=(42,3,5)),bg(1,depth=8,child=(42,3,5)),73],np.uint64)
        parents=np.array([bg(0,depth=9,child=(1,2,3)),bg(1,depth=9,child=(1,2,3)),7],np.uint64)
        current=dict(cells=cells,parent_cells=parents,parent_ids=np.array([10,11,12]))
        pending=dict(cells=np.r_[cells[[1,0,1]],cells[0]],parent_cells=np.r_[parents[[1,0,1]],parents[1]])
        keep,ids=match_growth_parents(current,pending)
        np.testing.assert_array_equal(keep,[True,True,True,False])
        np.testing.assert_array_equal(ids,[11,10,11])

    def test_core_behavior_unchanged(self):
        from reconstruction_topology import TopologyManager
        old=core_checkpoint([73,75,90,100])
        m=manager(old,growth_use_26_neighbors=True)
        expected=TopologyManager._generate_growth(m,old,10)
        old['model_metadata'].update(background_refinement='adaptive',background_shell_levels=2)
        actual=growth.generate_adaptive_growth_numpy(m,old,10)
        order=np.lexsort((expected['parent_cells'],expected['cells']))
        for name in growth._WIDTHS:
            np.testing.assert_allclose(actual[name],expected[name][order],rtol=1e-13,atol=1e-13)

    def test_coarse_background_to_many_core_neighbors(self):
        state=fixture([bg()])
        result=growth.generate_adaptive_growth_numpy(manager(state),state,10)
        core=result['cells'][result['cells']<8**3]
        self.assertGreater(len(core),1)
        self.assertTrue(np.all(core%8==7))
        assert_disjoint(self,state,result)

    def test_fine_background_promotes_core_target_without_duplicate_pair(self):
        state=fixture([bg(depth=6,child=(0,30,30))])
        result=growth.generate_adaptive_growth_numpy(manager(state),state,10)
        self.assertEqual(np.sum(result['cells']<8**3),1)
        self.assertEqual(len(np.unique(result['cells'])),len(result['cells']))
        row=int(np.flatnonzero(result['cells']<8**3)[0])
        # The child still crosses the shared x face after promotion into a
        # much larger core cell; a fixed 0.2*target_edge offset would detach it.
        radius=np.exp(result['log_scales'][row,0])
        face_distance=(1-result['centers'][row,0])*(2/8)
        self.assertLess(face_distance,radius)
        assert_disjoint(self,state,result)

    def test_cross_shell_neighbors_keep_world_edge(self):
        source=bg(coordinate=(3,2,2))
        state=fixture([source])
        result=growth.generate_adaptive_growth_numpy(manager(state),state,10)
        expected=bg(1,coordinate=(3,2,2),depth=1)
        self.assertIn(expected,result['cells'])
        lower,upper=leaf_boxes_numpy(np.array([source,expected],np.uint64),8,[-1]*3,[1]*3,3,np)
        self.assertEqual(upper[0,0],lower[1,0])
        np.testing.assert_array_equal(upper-lower,np.ones((2,3)))
        assert_disjoint(self,state,result)

    def test_finer_occupied_target_splits_empty_face_gaps(self):
        source=bg(coordinate=(3,2,2))
        occupied=bg(1,coordinate=(3,2,2),depth=2,child=(0,0,0))
        state=fixture([source,occupied])
        m=manager(state)
        m.candidates[1]=True
        result=growth.generate_adaptive_growth_numpy(m,state,10)
        parent_target=bg(1,coordinate=(3,2,2),depth=1)
        self.assertNotIn(parent_target,result['cells'])
        self.assertNotIn(occupied,result['cells'])
        siblings=[bg(1,coordinate=(3,2,2),depth=2,child=(0,y,z)) for y,z in ((1,0),(0,1),(1,1))]
        self.assertTrue(any(c in result['cells'] for c in siblings))
        assert_disjoint(self,state,result)

    def test_occupied_and_cooldown_ancestors_block_fine_targets(self):
        source=bg(depth=4,child=(0,8,8))
        state=fixture([source])
        m=manager(state)
        # Adjacent negative-x core leaf, plus a background ancestor covering
        # its positive-x neighbor and the source itself.
        m.cooldown={int(bg()):20, int(7+8*(2+8*2)):20}
        result=growth.generate_adaptive_growth_numpy(m,state,10)
        self.assertEqual(len(result['cells']),0)

    def test_candidate_partition_prefers_fine_targets(self):
        parent=bg()
        a=bg(depth=1,child=(0,0,0))
        b=bg(depth=1,child=(1,0,0))
        leaf=bg(depth=2,child=(0,0,0))
        state=fixture([])
        cells=np.array([parent,b,a,leaf,leaf,3,4],np.uint64)
        keep=growth.filter_nonoverlapping_growth_numpy(cells,state['model_metadata'],np)
        np.testing.assert_array_equal(keep,[1,3,5,6])

    def test_mixed_parents_candidate_tree_is_nonoverlapping(self):
        state=fixture([bg(coordinate=(3,2,2)),bg(1,coordinate=(3,2,2),depth=3,child=(4,0,0))])
        result=growth.generate_adaptive_growth_numpy(manager(state,growth_use_26_neighbors=True),state,10)
        assert_disjoint(self,state,result)
        validate_adaptive_cells_numpy(np.unique(result['cells']),8,3,np)

    def test_proposal_cap_and_target_subset(self):
        state=fixture([bg(),bg(1,coordinate=(3,2,2))])
        m=manager(state,background_growth_max_candidates=5)
        result=growth.generate_adaptive_growth_numpy(m,state,10)
        self.assertLessEqual(len(result['cells']),5)
        self.assertGreater(len(result['cells']),0)
        targets=result['cells'][::2]
        subset=growth.generate_adaptive_growth_numpy(m,state,10,targets)
        self.assertTrue(np.isin(subset['cells'],targets).all())
        assert_disjoint(self,state,subset)

    def test_empty_and_max_depth(self):
        state=fixture([])
        self.assertEqual(len(growth.generate_adaptive_growth_numpy(manager(state),state,10)['cells']),0)
        state=fixture([bg(depth=8,child=(1,2,3))],core=1024)
        state['model_metadata']['background_max_depth']=8
        result=growth.generate_adaptive_growth_numpy(manager(state),state,10)
        self.assertGreater(len(result['cells']),0)
        assert_disjoint(self,state,result)


@unittest.skipUnless('--gpu' in sys.argv,'explicit --gpu required')
class Cuda(unittest.TestCase):
    def test_geometry_and_parent_order_matches_cpu(self):
        from reconstruction_growth_candidates_gpu import generate_growth_gpu
        for cells in ([bg()], [bg(depth=6,child=(0,30,30))],
                      [bg(coordinate=(3,2,2)),bg(1,coordinate=(3,2,2),depth=2)],
                      [7+8*(2+8*2),bg(depth=4,child=(0,8,8)),bg(1,depth=8,child=(5,4,3))]):
            for neighbors in (False,True):
                state=fixture(cells)
                m=manager(state,growth_use_26_neighbors=neighbors)
                expected=growth.generate_adaptive_growth_numpy(m,state,10)
                actual=generate_growth_gpu(m,state,10)
                for name,width in growth._WIDTHS.items():
                    got=np.asarray(actual[name]).reshape(-1,width)
                    desired=expected[name].reshape(-1,width)
                    if width==1:
                        np.testing.assert_array_equal(got,desired,err_msg=name)
                    else:
                        np.testing.assert_allclose(got,desired,rtol=3e-12,atol=3e-12,err_msg=name)
                assert_disjoint(self,state,expected)
                model=mi.VoxelReconstruction.from_state(8,[-1.]*3,[1.]*3,
                    {k:mi.UInt64(v) if k=='cell_indices' else mi.Float(v) for k,v in state['model_state'].items()},
                    sh_count=4,opacity_sh_count=1,background_shell_levels=3,background_adaptive=True)
                from reconstruction_growth_rgb import upload_rgb_growth_candidates
                upload_rgb_growth_candidates(model,actual['candidate_state'])
                model.clear_growth_candidates()
                del model

    def test_empty_subset_cap_and_cooldown(self):
        from reconstruction_growth_candidates_gpu import generate_growth_gpu
        cases=[fixture([]),fixture([bg()]),fixture([bg(),bg(1,coordinate=(3,2,2))])]
        for state in cases:
            m=manager(state,background_growth_max_candidates=7)
            m.cooldown={int(bg(depth=2,child=(1,1,1))):12}
            expected=growth.generate_adaptive_growth_numpy(m,state,10)
            targets=expected['cells'][::2]
            expected=growth.generate_adaptive_growth_numpy(m,state,10,targets)
            actual=generate_growth_gpu(m,state,10,dr.cuda.UInt64(targets))
            for name,width in growth._WIDTHS.items():
                np.testing.assert_allclose(np.asarray(actual[name]).reshape(-1,width),expected[name].reshape(-1,width),
                                           atol=3e-12,rtol=3e-12,err_msg=name)

    def test_partition_and_pair_matching_gpu(self):
        from reconstruction_growth_filter import match_growth_pairs_gpu
        state=fixture([])
        cells=np.array([bg(),bg(depth=2),bg(depth=1),73,74,bg(1,depth=8)],np.uint64)
        expected=growth.filter_nonoverlapping_growth_numpy(cells,state['model_metadata'],np)
        got=growth.filter_nonoverlapping_growth_gpu(dr.cuda.UInt64(cells),state['model_metadata'],dr)
        np.testing.assert_array_equal(np.asarray(got),expected)
        parents=cells[::-1].copy()
        found,rows=match_growth_pairs_gpu(dr.cuda.UInt64(cells),dr.cuda.UInt64(parents),
                                        dr.cuda.UInt64(cells[[4,2,1]]),dr.cuda.UInt64(parents[[4,2,0]]),dr)
        np.testing.assert_array_equal(np.asarray(found),[True,True,False])
        np.testing.assert_array_equal(np.asarray(rows)[:2],[4,2])


if __name__=='__main__':
    use_gpu='--gpu' in sys.argv
    if use_gpu:
        sys.argv.remove('--gpu')
        from reconstruction_common import initialize_runtime
        from preview import RuntimeConfig
        directory=Path(__file__).resolve().parents[3]/'.cache'/'adaptive-growth-tests'
        directory.mkdir(parents=True,exist_ok=True)
        mi,dr,_,handles=initialize_runtime(RuntimeConfig(),directory,{})
    unittest.main()
