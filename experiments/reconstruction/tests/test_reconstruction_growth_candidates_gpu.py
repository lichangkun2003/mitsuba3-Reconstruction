"""CUDA proposal geometry/ordering versus the independent NumPy generator."""
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from reconstruction_topology import TopologyManager, _growth_opacity_coefficients
from test_reconstruction_growth_lookup import checkpoint, manager_for
from reconstruction_growth_candidates_gpu import generate_growth_gpu
import reconstruction_growth_candidates_gpu as growth_gpu


class Proposals(unittest.TestCase):
    def compare(self, manager, state, boundary=10, target_cells=None, **device_options):
        expected=TopologyManager._generate_growth(manager,state,boundary)
        if target_cells is not None:
            keep=np.isin(expected['cells'],target_cells)
            expected={key:value[keep] if isinstance(value,np.ndarray) else value
                      for key,value in expected.items()}
        order=np.lexsort((expected['parent_cells'],expected['cells']))
        got=generate_growth_gpu(manager,state,boundary,
                                target_cells=None if target_cells is None else dr.cuda.UInt32(target_cells),
                                **device_options)
        for key,width in [('cells',1),('parent_cells',1),('parent_ids',1),
                          ('centers',3),('log_scales',3),('rotations',4)]:
            actual=np.array(got[key]).reshape(-1,width)
            desired=expected[key][order].reshape(-1,width)
            if width==1: np.testing.assert_array_equal(actual,desired,err_msg=key)
            else: np.testing.assert_allclose(actual,desired,atol=3e-12,rtol=3e-12,err_msg=key)
        opacity=_growth_opacity_coefficients(state['model_state'],state['model_metadata'],
                                             expected['parent_ids'][order],manager.settings,np)
        np.testing.assert_allclose(np.array(got['candidate_state']['opacity_sh']),
                                   opacity.ravel(),rtol=1e-6,atol=1e-6)
        sh_count=state['model_metadata']['sh_count']
        radiance=np.zeros((len(order),sh_count*3),dtype=np.float32)
        radiance[:,:3]=state['model_state']['radiance_sh'].reshape(-1,sh_count*3)[
            expected['parent_ids'][order],:3]
        np.testing.assert_array_equal(np.array(got['candidate_state']['radiance_sh']),radiance.ravel())
        _,inverse,counts=np.unique(expected['cells'][order],return_inverse=True,return_counts=True)
        np.testing.assert_array_equal(np.array(got['geometric_support']),counts[inverse])
        return got

    def test_random_rotated_geometry(self):
        rng=np.random.default_rng(525132)
        for neighbors in (False,True):
            for penetration in (0.,.05,.2):
                state=checkpoint(rng.choice(512,90,replace=False))
                state['model_state']['centers']=rng.uniform(.05,.95,270).astype(np.float32)
                state['model_state']['rotations']=rng.normal(size=360).astype(np.float32)
                state['model_state']['log_scales']+=rng.uniform(-.5,.5,270).astype(np.float32)
                state['model_state']['opacity_sh']=rng.normal(size=90).astype(np.float32)*8
                manager=manager_for(state,use_rgb_growth=True,growth_use_26_neighbors=neighbors,
                    growth_evidence_penetration=penetration,growth_min_supporting_parents=2,
                    growth_inherit_opacity=True,growth_initial_opacity=.3)
                manager.candidates=rng.random(90)<.1
                manager.growth_wait_start[rng.random(90)<.1]=9
                manager.cooldown={int(c):12 for c in rng.choice(512,20,replace=False)}
                with self.subTest(neighbors=neighbors,penetration=penetration):
                    self.compare(manager,state)

    def test_empty_full_and_subset(self):
        for cells in ([],list(range(512)),[73,75]):
            state=checkpoint(cells)
            manager=manager_for(state,use_rgb_growth=True,growth_min_supporting_parents=2)
            full=self.compare(manager,state)
            if len(cells)==2:
                subset=generate_growth_gpu(manager,state,10,target_cells=dr.cuda.UInt32([74]))
                self.assertEqual(list(np.array(subset['cells'])),[74,74])
                for key in ('centers','log_scales','rotations'):
                    width=4 if key=='rotations' else 3
                    np.testing.assert_array_equal(np.array(subset[key]).reshape(-1,width),
                        np.array(full[key]).reshape(-1,width)[np.array(full['cells'])==74])

    def test_all_proposals_fail_parent_support(self):
        state=checkpoint([73])
        manager=manager_for(state,use_rgb_growth=True,growth_min_supporting_parents=26)
        self.assertEqual(len(self.compare(manager,state)['cells']),0)

    def test_radial_shell_is_not_a_growth_region(self):
        state=checkpoint([217,219,221])
        state['model_metadata'].update(core_resolution=6,scene_mapping='radial_seed_shell')
        manager=manager_for(state,use_rgb_growth=True,growth_min_supporting_parents=2)
        result=self.compare(manager,state)
        self.assertNotIn(218,np.array(result['cells']))
        self.assertIn(220,np.array(result['cells']))
        # No core proposal may re-enter a deleted shell cell, including the
        # restricted post-vote regeneration path.
        subset=generate_growth_gpu(manager,state,10,target_cells=dr.cuda.UInt32([218,220,222]))
        self.assertEqual(list(np.array(subset['cells'])),[220,220])

    def test_shell_majority_preserves_original_parent_ids_and_occupancy(self):
        # Almost all occupied cells are forbidden shell parents. Scatter the
        # eligible core parents among them so compact IDs cannot be used as
        # original IDs for pair matching, radiance, opacity, or rotations.
        addresses=np.arange(512,dtype=np.uint32)
        coordinates=np.stack((addresses%8,addresses//8%8,addresses//64),axis=1).astype(np.int64)
        outside=np.sum((np.abs(2*coordinates+1-8)+1)**2,axis=1)>6**2
        occupied=np.concatenate((addresses[outside],[219,221,283,285])).astype(np.uint32)
        rng=np.random.default_rng(717023)
        rng.shuffle(occupied)
        state=checkpoint(occupied)
        state['model_metadata'].update(core_resolution=6,scene_mapping='radial_seed_shell')
        count=len(occupied)
        state['model_state']['radiance_sh']=np.arange(count*12,dtype=np.float32)/1000
        state['model_state']['opacity_sh']=np.linspace(-3,3,count,dtype=np.float32)
        state['model_state']['rotations']=rng.normal(size=count*4).astype(np.float32)
        manager=manager_for(state,use_rgb_growth=True,growth_use_26_neighbors=True,
            growth_min_supporting_parents=2,growth_evidence_penetration=0.,
            growth_inherit_opacity=True,growth_initial_opacity=.3)
        manager.candidates[occupied==283]=True
        manager.growth_wait_start[occupied==285]=10
        manager.cooldown={284:11}
        original_ids=np.flatnonzero(np.isin(occupied,[219,221]))
        self.assertFalse(np.array_equal(original_ids,np.arange(2)))
        with patch.object(growth_gpu,'geometry_gpu',wraps=growth_gpu.geometry_gpu) as geometry:
            result=self.compare(manager,state)
            packed_state=geometry.call_args.args[0]
            np.testing.assert_array_equal(np.array(packed_state['cell_indices']),occupied[original_ids])
            self.assertEqual(len(packed_state['centers']),6)
        self.assertGreater(len(result['cells']),0)
        self.assertTrue(np.isin(np.array(result['parent_ids']),original_ids).all())
        self.assertFalse(np.isin(np.array(result['cells']),occupied).any())
        self.assertNotIn(284,np.array(result['cells']))
        # Device-lifecycle eligibility and target revalidation retain every
        # eligible supporting parent, not only a previous winning proposal.
        eligible=~manager.candidates & (manager.growth_wait_start==0)
        subset=self.compare(manager,state,target_cells=[220,283,284,285],
            eligible=dr.cuda.Bool(eligible),cooldown_cells=dr.cuda.UInt32([284]))
        self.assertEqual(list(np.array(subset['cells'])),[220,220])

    def test_no_eligible_parent_skips_geometry(self):
        for reason in ('shell','candidate','warmup'):
            state=checkpoint([0,7,63,511] if reason=='shell' else [219,221])
            state['model_metadata'].update(core_resolution=6,scene_mapping='radial_seed_shell')
            manager=manager_for(state,use_rgb_growth=True,growth_min_supporting_parents=2)
            if reason=='candidate': manager.candidates[:]=True
            if reason=='warmup': manager.growth_wait_start[:]=10
            with self.subTest(reason=reason):
                with patch.object(growth_gpu,'geometry_gpu',side_effect=AssertionError('no parent geometry')):
                    result=self.compare(manager,state)
                self.assertEqual(len(result['cells']),0)
                self.assertTrue(all(len(value)==0 for value in result['candidate_state'].values()))


if __name__=='__main__':
    from reconstruction_common import initialize_runtime
    from preview import RuntimeConfig
    with tempfile.TemporaryDirectory(prefix='mitsuba-growth-device-') as directory:
        mi,dr,_,handles=initialize_runtime(RuntimeConfig(),Path(directory),{})
        unittest.main()
