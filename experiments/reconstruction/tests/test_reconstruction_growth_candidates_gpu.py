"""CUDA proposal geometry/ordering versus the independent NumPy generator."""
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from reconstruction_topology import TopologyManager, _growth_opacity_coefficients
from test_reconstruction_growth_lookup import checkpoint, manager_for
from reconstruction_growth_candidates_gpu import generate_growth_gpu


class Proposals(unittest.TestCase):
    def compare(self, manager, state, boundary=10):
        expected=TopologyManager._generate_growth(manager,state,boundary)
        order=np.lexsort((expected['parent_cells'],expected['cells']))
        got=generate_growth_gpu(manager,state,boundary)
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


if __name__=='__main__':
    from reconstruction_common import initialize_runtime
    from preview import RuntimeConfig
    with tempfile.TemporaryDirectory(prefix='mitsuba-growth-device-') as directory:
        mi,dr,_,handles=initialize_runtime(RuntimeConfig(),Path(directory),{})
        unittest.main()
