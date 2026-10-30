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
    def test_pruned_parameter_indirection_matches_compacted_fields(self):
        original = checkpoint([0, 73, 7, 75, 90])
        rng = np.random.default_rng(38491)
        original['model_state']['opacity_sh'] = np.linspace(-3, 3, 5, dtype=np.float32)
        original['model_state']['radiance_sh'] = rng.normal(size=5*12).astype(np.float32)
        original['model_state']['centers'] += rng.uniform(-.02, .02, 15).astype(np.float32)
        survivors = np.array([1, 3, 4], np.uint32)
        widths = dict(cell_indices=1, centers=3, log_scales=3, rotations=4,
                      radiance_sh=12, opacity_sh=1)
        compact = dict(model_metadata=dict(original['model_metadata']), model_state={
            name: value.reshape(-1, widths[name])[survivors].reshape(-1)
            for name, value in original['model_state'].items()})
        indirect = dict(model_metadata=dict(original['model_metadata']),
                        model_state=dict(original['model_state']), parameter_parent_rows=survivors)
        indirect['model_state']['cell_indices'] = compact['model_state']['cell_indices']
        manager = manager_for(compact, use_rgb_growth=True, growth_use_26_neighbors=True,
            growth_min_supporting_parents=2, growth_evidence_penetration=0.,
            growth_inherit_opacity=True, growth_initial_opacity=.3)
        expected = generate_growth_gpu(manager, compact, 10, target_cells=dr.cuda.UInt32([74]))
        actual = generate_growth_gpu(manager, indirect, 10, target_cells=dr.cuda.UInt32([74]))
        self.assertGreater(len(actual['cells']), 0)
        # Parent IDs remain in the compact topology row order. Every geometry,
        # opacity and inherited DC read must address the original source row.
        for name in (*widths, 'cells', 'parent_ids', 'parent_cells', 'geometric_support'):
            if name == 'cell_indices':
                continue
            np.testing.assert_array_equal(np.array(actual[name]), np.array(expected[name]), err_msg=name)

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

    def test_bitmap_word_boundaries_duplicates_and_masked_queries(self):
        # Concurrent writes to the same word must combine bits, including the
        # unsigned high bit and the last partial word of a non-power-of-two grid.
        first=np.concatenate((np.arange(32,dtype=np.uint32),[31,32,63])).astype(np.uint32)
        second=np.array([31,32,64,69,69],dtype=np.uint32)
        bitmap=growth_gpu._make_bitmap(70,(dr.cuda.UInt32(first),dr.cuda.UInt32(second)),dr)
        self.assertEqual(dr.width(bitmap),3)
        np.testing.assert_array_equal(np.array(bitmap),
            np.array([0xffffffff,0x80000001,0x21],dtype=np.uint32))
        queries=np.concatenate((np.arange(70,dtype=np.uint32),
                                np.array([31,96,0xffffffff],dtype=np.uint32)))
        active=np.arange(len(queries))<70
        actual=growth_gpu._bitmap_contains(bitmap,dr.cuda.UInt32(queries),dr.cuda.Bool(active),dr)
        expected=np.isin(queries,np.concatenate((first,second))) & active
        np.testing.assert_array_equal(np.array(actual),expected)

    def test_parent_batches_preserve_geometry_ids_and_order(self):
        rng=np.random.default_rng(831903)
        occupied=np.array([0,31,32,63,64,511,73,75,91,134,198,217,283,361,387,409,446],dtype=np.uint32)
        rng.shuffle(occupied)
        state=checkpoint(occupied)
        count=len(occupied)
        state['model_state']['centers']=rng.uniform(.05,.95,count*3).astype(np.float32)
        state['model_state']['rotations']=rng.normal(size=count*4).astype(np.float32)
        state['model_state']['log_scales']+=rng.uniform(-.3,.3,count*3).astype(np.float32)
        state['model_state']['radiance_sh']=np.arange(count*12,dtype=np.float32)/100
        state['model_state']['opacity_sh']=np.linspace(-4,4,count,dtype=np.float32)
        for neighbors in (False,True):
            manager=manager_for(state,use_rgb_growth=True,growth_use_26_neighbors=neighbors,
                growth_evidence_penetration=.05,growth_min_supporting_parents=1,
                growth_inherit_opacity=True,growth_initial_opacity=.3)
            manager.cooldown={33:12,74:12}
            with self.subTest(neighbors=neighbors):
                with patch.object(growth_gpu,'geometry_gpu',wraps=growth_gpu.geometry_gpu) as geometry:
                    result=self.compare(manager,state,parent_batch_size=7)
                    batches=[np.array(call.args[0]['cell_indices']) for call in geometry.call_args_list]
                self.assertEqual([len(batch) for batch in batches],[7,7,3])
                np.testing.assert_array_equal(np.concatenate(batches),occupied)
                self.assertGreater(len(result['cells']),0)
                # Compare a literal full-batch divisor with a runtime tail
                # divisor and disjoint component writes across several batches.
                single=generate_growth_gpu(manager,state,10,parent_batch_size=count)
                for key in ('cells','parent_cells','parent_ids','geometric_support'):
                    np.testing.assert_array_equal(np.array(result[key]),np.array(single[key]),err_msg=key)
                for key in ('centers','log_scales','rotations'):
                    np.testing.assert_allclose(np.array(result[key]),np.array(single[key]),
                                               atol=3e-12,rtol=3e-12,err_msg=key)
                for key in single['candidate_state']:
                    np.testing.assert_array_equal(np.array(result['candidate_state'][key]),
                        np.array(single['candidate_state'][key]),err_msg=key)

    def test_parent_support_is_counted_across_batches(self):
        state=checkpoint([75,73])
        manager=manager_for(state,use_rgb_growth=True,growth_min_supporting_parents=2)
        result=self.compare(manager,state,target_cells=[74],parent_batch_size=1)
        np.testing.assert_array_equal(np.array(result['cells']),[74,74])
        np.testing.assert_array_equal(np.array(result['parent_ids']),[1,0])
        np.testing.assert_array_equal(np.array(result['geometric_support']),[2,2])

    def test_empty_target_subset_skips_bitmaps_and_geometry(self):
        state=checkpoint([73,75])
        manager=manager_for(state,use_rgb_growth=True,growth_min_supporting_parents=2)
        with patch.object(growth_gpu,'geometry_gpu',side_effect=AssertionError('no parent geometry')):
            with patch.object(growth_gpu,'_make_bitmap',side_effect=AssertionError('no occupancy bitmap')):
                result=self.compare(manager,state,target_cells=[],parent_batch_size=1)
        self.assertEqual(len(result['cells']),0)
        self.assertTrue(all(len(value)==0 for value in result['candidate_state'].values()))

    def test_only_exposed_parents_reach_fp64_geometry(self):
        hole=219
        occupied=np.delete(np.arange(512,dtype=np.uint32),hole)
        state=checkpoint(occupied)
        coordinates=np.stack((occupied%8,occupied//8%8,occupied//64),axis=1).astype(np.int64)
        delta=np.abs(coordinates-np.array([hole%8,hole//8%8,hole//64]))
        for neighbors in (False,True):
            manager=manager_for(state,use_rgb_growth=True,growth_use_26_neighbors=neighbors,
                growth_min_supporting_parents=2,growth_evidence_penetration=0.)
            exposed=np.max(delta,axis=1)==1 if neighbors else np.sum(delta,axis=1)==1
            with self.subTest(neighbors=neighbors):
                with patch.object(growth_gpu,'geometry_gpu',wraps=growth_gpu.geometry_gpu) as geometry:
                    self.compare(manager,state,parent_batch_size=5)
                    batches=[np.array(call.args[0]['cell_indices']) for call in geometry.call_args_list]
                np.testing.assert_array_equal(np.concatenate(batches),occupied[exposed])
                self.assertTrue(all(len(batch)<=5 for batch in batches))

    def test_target_neighborhood_keeps_all_supporting_parents(self):
        # Requested targets occur more than once, and distant eligible parents
        # must not enter FP64 geometry. Both local supporting parents survive.
        state=checkpoint([0,7,73,217,75,511])
        manager=manager_for(state,use_rgb_growth=True,growth_use_26_neighbors=True,
            growth_min_supporting_parents=2,growth_evidence_penetration=0.)
        with patch.object(growth_gpu,'geometry_gpu',wraps=growth_gpu.geometry_gpu) as geometry:
            result=self.compare(manager,state,target_cells=[74,74],parent_batch_size=1)
            parents=np.concatenate([np.array(call.args[0]['cell_indices']) for call in geometry.call_args_list])
        np.testing.assert_array_equal(parents,[73,75])
        np.testing.assert_array_equal(np.array(result['parent_ids']),[2,4])
        np.testing.assert_array_equal(np.array(result['geometric_support']),[2,2])

    def test_fully_blocked_or_unreachable_targets_skip_geometry(self):
        cases=[(checkpoint(np.arange(512,dtype=np.uint32)),None,{}),
               (checkpoint([73,75]),[511],{}),
               (checkpoint([73,75]),[74],{74:12})]
        for state,targets,cooldown in cases:
            manager=manager_for(state,use_rgb_growth=True,growth_use_26_neighbors=True,
                growth_min_supporting_parents=1)
            manager.cooldown=cooldown
            with self.subTest(targets=targets,cooldown=cooldown):
                with patch.object(growth_gpu,'geometry_gpu',side_effect=AssertionError('no exposed parent')):
                    result=self.compare(manager,state,target_cells=targets)
                self.assertEqual(len(result['cells']),0)

    def test_pair_capacity_and_compaction_count_are_bounded(self):
        state=checkpoint([73,75,217,219,361])
        manager=manager_for(state,use_rgb_growth=True,growth_use_26_neighbors=True,
            growth_min_supporting_parents=1,growth_evidence_penetration=.05)
        # An oversized requested parent batch must still respect pair capacity.
        with patch.object(growth_gpu,'MAX_BATCH_PAIRS',26*2):
            with patch.object(growth_gpu,'geometry_gpu',wraps=growth_gpu.geometry_gpu) as geometry:
                with patch.object(dr,'compress',wraps=dr.compress) as compress:
                    result=self.compare(manager,state,parent_batch_size=1000)
            sizes=[len(call.args[0]['cell_indices']) for call in geometry.call_args_list]
        self.assertEqual(sizes,[2,2,1])
        self.assertGreater(len(result['cells']),0)
        # Parent filtering plus global group/support filtering, and only two
        # compressions per batch regardless of the 26 neighbor directions.
        self.assertLessEqual(compress.call_count,3+2*len(sizes))

    def test_ragged_field_concatenation_preserves_rows(self):
        dimensions=dict(parent_ids=1,parent_cells=1,cells=1,centers=3,log_scales=3,rotations=4)
        pieces,expected=[],{name:[] for name in dimensions}
        for ordinal,count in enumerate((3,7,1)):
            piece={}
            for name,width in dimensions.items():
                values=np.arange(count*width)+(ordinal+1)*100
                dtype=dr.cuda.UInt32 if width==1 else dr.cuda.Float64
                piece[name]=dtype(values)
                expected[name].append(values)
            pieces.append(piece)
        result=growth_gpu._concatenate_fields(pieces,dimensions,dr)
        self.assertEqual(pieces,[])
        for name in dimensions:
            np.testing.assert_array_equal(np.array(result[name]),np.concatenate(expected[name]),err_msg=name)


if __name__=='__main__':
    from reconstruction_common import initialize_runtime
    from preview import RuntimeConfig
    with tempfile.TemporaryDirectory(prefix='mitsuba-growth-device-') as directory:
        mi,dr,_,handles=initialize_runtime(RuntimeConfig(),Path(directory),{})
        unittest.main()
