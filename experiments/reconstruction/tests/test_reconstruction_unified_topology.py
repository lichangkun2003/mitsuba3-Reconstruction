"""CPU checks for mixed-depth core growth and the fixed-topology SH tail."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import reconstruction_adaptive_growth as growth
import reconstruction_topology as topology
from reconstruction_world_shell import (CORE_ADAPTIVE_TAG, ADAPTIVE_TAG,
    core_adaptive_encode_numpy, core_adaptive_decode_numpy, leaf_boxes_numpy,
    validate_adaptive_cells_numpy)
from test_reconstruction_adaptive_growth import fixture, manager, bg, assert_disjoint


def core(depth, xyz):
    morton = sum(((xyz[a] >> b) & 1) << (3*b+a)
                 for b in range(depth) for a in range(3))
    return int(core_adaptive_encode_numpy(depth, morton, np))


def unified(cells, resolution=256, levels=5, maximum=1024):
    checkpoint = fixture(cells, core=resolution, levels=levels)
    checkpoint['model_metadata'].update(world_shell_version=8,
        foreground_refinement='adaptive', foreground_max_resolution=maximum,
        background_resolution_mode='core', background_max_resolution=maximum)
    return checkpoint


class MixedCoreGrowth(unittest.TestCase):
    def test_all_core_depths_round_trip_and_world_boxes(self):
        cells = [core(d, (2**d-1, 2**(d-1), 1)) for d in range(1, 11)]
        checkpoint = unified(cells, resolution=2)
        metadata = checkpoint['model_metadata']
        op = growth._Ops(np)
        xyz, size = growth._decode(cells, metadata, op)
        actual, _, _, valid, split = growth._canonical(xyz, size, metadata, op)
        np.testing.assert_array_equal(actual, np.asarray(cells, np.uint64))
        np.testing.assert_array_equal(size, np.arange(9, -1, -1))
        self.assertTrue(valid.all())
        self.assertFalse(split.any())
        lower, upper = leaf_boxes_numpy(np.asarray(cells, np.uint64), 2,
                                       [-1.]*3, [1.]*3, 5, np)
        root = 1024 << 5
        np.testing.assert_allclose(np.stack(xyz, 1)/512 - root/1024, lower)
        np.testing.assert_allclose(np.exp2(size)/512, upper[:, 0]-lower[:, 0])

    def test_refined_core_growth_preserves_parent_scale(self):
        checkpoint = unified([core(9, (200, 200, 200))])
        result = growth.generate_adaptive_growth_numpy(manager(checkpoint), checkpoint, 30)
        self.assertEqual(len(result['cells']), 6)
        self.assertTrue(np.all((result['cells'] & CORE_ADAPTIVE_TAG) != 0))
        depth, _ = core_adaptive_decode_numpy(result['cells'], np)
        np.testing.assert_array_equal(depth, 9)
        assert_disjoint(self, checkpoint, result)

    def test_descendant_blocks_only_occupied_part_of_coarse_neighbor(self):
        checkpoint = unified([core(2, (1, 1, 1)), core(3, (4, 2, 2))], resolution=4)
        result = growth.generate_adaptive_growth_numpy(manager(checkpoint), checkpoint, 30)
        self.assertGreater(len(result['cells']), 6)
        assert_disjoint(self, checkpoint, result)
        self.assertNotIn(core(2, (2, 1, 1)), result['cells'])

    def test_core_shell_boundary_uses_the_same_interval_occupancy(self):
        checkpoint = unified([core(2, (3, 1, 1)), bg(depth=1, child=(0, 1, 1))], resolution=4)
        result = growth.generate_adaptive_growth_numpy(manager(checkpoint), checkpoint, 30)
        self.assertGreater(len(result['cells']), 0)
        self.assertTrue(np.any(result['cells'] >= ADAPTIVE_TAG))
        self.assertTrue(np.any(result['cells'] < ADAPTIVE_TAG))
        assert_disjoint(self, checkpoint, result)

    def test_growth_obeys_global_world_size_floor(self):
        checkpoint = unified([core(4, (5, 5, 5))], resolution=8, maximum=16)
        op = growth._Ops(np)
        xyz, _ = growth._decode(checkpoint['model_state']['cell_indices'], checkpoint['model_metadata'], op)
        target, _, size, _, _ = growth._canonical(xyz, np.array([0], np.int32), checkpoint['model_metadata'], op)
        np.testing.assert_array_equal(size, [6])
        self.assertEqual(core_adaptive_decode_numpy(target, np)[0][0], 4)

    def test_unified_growth_proposals_are_bounded(self):
        checkpoint = unified([core(9, (200, 200, 200))])
        result = growth.generate_adaptive_growth_numpy(
            manager(checkpoint, background_growth_max_candidates=3), checkpoint, 30)
        self.assertLessEqual(len(result['cells']), 3)

    def test_coarse_background_enters_core_at_initial_resolution(self):
        checkpoint = unified([bg()], resolution=8, levels=3)
        result = growth.generate_adaptive_growth_numpy(manager(checkpoint), checkpoint, 30)
        foreground = result['cells'][result['cells'] < ADAPTIVE_TAG]
        self.assertGreater(len(foreground), 1)
        depth, _ = core_adaptive_decode_numpy(foreground, np)
        np.testing.assert_array_equal(depth, 3)
        assert_disjoint(self, checkpoint, result)

    def test_fine_background_keeps_its_scale_when_entering_core(self):
        checkpoint = unified([bg(depth=6, child=(0, 30, 30))], resolution=8, levels=3)
        result = growth.generate_adaptive_growth_numpy(manager(checkpoint), checkpoint, 30)
        foreground = result['cells'][result['cells'] < ADAPTIVE_TAG]
        self.assertGreater(len(foreground), 0)
        depth, _ = core_adaptive_decode_numpy(foreground, np)
        np.testing.assert_array_equal(depth, 7)
        assert_disjoint(self, checkpoint, result)


class FrozenTopology(unittest.TestCase):
    def build(self, *, total=1500, active=1, fraction=.2):
        state = unified([core(8, (100, 100, 100))])['model_state']
        state['opacity_sh'] = np.zeros(9, np.float32)
        model = SimpleNamespace(resolution=256, core_resolution=256,
            foreground_refinement='adaptive', background_refinement='adaptive',
            scene_mapping='world_shell', active_voxel_count=1,
            opacity_sh_count=9, active_opacity_sh_count=active,
            opacity_sh_degree_progression=True, snapshot=lambda: state,
            clear_growth_candidates=lambda: None)
        settings = topology.TopologySettings(freeze_last_fraction=fraction,
            growth_freeze_last_epochs=0, use_rgb_growth=True, use_rgb_deletion=True,
            use_low_opacity_deletion=True, growth_inherit_opacity=True,
            use_growth_evidence=False, gpu_topology=False,
            deletion_opacity_confirmation_windows=1)
        schedule = topology.CoarseToFineConfig(refinement='adaptive',
            start_resolution=256, target_resolution=1024, total_epochs=total)
        return model, topology.TopologyManager(model, 3, settings, schedule)

    def test_no_global_level_transition_or_depth_gate(self):
        model, manager = self.build()
        self.assertFalse(manager.uniform_refinement)
        self.assertEqual(manager.budgets, {256: 1500})
        self.assertTrue(manager._growth_due(600))
        self.assertTrue(manager._scheduled_deletion_due(600))
        self.assertTrue(manager._scheduled_deletion_due(1050))

    def test_first_frozen_epoch_disables_every_evidence_path(self):
        model, manager = self.build()
        self.assertFalse(manager.topology_frozen(1200))
        self.assertTrue(manager._growth_due(1200))
        self.assertTrue(manager.collect_deletion(1199))
        for boundary in (1201, 1210, 1500):
            self.assertTrue(manager.topology_frozen(boundary))
            self.assertTrue(manager.growth_frozen(boundary))
            self.assertTrue(manager.deletion_frozen(boundary))
            self.assertFalse(manager._growth_due(boundary))
            self.assertFalse(manager.collect_deletion(boundary-1))
            self.assertFalse(manager._low_opacity_deletion_due(boundary))

    def test_fraction_rounding_matches_global_sh_boundary(self):
        _, manager = self.build(total=1501)
        self.assertFalse(manager.topology_frozen(1200))
        self.assertTrue(manager.topology_frozen(1201))

    def test_frozen_tail_never_commits_even_stale_topology(self):
        model, manager = self.build(active=9)
        manager.pending_growth = {'stale': True}
        manager.pending_rgb_deletion = {'stale': True}
        with (patch.object(topology, 'refine_state', side_effect=AssertionError('frozen refinement')),
              patch.object(manager, '_evaluate_low_opacity_deletion', side_effect=AssertionError('frozen deletion'))):
            replacement, event = manager.finish_epoch(model, None, 1201)
        self.assertIsNone(replacement)
        self.assertTrue(event['topology_frozen'])
        self.assertFalse(event['finished'])
        self.assertIsNone(manager.pending_growth)
        self.assertIsNone(manager.pending_rgb_deletion)
        _, final = manager.finish_epoch(model, None, 1500)
        self.assertTrue(final['finished'])

    def test_high_opacity_orders_cannot_modify_unfrozen_topology(self):
        model, manager = self.build(active=9)
        with self.assertRaisesRegex(ValueError, 'fixed-topology'):
            manager.require_isotropic_topology(model, 1200)
        manager.require_isotropic_topology(model, 1201)

    def test_low_opacity_rule_reads_dc_from_nine_coefficient_rows(self):
        model, manager = self.build()
        model.snapshot()['opacity_sh'][:] = 20.
        model.snapshot()['opacity_sh'][0] = -20.
        selected = manager._evaluate_low_opacity_deletion(model, 30, {})
        np.testing.assert_array_equal(selected, [True])

    def test_growth_opacity_initializes_only_dc_while_higher_orders_inactive(self):
        model, manager = self.build()
        model.snapshot()['opacity_sh'][0] = 1.
        coefficients = topology._growth_opacity_coefficients(model.snapshot(),
            dict(opacity_sh_count=9, active_opacity_sh_count=1), np.array([0]), manager.settings, np)
        np.testing.assert_array_equal(coefficients[:, 1:], 0.)
        with self.assertRaisesRegex(ValueError, 'active opacity'):
            topology._growth_opacity_coefficients(model.snapshot(),
                dict(opacity_sh_count=9, active_opacity_sh_count=9), np.array([0]), manager.settings, np)


@unittest.skipUnless('--gpu' in sys.argv, 'explicit --gpu required')
class CudaMixedCoreGrowth(unittest.TestCase):
    def test_core_and_shell_growth_match_cpu_oracle(self):
        from reconstruction_growth_candidates_gpu import generate_growth_gpu
        for cells in ([core(9, (200, 200, 200))],
                      [core(2, (1, 1, 1)), core(3, (4, 2, 2))],
                      [core(2, (3, 1, 1)), bg(depth=1, child=(0, 1, 1))]):
            for neighbors in (False, True):
                checkpoint = unified(cells, resolution=2)
                m = manager(checkpoint, growth_use_26_neighbors=neighbors)
                expected = growth.generate_adaptive_growth_numpy(m, checkpoint, 30)
                actual = generate_growth_gpu(m, checkpoint, 30)
                for name, width in growth._WIDTHS.items():
                    got = np.asarray(actual[name]).reshape(-1, width)
                    wanted = expected[name].reshape(-1, width)
                    if width == 1:
                        np.testing.assert_array_equal(got, wanted, err_msg=name)
                    else:
                        np.testing.assert_allclose(got, wanted, rtol=3e-12, atol=3e-12, err_msg=name)

    def test_gpu_core_depth_decode_round_trip(self):
        cells = [core(d, (2**d-1, 2**(d-1), 1)) for d in range(1, 11)]
        checkpoint = unified(cells, resolution=2)
        metadata = checkpoint['model_metadata']
        op = growth._Ops(dr, True)
        xyz, size = growth._decode(dr.cuda.UInt64(cells), metadata, op)
        actual, _, _, valid, split = growth._canonical(xyz, size, metadata, op)
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(cells, np.uint64))
        np.testing.assert_array_equal(np.asarray(size), np.arange(9, -1, -1))
        self.assertTrue(np.asarray(valid).all())
        self.assertFalse(np.asarray(split).any())

    def test_gpu_nine_opacity_coefficients_copy_dc_only(self):
        from reconstruction_growth_candidates_gpu import generate_growth_gpu
        checkpoint = unified([core(9, (200, 200, 200))])
        checkpoint['model_metadata'].update(opacity_sh_count=9, active_opacity_sh_count=1)
        checkpoint['model_state']['opacity_sh'] = np.arange(9, dtype=np.float32)
        m = manager(checkpoint, growth_inherit_opacity=True)
        actual = generate_growth_gpu(m, checkpoint, 30)
        opacity = np.asarray(actual['opacity_sh']).reshape(-1, 9)
        self.assertGreater(len(opacity), 0)
        np.testing.assert_array_equal(opacity[:, 1:], 0.)


if __name__ == '__main__':
    if '--gpu' in sys.argv:
        sys.argv.remove('--gpu')
        from reconstruction_common import initialize_runtime
        from preview import RuntimeConfig
        directory = Path(__file__).resolve().parents[3]/'.cache'/'unified-topology-tests'
        directory.mkdir(parents=True, exist_ok=True)
        mi, dr, _, handles = initialize_runtime(RuntimeConfig(), directory, {})
    unittest.main()
