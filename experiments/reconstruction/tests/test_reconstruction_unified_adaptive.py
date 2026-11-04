"""CPU oracles for shared foreground/background adaptive refinement."""
from pathlib import Path
import copy
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruction_world_shell import (
    CORE_ADAPTIVE_TAG, ADAPTIVE_TAG, adaptive_encode_numpy,
    core_adaptive_encode_numpy, core_adaptive_decode_numpy, core_cells_to_adaptive_numpy,
    leaf_boxes_numpy, validate_adaptive_cells_numpy, adaptive_depth_limit,
    world_shell_metadata_layout,
)
from reconstruction_adaptive_background import (select_adaptive_background, refine_adaptive_background_numpy,
                                                 prepare_adaptive_background)
from reconstruction_topology import CoarseToFineConfig, refine_state
from reconstruction_checkpoint import save_checkpoint, load_checkpoint
from reconstruction_export import ArraySource


def metadata():
    return dict(resolution=256, core_resolution=256, grid_min=[-1.]*3, grid_max=[1.]*3,
                sh_count=9, active_sh_count=1, opacity_sh_count=1, scene_mapping="world_shell",
                world_shell_version=8, background_shell_levels=5, background_geometry="ellipsoid",
                background_refinement="adaptive", background_subdivision_level=0,
                background_resolution_mode="core", background_max_resolution=1024,
                foreground_refinement="adaptive", foreground_max_resolution=1024)


def camera():
    return [dict(position=[0., 0., -3.], right=[1., 0., 0.], up=[0., 1., 0.],
                 forward=[0., 0., 1.], fx=1024., fy=1024., cx=512., cy=512., width=1024, height=1024)]


def state(cells):
    low, high = leaf_boxes_numpy(cells, 256, [-1.]*3, [1.]*3, 5, np)
    count = len(cells)
    return dict(cell_indices=np.asarray(cells, np.uint64), centers=np.full(count*3, .5, np.float32),
                log_scales=np.log((high-low)*.9).astype(np.float32).ravel(),
                rotations=np.tile(np.array([1., 0., 0., 0.], np.float32), count),
                radiance_sh=np.arange(count*27, dtype=np.float32)/100,
                opacity_sh=np.zeros(count, np.float32))


class UnifiedAdaptiveTests(unittest.TestCase):
    def test_core_round_trip_and_legacy_boxes(self):
        for depth in range(1, 11):
            n = 1 << depth
            flat = np.array([0, n**3//2, n**3-1], np.uint64)
            encoded = core_cells_to_adaptive_numpy(flat, n, np)
            decoded_depth, morton = core_adaptive_decode_numpy(encoded, np)
            np.testing.assert_array_equal(decoded_depth, depth)
            np.testing.assert_array_equal(core_adaptive_encode_numpy(decoded_depth, morton, np), encoded)
            for a, b in zip(leaf_boxes_numpy(flat, n, [-1.]*3, [1.]*3, 5, np),
                            leaf_boxes_numpy(encoded, 256, [-1.]*3, [1.]*3, 5, np)):
                np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all(encoded < ADAPTIVE_TAG))

    def test_core_overlap_rejected_across_encoding_generations(self):
        old = np.array([0], np.uint64)
        duplicate = core_cells_to_adaptive_numpy(old, 256, np)
        child = core_adaptive_encode_numpy([9], [0], np)
        for cells in (np.r_[old, duplicate], np.r_[duplicate, child]):
            with self.assertRaisesRegex(ValueError, "overlap"):
                validate_adaptive_cells_numpy(cells, 256, 5, np)
        validate_adaptive_cells_numpy(np.r_[duplicate, core_adaptive_encode_numpy([9], [8], np),
                                            adaptive_encode_numpy([0], [0], [0], np)], 256, 5, np)
        for bad in (CORE_ADAPTIVE_TAG, CORE_ADAPTIVE_TAG | 2, CORE_ADAPTIVE_TAG | (1 << 34),
                    CORE_ADAPTIVE_TAG | (1 << 63) | 8):
            with self.assertRaises(ValueError):
                core_adaptive_decode_numpy(np.array([bad], np.uint64), np)

    def test_global_ceiling_open_from_first_epoch(self):
        meta = metadata()
        world_shell_metadata_layout(meta)
        np.testing.assert_array_equal(adaptive_depth_limit(meta, np.arange(5)), np.arange(5)+9)
        cells = np.r_[core_adaptive_encode_numpy([8, 9, 10], [0, 8, 64], np),
                      adaptive_encode_numpy([4*56, 4*56+1], [12, 13], [0, 0], np)]
        selected = select_adaptive_background(dict(model_metadata=meta, model_state=dict(cell_indices=cells)),
            camera(), np.ones(len(cells)), np, max_new_leaves=100,
            projector=lambda low, high, views: np.full(len(low), 100., np.float32))
        self.assertEqual(set(selected.tolist()), {0, 1, 3})

    def test_background_preparation_uses_final_world_floor(self):
        with patch("reconstruction_adaptive_background.projected_leaf_pixels",
                   side_effect=lambda cells, *args: np.full(len(cells), 100., np.float32)):
            result = prepare_adaptive_background(np.empty((0,3)), np.empty((0,3)),
                4, [-1.]*3, [1.]*3, 1, camera(), np, initial_leaf_budget=4000,
                resolution_mode="core", max_resolution=8,
                foreground_refinement="adaptive", foreground_max_resolution=8)
        self.assertEqual(len(result["cell_indices"]), 56*64)
        self.assertEqual(result["foreground_max_resolution"], 8)
        self.assertEqual(result["max_depth"], 2)
        low, high = leaf_boxes_numpy(result["cell_indices"], 4, [-1.]*3, [1.]*3, 1, np)
        np.testing.assert_array_equal(high-low, .25)

    def test_foreground_budget_survives_high_background_priority(self):
        cells = np.r_[core_adaptive_encode_numpy(8, np.arange(4, dtype=np.uint64), np),
                      adaptive_encode_numpy(np.arange(8), 0, 0, np)]
        checkpoint = dict(model_metadata=metadata(), model_state=dict(cell_indices=cells))
        options = dict(projector=lambda low, high, views: np.full(len(low), 100., np.float32),
                       max_new_leaves=28)
        selected = select_adaptive_background(checkpoint, camera(), np.r_[np.ones(4), np.full(8, 10000.)], np,
                                              **options)
        self.assertEqual(len(selected), 4)
        self.assertEqual(np.count_nonzero(selected < 4), 1)
        selected = select_adaptive_background(checkpoint, camera(), np.ones(12), np,
                                              max_background_new_leaves=0, **options)
        np.testing.assert_array_equal(np.sort(selected), np.arange(4))

    def test_mixed_fitting_preserves_retained_rows_and_refines_each_leaf_size(self):
        cells = np.r_[core_adaptive_encode_numpy([8, 9], [0, 8], np),
                      adaptive_encode_numpy([45], [2], [0], np)]
        checkpoint = dict(model_metadata=metadata(), model_state=state(cells))
        old = copy.deepcopy(checkpoint)
        fitted = refine_adaptive_background_numpy(checkpoint, np.array([0, 2]), CoarseToFineConfig(),
                                                   100, refine_state, np)
        self.assertEqual(fitted["replaced"], 2)
        self.assertEqual(fitted["children"], 16)
        self.assertEqual(fitted["source_rows"][0], 1)
        for name, width in (("centers", 3), ("log_scales", 3), ("rotations", 4),
                            ("radiance_sh", 27), ("opacity_sh", 1)):
            np.testing.assert_array_equal(fitted["model_state"][name][:width],
                                          old["model_state"][name][width:2*width])
        validate_adaptive_cells_numpy(fitted["model_state"]["cell_indices"], 256, 5, np)
        np.testing.assert_array_equal(checkpoint["model_state"]["cell_indices"], old["model_state"]["cell_indices"])
        output = fitted["model_state"]["cell_indices"]
        depth, _ = core_adaptive_decode_numpy(output[(output < ADAPTIVE_TAG)], np)
        np.testing.assert_array_equal(depth, 9)

    def test_streamed_checkpoint_roundtrip_keeps_foreground_and_opacity_schedule(self):
        cells = np.r_[core_adaptive_encode_numpy([8,9],[0,8],np),
                      adaptive_encode_numpy([45],[2],[0],np)]
        arrays = state(cells)
        arrays["opacity_sh"] = np.arange(27, dtype=np.float32)/100
        md = dict(metadata(), opacity_sh_count=9, active_sh_count=9, active_opacity_sh_count=9)
        streamed = {name:ArraySource(value.shape, value.dtype,
            lambda start,count,value=value:value[start:start+count]) for name,value in arrays.items()}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"mixed.npz"
            with patch("reconstruction_export.CHUNK_BYTES", 8):
                save_checkpoint(path, streamed, md)
            restored = load_checkpoint(path)
            self.assertEqual(restored["model_metadata"], md)
            for name,value in arrays.items():
                np.testing.assert_array_equal(restored["model_state"][name], value)

    def test_streamed_core_overlap_across_chunks_preserves_last_good_checkpoint(self):
        cells = np.r_[np.uint64(0), adaptive_encode_numpy([45],[2],[0],np),
                      core_adaptive_encode_numpy([9],[8],np)]
        arrays = state(cells)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"mixed.npz"
            save_checkpoint(path, arrays, metadata())
            original = path.read_bytes()
            arrays["cell_indices"][-1] = core_adaptive_encode_numpy([9],[0],np)[0]
            streamed = {name:ArraySource(value.shape, value.dtype,
                lambda start,count,value=value:value[start:start+count]) for name,value in arrays.items()}
            with patch("reconstruction_export.CHUNK_BYTES", 8), self.assertRaisesRegex(ValueError, "overlap"):
                save_checkpoint(path, streamed, metadata())
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual([item.name for item in Path(directory).iterdir()], [path.name])


if __name__ == "__main__":
    unittest.main(verbosity=2)
