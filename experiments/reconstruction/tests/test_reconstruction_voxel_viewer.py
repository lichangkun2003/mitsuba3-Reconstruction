"""CPU checks for portable, exact leaf geometry in the standalone voxel viewer."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reconstruction_checkpoint import save_checkpoint
from reconstruction_world_shell import adaptive_encode_numpy
from visualize_voxels import export_viewer, iter_boxes, read_layout


class VoxelViewerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / "model.npz"

    def save(self, cells, metadata=None):
        cells = np.asarray(cells, np.uint64)
        metadata = metadata or dict(resolution=4, grid_min=[-2., -4., -6.],
                                    grid_max=[2., 4., 6.], sh_count=1, opacity_sh_count=1)
        count = len(cells)
        state = dict(cell_indices=cells, centers=np.full(count * 3, 99., np.float32),
                     log_scales=np.zeros(count * 3, np.float32),
                     rotations=np.tile(np.array([1., 0., 0., 0.], np.float32), count),
                     radiance_sh=np.zeros(count * metadata["sh_count"] * 3, np.float32),
                     opacity_sh=np.ones(count * metadata["opacity_sh_count"], np.float32))
        save_checkpoint(self.path, state, metadata,
                        optimizer_state={"moments": np.zeros((count, 3), np.float32)})
        return metadata

    def shell_metadata(self, **changes):
        metadata = dict(resolution=16, core_resolution=16, grid_min=[-1.] * 3,
                        grid_max=[1.] * 3, sh_count=1, opacity_sh_count=1,
                        scene_mapping="world_shell", background_shell_levels=2,
                        background_refinement="fixed", background_subdivision_level=0,
                        world_shell_version=5, background_geometry="ellipsoid")
        metadata.update(changes)
        return metadata

    def archive(self):
        with np.load(self.path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    def test_bounded_cells_use_grid_bounds_not_optimized_ellipsoid_centers(self):
        metadata = self.save([0, 21, 63])
        actual_metadata, cells = read_layout(self.path)
        self.assertEqual(actual_metadata, metadata)
        chunks = list(iter_boxes(cells, actual_metadata, chunk_rows=2))
        self.assertEqual([len(chunk) for chunk in chunks], [2, 1])
        self.assertTrue(all(chunk.dtype == np.float32 for chunk in chunks))
        np.testing.assert_array_equal(np.concatenate(chunks), [
            [-1.5, -3., -4.5, 1., 2., 3., 0., 2.],
            [-.5, -1., -1.5, 1., 2., 3., 0., 2.],
            [1.5, 3., 4.5, 1., 2., 3., 0., 2.],
        ])

    def test_fixed_and_uniform_shell_addresses_preserve_world_boxes(self):
        core_count = 16 ** 3
        cases = [
            (0, [core_count, core_count + 55, core_count + 56], [
                [-1.5, -1.5, -1.5, 1., 1., 1., 1., 0.],
                [1.5, 1.5, 1.5, 1., 1., 1., 1., 0.],
                [-3., -3., -3., 2., 2., 2., 2., 0.],
            ]),
            (1, [core_count, core_count + 7, core_count + 56 * 8], [
                [-1.75, -1.75, -1.75, .5, .5, .5, 1., 1.],
                [-1.25, -1.25, -1.25, .5, .5, .5, 1., 1.],
                [-3.5, -3.5, -3.5, 1., 1., 1., 2., 1.],
            ]),
        ]
        for depth, ids, expected in cases:
            with self.subTest(depth=depth):
                self.save(ids, self.shell_metadata(background_subdivision_level=depth,
                          background_refinement="uniform" if depth else "fixed"))
                metadata, cells = read_layout(self.path)
                np.testing.assert_array_equal(np.concatenate(list(iter_boxes(cells, metadata))), expected)

    def test_mixed_adaptive_depths_survive_uint64_tag_and_chunk_boundaries(self):
        background = adaptive_encode_numpy(np.array([0, 1, 56]), np.array([0, 1, 3]),
                                            np.array([0, 7, 5]), np)
        self.assertTrue(np.all(background >= np.uint64(1 << 63)))
        self.save(np.concatenate((np.array([17], np.uint64), background)),
                  self.shell_metadata(background_refinement="adaptive", background_max_depth=3))
        metadata, cells = read_layout(self.path)
        chunks = list(iter_boxes(cells, metadata, chunk_rows=3))
        self.assertEqual([len(chunk) for chunk in chunks], [3, 1])
        np.testing.assert_array_equal(np.concatenate(chunks), [
            [-.8125, -.8125, -.9375, .125, .125, .125, 0., 4.],
            [-1.5, -1.5, -1.5, 1., 1., 1., 1., 0.],
            [-.25, -1.25, -1.25, .5, .5, .5, 1., 1.],
            [-3.625, -3.875, -3.625, .25, .25, .25, 2., 3.],
        ])

    def test_layout_read_does_not_decompress_radiance_geometry_or_adam(self):
        self.save([0, 21, 63])
        with np.load(self.path, allow_pickle=False) as archive:
            archive_type = type(archive)
            header = json.loads(archive["metadata_json"].tobytes())
        cells_key = header["payload"]["items"]["model_state"]["items"]["cell_indices"]["key"]
        accessed = []
        original = archive_type.__getitem__

        def read_member(archive, key):
            accessed.append(key)
            if key not in ("metadata_json", cells_key):
                raise AssertionError(f"Unnecessary array decompression: {key}")
            return original(archive, key)

        with patch.object(archive_type, "__getitem__", read_member):
            _, cells = read_layout(self.path)
        self.assertEqual(set(accessed), {"metadata_json", cells_key})
        np.testing.assert_array_equal(cells, [0, 21, 63])

    def test_corrupted_cell_checksum_is_rejected_before_visualization(self):
        self.save([0, 21, 63])
        arrays = self.archive()
        header = json.loads(arrays["metadata_json"].tobytes())
        key = header["payload"]["items"]["model_state"]["items"]["cell_indices"]["key"]
        arrays[key][0] = 1
        np.savez_compressed(self.path, **arrays)
        with self.assertRaisesRegex(ValueError, "checksum"):
            read_layout(self.path)

    def test_legacy_compressed_mapping_is_not_misrepresented_as_world_boxes(self):
        self.save([0])
        arrays = self.archive()
        header = json.loads(arrays["metadata_json"].tobytes())
        header["payload"]["items"]["model_metadata"]["items"]["scene_mapping"] = "projective"
        arrays["metadata_json"] = np.frombuffer(json.dumps(header).encode(), np.uint8)
        np.savez_compressed(self.path, **arrays)
        with self.assertRaises(ValueError):
            read_layout(self.path)

    def test_empty_scene_exports_and_does_not_overwrite_existing_html(self):
        self.save([])
        metadata, cells = read_layout(self.path)
        self.assertEqual(sum(len(chunk) for chunk in iter_boxes(cells, metadata)), 0)
        output = self.directory / "empty.html"
        summary = export_viewer(self.path, output)
        self.assertIsInstance(summary, dict)
        original = output.read_bytes()
        self.assertIn(b"<html", original.lower())
        with self.assertRaises((FileExistsError, ValueError)):
            export_viewer(self.path, output)
        self.assertEqual(output.read_bytes(), original)

    def test_embedded_boxes_roundtrip_across_non_base64_aligned_chunks(self):
        self.save([0, 1, 21, 42, 63])
        metadata, cells = read_layout(self.path)
        expected = np.concatenate(list(iter_boxes(cells, metadata)))
        original_iterator = iter_boxes

        def small_chunks(cells, metadata):
            # Two 32-byte rows leave a byte pending across base64 writes.
            return original_iterator(cells, metadata, chunk_rows=2)

        output = self.directory / "layout.html"
        with patch("visualize_voxels.iter_boxes", side_effect=small_chunks):
            summary = export_viewer(self.path, output)
        match = re.search(r'"boxes_base64":"([A-Za-z0-9+/=]*)"', output.read_text(encoding="utf-8"))
        self.assertIsNotNone(match)
        raw = base64.b64decode(match.group(1), validate=True)
        np.testing.assert_array_equal(np.frombuffer(raw, dtype="<f4").reshape(-1, 8), expected)
        self.assertEqual(summary["total_voxels"], 5)
        self.assertEqual(summary["exported_voxels"], 5)
        self.assertFalse(summary["sampled"])
        self.assertEqual(summary["groups"], [dict(shell=0, depth=2., count=5)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
