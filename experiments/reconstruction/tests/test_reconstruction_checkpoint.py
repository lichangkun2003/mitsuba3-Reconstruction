"""CPU tests for portable checkpoints and evaluation metrics; no GPU imports."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import zipfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import reconstruction_checkpoint as checkpoint
import evaluate as evaluate
from reconstruction_common import ContourSampler, generate_ray_batch
from preview import RenderConfig


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "state.npz"
        self.metadata = dict(resolution=2, grid_min=[-1., -1., -1.], grid_max=[1., 1., 1.],
                             sh_count=1, opacity_sh_count=1)
        self.model = {
            "cell_indices": np.array([0, 4], np.uint32),
            "centers": np.zeros(6, np.float32), "log_scales": np.zeros(6, np.float32),
            "rotations": np.array([1, 0, 0, 0] * 2, np.float32),
            "radiance_sh": np.ones(6, np.float32), "opacity_sh": np.ones(2, np.float32),
        }

    def save(self, **kwargs):
        return checkpoint.save_checkpoint(self.path, self.model, self.metadata, **kwargs)

    def archive(self):
        with np.load(self.path, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}

    def test_model_optimizer_rng_round_trip(self):
        rng = np.random.default_rng(42)
        rng.random(7)
        self.save(training_state={"completed_epochs": 3, "rng": rng.bit_generator.state,
                                  "empty_contour": np.zeros((0, 16), np.float32),
                                  "tuple": (2, np.array([3], np.uint64))},
                  optimizer_state={"m": np.zeros(6, np.float32), "counter": np.array(3, np.uint32)},
                  config={"type": "array", "key": "ordinary user data"})
        loaded = checkpoint.load_checkpoint(self.path, self.metadata)
        for key, value in self.model.items():
            np.testing.assert_array_equal(loaded["model_state"][key], value)
            self.assertEqual(loaded["model_state"][key].dtype, value.dtype)
        self.assertEqual(loaded["optimizer_state"]["counter"].shape, ())
        self.assertEqual(loaded["training_state"]["empty_contour"].shape, (0, 16))
        self.assertIsInstance(loaded["training_state"]["tuple"], tuple)
        self.assertEqual(loaded["config"], {"type": "array", "key": "ordinary user data"})
        restored = np.random.default_rng()
        restored.bit_generator.state = loaded["training_state"]["rng"]
        np.testing.assert_array_equal(restored.random(10), rng.random(10))

    def test_atomic_failure_preserves_old_file_and_cleans_temporary(self):
        self.save(training_state={"completed_epochs": 1})
        old = self.path.read_bytes()
        with patch.object(checkpoint.os, "replace", side_effect=OSError("simulated replacement failure")):
            with self.assertRaises(OSError):
                self.save(training_state={"completed_epochs": 2})
        self.assertEqual(self.path.read_bytes(), old)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_empty_model_round_trip_after_full_topology_deletion(self):
        empty = {name: np.empty(0, dtype=array.dtype) for name, array in self.model.items()}
        checkpoint.save_checkpoint(self.path, empty, self.metadata,
                                   training_state={"completed_epochs": 2},
                                   optimizer_state={"count": 0, "moments": np.empty((0, 3), np.float32)})
        restored = checkpoint.load_checkpoint(self.path, self.metadata)
        for name, array in empty.items():
            np.testing.assert_array_equal(restored["model_state"][name], array)
            self.assertEqual(restored["model_state"][name].dtype, array.dtype)
        self.assertEqual(restored["optimizer_state"]["moments"].shape, (0, 3))
        invalid = {**empty, "centers": np.zeros(3, np.float32)}
        with self.assertRaisesRegex(ValueError, "centers"):
            checkpoint.save_checkpoint(self.path, invalid, self.metadata)

    def test_fixed_eight_layer_1024_model_round_trip_and_metadata_guards(self):
        extent = 1040/1024
        metadata = dict(self.metadata, resolution=1040, core_resolution=1024,
                        grid_min=[-extent]*3, grid_max=[extent]*3,
                        scene_mapping="radial_seed_shell", background_refinement="fixed_layers",
                        background_layers=8)
        checkpoint.save_checkpoint(self.path, self.model, metadata)
        self.assertEqual(checkpoint.load_checkpoint(self.path)["model_metadata"], metadata)
        for changes in ({"background_layers": 4}, {"background_layers": True},
                        {"background_refinement": "unknown"}, {"core_resolution": 1025},
                        {"resolution": 1041}, {"scene_mapping": "bounded"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                checkpoint.validate_model_state(self.model, {**metadata, **changes})
        legacy = dict(metadata, resolution=544, core_resolution=512,
                      grid_min=[-544/512]*3, grid_max=[544/512]*3)
        legacy.pop("background_refinement")
        legacy.pop("background_layers")
        checkpoint.save_checkpoint(self.path, self.model, legacy)
        self.assertEqual(checkpoint.load_checkpoint(self.path)["model_metadata"], legacy)

    def test_fixed_layer_metadata_survives_model_restore_and_resave(self):
        from reconstruction_common import restore_model, model_metadata
        metadata = dict(self.metadata, resolution=1040, core_resolution=1024,
                        grid_min=[-1040/1024]*3, grid_max=[1040/1024]*3,
                        scene_mapping="radial_seed_shell", background_refinement="fixed_layers",
                        background_layers=8)
        def from_state(resolution, lo, hi, state, sh, opacity_sh, **kwargs):
            return SimpleNamespace(resolution=resolution, grid_min=lo, grid_max=hi,
                                   sh_count=sh, opacity_sh_count=opacity_sh, **kwargs)
        mi = SimpleNamespace(UInt32=lambda v: v, Float=lambda v: v,
                             VoxelReconstruction=SimpleNamespace(from_state=from_state))
        restored = restore_model({"model_metadata": metadata, "model_state": self.model}, mi)
        self.assertEqual(model_metadata(restored), metadata)

    def test_failed_first_save_leaves_no_checkpoint(self):
        with patch.object(checkpoint.os, "replace", side_effect=OSError("simulated replacement failure")):
            with self.assertRaises(OSError):
                self.save()
        self.assertEqual(list(self.path.parent.iterdir()), [])

    def test_checksum_detects_modified_parameter(self):
        self.save()
        archive = self.archive()
        archive["a000001"][0] = 4.
        np.savez_compressed(self.path, **archive)
        with self.assertRaisesRegex(ValueError, "checksum"):
            checkpoint.load_checkpoint(self.path)

    def test_pickle_array_is_rejected(self):
        self.save()
        archive = self.archive()
        archive["a000001"] = np.array([{"unsafe": "object"}], dtype=object)
        np.savez_compressed(self.path, **archive)
        with self.assertRaisesRegex(ValueError, "allow_pickle=False"):
            checkpoint.load_checkpoint(self.path)
        with self.assertRaises(ValueError):
            self.save(optimizer_state={"object": np.array([object()], dtype=object)})

    def test_manifest_shape_mismatch_is_rejected(self):
        self.save()
        archive = self.archive()
        archive["a000001"] = archive["a000001"].reshape(2, 3)
        np.savez_compressed(self.path, **archive)
        with self.assertRaisesRegex(ValueError, "shape/dtype"):
            checkpoint.load_checkpoint(self.path)

    def test_metadata_and_format_checks(self):
        self.save()
        with self.assertRaisesRegex(ValueError, "resolution"):
            checkpoint.load_checkpoint(self.path, {"resolution": 3})
        archive = self.archive()
        header = json.loads(archive["metadata_json"].tobytes())
        header["version"] = 999
        archive["metadata_json"] = np.frombuffer(json.dumps(header).encode(), dtype=np.uint8)
        np.savez_compressed(self.path, **archive)
        with self.assertRaisesRegex(ValueError, "format/version"):
            checkpoint.load_checkpoint(self.path)
        np.savez_compressed(self.path, **self.model)
        with self.assertRaisesRegex(ValueError, "Not a reconstruction checkpoint"):
            checkpoint.load_checkpoint(self.path)

    def test_invalid_model_layout_and_values_are_rejected(self):
        changes = {
            "center shape": ("centers", np.zeros((2, 3), np.float32)),
            "zero quaternion": ("rotations", np.zeros(8, np.float32)),
            "nonfinite": ("opacity_sh", np.array([1, np.nan], np.float32)),
            "duplicate cells": ("cell_indices", np.array([4, 4], np.uint32)),
            "out of range": ("cell_indices", np.array([0, 8], np.uint32)),
        }
        for name, (key, value) in changes.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                state = {**self.model, key: value}
                checkpoint.save_checkpoint(self.path, state, self.metadata)
        for metadata in ({**self.metadata, "resolution": 0},
                         {**self.metadata, "grid_min": [1., 1., 1.]},
                         {**self.metadata, "sh_count": 17}):
            with self.subTest(metadata=metadata), self.assertRaises(ValueError):
                checkpoint.save_checkpoint(self.path, self.model, metadata)

    def source(self, array, calls=None, fail_at=None):
        from reconstruction_export import ArraySource
        array = np.asarray(array)

        def read(start, count):
            if calls is not None:
                calls.append((start, count))
            if fail_at is not None and start >= fail_at:
                raise OSError("simulated chunk read failure")
            return array.flat[start:start + count]

        return ArraySource(array.shape, array.dtype, read)

    def test_streamed_model_and_nested_state_round_trip_bit_exact(self):
        import reconstruction_export as export
        calls = {name: [] for name in self.model}
        state = {name: self.source(value, calls[name]) for name, value in self.model.items()}
        special = np.array([-0., np.finfo(np.float32).tiny, -1.25, 3.5], np.float32)
        training = {"nested": (self.source(special), [self.source(np.zeros((0, 16), np.float64))]),
                    "clock": self.source(np.array(19, np.uint64))}
        with patch.object(export, "CHUNK_BYTES", 16):
            checkpoint.save_checkpoint(self.path, state, self.metadata, training_state=training)
        loaded = checkpoint.load_checkpoint(self.path)
        for name, value in self.model.items():
            self.assertEqual(loaded["model_state"][name].tobytes(), value.tobytes())
            self.assertEqual(sum(count for _, count in calls[name]), value.size)
            self.assertLessEqual(max(count for _, count in calls[name]), 4)
        self.assertEqual(loaded["training_state"]["nested"][0].tobytes(), special.tobytes())
        self.assertEqual(loaded["training_state"]["nested"][1][0].shape, (0, 16))
        self.assertEqual(loaded["training_state"]["clock"].shape, ())

    def test_streaming_uses_compatible_zip64_npy_manifest_and_metadata_last(self):
        state = {name: self.source(value) for name, value in self.model.items()}
        checkpoint.save_checkpoint(self.path, state, self.metadata)
        with zipfile.ZipFile(self.path) as archive:
            self.assertEqual(archive.namelist()[-1], "metadata_json.npy")
            self.assertTrue(all(info.compress_type == zipfile.ZIP_DEFLATED for info in archive.infolist()))
            self.assertTrue(all(info.extract_version >= 45 for info in archive.infolist()))
        arrays = self.archive()
        header = json.loads(arrays.pop("metadata_json").tobytes())
        self.assertEqual(header["version"], checkpoint.FORMAT_VERSION)
        self.assertEqual(header["format"], checkpoint.FORMAT)
        self.assertEqual(set(arrays), set(header["arrays"]))
        for key, value in arrays.items():
            self.assertEqual(header["arrays"][key], {
                "shape": list(value.shape), "dtype": value.dtype.str,
                "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()})
        # An archive created by the historical NumPy writer still loads.
        np.savez_compressed(self.path, **arrays,
                            metadata_json=np.frombuffer(json.dumps(header).encode(), np.uint8))
        checkpoint.load_checkpoint(self.path, self.metadata)

    def test_noncontiguous_numpy_arrays_are_written_in_c_order(self):
        import reconstruction_export as export
        model = {name: np.column_stack([value, value]).ravel()[::2]
                 for name, value in self.model.items()}
        moments = np.arange(90, dtype=np.float64).reshape(9, 10).T[::-1, ::2]
        with patch.object(export, "CHUNK_BYTES", 32):
            checkpoint.save_checkpoint(self.path, model, self.metadata,
                                       optimizer_state={"strided": moments})
        restored = checkpoint.load_checkpoint(self.path)
        for name, value in model.items():
            np.testing.assert_array_equal(restored["model_state"][name], value)
        np.testing.assert_array_equal(restored["optimizer_state"]["strided"], moments)

    def test_stream_read_failure_is_atomic_and_cleans_temporary_files(self):
        import reconstruction_export as export
        self.save(training_state={"epoch": 1})
        old = self.path.read_bytes()
        source = self.source(np.arange(12, dtype=np.float32), fail_at=4)
        with patch.object(export, "CHUNK_BYTES", 16), self.assertRaisesRegex(OSError, "chunk read"):
            self.save(optimizer_state={"moments": source})
        self.assertEqual(self.path.read_bytes(), old)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_stream_validation_rejects_nonfinite_and_zero_rotation(self):
        import reconstruction_export as export
        for name, array, message in (
                ("centers", np.array([0., 0., 0., 0., np.nan, 0.], np.float32), "nonfinite"),
                ("rotations", np.array([1., 0., 0., 0., 0., 0., 0., 0.], np.float32), "zero quaternion")):
            with self.subTest(name=name), patch.object(export, "CHUNK_BYTES", 16):
                with self.assertRaisesRegex(ValueError, message):
                    checkpoint.save_checkpoint(self.path, {**self.model, name: self.source(array)}, self.metadata)
                self.assertEqual(list(self.path.parent.iterdir()), [])
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.save(optimizer_state={"moments": self.source(np.array([np.inf], np.float32))})

    def test_stream_cells_detect_duplicates_across_chunks_and_allow_unordered(self):
        import reconstruction_export as export
        count = 8
        model = {"cell_indices": np.array([7, 1, 3, 0, 6, 2, 5, 4], np.uint32),
                 "centers": np.zeros(count * 3, np.float32),
                 "log_scales": np.zeros(count * 3, np.float32),
                 "rotations": np.tile(np.array([1, 0, 0, 0], np.float32), count),
                 "radiance_sh": np.zeros(count * 3, np.float32),
                 "opacity_sh": np.zeros(count, np.float32)}
        with patch.object(export, "CHUNK_BYTES", 16):
            checkpoint.save_checkpoint(self.path, {key: self.source(value) for key, value in model.items()}, self.metadata)
            np.testing.assert_array_equal(checkpoint.load_checkpoint(self.path)["model_state"]["cell_indices"], model["cell_indices"])
            old = self.path.read_bytes()
            for cells in ([7, 1, 3, 0, 6, 2, 5, 7], [7, 1, 3, 0, 6, 2, 5, 8],
                          [7, 1, 3, 0, 6, 2, 5, -1]):
                with self.subTest(cells=cells), self.assertRaises(ValueError):
                    checkpoint.save_checkpoint(self.path,
                        {**model, "cell_indices": self.source(np.array(cells, np.int64))}, self.metadata)
                self.assertEqual(self.path.read_bytes(), old)
                self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_streamed_empty_model_and_source_chunk_contract(self):
        from reconstruction_export import ArraySource
        empty = {key: self.source(np.empty(0, value.dtype)) for key, value in self.model.items()}
        checkpoint.save_checkpoint(self.path, empty, self.metadata)
        restored = checkpoint.load_checkpoint(self.path)
        self.assertTrue(all(value.size == 0 for value in restored["model_state"].values()))
        for reader in (lambda start, count: np.zeros(count + 1, np.float32),
                       lambda start, count: np.zeros(count, np.float64)):
            source = ArraySource((6,), np.dtype(np.float32), reader)
            with self.assertRaisesRegex(ValueError, "chunk"):
                checkpoint.save_checkpoint(self.path, {**self.model, "centers": source}, self.metadata)

    def test_stream_preserves_projective_cell_region_validation(self):
        metadata = dict(self.metadata, resolution=4, core_resolution=2,
                        grid_min=[-2.] * 3, grid_max=[2.] * 3,
                        scene_mapping="projective_shell", background_layers=1,
                        background_refinement="preserve_physical_width")
        valid = np.array([21, 20], np.uint32)  # core (1,1,1), face (0,1,1).
        checkpoint.save_checkpoint(self.path, {**self.model, "cell_indices": self.source(valid)}, metadata)
        for cells in (np.array([21, 0], np.uint32), np.array([21, 16], np.uint32)):
            with self.subTest(cells=cells), self.assertRaisesRegex(ValueError, "atlas corners"):
                checkpoint.save_checkpoint(self.path, {**self.model, "cell_indices": self.source(cells)}, metadata)

    def test_small_source_limit_keeps_quaternion_validation_across_chunks(self):
        from reconstruction_export import ArraySource
        calls = []

        def source(array):
            def read(start, count):
                calls.append(count)
                return array[start:start + count]
            return ArraySource(array.shape, array.dtype, read, max_chunk_elements=3)

        checkpoint.save_checkpoint(self.path,
            {**self.model, "rotations": source(self.model["rotations"])}, self.metadata)
        self.assertLessEqual(max(calls), 3)
        invalid = np.array([1, 0, 0, 0, 0, 0, 0, 0], np.float32)
        with self.assertRaisesRegex(ValueError, "zero quaternion"):
            checkpoint.save_checkpoint(self.path, {**self.model, "rotations": source(invalid)}, self.metadata)


class MetricTests(unittest.TestCase):
    def test_known_psnr_values(self):
        black = np.zeros((2, 3, 3), np.float32)
        white = np.ones_like(black)
        self.assertEqual(evaluate.image_metrics(black, white, np)["linear"]["psnr_db"], 0.)
        half = evaluate.image_metrics(black, white * .5, np)["linear"]
        self.assertEqual(half["mse"], .25)
        self.assertAlmostEqual(half["psnr_db"], 6.020599913279624)

    def test_mean_psnr_and_psnr_of_mean_mse_are_distinct(self):
        black = np.zeros((1, 1, 3), np.float32)
        records = [{"metrics": evaluate.image_metrics(black, np.full_like(black, value), np)}
                   for value in (1., .5)]
        aggregate = evaluate.aggregate_metrics(records)["linear"]
        self.assertAlmostEqual(aggregate["mean_psnr_db"], 3.010299956639812)
        self.assertAlmostEqual(aggregate["psnr_from_mean_mse_db"], -10 * math.log10(.625))

    def test_perfect_images_have_explicit_infinity_flag(self):
        image = np.ones((1, 1, 3), np.float32)
        metrics = evaluate.image_metrics(image, image, np)
        self.assertIsNone(metrics["linear"]["psnr_db"])
        self.assertTrue(metrics["linear"]["perfect_match"])
        aggregate = evaluate.aggregate_metrics([{"metrics": metrics}])
        self.assertIsNone(aggregate["linear"]["mean_psnr_db"])
        self.assertTrue(aggregate["linear"]["mean_psnr_is_infinite"])
        json.dumps(aggregate, allow_nan=False)

    def test_srgb_transfer_and_clipping_protocol(self):
        result = evaluate.linear_to_display_srgb(np.array([0., .0031308, 1., -1., 2.]), np)
        np.testing.assert_allclose(result, [0., .040449936, 1., 0., 1.])
        prediction = np.full((1, 1, 3), 2., np.float32)
        metrics = evaluate.image_metrics(prediction, np.ones_like(prediction), np)
        self.assertEqual(metrics["linear"]["mse"], 1.)
        self.assertTrue(metrics["srgb"]["perfect_match"])

    def test_metric_rejects_mismatched_or_nonfinite_inputs(self):
        image = np.zeros((2, 3, 3), np.float32)
        for reference in (np.zeros((3, 2, 3)), np.full_like(image, np.nan)):
            with self.subTest(shape=reference.shape), self.assertRaises(ValueError):
                evaluate.image_metrics(image, reference, np)
        with self.assertRaises(ValueError):
            evaluate.aggregate_metrics([])


class ContourSamplerTests(unittest.TestCase):
    def test_proposal_normalization_and_pdf_corrected_integrals(self):
        # Query each of the 16 bins exactly once with an inverse-CDF midpoint;
        # this checks the discrete expectation without Monte Carlo tolerances.
        mixture = .2
        weights = np.arange(1, 17, dtype=np.float32) ** 2
        probability = mixture / 16 + (1 - mixture) * weights.astype(np.float64) / float(weights.sum())
        cdf = np.cumsum(probability)
        midpoints = cdf - .5 * probability

        class StratifiedRng:
            def random(self, shape):
                if shape == (16, 2):
                    return np.full(shape, .5)
                if shape == (16, 3):
                    return np.column_stack([midpoints, np.full(16, .25), np.full(16, .75)])
                raise AssertionError(f"Unexpected random request: {shape}")

        sampler = ContourSampler(np, enabled=True, uniform_fraction=mixture)
        sampler.views["4"] = {"pixels": np.arange(16), "mean_alpha": np.tile(weights, (16, 1))}
        offsets, inverse_pdf, token = sampler.sample(4, np.full(16, .5, np.float32), StratifiedRng())
        np.testing.assert_array_equal(token[1], np.arange(16))
        recovered_probability = 1 / (16 * inverse_pdf.astype(np.float64))
        np.testing.assert_allclose(recovered_probability, probability, rtol=2e-7)
        self.assertAlmostEqual(float(recovered_probability.sum()), 1., places=6)
        self.assertTrue(np.all(recovered_probability >= mixture / 16))
        constant_expectation = np.sum(probability * inverse_pdf)
        alpha_by_bin = np.linspace(0, 1, 16)
        alpha_expectation = np.sum(probability * inverse_pdf * alpha_by_bin)
        self.assertAlmostEqual(float(constant_expectation), 1., places=6)
        self.assertAlmostEqual(float(alpha_expectation), float(alpha_by_bin.mean()), places=6)
        np.testing.assert_allclose(offsets[:, 0], (np.arange(16) % 4 + .25) / 4)
        np.testing.assert_allclose(offsets[:, 1], (np.arange(16) // 4 + .75) / 4)

    def test_uniform_fallback_and_noncontour_pixels(self):
        sampler = ContourSampler(np, enabled=True)
        reference = np.array([0., .4, 1.], np.float32)
        first_rng = np.random.default_rng(42)
        expected_uniform_offsets = first_rng.random((3, 2)).astype(np.float32)
        sampler.views["0"] = {"pixels": np.array([1]), "mean_alpha": np.zeros((1, 16), np.float32)}
        offsets, inverse_pdf, token = sampler.sample(0, reference, np.random.default_rng(42))
        np.testing.assert_array_equal(offsets[[0, 2]], expected_uniform_offsets[[0, 2]])
        np.testing.assert_array_equal(inverse_pdf, np.ones(3, np.float32))
        self.assertEqual(len(token[1]), 1)
        self.assertTrue(np.all((offsets >= 0) & (offsets < 1)))

    def test_history_uses_raw_alpha_and_updates_only_selected_bin(self):
        sampler = ContourSampler(np, enabled=True, history_rate=.3)
        reference = np.array([0., .3, 1., .7], np.float32)
        sampler.sample(5, reference, np.random.default_rng(11))
        old = sampler.views["5"]["mean_alpha"].copy()
        token = ("5", np.array([1, 9]))
        sampler.observe(token, np.array([.9, .8, .2, .1], np.float32))
        expected = old.copy()
        expected[0, 1] += .3 * (.8 - old[0, 1])
        expected[1, 9] += .3 * (.1 - old[1, 9])
        np.testing.assert_allclose(sampler.views["5"]["mean_alpha"], expected)
        sampler.observe(None, np.zeros(4))
        np.testing.assert_allclose(sampler.views["5"]["mean_alpha"], expected)

    def test_checkpoint_rng_resume_preserves_sampling_and_rays(self):
        sampler = ContourSampler(np, enabled=True)
        rng = np.random.default_rng(123)
        reference = np.array([[0., .2], [.6, 1.]], np.float32)
        _, _, token = sampler.sample(2, reference, rng)
        sampler.observe(token, np.array([[0., .8], [.1, 1.]], np.float32))
        metadata = dict(resolution=1, grid_min=[-1., -1., -1.], grid_max=[1., 1., 1.],
                        sh_count=1, opacity_sh_count=1)
        model = dict(cell_indices=np.array([0], np.uint32), centers=np.zeros(3, np.float32),
                     log_scales=np.zeros(3, np.float32), rotations=np.array([1., 0, 0, 0], np.float32),
                     radiance_sh=np.zeros(3, np.float32), opacity_sh=np.zeros(1, np.float32))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contour.npz"
            checkpoint.save_checkpoint(path, model, metadata,
                training_state={"contour": sampler.state_dict(), "rng": rng.bit_generator.state})
            restored_state = checkpoint.load_checkpoint(path)["training_state"]
        resumed = ContourSampler(np, enabled=True)
        resumed.load_state_dict(restored_state["contour"])
        resumed_rng = np.random.default_rng()
        resumed_rng.bit_generator.state = restored_state["rng"]
        uninterrupted = sampler.sample(2, reference, rng)
        restored = resumed.sample(2, reference, resumed_rng)
        np.testing.assert_array_equal(uninterrupted[0], restored[0])
        np.testing.assert_array_equal(uninterrupted[1], restored[1])
        np.testing.assert_array_equal(uninterrupted[2][1], restored[2][1])

        # Execute the shared ray builder using NumPy stand-ins for Dr.Jit types.
        # This verifies the actual builder's origins/directions/ranges, without GPU.
        vector = lambda value: np.asarray(value, np.float32).reshape(3, 1)
        norm = lambda value: np.sqrt(np.sum(value * value, axis=0))
        mi = SimpleNamespace(UInt32=lambda value: np.asarray(value, np.uint32),
                             Float=lambda value: np.asarray(value, np.float32),
                             Vector3f=vector, Point3f=vector,
                             Ray3f=lambda origin, direction: SimpleNamespace(o=origin, d=direction))
        dr = SimpleNamespace(norm=norm, normalize=lambda value: value / norm(value),
                             opaque=lambda dtype, value: dtype(value))
        camera = dict(camera_angle_x=.7, position=[0., 0., 3.], forward=[0., 0., -1.],
                      right=[1., 0., 0.], up=[0., 1., 0.])
        render = RenderConfig(width=2, height=2)
        rays = [generate_ray_batch(render, camera, np.arange(4, dtype=np.uint32),
                                   offsets, mi, dr) for offsets in (uninterrupted[0], restored[0])]
        for field in ("o", "d", "maxt"):
            np.testing.assert_array_equal(getattr(rays[0], field), getattr(rays[1], field))


if __name__ == "__main__":
    unittest.main(verbosity=2)
