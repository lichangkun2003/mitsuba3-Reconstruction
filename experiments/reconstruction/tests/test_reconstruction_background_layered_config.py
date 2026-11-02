"""CPU coverage for opt-in layer sampling, metadata migration and legacy saves."""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import preview
import train
from reconstruction_background import require_layered_background
from reconstruction_checkpoint import save_checkpoint, load_checkpoint, validate_model_state
from reconstruction_common import model_metadata, restore_model, initialize_model
from reconstruction_projective import refine_projective_numpy
from reconstruction_topology import CoarseToFineConfig, model_checkpoint
from reconstruction_topology_gpu import device_checkpoint
from test_reconstruction_projective_runtime import payload


class BackgroundLayeredConfigTests(unittest.TestCase):
    def test_strict_mode_validation_and_legacy_default(self):
        self.assertIs(preview.ModelConfig().background_layered_sampling, False)
        for value in (None, 0, 1, "true", [], {}):
            config = preview.PreviewConfig()
            config.model.background_layered_sampling = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "must be a boolean"):
                preview.validate_config(config)
        config = preview.PreviewConfig()
        config.model.background_layered_sampling = True
        with self.assertRaisesRegex(ValueError, "background_use_ellipsoid=false"):
            preview.validate_config(config)
        config.model.background_use_ellipsoid = False
        config.data.scene_mode = "unbounded"
        preview.validate_config(config)
        for mapping in ("bounded", "radial_seed_shell"):
            config.data.initialization = {"scene_mapping": mapping}
            with self.subTest(mapping=mapping), self.assertRaisesRegex(ValueError, "projective mapping"):
                preview.validate_config(config)

    def test_legacy_resume_normalizes_false_but_true_changes_algorithm(self):
        values = asdict(preview.ModelConfig())
        values.pop("background_layered_sampling")
        old = train.normalized_algorithm_group("model", values)
        self.assertIs(old["background_layered_sampling"], False)
        self.assertEqual(old, train.normalized_algorithm_group("model", {**values,
            "background_layered_sampling": False}))
        self.assertNotEqual(old, train.normalized_algorithm_group("model", {**values,
            "background_layered_sampling": True}))

    def test_cli_opt_in_and_opt_out_preserve_legacy(self):
        for options, expected in (([], False), (["--background-layered-sampling"], True),
                                  (["--no-background-layered-sampling"], False)):
            with self.subTest(options=options), patch.object(sys, "argv", ["train.py", *options]):
                config = copy.deepcopy(train.CONFIG)
                config.model.background_use_ellipsoid = False
                train.apply_arguments(config, train.parse_arguments(), {})
                self.assertIs(config.model.background_layered_sampling, expected)

    def test_reporting_distinguishes_layer_queries_from_cell_traversal(self):
        data = preview.DataConfig(initialization={"version": 6, "scene_mode": "unbounded",
            "scene_mapping": "projective_shell", "background_far_distance": 8.,
            "initial_background_layers": 2, "shell_layers": 8})
        for layered, expected in ((False, "cell_traversal"), (True, "layer_midpoint")):
            protocol = preview.scene_background_protocol(data, preview.RenderConfig(),
                preview.ModelConfig(background_use_ellipsoid=False, background_layered_sampling=layered))
            self.assertEqual(protocol["background_sampling"], expected)
            self.assertEqual(protocol["background_geometry"], "occupied_cell")
            self.assertEqual(protocol["ray_extent"], "finite_scene_boundary")

    def test_stale_build_and_invalid_mapping_fail_before_initialization(self):
        state, metadata = payload()
        metadata.update(background_use_ellipsoid=False, background_layered_sampling=True)
        native = SimpleNamespace(from_state=lambda *a, **k: self.fail("allocated before capability guard"))
        runtime = SimpleNamespace(VoxelReconstruction=native)
        with self.assertRaisesRegex(RuntimeError, "Rebuild.*layer-midpoint"):
            restore_model({"model_metadata": metadata, "model_state": state}, runtime)
        config = preview.PreviewConfig()
        config.model.background_use_ellipsoid = False
        config.model.background_layered_sampling = True
        with self.assertRaisesRegex(ValueError, "projective mapping"):
            initialize_model(config, Path("unused"), runtime, np)
        require_layered_background(SimpleNamespace(), "bounded", True, False)

    def test_checkpoint_roundtrip_export_and_refine_keep_sampling(self):
        state, metadata = payload()
        metadata.update(background_use_ellipsoid=False, background_layered_sampling=True)
        config = {"model": {"background_use_ellipsoid": False, "background_layered_sampling": True}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            save_checkpoint(path, state, metadata, config=config)
            saved = load_checkpoint(path)
            self.assertEqual(saved["model_metadata"], metadata)
            for value in (False, None, 1):
                bad = {"model": {**config["model"], "background_layered_sampling": value}}
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "background_layered_sampling"):
                    save_checkpoint(path, state, metadata, config=bad)

        def factory(resolution, lo, hi, values, sh, opacity_sh, **kwargs):
            model = SimpleNamespace(resolution=resolution, grid_min=lo, grid_max=hi,
                sh_count=sh, opacity_sh_count=opacity_sh, core_resolution=kwargs["core_resolution"],
                set_scene_mapping=lambda value: None, use_ellipsoid=True, layered=False,
                snapshot=lambda: values)
            model.set_background_use_ellipsoid = lambda value: setattr(model, "use_ellipsoid", value)
            model.native_background_use_ellipsoid = lambda: model.use_ellipsoid
            model.set_background_layered_sampling = lambda value: setattr(model, "layered", value)
            model.native_background_layered_sampling = lambda: model.layered
            return model

        native = SimpleNamespace(from_state=factory, background_layered_sampling_version=1,
            set_background_layered_sampling=lambda value: None, native_background_layered_sampling=lambda: False)
        runtime = SimpleNamespace(UInt32=np.asarray, Float=np.asarray, VoxelReconstruction=native)
        restored = restore_model(saved, runtime)
        self.assertTrue(restored.layered)
        self.assertEqual(model_metadata(restored), metadata)
        dr = SimpleNamespace(eval=lambda *a: None, sync_thread=lambda: None,
                             cuda=SimpleNamespace(UInt32=np.asarray, Float32=np.asarray))
        with patch.dict(sys.modules, {"drjit": dr}):
            self.assertEqual(model_checkpoint(restored)["model_metadata"], metadata)
            self.assertEqual(device_checkpoint(restored)["model_metadata"], metadata)

        def fit_core(core, *args):
            self.assertNotIn("background_layered_sampling", core["model_metadata"])
            self.assertNotIn("background_use_ellipsoid", core["model_metadata"])
            return {"model_state": {key: value[:0] for key, value in state.items()}}

        refined = refine_projective_numpy(saved, CoarseToFineConfig(), 1000, fit_core, np)
        self.assertTrue(refined["model_metadata"]["background_layered_sampling"])
        self.assertEqual(refined["model_metadata"]["background_layers"], 2)
        legacy = {**saved, "model_metadata": {k: v for k, v in metadata.items()
                  if k != "background_layered_sampling"}}
        self.assertFalse(restore_model(legacy, runtime).layered)

    def test_metadata_rejects_layered_ellipsoid_and_non_boolean(self):
        state, metadata = payload()
        for values in ({"background_layered_sampling": True},
                       {"background_layered_sampling": 1, "background_use_ellipsoid": False},
                       {"background_layered_sampling": "false", "background_use_ellipsoid": False}):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "background_layered_sampling"):
                validate_model_state(state, {**metadata, **values})


if __name__ == "__main__":
    unittest.main()
