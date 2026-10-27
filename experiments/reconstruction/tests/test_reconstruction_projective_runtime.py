"""Portable projective metadata and camera contracts, without a CUDA runtime."""

from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reconstruction_checkpoint import save_checkpoint, load_checkpoint, validate_model_state
from reconstruction_common import generate_ray_batch, restore_model, model_metadata
from preview import RenderConfig
from test_reconstruction_initialization import numpy_runtime


def payload():
    # One core voxel and one +X slab voxel in a 4 + 2*1 padded grid.
    metadata = dict(resolution=6, core_resolution=4, grid_min=[-1.5]*3, grid_max=[1.5]*3,
                    sh_count=1, opacity_sh_count=1, scene_mapping="projective_shell",
                    background_refinement="preserve_physical_width", background_layers=1)
    state = dict(cell_indices=np.array([2+6*2+36*2, 5+6*2+36*2], np.uint32),
                 centers=np.full(6, .5, np.float32), log_scales=np.full(6, -1., np.float32),
                 rotations=np.array([1, 0, 0, 0]*2, np.float32),
                 radiance_sh=np.ones(6, np.float32), opacity_sh=np.zeros(2, np.float32))
    return state, metadata


class ProjectiveRuntimeTests(unittest.TestCase):
    def test_checkpoint_roundtrip_activates_native_mapping(self):
        state, metadata = payload()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            save_checkpoint(path, state, metadata)
            saved = load_checkpoint(path)
        calls = []

        def factory(resolution, lo, hi, values, sh, opacity_sh, **kwargs):
            return SimpleNamespace(resolution=resolution, grid_min=lo, grid_max=hi,
                sh_count=sh, opacity_sh_count=opacity_sh, core_resolution=kwargs["core_resolution"],
                set_scene_mapping=calls.append)

        runtime = SimpleNamespace(UInt32=np.asarray, Float=np.asarray,
                                  VoxelReconstruction=SimpleNamespace(from_state=factory))
        restored = restore_model(saved, runtime)
        self.assertEqual(calls, ["projective_shell"])
        self.assertEqual(model_metadata(restored), metadata)
        legacy = {**saved, "model_metadata": {**metadata, "scene_mapping": "radial_seed_shell",
                                             "background_refinement": "fixed_layers"}}
        calls.clear()
        restore_model(legacy, runtime)
        self.assertEqual(calls, [])  # Old data must not silently receive new ray semantics.

    def test_rejects_atlas_corner_and_inconsistent_layers(self):
        state, metadata = payload()
        validate_model_state(state, metadata)
        for change in ({"background_layers": 2}, {"background_refinement": "fixed_layers"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_model_state(state, {**metadata, **change})
        state["cell_indices"][1] = 5 + 6*5 + 36*2
        with self.assertRaisesRegex(ValueError, "atlas corners"):
            validate_model_state(state, metadata)

    def test_infinite_projective_ray_keeps_world_origin_direction(self):
        mi, dr = numpy_runtime()
        render = RenderConfig(width=2, height=2, near_clip=.1, far_clip=1000.)
        camera = dict(position=[.8, 0, 0], forward=[.6, 0, .8], right=[.8, 0, -.6],
                      up=[0, 1, 0], camera_angle_x=.7)
        indices = np.arange(4, dtype=np.uint32)
        offsets = np.full((4, 2), .5, np.float32)
        legacy = generate_ray_batch(render, camera, indices, offsets, mi, dr)
        projective = generate_ray_batch(render, {**camera, "projective_unbounded": True},
                                        indices, offsets, mi, dr)
        np.testing.assert_array_equal(legacy.o, projective.o)
        np.testing.assert_array_equal(legacy.d, projective.d)
        self.assertTrue(np.isfinite(legacy.maxt).all())
        self.assertTrue(np.isposinf(projective.maxt))
        self.assertEqual(render.far_clip, 1000.)  # JSON configuration stays finite.


if __name__ == "__main__":
    unittest.main(verbosity=2)
