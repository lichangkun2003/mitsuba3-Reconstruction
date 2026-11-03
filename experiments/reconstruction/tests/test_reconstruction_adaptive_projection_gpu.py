"""CPU oracle and optional CUDA regression for bounded all-camera projection."""

from pathlib import Path
from types import SimpleNamespace
from contextlib import nullcontext
import argparse
import math
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruction_adaptive_background import prepare_projection_cameras, projected_boxes_pixels
from reconstruction_adaptive_projection_gpu import make_adaptive_projector, projection_batch_layout


class ArrayDevice:
    """Execute the projection's bounded array operations on the CPU."""

    cuda = SimpleNamespace(Float64=lambda x: np.asarray(x, np.float64),
                           Float32=lambda x: np.asarray(x, np.float32),
                           UInt32=lambda x: np.asarray(x, np.uint32))
    JitFlag = SimpleNamespace(FastMath="fast_math")
    ReduceOp = SimpleNamespace(Max="max")
    minimum, maximum, select = staticmethod(np.minimum), staticmethod(np.maximum), staticmethod(np.where)

    @staticmethod
    def scoped_set_flag(*args):
        return nullcontext()

    @staticmethod
    def opaque(dtype, value):
        return dtype(value)

    @staticmethod
    def arange(dtype, size):
        return dtype(np.arange(size))

    @staticmethod
    def zeros(dtype, size):
        return dtype(np.zeros(size))

    @staticmethod
    def gather(dtype, values, indices, active=True):
        active = np.broadcast_to(active, indices.shape)
        result = dtype(np.zeros(indices.shape))
        result[active] = values[indices[active]]
        return result

    @staticmethod
    def block_reduce(operation, values, size):
        assert operation == "max"
        return values.reshape(-1, size).max(axis=1)

    @staticmethod
    def eval(*values):
        pass


def cameras(count=169):
    result = []
    for index in range(count):
        angle = index*2*math.pi/max(count, 1)
        position = np.array([4*math.cos(angle), .3*math.sin(2*angle), 4*math.sin(angle)])
        forward = -position/np.linalg.norm(position)
        right = np.cross([0., 1., 0.], forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        result.append(dict(position=position, right=right, up=up, forward=forward,
                           fx=130.+index%7, fy=117.+index%5, cx=63., cy=49.,
                           width=128, height=96, near_clip=.05))
    return prepare_projection_cameras(result, np)


def random_boxes(count=83):
    rng = np.random.default_rng(735)
    center = rng.uniform(-8., 8., (count, 3))
    half = np.exp(rng.uniform(-8., 1., (count, 3)))
    return center-half, center+half


def axis_camera(position=(0., 0., 0.)):
    return prepare_projection_cameras([dict(position=position, right=[1., 0., 0.],
        up=[0., 1., 0.], forward=[0., 0., 1.], fx=2., fy=2., cx=2., cy=2.,
        width=4, height=4, near_clip=.1)], np)[0]


def check_equivalent(projector, low, high, views):
    expected = projected_boxes_pixels(low, high, views, np, chunk_size=19)
    actual = projector(low, high, views)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=2e-7, atol=2e-6)
    np.testing.assert_array_equal(actual == 0, expected == 0)
    assert projector.last_plan["estimated_workspace_bytes"] < 512*1024*1024
    return actual


class AdaptiveProjectionTests(unittest.TestCase):
    def test_plan_bounds_even_for_billions_of_leaf_camera_pairs(self):
        plan = projection_batch_layout(4_000_000, 169)
        self.assertEqual(plan["pair_capacity"], 65536)
        self.assertLess(plan["estimated_workspace_bytes"], 256*1024*1024)
        self.assertLessEqual(plan["leaf_batch_size"]*plan["camera_batch_size"], 65536)
        small = projection_batch_layout(4_000_000, 169, workspace_limit_bytes=8*1024*1024)
        self.assertLess(small["estimated_workspace_bytes"], 8*1024*1024)
        for options in (dict(max_pairs=0), dict(camera_batch_size=0), dict(max_pairs=1.5),
                        dict(workspace_limit_bytes=512*1024*1024), dict(workspace_limit_bytes=1024)):
            with self.assertRaises(ValueError):
                projection_batch_layout(5, 169, **options)

    def test_random_boxes_all_169_cameras_and_partial_batches(self):
        low, high = random_boxes()
        projector = make_adaptive_projector(ArrayDevice(), np, max_pairs=112, camera_batch_size=16)
        check_equivalent(projector, low, high, cameras())
        self.assertEqual(projector.last_batch_count, math.ceil(83/7)*math.ceil(169/16))

    def test_near_plane_screen_edges_behind_camera_and_span_cap(self):
        low = np.array([[-.1, -.1, .1], [-1., -1., -.2], [-.1, -.1, -3.],
                        [1., -.1, 1.], [-2., -.1, 1.], [-1e9, -1e9, -.1]])
        high = np.array([[.1, .1, .2], [1., 1., .2], [.1, .1, -1.],
                         [2., .1, 1.], [-1., .1, 1.], [1e9, 1e9, .2]])
        actual = check_equivalent(make_adaptive_projector(ArrayDevice(), np, max_pairs=2),
                                  low, high, [axis_camera()])
        self.assertEqual(actual[2], 0.)
        self.assertEqual(actual[3], 0.)  # min_x == image width is outside.
        self.assertGreater(actual[4], 0.)  # max_x == 0 remains visible.
        self.assertEqual(actual[-1], 16.)

    def test_last_camera_is_not_subsampled_and_cache_updates(self):
        good = axis_camera()
        bad = dict(good, forward=np.array([0., 0., -1.]))
        views = [bad]*168+[good]
        low, high = np.array([[-.1, -.1, 1.]]), np.array([[.1, .1, 2.]])
        projector = make_adaptive_projector(ArrayDevice(), np)
        actual = check_equivalent(projector, low, high, views)
        self.assertGreater(actual[0], 0.)
        self.assertEqual(projector.camera_uploads, 1)
        check_equivalent(projector, low, high, views)
        self.assertEqual(projector.camera_uploads, 1)
        views[-1] = dict(good, fx=3.)
        check_equivalent(projector, low, high, views)
        self.assertEqual(projector.camera_uploads, 2)

    def test_float64_projection_preserves_small_boxes_at_large_world_coordinates(self):
        position = np.full(3, 1e8, np.float64)
        low = position+np.array([[-.1, -.1, 1.]])
        high = position+np.array([[.1, .1, 2.]])
        actual = check_equivalent(make_adaptive_projector(ArrayDevice(), np), low, high,
                                  [axis_camera(position)])
        self.assertGreater(actual[0], .39)
        self.assertLess(actual[0], .41)

    def test_empty_inputs_do_not_upload_or_launch(self):
        projector = make_adaptive_projector(ArrayDevice(), np)
        np.testing.assert_array_equal(projector(np.empty((0, 3)), np.empty((0, 3)), cameras()), [])
        np.testing.assert_array_equal(projector(np.zeros((3, 3)), np.ones((3, 3)), []), np.zeros(3))
        self.assertEqual(projector.camera_uploads, 0)
        self.assertEqual(projector.last_batch_count, 0)
        with self.assertRaises(ValueError):
            projector(np.zeros(3), np.ones(3), cameras())


def check_cuda(dr):
    projector = make_adaptive_projector(dr, np, max_pairs=8192, camera_batch_size=16)
    low, high = random_boxes(1031)
    check_equivalent(projector, low, high, cameras())
    assert projector.last_batch_count == 3*11
    for case in ("test_near_plane_screen_edges_behind_camera_and_span_cap",
                 "test_last_camera_is_not_subsampled_and_cache_updates",
                 "test_float64_projection_preserves_small_boxes_at_large_world_coordinates"):
        # Reuse the exact boundary fixtures with the real backend, without
        # importing or constructing a Mitsuba model or any optimizer state.
        from unittest.mock import patch
        with patch(__name__+".ArrayDevice", return_value=dr):
            getattr(AdaptiveProjectionTests(case), case)()
    dr.sync_thread()
    print("PASS CUDA FP64 all-camera projection, partial batches and camera cache", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--build-dir", default=None)
    args = parser.parse_args()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AdaptiveProjectionTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.cuda:
        from preview import RuntimeConfig
        from reconstruction_common import initialize_runtime, WORKSPACE
        runtime = RuntimeConfig(gpu=args.gpu)
        if args.build_dir:
            runtime.build_dir = args.build_dir
        _, dr, _, _ = initialize_runtime(runtime, WORKSPACE/".cache"/"adaptive-projection", {})
        check_cuda(dr)
