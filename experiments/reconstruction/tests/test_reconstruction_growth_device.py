"""CUDA growth votes versus the original float64 NumPy projection oracle.

Uses generated proposal points and alpha planes, no dataset or image render.
Checks clipped frusta, pixel/alpha thresholds, background veto, unique views,
checkpoint restoration, and the absence of per-view GPU-to-host transfers.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preview import RuntimeConfig
from reconstruction_common import initialize_runtime
from reconstruction_topology_device import GrowthEvidenceDevice


SETTINGS = SimpleNamespace(foreground_alpha_min=.9, background_alpha_max=.1,
                           growth_min_alpha_deficit=.05)


def camera():
    return dict(position=[0., 0., 0.], right=[1., 0., 0.], up=[0., 1., 0.],
                forward=[0., 0., 1.], camera_angle_x=math.pi / 2,
                near_clip=.1, far_clip=2.)


def pending(count):
    return dict(foreground=np.zeros(count, dtype=np.uint32),
                background=np.zeros(count, dtype=np.uint32), seen_views=set())


def cpu_observe(points, state, cam, reference, rendered, view_id, settings=SETTINGS):
    """Original NumPy formula, kept independent of the device implementation."""
    if view_id in state["seen_views"]:
        return
    height, width = reference.shape
    position, right, up, forward = [np.asarray(cam[key])
                                   for key in ("position", "right", "up", "forward")]
    tan_x = math.tan(cam["camera_angle_x"] * .5)
    tan_y = tan_x * height / width
    fg, bg = np.zeros(len(points[0]), bool), np.zeros(len(points[0]), bool)
    for world in points:
        relative = world - position
        depth = relative @ forward
        u = .5 + .5 * (relative @ right) / np.maximum(depth * tan_x, 1e-30)
        v = .5 - .5 * (relative @ up) / np.maximum(depth * tan_y, 1e-30)
        visible = ((depth >= cam.get("near_clip", .1)) & (depth <= cam.get("far_clip", 1000.))
                   & (u >= 0) & (u < 1) & (v >= 0) & (v < 1))
        ids = np.flatnonzero(visible)
        x, y = (u[ids] * width).astype(np.int64), (v[ids] * height).astype(np.int64)
        ref, pred = reference[y, x], rendered[y, x]
        bg[ids] |= ref <= settings.background_alpha_max
        fg[ids] |= ((ref >= settings.foreground_alpha_min)
                    & (ref - pred >= np.clip(settings.growth_min_alpha_deficit, .001, 1.)))
    state["foreground"] += fg & ~bg
    state["background"] += bg
    state["seen_views"].add(int(view_id))


class NumpyGuard:
    """Forbid host conversion of every Dr.Jit array during observe()."""

    active = False

    def __getattr__(self, name):
        value = getattr(np, name)
        if name not in ("array", "asarray", "asanyarray", "ascontiguousarray"):
            return value

        def checked(*args, **kwargs):
            if self.active and args and dr.is_array_v(type(args[0])):
                raise AssertionError("Per-view growth read back a GPU array")
            return value(*args, **kwargs)

        return checked


class DrJitGuard:
    active = False

    def __getattr__(self, name):
        if name == "sync_thread" and self.active:
            raise AssertionError("Per-view growth explicitly synchronized the GPU")
        return getattr(dr, name)


class GrowthDeviceTests(unittest.TestCase):
    def assert_counters(self, device, oracle):
        got = device.materialize()
        for field in ("foreground", "background"):
            np.testing.assert_array_equal(got[field], oracle[field], err_msg=field)
        self.assertEqual(device.pending["seen_views"], oracle["seen_views"])

    def test_random_projection_checkpoint_and_unique_views(self):
        rng = np.random.default_rng(8326)
        points = tuple(rng.uniform([-2., -2., -.5], [2., 2., 3.], (1003, 3)) for _ in range(7))
        state, oracle = pending(1003), pending(1003)
        device = GrowthEvidenceDevice(points, state, mi, dr, np)
        original_points = device.points
        for view in range(9):
            cam = camera()
            cam["position"] = rng.uniform(-.1, .1, 3).tolist()
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            cam["right"], cam["up"], cam["forward"] = rotation.T.tolist()
            reference = rng.choice(np.array([0., .1, .5, .9, 1.], dtype=np.float32), (17, 23))
            rendered = rng.uniform(0, 1, reference.shape).astype(np.float32)
            ref, pred = mi.Float(reference.ravel()), mi.Float(rendered.ravel())
            for _ in range(2):
                cpu_observe(points, oracle, cam, reference, rendered, view)
                device.observe(cam, ref, pred, 23, 17, view, SETTINGS)
            self.assertIs(device.points, original_points)
            if view == 4:
                self.assert_counters(device, oracle)
                state = copy.deepcopy(state)
                device = GrowthEvidenceDevice(points, state, mi, dr, np)
                original_points = device.points
        self.assert_counters(device, oracle)

    def test_frustum_boundaries_background_veto_and_alpha_thresholds(self):
        cam = camera()
        tangent = math.tan(cam["camera_angle_x"] * .5)
        coordinates = np.array([
            [0., 0., .1], [0., 0., 2.], [0., 0., np.nextafter(.1, 0.)],
            [0., 0., np.nextafter(2., 3.)], [-tangent, 0., 1.], [tangent, 0., 1.],
            [0., tangent, 1.], [0., -tangent, 1.], [0., 0., -1.], [0., 0., 0.],
            [-.75 * tangent, .75 * tangent, 1.], [-.25 * tangent, .75 * tangent, 1.],
            [.25 * tangent, .75 * tangent, 1.], [.75 * tangent, .75 * tangent, 1.],
            [0., 0., 1.], [0., 0., 1.],
        ])
        points = [coordinates.copy() for _ in range(7)]
        # Proposal 14 sees both foreground and background in the same view.
        points[6][14] = coordinates[10]
        reference = np.ones((4, 4), dtype=np.float32)
        reference[0] = [.1, np.nextafter(np.float32(.9), np.float32(0)), .9, 1.]
        rendered = np.zeros_like(reference)
        rendered[0, 2] = np.float32(.85)
        rendered[0, 3] = np.float32(.95)
        state, oracle = pending(len(coordinates)), pending(len(coordinates))
        device = GrowthEvidenceDevice(points, state, mi, dr, np)
        cpu_observe(points, oracle, cam, reference, rendered, 0)
        device.observe(cam, mi.Float(reference.ravel()), mi.Float(rendered.ravel()), 4, 4, 0, SETTINGS)
        self.assert_counters(device, oracle)
        self.assertEqual(int(state["foreground"][14]), 0)
        self.assertEqual(int(state["background"][14]), 1)
        for index in (2, 3, 5, 7, 8, 9):
            self.assertEqual(int(state["foreground"][index] + state["background"][index]), 0)

    def test_observe_has_no_readback_or_sync_and_cpu_reference_fallback(self):
        numpy_guard, drjit_guard = NumpyGuard(), DrJitGuard()
        points = [np.array([[0., 0., 1.], [.25, .25, 1.]])] * 7
        state, oracle = pending(2), pending(2)
        device = GrowthEvidenceDevice(points, state, mi, drjit_guard, numpy_guard)
        reference = np.ones((4, 6), dtype=np.float32)
        rendered = np.zeros_like(reference)
        gpu_ref, gpu_pred = mi.Float(reference.ravel()), mi.Float(rendered.ravel())
        numpy_guard.active = drjit_guard.active = True
        for view, ref in enumerate((gpu_ref, reference, gpu_ref)):
            device.observe(camera(), ref, gpu_pred, 6, 4, view, SETTINGS)
            cpu_observe(points, oracle, camera(), reference, rendered, view)
        self.assertEqual(device.materializations, 0)
        numpy_guard.active = drjit_guard.active = False
        self.assert_counters(device, oracle)

    def test_empty_batch_and_invalid_images(self):
        points = [np.empty((0, 3))] * 7
        state = pending(0)
        device = GrowthEvidenceDevice(points, state, mi, dr, np)
        device.observe(camera(), mi.Float([1]), mi.Float([0]), 1, 1, 5, SETTINGS)
        self.assertEqual(state["seen_views"], {5})
        self.assert_counters(device, state)
        for reference, rendered, width, height in ((mi.Float([1]), mi.Float([0]), 2, 2),
                (np.ones((1, 1), dtype=np.float64), mi.Float([0]), 1, 1),
                (mi.Float([1]), mi.Float([0, 1]), 1, 1), (mi.Float([1]), mi.Float([0]), 0, 1)):
            with self.assertRaises(ValueError):
                device.observe(camera(), reference, rendered, width, height, 6, SETTINGS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    global mi, dr, np
    report = {}
    mi, dr, np, handles = initialize_runtime(RuntimeConfig(build_dir=args.build_dir, gpu=args.gpu), None, report)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(GrowthDeviceTests))
    print(json.dumps({"status": "passed" if result.wasSuccessful() else "failed", "checks": result.testsRun}), flush=True)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
