"""GPU topology evidence vs. the unchanged NumPy lifecycle, on tiny arrays.

No dataset or training render is required. Tests reject per-view device-to-host
array conversion and cover duplicate views, 32-bit boundaries, protection,
checkpoint resume, remapping, and switching between host/device evidence.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preview import RuntimeConfig
from reconstruction_common import initialize_runtime
from reconstruction_topology import TopologyManager, TopologySettings, CoarseToFineConfig, _cuda_array


class NoDeviceReads:
    def __init__(self, np):
        self.np = np

    def __getattr__(self, name):
        return getattr(self.np, name)

    def array(self, value, *args, **kwargs):
        if _cuda_array(value) is not None:
            raise AssertionError("Unexpected per-view device array readback")
        return self.np.array(value, *args, **kwargs)

    def asarray(self, value, *args, **kwargs):
        if _cuda_array(value) is not None:
            raise AssertionError("Unexpected per-view device array readback")
        return self.np.asarray(value, *args, **kwargs)


def manager(count=17, views=70):
    model = SimpleNamespace(resolution=8, active_voxel_count=count,
                            grid_min=[0., 0., 0.], grid_max=[8., 8., 8.])
    return model, TopologyManager(model, views,
        TopologySettings(enable_growth=False),
        CoarseToFineConfig(enabled=False, total_epochs=40), 0, 0)


def assert_state_equal(actual, expected, np):
    for field in ("foreground_masks", "background_masks", "qualified", "candidates",
                  "deletion_eligible", "growth_birth", "growth_wait_start"):
        np.testing.assert_array_equal(actual[field], expected[field], err_msg=field)
    assert actual["completed_windows"] == expected["completed_windows"]
    assert actual["count"] == expected["count"]


def observe(target, model, fg, bg, view, epoch):
    target.observe_view(model, {"foreground_evidence": fg, "background_evidence": bg},
                        view_id=view, epoch=epoch, camera={}, reference_alpha=None, rendered_alpha=None)


def deletion_equivalence(mi, dr, np):
    model, actual = manager()
    _, expected = manager()
    for target in (actual, expected):
        target.deletion_eligible[::3] = 5
        target.deletion_eligible[-1] = 200
    rng = np.random.default_rng(42)
    for epoch in range(10):
        for view in (0, 31, 32, 63, 69, 31, 0):
            fg = rng.integers(0, 2, actual.count, dtype=np.uint32)
            bg = rng.integers(0, 2, actual.count, dtype=np.uint32)
            fg[0], bg[0] = 0, 1
            expected_flags = (fg.copy(), bg.copy())
            actual.np = NoDeviceReads(np)
            observe(actual, model, mi.UInt32(fg), mi.UInt32(bg), view, epoch)
            actual.np = np
            if actual._deletion_device is not None:
                actual._deletion_device.np = np
            observe(expected, model, *expected_flags, view, epoch)
        if epoch in (4, 9):
            assert actual._evaluate_deletion(epoch + 1) == expected._evaluate_deletion(epoch + 1)
            np.testing.assert_array_equal(actual.qualified, expected.qualified)
            np.testing.assert_array_equal(actual.candidates, expected.candidates)
            # No raw N-voxel flags or masks were copied by observe_view.
            assert actual.statistics()["deletion_mask_readback_bytes"] == 0
    assert actual.candidates[0] == expected.candidates[0]
    assert actual.statistics()["deletion_device_views"] == 70
    assert actual.statistics()["deletion_count_readback_bytes"] == 2 * 17 * 4 * 2
    assert_state_equal(actual.state_dict(), expected.state_dict(), np)


def checkpoint_remap_and_mutable_state(mi, dr, np):
    model, actual = manager(5)
    _, expected = manager(5)
    for view in (0, 32, 69, 32):
        fg, bg = np.array([1, 0, 1, 0, 1], np.uint32), np.ones(5, np.uint32)
        observe(actual, model, mi.UInt32(fg), mi.UInt32(bg), view, 0)
        observe(expected, model, fg, bg, view, 0)
    saved = actual.state_dict()
    assert_state_equal(saved, expected.state_dict(), np)
    _, restored = manager(5)
    restored.load_state_dict(saved)
    assert restored._deletion_device is None
    for target in (restored, expected):
        target.foreground_masks[1, 1] = np.uint32(1 << 31)
        target.background_masks[2, 2] = np.uint32(1 << 5)
        target.deletion_eligible[3] = 4
    for view in (1, 63):
        observe(restored, model, mi.UInt32([1]*5), mi.UInt32([1]*5), view, 1)
        observe(expected, model, np.ones(5, np.uint32), np.ones(5, np.uint32), view, 1)
    survivors = np.array([4, 1, 3], np.uint32)
    restored._remap_lifecycle(survivors, 4, 2)
    expected._remap_lifecycle(survivors, 4, 2)
    model.active_voxel_count = 4
    # Newborn deletion protection must also apply after remapping the cache.
    observe(restored, model, mi.UInt32([1]*4), mi.UInt32([1]*4), 31, 3)
    observe(expected, model, np.ones(4, np.uint32), np.ones(4, np.uint32), 31, 3)
    assert_state_equal(restored.state_dict(), expected.state_dict(), np)
    # Mixing a NumPy evidence batch with prior device evidence is supported.
    observe(restored, model, mi.UInt32([1]*4), mi.UInt32([1]*4), 32, 4)
    observe(expected, model, np.ones(4, np.uint32), np.ones(4, np.uint32), 32, 4)
    for target in (restored, expected):
        observe(target, model, np.zeros(4, np.uint32), np.ones(4, np.uint32), 69, 4)
    assert_state_equal(restored.state_dict(), expected.state_dict(), np)
    for target in (restored, expected):
        target._initialize_lifecycle(0)
    model.active_voxel_count = 0
    observe(restored, model, mi.UInt32([]), mi.UInt32([]), 0, 0)
    observe(expected, model, np.zeros(0, np.uint32), np.zeros(0, np.uint32), 0, 0)
    assert restored._evaluate_deletion(5) == expected._evaluate_deletion(5)
    assert_state_equal(restored.state_dict(), expected.state_dict(), np)


def growth_manager_resume(mi, dr, np):
    model, actual = manager(2, 8)
    _, expected = manager(2, 8)
    pending = dict(parent_ids=np.array([0, 1]), parent_cells=np.array([73, 75]),
                   cells=np.array([81, 83]), centers=np.full((2, 3), .5),
                   log_scales=np.full((2, 3), -1.), rotations=np.array([[1., 0, 0, 0]]*2),
                   foreground=np.zeros(2, np.uint32), background=np.zeros(2, np.uint32),
                   seen_views=set(), boundary=1, resolution=8)
    for target in (actual, expected):
        target.settings.enable_deletion = False
        target.settings.enable_growth = True
        target.settings.growth_interval = 1
        target.pending_growth = copy.deepcopy(pending)
    camera = dict(position=[4., 4., -10.], right=[1., 0., 0.], up=[0., 1., 0.],
                  forward=[0., 0., 1.], camera_angle_x=1.2)
    rng = np.random.default_rng(2026)
    for view in (0, 1, 2, 2, 3, 4, 5, 6, 7):
        reference = rng.choice([0., 1.], (16, 16)).astype(np.float32)
        rendered = rng.uniform(0, 1, (16, 16)).astype(np.float32)
        actual.observe_view(model, {}, view_id=view, epoch=0, camera=camera,
                            reference_alpha=mi.Float(reference.ravel()), rendered_alpha=mi.Float(rendered.ravel()),
                            image_shape=(16, 16))
        expected.observe_view(model, {}, view_id=view, epoch=0, camera=camera,
                              reference_alpha=reference, rendered_alpha=rendered)
        if view == 3:
            saved = actual.state_dict()
            _, actual = manager(2, 8)
            actual.load_state_dict(saved)
    saved, oracle = actual.state_dict(), expected.state_dict()
    for field in ("foreground", "background"):
        np.testing.assert_array_equal(saved["pending_growth"][field], oracle["pending_growth"][field])
    assert saved["pending_growth"]["seen_views"] == list(range(8))
    assert actual._growth_device is None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="mitsuba-topology-device-") as temporary:
        mi, dr, np, handles = initialize_runtime(RuntimeConfig(build_dir=args.build_dir, gpu=args.gpu), Path(temporary), {})
        deletion_equivalence(mi, dr, np)
        checkpoint_remap_and_mutable_state(mi, dr, np)
        growth_manager_resume(mi, dr, np)
        dr.sync_thread()
    print("passed: GPU deletion masks, no per-view readback, CPU equivalence, checkpoint/remap, growth resume")


if __name__ == "__main__":
    main()
