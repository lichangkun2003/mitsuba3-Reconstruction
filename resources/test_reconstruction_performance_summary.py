"""CPU-only checks for mixed legacy and native-stage performance records."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconstruction_performance import PerformanceRecorder


class PerformanceSummaryTests(unittest.TestCase):
    def test_legacy_profiles_have_no_native_stage_samples(self):
        with tempfile.TemporaryDirectory(prefix="mitsuba-performance-summary-") as directory:
            recorder = PerformanceRecorder(Path(directory) / "summary.json", {})
            recorder.record_profile({"wall_seconds": .5, "device_ms": 12., "training_resolution": 256})
            recorder.record_profile({"wall_seconds": .75, "device_ms": 18., "training_resolution": 256})
            summary = recorder.write("passed")["kernel_profiles"]["overall"]
            self.assertEqual(summary["sampled_views"], 2)
            self.assertEqual(summary["wall_seconds"]["total"], 1.25)
            self.assertEqual(summary["totals"]["device_ms"], 30.)
            self.assertEqual(summary["native_stages"], {})

    def test_native_stages_are_separate_from_view_totals_and_grouped_by_resolution(self):
        with tempfile.TemporaryDirectory(prefix="mitsuba-performance-summary-") as directory:
            path = Path(directory) / "summary.json"
            recorder = PerformanceRecorder(path, {})
            stages = {
                "forward": {"wall_seconds": .1, "device_ms": 4., "codegen_ms": 1.,
                            "backend_compile_ms": 2., "operations": 3, "jit_kernels": 2,
                            "cache_hits": 1, "cache_misses": 1},
                "hit_backward": {"wall_seconds": .2, "device_ms": 5.},
                "candidate_backward": {"wall_seconds": .3, "device_ms": 6.},
            }
            for resolution, detail in ((256, None), (256, stages), (512, stages)):
                record = {"wall_seconds": 1., "device_ms": 20., "operations": 10,
                          "training_resolution": resolution,
                          "native_execution": "direct_stage_profile" if detail else "frozen"}
                if detail is not None:
                    record["native_stages"] = detail
                recorder.record_profile(record)
            report = recorder.write("passed")
            overall = report["kernel_profiles"]["overall"]
            self.assertEqual(overall["sampled_views"], 3)
            self.assertEqual(overall["wall_seconds"]["total"], 3.)
            self.assertEqual(overall["totals"]["device_ms"], 60.)
            self.assertEqual(overall["totals"]["operations"], 30.)
            for name, metrics in stages.items():
                stage = overall["native_stages"][name]
                self.assertEqual(stage["sampled_views"], 2)
                self.assertEqual(stage["wall_seconds"]["count"], 2)
                self.assertEqual(stage["wall_seconds"]["total"], 2 * metrics["wall_seconds"])
                self.assertEqual(stage["wall_seconds"]["p95"], metrics["wall_seconds"])
                for field, value in metrics.items():
                    if field != "wall_seconds":
                        self.assertEqual(stage["totals"][field], 2 * value)
            grouped = report["kernel_profiles"]["by_resolution"]
            self.assertEqual(grouped["256"]["sampled_views"], 2)
            self.assertEqual(grouped["256"]["native_stages"]["forward"]["sampled_views"], 1)
            self.assertEqual(grouped["512"]["native_stages"]["forward"]["sampled_views"], 1)
            execution = report["kernel_profiles"]["by_native_execution"]
            self.assertEqual(execution["frozen"]["sampled_views"], 1)
            self.assertEqual(execution["frozen"]["native_stages"], {})
            self.assertEqual(execution["direct_stage_profile"]["sampled_views"], 2)
            self.assertEqual(execution["direct_stage_profile"]["totals"]["device_ms"], 40.)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), report)

    def test_partial_and_invalid_stage_values_do_not_create_zero_samples(self):
        with tempfile.TemporaryDirectory(prefix="mitsuba-performance-summary-") as directory:
            recorder = PerformanceRecorder(Path(directory) / "summary.json", {})
            recorder.record_profile({"native_stages": {
                "forward": {"wall_seconds": math.nan, "device_ms": 7., "codegen_ms": -1.,
                            "backend_compile_ms": math.inf, "cache_hits": True,
                            "operations": "3"},
                "hit_backward": None,
                "unexpected": {"wall_seconds": 100.},
            }})
            recorder.record_profile({"native_stages": {"forward": {"wall_seconds": .25, "device_ms": 9.}}})
            recorder.record_profile({"native_stages": None})
            recorder.record_profile({})
            summary = recorder.write("passed")["kernel_profiles"]["overall"]
            self.assertEqual(summary["sampled_views"], 4)
            self.assertEqual(set(summary["native_stages"]), {"forward"})
            forward = summary["native_stages"]["forward"]
            self.assertEqual(forward["sampled_views"], 2)
            self.assertEqual(forward["wall_seconds"]["count"], 1)
            self.assertEqual(forward["wall_seconds"]["mean"], .25)
            self.assertEqual(forward["totals"]["device_ms"], 16.)
            self.assertEqual(forward["totals"]["codegen_ms"], 0.)
            self.assertEqual(forward["totals"]["backend_compile_ms"], 0.)
            self.assertEqual(forward["totals"]["cache_hits"], 0.)
            self.assertEqual(forward["totals"]["operations"], 0.)
            self.assertIsNone(recorder.profiles[0]["native_stages"]["forward"]["wall_seconds"])


if __name__ == "__main__":
    unittest.main()
