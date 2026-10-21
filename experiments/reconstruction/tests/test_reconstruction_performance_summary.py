"""CPU-only checks for mixed legacy and native-stage performance records."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconstruction_performance import PerformanceRecorder


class PerformanceSummaryTests(unittest.TestCase):
    def test_growth_preparation_in_epoch_total_and_separate_nested_breakdown(self):
        with tempfile.TemporaryDirectory(prefix="mitsuba-performance-summary-") as directory:
            recorder = PerformanceRecorder(Path(directory) / "summary.json", {})
            for epoch, resolution, preparation, detail, growth in (
                (1, 256, None, None, False),  # Legacy entry has no preparation timings.
                (2, 256, 5., {"snapshot": 1., "candidate_generation": 3.}, True),
                (3, 512, 2., {"snapshot": .5, "candidate_generation": 1.}, True),
                (4, 512, .25, {"candidate_clear": .125}, False),
            ):
                entry = {"epoch": epoch, "training_resolution": resolution, "seconds": 10.,
                         "view_stage_seconds": {"gradients": 2.}, "epoch_topology_seconds": 1.,
                         "growth_evidence_requested": growth}
                if preparation is not None:
                    entry["epoch_growth_prepare_seconds"] = preparation
                    entry["epoch_growth_prepare_stage_seconds"] = detail
                recorder.record_epoch(entry)
            report = recorder.write("passed")
            overall = report["overall"]
            self.assertEqual(overall["epoch_seconds"]["total"], 40.)
            self.assertEqual(overall["stage_seconds"],
                             {"gradients": 8., "epoch_topology": 4., "epoch_growth_prepare": 7.25})
            self.assertEqual(sum(overall["stage_seconds"].values()), 19.25)
            self.assertEqual(overall["stage_fraction_of_epoch_time"]["epoch_growth_prepare"], 7.25 / 40.)
            detail = overall["growth_prepare_stages"]
            self.assertEqual(detail["snapshot"]["count"], 2)
            self.assertEqual(detail["snapshot"]["total"], 1.5)
            self.assertEqual(detail["candidate_generation"]["mean"], 2.)
            self.assertEqual(detail["candidate_clear"]["count"], 1)
            grouped = report["by_resolution"]
            self.assertEqual(grouped["256"]["stage_seconds"]["epoch_growth_prepare"], 5.)
            self.assertEqual(grouped["256"]["growth_prepare_stages"]["snapshot"]["count"], 1)
            self.assertEqual(grouped["256"]["normal_epochs"]["growth_prepare_stages"], {})
            self.assertEqual(grouped["512"]["topology_event_epochs"]["growth_prepare_stages"]
                             ["candidate_generation"]["total"], 1.)
            self.assertEqual(grouped["512"]["normal_epochs"]["growth_prepare_stages"]
                             ["candidate_clear"]["total"], .125)

    def test_missing_and_invalid_growth_prepare_timings_are_not_zero_samples(self):
        with tempfile.TemporaryDirectory(prefix="mitsuba-performance-summary-") as directory:
            recorder = PerformanceRecorder(Path(directory) / "summary.json", {})
            for detail in (None, {}, {"snapshot": math.nan, "candidate_generation": math.inf,
                                     "proposal_packing": -1., "candidate_install": True},
                           {"snapshot": .25, "candidate_generation": "3"}):
                recorder.record_epoch({"seconds": 1., "epoch_growth_prepare_seconds": math.nan,
                                       "epoch_growth_prepare_stage_seconds": detail})
            report = recorder.write("passed")
            summary = report["overall"]
            self.assertEqual(summary["stage_seconds"], {})
            self.assertEqual(summary["growth_prepare_stages"]["snapshot"]["count"], 1)
            self.assertEqual(summary["growth_prepare_stages"]["snapshot"]["total"], .25)
            for name in ("candidate_generation", "proposal_packing", "candidate_install"):
                self.assertEqual(summary["growth_prepare_stages"][name]["count"], 0)
                self.assertIsNone(summary["growth_prepare_stages"][name]["mean"])
            self.assertNotIn("NaN", json.dumps(report, allow_nan=False))

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
