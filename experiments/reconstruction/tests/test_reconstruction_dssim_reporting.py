"""CPU regression for D-SSIM summaries, legacy packing, and validation."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruction_checks import DeferredViewChecks


class ReportingTests(unittest.TestCase):
    def checks(self):
        mi = SimpleNamespace(Float=lambda value: np.atleast_1d(np.float32(value)),
                             Bool=lambda value: np.atleast_1d(np.bool_(value)))
        transfers = []

        def asarray(value):
            transfers.append(value.size)
            return np.asarray(value)

        dr = SimpleNamespace(width=np.size, concat=np.concatenate, eval=lambda *args: None)
        return DeferredViewChecks(mi, dr, SimpleNamespace(asarray=asarray)), transfers

    def test_enabled_component_has_single_epoch_readback_and_correct_weight(self):
        checks, transfers = self.checks()
        for index in range(2):
            checks.add(index, np.array([1.3, 1., 1., .2, .5], np.float32))
            checks.updated(np.array([True]))
        self.assertEqual(transfers, [])
        rows = checks.read(rgb_only=True, distortion_weight=.25, dssim_weight=.5)
        self.assertEqual(transfers, [12])
        for row in rows:
            self.assertEqual(row["rgb_loss"], 1.)
            self.assertEqual(row["dssim_loss"], .5)
            self.assertEqual(row["weighted_dssim_loss"], .25)
            self.assertAlmostEqual(row["weighted_distortion_loss"], .05)

    def test_disabled_layout_keeps_original_packed_width(self):
        checks, transfers = self.checks()
        checks.add(0, np.array([1., 1., 1., 0.], np.float32))
        row = checks.read(rgb_only=True, distortion_weight=0.)[0]
        self.assertEqual(transfers, [5])
        self.assertEqual(row["dssim_loss"], 0.)
        self.assertEqual(row["weighted_dssim_loss"], 0.)
        with self.assertRaisesRegex(ValueError, "own view loss"):
            checks.read(rgb_only=True, distortion_weight=0., dssim_weight=.1)

    def test_nonfinite_component_and_inconsistent_summary_rejected(self):
        checks, _ = self.checks()
        checks.add(7, np.array([1., 1., 1., 0., np.nan], np.float32))
        with self.assertRaisesRegex(RuntimeError, "view 7"):
            checks.read(rgb_only=True, distortion_weight=0., dssim_weight=.1)
        with self.assertRaisesRegex(ValueError, "consistent"):
            checks.add(8, np.array([1., 1., 1., 0.], np.float32))


if __name__ == "__main__":
    unittest.main()
