"""Small device-resident view summaries with explicit reporting boundaries.

This changes reporting/synchronization only. Each view still updates Adam once.
The optimizer owns a latched device guard; rejected updates never change its
parameters or moments, and validation happens before any epoch topology/save.
"""

from __future__ import annotations

import math


class DeferredViewChecks:
    def __init__(self, mi, dr, np):
        self.mi, self.dr, self.np = mi, dr, np
        self.records = []
        self.summary_width = None

    def add(self, view_id, summary):
        width = self.dr.width(summary)
        if width not in (4, 5):
            raise ValueError("View summaries require loss, validity, RGB, distortion, and optional D-SSIM")
        if self.summary_width is not None and width != self.summary_width:
            raise ValueError("View summary components must remain consistent within an epoch")
        self.summary_width = width
        self.dr.eval(summary)
        self.records.append([int(view_id), summary, self.mi.Bool(True)])

    def updated(self, accepted):
        self.dr.eval(accepted)
        self.records[-1][2] = accepted

    def read(self, *, rgb_only, distortion_weight, dssim_weight=0.):
        """One small transfer, including all earlier failures, at a boundary."""
        dr, mi, np = self.dr, self.mi, self.np
        if not self.records:
            return []
        if dssim_weight and self.summary_width != 5:
            raise ValueError("Enabled D-SSIM requires its own view loss component")
        packed = dr.concat([dr.concat([summary, mi.Float(accepted)])
                            for _, summary, accepted in self.records])
        values = np.asarray(packed).reshape(len(self.records), self.summary_width + 1)
        result = []
        for (view_id, _, _), value in zip(self.records, values):
            loss, finite, rgb, distortion = map(float, value[:4])
            dssim = float(value[4]) if self.summary_width == 5 else 0.
            accepted = float(value[-1])
            if not finite or not math.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss/alpha/gradient at training view {view_id}; "
                                   "GPU guard blocked this and all later updates")
            if loss < 0:
                raise RuntimeError(f"Negative squared-error loss at training view {view_id}: {loss}; "
                                   "GPU guard blocked this and all later updates")
            if not accepted:
                raise RuntimeError(f"Rejected nonfinite optimizer update at training view {view_id}; "
                                   "GPU guard preserved parameters and Adam state")
            if not all(math.isfinite(component) for component in (rgb, distortion, dssim)):
                raise RuntimeError(f"Nonfinite loss component at training view {view_id}")
            result.append(dict(loss=loss,
                               rgb_loss=rgb if rgb_only or distortion_weight or dssim_weight else None,
                               distortion_loss=distortion,
                               weighted_distortion_loss=distortion_weight * distortion,
                               dssim_loss=dssim,
                               weighted_dssim_loss=dssim_weight * dssim))
        return result
