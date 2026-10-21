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

    def add(self, view_id, summary):
        self.dr.eval(summary)
        self.records.append([int(view_id), summary, self.mi.Bool(True)])

    def updated(self, accepted):
        self.dr.eval(accepted)
        self.records[-1][2] = accepted

    def read(self, *, rgb_only, distortion_weight):
        """One small transfer, including all earlier failures, at a boundary."""
        dr, mi, np = self.dr, self.mi, self.np
        packed = dr.concat([dr.concat([summary, mi.Float(accepted)])
                            for _, summary, accepted in self.records])
        values = np.asarray(packed).reshape(len(self.records), 5)
        result = []
        for (view_id, _, _), value in zip(self.records, values):
            loss, finite, rgb, distortion, accepted = map(float, value)
            if not finite or not math.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss/alpha/gradient at training view {view_id}; "
                                   "GPU guard blocked this and all later updates")
            if loss < 0:
                raise RuntimeError(f"Negative squared-error loss at training view {view_id}: {loss}; "
                                   "GPU guard blocked this and all later updates")
            if not accepted:
                raise RuntimeError(f"Rejected nonfinite optimizer update at training view {view_id}; "
                                   "GPU guard preserved parameters and Adam state")
            if not math.isfinite(rgb) or not math.isfinite(distortion):
                raise RuntimeError(f"Nonfinite loss component at training view {view_id}")
            result.append(dict(loss=loss,
                               rgb_loss=rgb if rgb_only or distortion_weight else None,
                               distortion_loss=distortion,
                               weighted_distortion_loss=distortion_weight * distortion))
        return result
