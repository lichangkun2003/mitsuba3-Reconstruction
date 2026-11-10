"""Small ray-category counters; no host reads until the epoch boundary."""

EVIDENCE_RAY_KEYS = (
    "growth_vote_known_rays",
    "growth_vote_recovered_stop_rays",
    "growth_vote_unknown_record_budget_rays",
    "growth_vote_unknown_trace_rays",
    "growth_vote_unknown_nonfinite_rays",
    "deletion_vote_known_rays",
    "deletion_vote_recovered_stop_rays",
    "deletion_vote_unknown_tail_rays",
    "deletion_vote_unknown_trace_rays",
    "deletion_vote_unknown_nonfinite_rays",
)


def accumulate_evidence_rays(totals, result, mi, dr):
    """Consume per-view UInt32 counts into UInt64 epoch totals on device."""
    for name in EVIDENCE_RAY_KEYS:
        if name in result:
            totals[name] = totals.get(name, 0) + mi.UInt64(result.pop(name))
    if totals:
        dr.eval(*totals.values())


def read_evidence_rays(totals):
    if not totals:
        return None
    return {
        "unit": "unique rays per view/category, summed across views; categories may overlap",
        **{name: int(value[0]) for name, value in totals.items()},
    }
