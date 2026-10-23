"""CPU equivalence checks for vote-first growth revalidation.

Optional ``--benchmark`` times full and restricted candidate regeneration on a
synthetic grid; it does not render, train, or require a CUDA installation.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import reconstruction_topology as topology
from reconstruction_growth_filter import generate_growth_for_targets, match_growth_parents


def checkpoint(cells, *, resolution=8, axes=(.8, .7, .6)):
    cells = np.asarray(cells, dtype=np.uint32)
    count = len(cells)
    return {"model_metadata": {"resolution": resolution,
            "grid_min": [0., 0., 0.], "grid_max": [float(resolution)] * 3,
            "sh_count": 4, "opacity_sh_count": 4},
            "model_state": {"cell_indices": cells,
                "centers": np.tile([.5, .5, .5], count).astype(np.float32),
                "log_scales": np.tile(np.log(axes), count).astype(np.float32),
                "rotations": np.tile([1., 0., 0., 0.], count).astype(np.float32),
                "radiance_sh": np.zeros(count * 12, dtype=np.float32),
                "opacity_sh": np.zeros(count * 4, dtype=np.float32)}}


def manager_for(state, **settings):
    count = len(state["model_state"]["cell_indices"])
    return SimpleNamespace(np=np, settings=topology.TopologySettings(**settings),
        schedule=topology.CoarseToFineConfig(enabled=False), cooldown={},
        candidates=np.zeros(count, dtype=bool), growth_wait_start=np.zeros(count, dtype=np.int64))


def full_generate(manager, state, boundary):
    return topology.TopologyManager._generate_growth(manager, state, boundary)


def subset(batch, targets):
    keep = np.isin(batch["cells"], targets)
    return {key: value[keep] if isinstance(value, np.ndarray) else copy.deepcopy(value)
            for key, value in batch.items()}


def selected_proposals(current, pending):
    keep = (pending["foreground"] >= 3) & (pending["background"] < 2)
    keep &= topology._matching_growth_pairs(current, pending, np)
    selected = {key: value[keep] for key, value in pending.items()
                if isinstance(value, np.ndarray)}
    _, winners = np.unique(selected["cells"], return_index=True)
    return {key: value[winners] for key, value in selected.items()}


class GrowthFilterTests(unittest.TestCase):
    def assert_batch_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key in actual:
            if isinstance(actual[key], np.ndarray):
                np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
            else:
                self.assertEqual(actual[key], expected[key], key)

    def assert_subset_equivalence(self, manager, state, targets, boundary=10):
        expected = subset(full_generate(manager, state, boundary), targets)
        actual = generate_growth_for_targets(manager, state, boundary, targets)
        self.assert_batch_equal(actual, expected)
        return actual

    def test_all_and_partial_targets_preserve_exact_order_and_geometry(self):
        rng = np.random.default_rng(1807)
        for use_26 in (False, True):
            for trial in range(12):
                state = checkpoint(rng.choice(512, 70, replace=False), axes=(1., .9, .8))
                values = state["model_state"]
                values["centers"] = rng.uniform(.05, .95, 210).astype(np.float32)
                values["rotations"] = rng.normal(size=280).astype(np.float32)
                values["log_scales"] += rng.uniform(-.4, .4, 210).astype(np.float32)
                manager = manager_for(state, growth_use_26_neighbors=use_26,
                                      growth_min_supporting_parents=1 + trial % 3)
                manager.candidates = rng.random(70) < .1
                manager.growth_wait_start[rng.random(70) < .2] = 9
                manager.cooldown = {int(cell): 11 for cell in rng.choice(512, 20, replace=False)}
                full = full_generate(manager, state, 10)
                self.assertGreater(len(full["cells"]), 0)
                self.assert_batch_equal(generate_growth_for_targets(
                    manager, state, 10, np.arange(512)), full)
                chosen = np.unique(full["cells"])[::3]
                self.assert_subset_equivalence(manager, state, np.r_[chosen, chosen, -1, 512, 999])

    def test_nonvoting_supporter_still_counts_and_frozen_attributes_survive(self):
        state = checkpoint([73, 75])
        manager = manager_for(state, growth_min_supporting_parents=2)
        pending = full_generate(manager, state, 10)
        pending["foreground"][:] = np.where(pending["parent_cells"] == 73, 4, 0)
        # Continuous parameters changed after the frozen first-view proposals.
        state["model_state"]["centers"] += np.float32(.03)
        current = self.assert_subset_equivalence(manager, state, [74])
        self.assertEqual(set(current["parent_cells"]), {73, 75})
        expected = selected_proposals(full_generate(manager, state, 10), pending)
        actual = selected_proposals(current, pending)
        self.assert_batch_equal(actual, expected)
        self.assertEqual(len(actual["cells"]), 1)
        chosen = np.flatnonzero((pending["cells"] == 74) & (pending["parent_cells"] == 73))[0]
        np.testing.assert_array_equal(actual["centers"][0], pending["centers"][chosen])

    def test_radial_background_never_supplies_parents_or_targets(self):
        # Core radius is three cell widths. Parent x=1 intersects the shell;
        # x=3 and x=5 are core cells. The shell must not count as the second
        # supporting parent for x=2, while the interior hole x=4 stays legal.
        state = checkpoint([217, 219, 221])
        state["model_metadata"].update(core_resolution=6)
        manager = manager_for(state, growth_min_supporting_parents=2)
        legacy = full_generate(manager, state, 10)
        self.assertIn(218, legacy["cells"])
        state["model_metadata"]["scene_mapping"] = "radial_seed_shell"
        restricted = full_generate(manager, state, 10)
        self.assertNotIn(218, restricted["cells"])
        self.assertIn(220, restricted["cells"])
        for name in ("cells", "parent_cells"):
            coordinates = topology._coordinates(restricted[name], 8, np)
            self.assertTrue(np.all(topology._growth_coordinates_allowed(
                coordinates, state["model_metadata"], np)), name)
        self.assert_subset_equivalence(manager, state, np.arange(512))
        self.assert_subset_equivalence(manager, state, [218, 220, 222])
        self.assertEqual(len(self.assert_subset_equivalence(manager, state, [222])["cells"]), 0)

    def test_radial_growth_region_follows_refinement(self):
        state = checkpoint([217, 219, 221])
        state["model_metadata"].update(core_resolution=6, scene_mapping="radial_seed_shell")
        refined = topology.refine_state(state, topology.CoarseToFineConfig())
        self.assertEqual(refined["model_metadata"]["scene_mapping"], "radial_seed_shell")
        self.assertEqual(topology._growth_core_radius_squared(refined["model_metadata"]), 144)
        manager = manager_for(refined, growth_min_supporting_parents=1)
        result = full_generate(manager, refined, 10)
        for name in ("cells", "parent_cells"):
            coordinates = topology._coordinates(result[name], 16, np)
            self.assertTrue(np.all(topology._growth_coordinates_allowed(
                coordinates, refined["model_metadata"], np)), name)
        self.assert_subset_equivalence(manager, refined, np.arange(16 ** 3))

    def test_current_contact_wait_and_deleted_parent_are_rechecked(self):
        state = checkpoint([73, 75])
        manager = manager_for(state, growth_min_supporting_parents=2)
        self.assertEqual(len(self.assert_subset_equivalence(manager, state, [74])["cells"]), 2)
        manager.candidates[1] = True
        self.assertEqual(len(self.assert_subset_equivalence(manager, state, [74])["cells"]), 0)
        manager.candidates[1] = False
        manager.growth_wait_start[1] = 9
        self.assertEqual(len(self.assert_subset_equivalence(manager, state, [74])["cells"]), 0)
        manager.growth_wait_start[1] = 6  # Exactly five elapsed epochs.
        self.assertEqual(len(self.assert_subset_equivalence(manager, state, [74])["cells"]), 2)
        state["model_state"]["log_scales"][3:6] = np.log(.1)
        self.assertEqual(len(self.assert_subset_equivalence(manager, state, [74])["cells"]), 0)
        deleted = checkpoint([73])
        self.assertEqual(len(self.assert_subset_equivalence(manager_for(deleted), deleted, [74])["cells"]), 0)

    def test_occupied_cooldown_expiry_and_grid_edge(self):
        state = checkpoint([73, 75])
        manager = manager_for(state)
        manager.cooldown[74] = 11
        self.assertEqual(len(self.assert_subset_equivalence(manager, state, [74])["cells"]), 0)
        self.assertEqual(len(self.assert_subset_equivalence(manager, state, [74], boundary=11)["cells"]), 2)
        occupied = checkpoint([73, 74, 75])
        self.assertEqual(len(self.assert_subset_equivalence(manager_for(occupied), occupied, [74])["cells"]), 0)
        edge = checkpoint([7, 9, 64])
        self.assert_subset_equivalence(manager_for(edge, growth_min_supporting_parents=1), edge, [0, 8, 63])

    def test_no_eligible_target_skips_all_geometry(self):
        state = checkpoint([73, 75])
        manager = manager_for(state)
        with patch.object(topology, "_geometry", side_effect=AssertionError("unexpected geometry")):
            for targets in ([], [73, 75], [-1, 512]):
                result = generate_growth_for_targets(manager, state, 10, targets)
                self.assertEqual(len(result["cells"]), 0)

    def test_only_nearby_parent_geometry_is_evaluated(self):
        resolution = 16
        target = 2 + resolution + resolution ** 2
        state = checkpoint([target - 1, target + 1, *range(2000, 2200)], resolution=resolution)
        manager = manager_for(state)
        with patch.object(topology, "_geometry", wraps=topology._geometry) as geometry:
            result = generate_growth_for_targets(manager, state, 10, [target])
            self.assertEqual(len(result["cells"]), 2)
            self.assertEqual(len(geometry.call_args.args[0]["centers"]), 2)

    def test_non_evidence_geometry_and_coarse_penetration_match(self):
        state = checkpoint([73, 75], axes=(3., 2., 2.))
        manager = manager_for(state, use_growth_evidence=False, growth_face_penetration=.25)
        self.assert_subset_equivalence(manager, state, np.arange(512))
        manager.schedule.enabled = True
        manager.schedule.start_resolution = 8
        manager.schedule.coarsest_growth_face_penetration = .2
        self.assert_subset_equivalence(manager, state, np.arange(512))

    def test_matching_remaps_after_deletion_and_preserves_duplicate_proposals(self):
        current = {"parent_cells": np.asarray([75, 73, 2 ** 32 - 1], dtype=np.uint32),
                   "cells": np.asarray([74, 74, 2 ** 32 - 2], dtype=np.uint32),
                   "parent_ids": np.asarray([1, 0, 2], dtype=np.int64)}
        pending = {"parent_cells": np.asarray([73, 99, 75, 73, 2 ** 32 - 1], dtype=np.uint32),
                   "cells": np.asarray([74, 74, 74, 74, 2 ** 32 - 2], dtype=np.uint32)}
        keep, parent_ids = match_growth_parents(current, pending)
        np.testing.assert_array_equal(keep, topology._matching_growth_pairs(current, pending, np))
        np.testing.assert_array_equal(parent_ids, [0, 1, 0, 2])
        empty = {key: value[:0] for key, value in current.items()}
        keep, parent_ids = match_growth_parents(empty, pending)
        self.assertFalse(keep.any())
        self.assertEqual(len(parent_ids), 0)


def benchmark():
    rng = np.random.default_rng(271)
    state = checkpoint(rng.choice(32 ** 3, 6000, replace=False), resolution=32,
                       axes=(1.1, 1., .9))
    manager = manager_for(state)
    pending = full_generate(manager, state, 10)
    targets = np.unique(pending["cells"])[::200]
    pending["foreground"][:] = np.where(np.isin(pending["cells"], targets), 3, 0)
    times = {"full_seconds": [], "filtered_seconds": []}
    current = filtered = None
    for _ in range(3):
        started = time.perf_counter()
        current = full_generate(manager, state, 10)
        times["full_seconds"].append(time.perf_counter() - started)
        started = time.perf_counter()
        filtered = generate_growth_for_targets(manager, state, 10, targets)
        times["filtered_seconds"].append(time.perf_counter() - started)
    test = GrowthFilterTests()
    test.assert_batch_equal(filtered, subset(current, targets))
    test.assert_batch_equal(selected_proposals(filtered, pending), selected_proposals(current, pending))
    print(json.dumps({"parents": 6000, "eligible_targets": len(targets),
                      "all_proposals": len(current["cells"]),
                      "filtered_proposals": len(filtered["cells"]),
                      "identical_proposals_and_winners": True,
                      **{key: float(np.median(value)) for key, value in times.items()}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(GrowthFilterTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.benchmark:
        benchmark()
