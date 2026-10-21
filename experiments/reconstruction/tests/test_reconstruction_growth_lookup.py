"""CPU equivalence checks for reusable blocked-cell membership during growth.

The reference below is the complete pre-optimization candidate generator from
4abf1b0be2546ce355bc11d1e914c2318b10ff51. It intentionally retains its original
per-direction np.isin calls. Tests need neither Git nor a GPU at runtime.
"""

from __future__ import annotations

import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import reconstruction_topology as topology
from reconstruction_topology import (_geometry, _coordinates, _addresses,
                                      _contact, _neighbor_offsets)


def legacy_generate_growth(self, checkpoint, boundary):
    np, cfg = self.np, self.settings
    state, metadata = checkpoint["model_state"], checkpoint["model_metadata"]
    n, resolution = len(state["cell_indices"]), metadata["resolution"]
    center, axes, rotation, quaternion, size, metric, extent = _geometry(state, metadata, np)
    cells = _coordinates(state["cell_indices"], resolution, np)
    eligible = ~self.candidates
    eligible &= (self.growth_wait_start == 0) | ((boundary >= self.growth_wait_start - 1) &
                                                (boundary - (self.growth_wait_start - 1) >= cfg.growth_wait_epochs))
    occupied = np.asarray(state["cell_indices"], dtype=np.int64)
    evidence = cfg.use_growth_evidence or cfg.use_rgb_growth
    if evidence:
        penetration = float(np.clip(cfg.growth_evidence_penetration, 0., 4.))
    elif self.schedule.enabled:
        penetration = self.schedule.coarsest_growth_face_penetration * (1 if resolution == self.schedule.start_resolution
                                                                       else self.schedule.growth_finer_level_multiplier)
        penetration = float(np.clip(penetration, 0., 4.))
    else:
        penetration = float(np.clip(cfg.growth_face_penetration, 0., 4.))
    pieces = []
    forbidden_cooldown = np.asarray([cell for cell, expiry in self.cooldown.items() if boundary < expiry], dtype=np.int64)
    for offset in _neighbor_offsets(evidence and cfg.growth_use_26_neighbors, np):
        neighbors = cells + offset
        targets = _addresses(neighbors, resolution)
        possible = eligible & np.all(neighbors >= 0, axis=1) & np.all(neighbors < resolution, axis=1)
        possible &= ~np.isin(targets, occupied) & ~np.isin(targets, forbidden_cooldown)
        ids = np.flatnonzero(possible)
        if not len(ids):
            continue
        touched, contact = _contact(center[ids], metric[ids], offset, 0., np)
        if penetration > 0:
            deep, _ = _contact(center[ids], metric[ids], offset, penetration, np)
            touched &= deep
        child_center = np.clip(contact, .05, .95)
        for axis in np.flatnonzero(offset):
            child_center[:, axis] = np.clip(cfg.growth_contact_offset, .01, .49) if offset[axis] > 0 else 1 - np.clip(cfg.growth_contact_offset, .01, .49)
        margin = np.minimum(child_center, 1 - child_center)
        for axis in np.flatnonzero(offset):
            margin[:, axis] = 1 - child_center[:, axis] if offset[axis] > 0 else child_center[:, axis]
        factor = np.minimum(np.clip(cfg.growth_shrink, .01, .99), np.min(np.maximum(margin - .005, 0) / np.maximum(extent[ids], 1e-20), axis=1))
        touched &= (factor > 0) & np.all(axes[ids] * factor[:, None] >= size * 1e-4, axis=1)
        selected = ids[touched]
        if len(selected):
            pieces.append({"parent_ids": selected, "parent_cells": occupied[selected], "cells": targets[selected],
                           "centers": child_center[touched], "log_scales": np.log(axes[selected] * factor[touched, None]),
                           "rotations": quaternion[selected]})
    if pieces:
        result = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
    else:
        # Freeze even an empty first-view batch for the whole epoch.
        result = {"parent_ids": np.empty(0, dtype=np.int64), "parent_cells": np.empty(0, dtype=np.int64),
                  "cells": np.empty(0, dtype=np.int64), "centers": np.empty((0, 3)),
                  "log_scales": np.empty((0, 3)), "rotations": np.empty((0, 4))}
    if evidence:
        _, inverse, counts = np.unique(result["cells"], return_inverse=True, return_counts=True)
        supported = counts[inverse] >= max(1, min(26, cfg.growth_min_supporting_parents))
        result = {key: value[supported] for key, value in result.items()}
    count = len(result["cells"])
    result.update(foreground=np.zeros(count, dtype=np.uint32), background=np.zeros(count, dtype=np.uint32),
                  seen_views=set(), boundary=boundary, resolution=resolution)
    return result


def checkpoint(cells, *, resolution=8, axes=(.9, .8, .7)):
    cells = np.asarray(cells, dtype=np.uint32)
    count = len(cells)
    return {"model_metadata": {"resolution": resolution,
            "grid_min": [0., 0., 0.], "grid_max": [float(resolution)] * 3,
            "sh_count": 4, "opacity_sh_count": 1},
            "model_state": {"cell_indices": cells,
                "centers": np.tile([.5, .5, .5], count).astype(np.float32),
                "log_scales": np.tile(np.log(axes), count).astype(np.float32),
                "rotations": np.tile([1., 0., 0., 0.], count).astype(np.float32),
                "radiance_sh": np.zeros(count * 12, dtype=np.float32),
                "opacity_sh": np.zeros(count, dtype=np.float32)}}


def manager_for(state, **settings):
    count = len(state["model_state"]["cell_indices"])
    return SimpleNamespace(np=np, settings=topology.TopologySettings(**settings),
        schedule=topology.CoarseToFineConfig(enabled=False), cooldown={},
        candidates=np.zeros(count, dtype=bool),
        growth_wait_start=np.zeros(count, dtype=np.int64))


class GrowthLookupTests(unittest.TestCase):
    def assert_batch_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for key, value in expected.items():
            if isinstance(value, np.ndarray):
                self.assertEqual(actual[key].dtype, value.dtype, key)
                np.testing.assert_array_equal(actual[key], value, err_msg=key)
            else:
                self.assertEqual(actual[key], value, key)

    def assert_generator_equivalence(self, manager, state, boundary=10):
        original_state = copy.deepcopy(state)
        expected = legacy_generate_growth(manager, state, boundary)
        constructor = topology._GrowthBlockedCells
        for dense_budget in (0, 128 * 1024 * 1024):
            with self.subTest(dense_budget=dense_budget):
                def factory(*args, **kwargs):
                    kwargs["max_dense_bytes"] = dense_budget
                    return constructor(*args, **kwargs)
                with patch.object(topology, "_GrowthBlockedCells", side_effect=factory):
                    actual = topology.TopologyManager._generate_growth(manager, state, boundary)
                self.assert_batch_equal(actual, expected)
                self.assertEqual(state["model_metadata"], original_state["model_metadata"])
                self.assert_batch_equal(state["model_state"], original_state["model_state"])
        return expected

    def test_membership_duplicate_empty_and_out_of_range_queries(self):
        resolution = 16
        occupied = np.array([0, 3000, 1, 2048, 3000, resolution ** 3 - 1], dtype=np.int64)
        cooldown = np.array([3, 4, 3, 3000], dtype=np.int64)
        queries = np.array([0, 2, 3, 4, 2048, 3000, 3000, resolution ** 3 - 1,
                            -1, -300, resolution ** 3, 2 ** 40], dtype=np.int64)
        for occupied_values, cooldown_values in ((occupied, cooldown), (occupied, []),
                                                 ([], cooldown), ([], [])):
            keys = np.asarray([*occupied_values, *cooldown_values], dtype=np.int64)
            expected = np.isin(queries, keys)
            for budget in (0, 128 * 1024 * 1024):
                with self.subTest(occupied=len(occupied_values), cooldown=len(cooldown_values), budget=budget):
                    old_occupied, old_cooldown = np.copy(occupied_values), np.copy(cooldown_values)
                    lookup = topology._GrowthBlockedCells(occupied_values, cooldown_values,
                                                          resolution, np, max_dense_bytes=budget)
                    np.testing.assert_array_equal(lookup.contains(queries), expected)
                    np.testing.assert_array_equal(lookup.contains(queries[::-1]), expected[::-1])
                    self.assertEqual(lookup.contains(np.empty(0, dtype=np.int64)).shape, (0,))
                    np.testing.assert_array_equal(occupied_values, old_occupied)
                    np.testing.assert_array_equal(cooldown_values, old_cooldown)

    def test_membership_uint32_and_large_sparse_grid(self):
        # A dense grid would require 1 GiB; sorted membership must not wrap or
        # truncate the address, including probes beyond the legal range.
        resolution = 1024
        occupied = np.array([1, resolution ** 3 - 1, 2 ** 28], dtype=np.uint32)
        queries = np.array([0, 1, 2 ** 28, resolution ** 3 - 1, 2 ** 32 - 1], dtype=np.uint32)
        lookup = topology._GrowthBlockedCells(occupied, [], resolution, np,
                                              max_dense_bytes=128 * 1024 * 1024)
        np.testing.assert_array_equal(lookup.contains(queries), np.isin(queries, occupied))

    def test_dense_allocation_failure_uses_equivalent_sparse_lookup(self):
        zeros = np.zeros

        def fail_dense(shape, *args, **kwargs):
            if shape == 512:
                raise MemoryError("simulated dense-table memory pressure")
            return zeros(shape, *args, **kwargs)

        class AllocationFailureNumpy:
            zeros = staticmethod(fail_dense)

            def __getattr__(self, name):
                return getattr(np, name)

        lookup = topology._GrowthBlockedCells([7, 2, 7, 511], [9, 511], 8,
                                              AllocationFailureNumpy())
        queries = np.array([-1, 0, 2, 7, 9, 510, 511, 512], dtype=np.int64)
        np.testing.assert_array_equal(lookup.contains(queries),
                                      np.isin(queries, [7, 2, 7, 511, 9, 511]))

    def test_random_rotated_anisotropic_geometry_exactly_matches_original(self):
        rng = np.random.default_rng(845211)
        for use_26 in (False, True):
            for trial in range(12):
                state = checkpoint(rng.choice(512, 90, replace=False), axes=(1., .9, .7))
                values = state["model_state"]
                values["centers"] = rng.uniform(.05, .95, 270).astype(np.float32)
                values["rotations"] = rng.normal(size=360).astype(np.float32)
                values["log_scales"] += rng.uniform(-.4, .4, 270).astype(np.float32)
                manager = manager_for(state, growth_use_26_neighbors=use_26,
                                      growth_min_supporting_parents=1 + trial % 3,
                                      growth_evidence_penetration=(0., .05, .2)[trial % 3],
                                      use_rgb_growth=trial % 2 == 0)
                manager.candidates = rng.random(90) < .1
                manager.growth_wait_start[rng.random(90) < .2] = 9
                manager.cooldown = {int(cell): 9 + index % 3 for index, cell in
                                    enumerate(rng.choice(512, 35, replace=False))}
                with self.subTest(use_26=use_26, trial=trial):
                    result = self.assert_generator_equivalence(manager, state)
                    self.assertGreater(len(result["cells"]), 0)

    def test_support_maturity_and_cooldown_expiry(self):
        state = checkpoint([73, 75])
        manager = manager_for(state, growth_min_supporting_parents=2)
        result = self.assert_generator_equivalence(manager, state)
        self.assertEqual(np.count_nonzero(result["cells"] == 74), 2)
        manager.cooldown[74] = 11
        self.assertNotIn(74, self.assert_generator_equivalence(manager, state)["cells"])
        result = self.assert_generator_equivalence(manager, state, boundary=11)
        self.assertEqual(np.count_nonzero(result["cells"] == 74), 2)
        manager.cooldown.clear()
        manager.growth_wait_start[1] = 7
        self.assertNotIn(74, self.assert_generator_equivalence(manager, state)["cells"])
        manager.growth_wait_start[1] = 6  # Five elapsed epochs, exactly mature.
        result = self.assert_generator_equivalence(manager, state)
        self.assertEqual(np.count_nonzero(result["cells"] == 74), 2)
        manager.candidates[1] = True
        self.assertNotIn(74, self.assert_generator_equivalence(manager, state)["cells"])

    def test_empty_model_full_occupancy_and_all_cooldown(self):
        for cells in ([], list(range(8 ** 3)), [73, 75]):
            state = checkpoint(cells)
            manager = manager_for(state)
            if cells == [73, 75]:
                manager.cooldown = {cell: 11 for cell in range(8 ** 3)}
            with self.subTest(count=len(cells)):
                result = self.assert_generator_equivalence(manager, state)
                self.assertEqual(len(result["cells"]), 0)

    def test_grid_edges_do_not_wrap_to_adjacent_row_or_plane(self):
        # Flat addresses of invalid neighbors can alias legal cells: xyz bounds
        # must therefore be checked before relying on membership alone.
        cells = [0, 7, 8, 56, 63, 64, 448, 455, 504, 511]
        state = checkpoint(cells, axes=(2., 1.8, 1.6))
        manager = manager_for(state, growth_min_supporting_parents=1,
                              growth_evidence_penetration=0.)
        result = self.assert_generator_equivalence(manager, state)
        parents = _coordinates(result["parent_cells"], 8, np)
        children = _coordinates(result["cells"], 8, np)
        self.assertTrue(np.all(np.abs(parents - children) <= 1))
        self.assertTrue(np.all((children >= 0) & (children < 8)))
        self.assertFalse(np.isin(result["cells"], cells).any())

    def test_legacy_and_coarse_fine_penetration_paths(self):
        state = checkpoint([73, 75, 139, 191], axes=(3., 2., 1.8))
        manager = manager_for(state, use_growth_evidence=False,
                              growth_face_penetration=.2)
        self.assert_generator_equivalence(manager, state)
        manager.schedule.enabled = True
        manager.schedule.start_resolution = 8
        manager.schedule.coarsest_growth_face_penetration = .2
        self.assert_generator_equivalence(manager, state)
        manager.schedule.start_resolution = 4
        self.assert_generator_equivalence(manager, state)

    def test_single_lookup_is_reused_and_queries_are_valid(self):
        state = checkpoint([0, 7, 8, 64, 73, 75, 511])
        manager = manager_for(state, growth_min_supporting_parents=1)
        constructor = topology._GrowthBlockedCells
        constructions, queries = [], []

        class RecordingLookup:
            def __init__(self, *args, **kwargs):
                constructions.append(1)
                self.inner = constructor(*args, **kwargs)

            def contains(self, addresses):
                queries.append(addresses.copy())
                if np.any(addresses < 0) or np.any(addresses >= 512):
                    raise AssertionError("Lookup received an out-of-grid candidate")
                return self.inner.contains(addresses)

        expected = legacy_generate_growth(manager, state, 10)
        with patch.object(topology, "_GrowthBlockedCells", RecordingLookup):
            with patch.object(np, "isin", side_effect=AssertionError("repeated np.isin")):
                actual = topology.TopologyManager._generate_growth(manager, state, 10)
        self.assertEqual(len(constructions), 1)
        self.assertGreater(len(queries), 1)
        self.assert_batch_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
