"""Revalidate only growth targets whose frozen proposals received enough votes.

The first-view candidate generator still creates the proposal geometry. At the
epoch boundary this helper checks the current model around eligible targets,
including parents whose own proposals did not receive votes. Its result is the
same ordered subset of ``TopologyManager._generate_growth`` for those targets;
callers retain the frozen proposal attributes when matching parent/target pairs.
"""

from __future__ import annotations


def match_growth_parents(current, pending):
    """Match frozen pairs and return their current parent IDs in pending order.

    The candidate generator already stores current IDs. Reusing them avoids
    constructing a Python dictionary for every occupied cell at the boundary.
    ``pending`` may contain duplicate pairs; each retains its original order.
    """
    import numpy as np

    def keys(batch):
        return ((batch["parent_cells"].astype(np.uint64) << np.uint64(32)) |
                batch["cells"].astype(np.uint64))

    allowed, proposed = keys(current), keys(pending)
    order = np.argsort(allowed, kind="stable")
    allowed = allowed[order]
    locations = np.searchsorted(allowed, proposed)
    keep = locations < len(allowed)
    keep[keep] &= allowed[locations[keep]] == proposed[keep]
    return keep, current["parent_ids"][order[locations[keep]]]


def generate_growth_for_targets(manager, checkpoint, boundary, target_cells):
    """Return current candidates for target cells without scanning all geometry.

    Parent addresses are looked up before quaternion/ellipsoid/contact math.
    All occupied cells still block growth, and all eligible neighboring parents
    still contribute to the support threshold. Sorting each offset's parent IDs
    preserves the full generator's deterministic proposal order and winner.
    """
    from reconstruction_topology import (
        _addresses, _contact, _coordinates, _geometry, _neighbor_offsets)

    np, cfg = manager.np, manager.settings
    state, metadata = checkpoint["model_state"], checkpoint["model_metadata"]
    resolution = metadata["resolution"]
    occupied = np.asarray(state["cell_indices"], dtype=np.int64)

    def empty_result():
        return {"parent_ids": np.empty(0, dtype=np.int64),
                "parent_cells": np.empty(0, dtype=np.int64),
                "cells": np.empty(0, dtype=np.int64),
                "centers": np.empty((0, 3)), "log_scales": np.empty((0, 3)),
                "rotations": np.empty((0, 4)),
                "foreground": np.zeros(0, dtype=np.uint32),
                "background": np.zeros(0, dtype=np.uint32),
                "seen_views": set(), "boundary": boundary,
                "resolution": resolution}

    targets = np.unique(np.asarray(target_cells, dtype=np.int64).reshape(-1))
    targets = targets[(targets >= 0) & (targets < resolution ** 3)]
    if not len(targets) or not len(occupied):
        return empty_result()

    # One sorted address index serves occupancy tests and all neighbor lookups.
    # This index contains the whole model, not merely the voting parents.
    order = np.argsort(occupied, kind="stable")
    sorted_cells = occupied[order]

    def lookup(addresses):
        locations = np.searchsorted(sorted_cells, addresses)
        found = locations < len(sorted_cells)
        found[found] &= sorted_cells[locations[found]] == addresses[found]
        return locations, found

    _, blocked = lookup(targets)
    targets = targets[~blocked]
    forbidden = np.asarray([cell for cell, expiry in manager.cooldown.items()
                            if boundary < expiry], dtype=np.int64)
    targets = targets[~np.isin(targets, forbidden)]
    if not len(targets):
        return empty_result()

    candidates = np.asarray(manager.candidates, dtype=bool).reshape(-1)
    wait_start = np.asarray(manager.growth_wait_start, dtype=np.int64).reshape(-1)
    eligible = ~candidates
    eligible &= ((wait_start == 0) |
                 ((boundary >= wait_start - 1) &
                  (boundary - (wait_start - 1) >= cfg.growth_wait_epochs)))
    target_coordinates = _coordinates(targets, resolution, np)
    evidence = cfg.use_growth_evidence
    offsets = _neighbor_offsets(evidence and cfg.growth_use_26_neighbors, np)
    neighborhoods = []
    for offset in offsets:
        parent_coordinates = target_coordinates - offset
        inside = (np.all(parent_coordinates >= 0, axis=1) &
                  np.all(parent_coordinates < resolution, axis=1))
        target_ids = np.flatnonzero(inside)
        locations, found = lookup(_addresses(parent_coordinates[inside], resolution))
        parent_ids = order[locations[found]]
        target_ids = target_ids[found]
        accepted = eligible[parent_ids]
        parent_ids, target_ids = parent_ids[accepted], target_ids[accepted]
        if len(parent_ids):
            parent_order = np.argsort(parent_ids)
            neighborhoods.append((offset, parent_ids[parent_order],
                                  targets[target_ids[parent_order]]))
    if not neighborhoods:
        return empty_result()

    geometry_ids = np.unique(np.concatenate([item[1] for item in neighborhoods]))
    n = len(occupied)
    geometry_state = {key: np.asarray(state[key]).reshape(n, dimension)[geometry_ids]
                      for key, dimension in (("centers", 3), ("log_scales", 3),
                                             ("rotations", 4))}
    center, axes, _, quaternion, size, metric, extent = _geometry(geometry_state, metadata, np)
    if evidence:
        penetration = float(np.clip(cfg.growth_evidence_penetration, 0., 4.))
    elif manager.schedule.enabled:
        penetration = manager.schedule.coarsest_growth_face_penetration * (
            1 if resolution == manager.schedule.start_resolution
            else manager.schedule.growth_finer_level_multiplier)
        penetration = float(np.clip(penetration, 0., 4.))
    else:
        penetration = float(np.clip(cfg.growth_face_penetration, 0., 4.))

    pieces = []
    for offset, parent_ids, addresses in neighborhoods:
        ids = np.searchsorted(geometry_ids, parent_ids)
        touched, contact = _contact(center[ids], metric[ids], offset, 0., np)
        if penetration > 0:
            deep, _ = _contact(center[ids], metric[ids], offset, penetration, np)
            touched &= deep
        child_center = np.clip(contact, .05, .95)
        for axis in np.flatnonzero(offset):
            child_center[:, axis] = (np.clip(cfg.growth_contact_offset, .01, .49)
                                     if offset[axis] > 0 else
                                     1 - np.clip(cfg.growth_contact_offset, .01, .49))
        margin = np.minimum(child_center, 1 - child_center)
        for axis in np.flatnonzero(offset):
            margin[:, axis] = (1 - child_center[:, axis] if offset[axis] > 0
                              else child_center[:, axis])
        factor = np.minimum(np.clip(cfg.growth_shrink, .01, .99), np.min(
            np.maximum(margin - .005, 0) / np.maximum(extent[ids], 1e-20), axis=1))
        touched &= ((factor > 0) &
                    np.all(axes[ids] * factor[:, None] >= size * 1e-4, axis=1))
        if np.any(touched):
            selected = ids[touched]
            pieces.append({"parent_ids": parent_ids[touched],
                           "parent_cells": occupied[parent_ids[touched]],
                           "cells": addresses[touched],
                           "centers": child_center[touched],
                           "log_scales": np.log(axes[selected] * factor[touched, None]),
                           "rotations": quaternion[selected]})
    if not pieces:
        return empty_result()
    result = {key: np.concatenate([piece[key] for piece in pieces])
              for key in pieces[0]}
    if evidence:
        _, inverse, counts = np.unique(result["cells"], return_inverse=True, return_counts=True)
        supported = counts[inverse] >= max(1, min(26, cfg.growth_min_supporting_parents))
        result = {key: value[supported] for key, value in result.items()}
    count = len(result["cells"])
    result.update(foreground=np.zeros(count, dtype=np.uint32),
                  background=np.zeros(count, dtype=np.uint32), seen_views=set(),
                  boundary=boundary, resolution=resolution)
    return result
