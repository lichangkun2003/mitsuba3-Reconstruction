"""Small deterministic world-space leaves shared by storage regressions."""

import math


def fixture(np, *, core=4, levels=2, background_geometry="occupied_cell"):
    count = core**3 + 56 * levels
    rng = np.random.default_rng(415)
    axes = rng.normal(size=(count, 3))
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    angle = rng.uniform(-.4, .4, size=count)
    rotations = np.column_stack((np.cos(angle/2), axes*np.sin(angle/2)[:, None]))
    y00 = 1 / (2 * math.sqrt(math.pi))
    result = {"model_metadata": {
        "resolution": core, "core_resolution": core,
        "grid_min": [-1.]*3, "grid_max": [1.]*3,
        "scene_mapping": "world_shell", "background_shell_levels": levels,
        "background_refinement": "fixed",
        "background_geometry": background_geometry,
        "world_shell_version": 3 if background_geometry == "ellipsoid" else 2,
        "sh_count": 1, "opacity_sh_count": 1},
        "model_state": {
            "cell_indices": np.arange(count, dtype=np.uint32),
            "centers": np.tile([.53, .46, .48], count),
            "log_scales": np.tile(np.log([.46, .35, .42]), count),
            "rotations": rotations.ravel(),
            "radiance_sh": (rng.uniform(.15, .85, size=(count, 3))/y00).ravel(),
            "opacity_sh": np.full(count, math.log(.12/.88)/y00)}}
    if background_geometry == "ellipsoid":
        from reconstruction_world_shell import leaf_boxes_numpy
        cells = result["model_state"]["cell_indices"]
        lower, upper = leaf_boxes_numpy(cells, core, [-1.] * 3, [1.] * 3, levels, np)
        background = cells >= core ** 3
        # Geometry is relative to the actual coarse leaf, irrespective of the
        # foreground grid spacing. Retain anisotropy and the seeded rotations.
        result["model_state"]["log_scales"].reshape(-1, 3)[background] = np.log(
            (upper[background] - lower[background]) * np.array([.46, .35, .42]))
    return result


def make_ray(mi, np, origins, directions):
    origins = np.asarray(origins, np.float32).reshape(-1, 3)
    directions = np.asarray(directions, np.float32).reshape(-1, 3)
    directions = directions / np.linalg.norm(directions, axis=1)[:, None]
    ray = mi.Ray3f(mi.Point3f(*(mi.Float(origins[:, axis].copy()) for axis in range(3))),
                  mi.Vector3f(*(mi.Float(directions[:, axis].copy()) for axis in range(3))))
    ray.maxt = 1.e6
    return ray
