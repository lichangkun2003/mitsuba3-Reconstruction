"""CUDA refinement matches shell ellipsoid fits and legacy occupied-cell shells."""
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from preview import RuntimeConfig
from reconstruction_common import initialize_runtime, restore_model
from reconstruction_optimizer import FalcorOptimizer
from reconstruction_refinement_gpu import refine_state_gpu
from reconstruction_topology import CoarseToFineConfig, NoRepresentableChildren, TopologyManager, TopologySettings, refine_state
from test_reconstruction_world_refinement import (fixture, sparse_ellipsoid_fixture,
    empty_first_shell_fixture, high_address_fixture, growth_address_alias_fixture)


def run(mi, dr, np):
    from reconstruction_growth_candidates_gpu import generate_growth_gpu
    growth_state, growth_manager = growth_address_alias_fixture()
    proposals = generate_growth_gpu(growth_manager, growth_state, 10, target_cells=dr.cuda.UInt64([74]))
    np.testing.assert_array_equal(np.array(proposals["cells"]), [74, 74])
    np.testing.assert_array_equal(np.array(proposals["parent_ids"]), [0, 1])
    assert np.array(proposals["candidate_state"]["cell_indices"]).dtype == np.uint64
    far_targets = generate_growth_gpu(growth_manager, growth_state, 10,
                                      target_cells=dr.cuda.UInt64([2 ** 32 + 74]))
    assert len(far_targets["cells"]) == 0
    # B=128 produces S6 initially; only sparse leaves are fitted at S7/S8.
    current = high_address_fixture()
    for resolution, subdivision in ((512, 7), (1024, 8)):
        expected = refine_state(current, CoarseToFineConfig())
        actual = refine_state_gpu(current, CoarseToFineConfig())
        assert actual["model_metadata"]["background_subdivision_level"] == subdivision
        cells = np.asarray(actual["model_state"]["cell_indices"])
        assert cells.dtype == np.uint64
        for key, values in expected["model_state"].items():
            if key == "cell_indices":
                np.testing.assert_array_equal(np.asarray(actual["model_state"][key]), values)
            else:
                np.testing.assert_allclose(np.asarray(actual["model_state"][key]), values,
                    atol=5e-7, rtol=7e-6, err_msg=f"high address {key}")
        model = restore_model(actual, mi)
        assert model.core_resolution == resolution
        np.testing.assert_array_equal(np.asarray(model.snapshot()["cell_indices"]), cells)
        current = expected
    assert np.max(cells) > np.iinfo(np.uint32).max
    # A geometrically valid shell may have no representable child ellipsoids.
    # Only that shell is omitted; normal outer shells and core must still refine.
    checkpoint = empty_first_shell_fixture()
    expected = refine_state(checkpoint, CoarseToFineConfig())
    actual = refine_state_gpu(checkpoint, CoarseToFineConfig())
    for key, values in expected["model_state"].items():
        if key == "cell_indices":
            np.testing.assert_array_equal(np.asarray(actual["model_state"][key]), values)
        else:
            np.testing.assert_allclose(np.asarray(actual["model_state"][key]), values,
                                       atol=4e-7, rtol=5e-6, err_msg=f"empty first shell {key}")
    cells = np.asarray(actual["model_state"]["cell_indices"])
    assert np.any(cells < 8 ** 3) and np.any(cells >= 8 ** 3 + 56 * 8)
    assert not np.any((cells >= 8 ** 3) & (cells < 8 ** 3 + 56 * 8))
    state, cells = checkpoint["model_state"], checkpoint["model_state"]["cell_indices"]
    selected = (cells >= 4 ** 3) & (cells < 4 ** 3 + 56)
    tiny_only = {key: values.reshape(len(cells), -1)[selected].ravel() for key, values in state.items()}
    try:
        refine_state_gpu({"model_metadata": checkpoint["model_metadata"], "model_state": tiny_only},
                         CoarseToFineConfig())
    except NoRepresentableChildren:
        pass
    else:
        raise AssertionError("A nonempty whole world without fitted children must preserve the old grid")
    for refinement in ("fixed", "uniform"):
        for start, shells in ((4, 5), (256, 8)):
            current = fixture(start, shells, refinement)
            current["model_metadata"]["active_sh_count"] = 1
            background_count = 56 * shells
            for subdivision, resolution in enumerate((start * 2, start * 4), 1):
                if refinement == "uniform":
                    background_count *= 8
                requests = []
                expected = refine_state(current, CoarseToFineConfig())
                actual = refine_state_gpu(current, CoarseToFineConfig(),
                                          before_allocate=lambda request: requests.append(request))
                assert expected["model_metadata"] == actual["model_metadata"]
                assert len(requests) == 2
                assert [request["stage"] for request in requests] == ["world_core_appearance", "world_model"]
                for request in requests:
                    assert request["background_child_count"] == background_count
                    assert request["active_voxels"] == len(actual["model_state"]["cell_indices"])
                    assert request["model_metadata"].get("background_subdivision_level", 0) == (
                        subdivision if refinement == "uniform" else 0)
                np.testing.assert_array_equal(np.asarray(actual["refinement_background_rows"]),
                                              expected["refinement_background_rows"])
                for key, values in expected["model_state"].items():
                    if key == "cell_indices":
                        np.testing.assert_array_equal(np.asarray(actual["model_state"][key]), values)
                    else:
                        np.testing.assert_allclose(np.asarray(actual["model_state"][key]), values,
                                                   atol=3e-7, rtol=4e-6, err_msg=key)
                model = restore_model(actual, mi)
                assert model.core_resolution == resolution
                assert np.count_nonzero(np.asarray(model.snapshot()["cell_indices"]) >= resolution**3) == background_count
                current = expected

    # Learned background centers/rotations and physical scales must reach the
    # same ellipsoid fitter as foreground, at both coarse and production depths.
    for start in (4, 256):
        current = sparse_ellipsoid_fixture(start)
        current["model_metadata"]["active_sh_count"] = 1
        config = CoarseToFineConfig(opacity_optical_depth_scale=.37)
        for resolution in (start * 2, start * 4):
            requests = []
            expected = refine_state(current, config)
            actual = refine_state_gpu(current, config, before_allocate=requests.append)
            assert expected["model_metadata"] == actual["model_metadata"]
            stages = [request["stage"] for request in requests]
            assert stages == ["world_background_appearance"] * 3 + ["world_core_appearance", "world_model"]
            for request in requests[:3]:
                assert not request["exact_count"]
                assert request["allocation_voxels"] <= 8 * request["background_parent_count"]
                assert request["active_voxels"] == (request["existing_background_child_count"]
                                                     + request["allocation_voxels"])
            bg_count = np.count_nonzero(expected["model_state"]["cell_indices"] >= resolution ** 3)
            for request in requests[-2:]:
                assert request["existing_background_child_count"] == bg_count
                assert request["active_voxels"] == len(actual["model_state"]["cell_indices"])
            for key, values in expected["model_state"].items():
                if key == "cell_indices":
                    np.testing.assert_array_equal(np.asarray(actual["model_state"][key]), values)
                else:
                    np.testing.assert_allclose(np.asarray(actual["model_state"][key]), values,
                                               atol=4e-7, rtol=5e-6, err_msg=f"ellipsoid {key}")
            model = restore_model(actual, mi)
            assert model.core_resolution == resolution
            assert model.native_background_use_ellipsoid()
            assert model.background_geometry == "ellipsoid"
            current = expected

    # Exercise both topology lifecycle paths, with real nonzero moment state.
    for refinement, geometry in (("fixed", None), ("uniform", None),
                                 ("fixed", "ellipsoid"), ("uniform", "ellipsoid"),
                                 ("uniform", "high_address")):
        original = (high_address_fixture() if geometry == "high_address" else
            sparse_ellipsoid_fixture(refinement=refinement) if geometry else fixture(refinement=refinement))
        start = original["model_metadata"]["resolution"]
        for gpu_topology in (False, True):
            model = restore_model(original, mi)
            optimizer = FalcorOptimizer(model, radiance_degree_steps=True)
            saved = optimizer.state_dict()
            for name, group in saved["groups"].items():
                for key, values in group.items():
                    values[:] = (7 if key == "steps" else .2 if key == "mean" else .04)
            optimizer.load_state_dict(saved)
            settings = TopologySettings(enable_growth=False, enable_deletion=False,
                use_rgb_growth=True, use_rgb_deletion=True, gpu_topology=gpu_topology)
            schedule = CoarseToFineConfig(start_resolution=start, target_resolution=start * 4,
                total_epochs=3, level_weights=(1, 1, 1), save_each_level=False)
            manager = TopologyManager(model, 3, settings, schedule)
            old_bg = np.flatnonzero(original["model_state"]["cell_indices"] >= start ** 3)
            bg_count = len(old_bg)
            for epoch, resolution in ((1, start * 2), (2, start * 4)):
                replacement, event = manager.finish_epoch(model, optimizer, epoch)
                assert event["refined"]
                if gpu_topology:
                    assert event["refinement_backend"] == "cuda"
                bg_count = np.count_nonzero(np.asarray(replacement["model_state"]["cell_indices"]) >= resolution ** 3)
                restored = optimizer.state_dict()
                for name, group in saved["groups"].items():
                    for key, values in group.items():
                        width = len(values) // saved["count"]
                        if refinement == "uniform":
                            np.testing.assert_array_equal(restored["groups"][name][key], 0)
                        else:
                            expected = values.reshape(saved["count"], width)[old_bg].ravel()
                            np.testing.assert_array_equal(restored["groups"][name][key][:bg_count * width], expected)
                            np.testing.assert_array_equal(restored["groups"][name][key][bg_count * width:], 0)
                model = restore_model(replacement, mi)
                assert model.core_resolution == resolution
                assert model.native_background_shell_levels() == original["model_metadata"]["background_shell_levels"]
                assert np.count_nonzero(np.asarray(model.snapshot()["cell_indices"]) >= resolution**3) == bg_count
    dr.sync_thread()
    print("PASS: two CUDA fits match NumPy, including learned shell ellipsoids; fixed shells retain Adam history, uniform children reset it")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="world-shell-refinement-") as directory:
        mi, dr, np, handles = initialize_runtime(RuntimeConfig(), Path(directory), {})
        run(mi, dr, np)
