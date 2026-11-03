"""CPU-only mixed-depth addressing, camera preparation and selective fitting."""
from pathlib import Path
import copy
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconstruction_world_shell import (ADAPTIVE_TAG, adaptive_decode_numpy, adaptive_encode_numpy,
    adaptive_intervals_numpy, leaf_boxes_numpy, validate_adaptive_cells_numpy, world_shell_metadata_layout)
from reconstruction_adaptive_background import (prepare_adaptive_background, prepare_projection_cameras,
    projected_leaf_pixels, select_adaptive_background, refine_adaptive_background_numpy,
    adaptive_child_coverage_counts, cached_leaf_projection, prune_projection_cache)
from reconstruction_topology import CoarseToFineConfig, refine_state
from reconstruction_checkpoint import save_checkpoint, load_checkpoint, validate_model_state
from reconstruction_export import ArraySource


def cameras():
    return [dict(position=[0., 0., 0.], right=[1., 0., 0.], up=[0., 1., 0.], forward=[0., 0., 1.],
                 fx=32., fy=32., cx=32., cy=32., width=64, height=64)]


def fixture():
    cells = np.r_[np.uint64(21), adaptive_encode_numpy([3, 63, 64], [1, 2, 9], [3, 31, 2**27-1], np)]
    metadata = dict(resolution=4, core_resolution=4, grid_min=[-1.]*3, grid_max=[1.]*3,
        sh_count=9, active_sh_count=1, opacity_sh_count=1, scene_mapping="world_shell",
        background_shell_levels=2, background_subdivision_level=0, background_geometry="ellipsoid",
        background_refinement="adaptive", world_shell_version=5)
    low, high = leaf_boxes_numpy(cells, 4, [-1.]*3, [1.]*3, 2, np)
    rng = np.random.default_rng(183)
    count = len(cells)
    state = dict(cell_indices=cells, centers=np.full(count*3, .5, np.float32),
        log_scales=np.log((high-low)*.9).astype(np.float32).ravel(),
        rotations=np.tile(np.array([1.,0,0,0], np.float32), count),
        radiance_sh=rng.normal(size=count*27).astype(np.float32), opacity_sh=np.zeros(count, np.float32))
    return dict(model_metadata=metadata, model_state=state)


class AdaptiveAddressTests(unittest.TestCase):
    def test_tagged_mixed_depth_boxes_independent_of_core_resolution(self):
        cells = adaptive_encode_numpy([0, 55, 559], [0, 4, 9], [0, 4095, 2**27-1], np)
        coarse, depth, morton = adaptive_decode_numpy(cells, 10, np)
        np.testing.assert_array_equal(coarse, [0, 55, 559])
        np.testing.assert_array_equal(depth, [0, 4, 9])
        np.testing.assert_array_equal(morton, [0, 4095, 2**27-1])
        low, high = leaf_boxes_numpy(cells, 4, [-1.]*3, [1.]*3, 10, np)
        np.testing.assert_array_equal(low, [[-2.]*3, [1.9375]*3, [1023.]*3])
        np.testing.assert_array_equal(high, [[-1.]*3, [2.]*3, [1024.]*3])
        for n in (256, 512, 1024):
            for actual, expected in zip(leaf_boxes_numpy(cells, n, [-1.]*3, [1.]*3, 10, np), (low, high)):
                np.testing.assert_array_equal(actual, expected)

    def test_invalid_addresses_and_parent_child_overlap(self):
        parent = adaptive_encode_numpy([17], [2], [31], np)
        child = adaptive_encode_numpy([17], [3], [31*8+4], np)
        sibling = adaptive_encode_numpy([17], [2], [30], np)
        validate_adaptive_cells_numpy(np.r_[np.uint64(7), parent, sibling], 4, 1, np)
        for cells, expression in ((np.r_[parent, child], "overlap"), (np.r_[parent, parent], "unique"),
                                  (np.array([64],np.uint64), "legacy")):
            with self.assertRaisesRegex(ValueError, expression):
                validate_adaptive_cells_numpy(cells, 4, 1, np)
        for invalid in (ADAPTIVE_TAG, ADAPTIVE_TAG|2, ADAPTIVE_TAG|(1<<30),
                        ADAPTIVE_TAG|(56<<32)|1, 2**32):
            with self.assertRaises(ValueError):
                adaptive_decode_numpy(np.array([invalid], np.uint64), 1, np)
        for metadata in (dict(background_refinement="adaptive", world_shell_version=4, background_geometry="ellipsoid"),
                         dict(background_refinement="adaptive", world_shell_version=5, background_geometry="ellipsoid", background_subdivision_level=1)):
            with self.assertRaises(ValueError):
                world_shell_metadata_layout(metadata)


class AdaptivePreparationTests(unittest.TestCase):
    def test_core_mode_initialization_reaches_foreground_size_in_every_shell(self):
        with patch("reconstruction_adaptive_background.projected_leaf_pixels",
                   side_effect=lambda cells, *args: np.full(len(cells), 100., np.float32)):
            data = prepare_adaptive_background(np.empty((0,3)), np.empty((0,3)),
                4, [-1.]*3, [1.]*3, 2, cameras(), np, initial_leaf_budget=5000,
                pixel_threshold=8., resolution_mode="core", max_resolution=1024)
        coarse, depth, _ = adaptive_decode_numpy(data["cell_indices"], 2, np)
        self.assertEqual(len(data["cell_indices"]), 56*8 + 56*64)
        np.testing.assert_array_equal(depth[coarse < 56], 1)
        np.testing.assert_array_equal(depth[coarse >= 56], 2)
        low, high = leaf_boxes_numpy(data["cell_indices"], 4, [-1.]*3, [1.]*3, 2, np)
        np.testing.assert_array_equal(high-low, .5)
        self.assertEqual(data["background_resolution_mode"], "core")
        self.assertEqual(data["background_max_resolution"], 1024)

    def test_core_mode_all_ten_shells_unlock_depth_18_at_1024(self):
        coarse = np.arange(10)*56 + 45
        cells = adaptive_encode_numpy(coarse, np.arange(10)+8, 0, np)
        metadata = dict(fixture()["model_metadata"], resolution=512, core_resolution=512,
            background_shell_levels=10, world_shell_version=7, background_max_depth=17,
            background_resolution_mode="core", background_max_resolution=1024)
        state = dict(cell_indices=cells)
        projector = lambda low, high, cameras: np.full(len(low), 100., np.float32)
        checkpoint = dict(model_metadata=metadata, model_state=state)
        self.assertEqual(len(select_adaptive_background(checkpoint, cameras(), np.ones(10), np,
                                                        projector=projector)), 0)
        metadata.update(resolution=1024, core_resolution=1024, background_max_depth=18)
        selected = select_adaptive_background(checkpoint, cameras(), np.ones(10), np,
                                              projector=projector)
        np.testing.assert_array_equal(np.sort(selected), np.arange(10))
        state["cell_indices"] = adaptive_encode_numpy(coarse, np.arange(10)+9, 0, np)
        self.assertEqual(len(select_adaptive_background(checkpoint, cameras(), np.ones(10), np,
                                                        projector=projector)), 0)

    def test_core_parent_initialization_has_one_world_size_floor_across_shells(self):
        # N=4 has core edge .5; all shell leaves may become edge 1, not .5.
        # Making projection demand uniform refines only the coarser outer shell.
        with patch("reconstruction_adaptive_background.projected_leaf_pixels",
                   side_effect=lambda cells, *args: np.full(len(cells), 100., np.float32)):
            data = prepare_adaptive_background(np.empty((0,3)), np.empty((0,3)),
                4, [-1.]*3, [1.]*3, 2, cameras(), np, initial_leaf_budget=1000,
                pixel_threshold=8., resolution_mode="core_parent", max_resolution=512)
        coarse, depth, _ = adaptive_decode_numpy(data["cell_indices"], 2, np)
        self.assertEqual(len(data["cell_indices"]), 56 + 56*8)
        np.testing.assert_array_equal(depth[coarse < 56], 0)
        np.testing.assert_array_equal(depth[coarse >= 56], 1)
        low, high = leaf_boxes_numpy(data["cell_indices"], 4, [-1.]*3, [1.]*3, 2, np)
        np.testing.assert_array_equal(high-low, 1.)
        self.assertEqual(data["background_resolution_mode"], "core_parent")
        self.assertEqual(data["max_depth"], 1)

    def test_core_refinement_unlocks_each_shell_without_a_scalar_depth_cap(self):
        # All five shells are at their common world floor with a 256 core.
        # A 512 core permits one more split in every shell, even depths > 9.
        cells = adaptive_encode_numpy([45, 101, 157, 213, 269], [6, 7, 8, 9, 10], 0, np)
        metadata = dict(fixture()["model_metadata"], resolution=256, core_resolution=256,
            background_shell_levels=5, world_shell_version=6, background_max_depth=10,
            background_resolution_mode="core_parent", background_max_resolution=512)
        state = dict(cell_indices=cells)
        projector = lambda low, high, cameras: np.full(len(low), 100., np.float32)
        checkpoint = dict(model_metadata=metadata, model_state=state)
        selected = select_adaptive_background(checkpoint, cameras(), np.ones(5), np,
                                              projector=projector)
        self.assertEqual(len(selected), 0)
        metadata.update(resolution=512, core_resolution=512, background_max_depth=11)
        selected = select_adaptive_background(checkpoint, cameras(), np.ones(5), np,
                                              projector=projector)
        self.assertEqual(set(selected.tolist()), set(range(5)))
        metadata.update(resolution=1024, core_resolution=1024, background_max_depth=12)
        state["cell_indices"] = adaptive_encode_numpy([45, 101, 157, 213, 269], [8, 9, 10, 11, 12], 0, np)
        selected = select_adaptive_background(checkpoint, cameras(), np.ones(5), np,
                                              projector=projector)
        self.assertEqual(len(selected), 0)

    def prepare(self, positions=(), colors=(), budget=252):
        return prepare_adaptive_background(np.asarray(positions).reshape(-1,3), np.asarray(colors).reshape(-1,3),
            4, [-1.]*3, [1.]*3, 2, cameras(), np, initial_leaf_budget=budget, pixel_threshold=8, max_depth=3)

    def test_empty_point_cloud_still_covers_all_shell_space_under_budget(self):
        data = self.prepare()
        cells = data["cell_indices"]
        self.assertGreater(len(cells), 112)
        self.assertLessEqual(len(cells), 252)
        self.assertEqual((len(cells)-112)%7, 0)
        low, high = leaf_boxes_numpy(cells, 4, [-1.]*3, [1.]*3, 2, np)
        self.assertEqual(float(np.prod(high-low, axis=1).sum()), 8**3-2**3)
        np.testing.assert_array_equal(data["mean_linear_colors"], .5)
        np.testing.assert_array_equal(data["point_counts"], 0)
        self.assertGreater(data["actual_max_depth"], 0)

    def test_empty_leaf_native_initialization_uses_positive_gray_dc(self):
        from test_reconstruction_background import BackgroundPointColorTests
        from reconstruction_background import initialize_background_model
        seed, mi, dr, original = BackgroundPointColorTests().runtime()
        prepared = self.prepare()
        model = initialize_background_model(seed,mi,dr,np,shell_levels=2,
            background_refinement="adaptive",adaptive_background=prepared)
        coefficients = model.snapshot()["radiance_sh"].reshape(-1,9,3)[1:]
        np.testing.assert_allclose(coefficients[:,0], .5/.28209479177387814, rtol=1e-7)
        np.testing.assert_array_equal(coefficients[:,1:], 0.)
        self.assertEqual(model.background_initialization["empty_leaf_color"],[.5,.5,.5])
        self.assertEqual(model.background_max_depth, prepared["max_depth"])
        self.assertEqual(prepared["background_max_depth"], prepared["max_depth"])

    def test_points_only_change_color_and_half_open_ownership(self):
        points = [[-1,0,0],[-2,0,0],[1,0,0],[2,0,0],[-4,0,0],[4,0,0]]
        data = self.prepare(points, [[.5,.25,.75]]*6)
        np.testing.assert_array_equal(data["cell_indices"], self.prepare()["cell_indices"])
        self.assertEqual(int(data["point_counts"].sum()), 4)
        from reconstruction_color import srgb_to_linear
        occupied = data["point_counts"]>0
        np.testing.assert_allclose(data["mean_linear_colors"][occupied],
            np.broadcast_to(srgb_to_linear(np.array([.5,.25,.75]),np), (occupied.sum(),3)), atol=1e-7)

    def test_selection_requires_projection_and_observed_priority(self):
        data = self.prepare(budget=112)
        metadata = fixture()["model_metadata"]
        cells = data["cell_indices"]
        scores = projected_leaf_pixels(cells, metadata, prepare_projection_cameras(cameras(),np), np)
        visible = np.flatnonzero(scores>8)
        signal = np.zeros(len(cells))
        signal[visible[:2]] = [1.,100.]
        checkpoint = dict(model_metadata=metadata,model_state=dict(cell_indices=cells))
        selected = select_adaptive_background(checkpoint,cameras(),signal,np,pixel_threshold=8,max_new_leaves=7,max_depth=3)
        np.testing.assert_array_equal(selected,[visible[1]])
        self.assertEqual(len(select_adaptive_background(checkpoint,cameras(),signal*0,np)),0)

    def test_selection_uses_real_child_coverage_cost_after_projection_gate(self):
        metadata = fixture()["model_metadata"]
        cells = adaptive_encode_numpy([45,46],0,0,np)
        low,high = leaf_boxes_numpy(cells,4,[-1.]*3,[1.]*3,2,np)
        state = dict(cell_indices=cells, centers=np.array([[.5,.5,.5],[.25,.25,.25]],np.float32).ravel(),
            log_scales=np.log((high-low)*np.array([[.9],[.1]])).astype(np.float32).ravel(),
            rotations=np.tile(np.array([1.,0,0,0],np.float32),2))
        np.testing.assert_array_equal(adaptive_child_coverage_counts(state,metadata,np),[8,1])
        selected = select_adaptive_background(dict(model_metadata=metadata,model_state=state),
            cameras(),np.array([1.,1.]),np,max_new_leaves=7,pixel_threshold=1,max_depth=3)
        np.testing.assert_array_equal(selected,[1])

    def test_all_views_include_a_leaf_visible_only_in_omitted_view_33(self):
        views=[dict(cameras()[0],forward=[0.,0.,-1.]) for _ in range(65)]
        views[32]["forward"]=[0.,0.,1.]
        cells=adaptive_encode_numpy([45],0,0,np)
        metadata=fixture()["model_metadata"]
        sampled=projected_leaf_pixels(cells,metadata,prepare_projection_cameras(views,np,32),np)
        complete=projected_leaf_pixels(cells,metadata,prepare_projection_cameras(views,np),np)
        np.testing.assert_array_equal(sampled,0.)
        self.assertGreater(complete[0],8.)
        selected=select_adaptive_background(dict(model_metadata=metadata,model_state=dict(cell_indices=cells)),
            views,np.ones(1),np,max_new_leaves=7,pixel_threshold=8)
        np.testing.assert_array_equal(selected,[0])
        data=prepare_adaptive_background(np.empty((0,3)),np.empty((0,3)),4,[-1.]*3,[1.]*3,
            1,views,np,initial_leaf_budget=63,pixel_threshold=8,max_depth=1)
        self.assertEqual(data["camera_sample_count"],65)
        self.assertEqual(len(data["cell_indices"]),63)

    def test_sorted_array_cache_reuses_ids_through_core_refinement_and_prunes(self):
        cells=adaptive_encode_numpy([45,46],0,0,np)
        metadata=fixture()["model_metadata"]
        prepared=prepare_projection_cameras(cameras(),np)
        cache={}
        with patch("reconstruction_adaptive_background.projected_leaf_pixels",wraps=projected_leaf_pixels) as projected:
            first=cached_leaf_projection(cells,metadata,prepared,np,cache)
            np.testing.assert_array_equal(cached_leaf_projection(cells[::-1],metadata,prepared,np,cache),first[::-1])
            np.testing.assert_array_equal(cached_leaf_projection(cells,dict(metadata,resolution=8),prepared,np,cache),first)
            self.assertEqual(projected.call_count,1)
            self.assertEqual(cache["cells"].dtype,np.dtype(np.uint64))
            self.assertEqual(cache["pixels"].dtype,np.dtype(np.float32))
            prune_projection_cache(cache,cells[1:],np)
            np.testing.assert_array_equal(cache["cells"],cells[1:])
            changed=prepare_projection_cameras([dict(cameras()[0],fx=64.)],np)
            cached_leaf_projection(cells[1:],metadata,changed,np,cache)
            self.assertEqual(projected.call_count,2)

    def test_small_dynamic_ellipsoid_fails_projection_even_with_cached_large_leaf(self):
        metadata=fixture()["model_metadata"]
        state=dict(cell_indices=adaptive_encode_numpy([45],0,0,np),
            centers=np.full(3,.5,np.float32),log_scales=np.full(3,np.log(.0001),np.float32),
            rotations=np.array([1.,0,0,0],np.float32))
        checkpoint=dict(model_metadata=metadata,model_state=state)
        cache={}
        self.assertEqual(len(select_adaptive_background(checkpoint,cameras(),np.ones(1),np,
            max_new_leaves=7,pixel_threshold=8,projection_cache=cache)),0)
        self.assertGreater(cache["pixels"][0],8.)
        state["log_scales"][:]=np.log(.9)
        selected=select_adaptive_background(checkpoint,cameras(),np.ones(1),np,
            max_new_leaves=7,pixel_threshold=8,projection_cache=cache)
        np.testing.assert_array_equal(selected,[0])

    def test_projector_callback_preserves_selection_and_only_caches_static_boxes(self):
        from reconstruction_adaptive_background import projected_boxes_pixels
        metadata=fixture()["model_metadata"]
        state=dict(cell_indices=adaptive_encode_numpy([45],0,0,np),
            centers=np.full(3,.5,np.float32),log_scales=np.full(3,np.log(.9),np.float32),
            rotations=np.array([1.,0,0,0],np.float32))
        payload=dict(model_metadata=metadata,model_state=state)
        calls=[]
        def projector(low,high,prepared):
            calls.append((low.copy(),high.copy(),len(prepared)))
            return projected_boxes_pixels(low,high,prepared,np)
        cache={}
        for iteration in range(2):
            actual=select_adaptive_background(payload,cameras(),np.ones(1),np,max_new_leaves=7,
                pixel_threshold=8,projection_cache=cache,projector=projector)
            expected=select_adaptive_background(payload,cameras(),np.ones(1),np,max_new_leaves=7,pixel_threshold=8)
            np.testing.assert_array_equal(actual,expected)
            self.assertEqual(len(calls),2+iteration)  # leaf once; learned geometry each round
        self.assertTrue(all(count==1 for _,_,count in calls))


class AdaptiveRefinementTests(unittest.TestCase):
    def test_coverage_counts_real_ellipsoid_box_contact_not_child_centers(self):
        source = fixture()
        state,metadata = source["model_state"],source["model_metadata"]
        low,high = leaf_boxes_numpy(state["cell_indices"],4,[-1.]*3,[1.]*3,2,np)
        state["centers"].reshape(4,3)[:] = [[.25,.25,.25],[.5,.25,.25],[.5,.5,.5],[2.,2.,2.]]
        state["log_scales"] = np.log((high-low)*.02).astype(np.float32).ravel()
        np.testing.assert_array_equal(adaptive_child_coverage_counts(state,metadata,np),[1,2,8,0])

    def test_selected_parent_only_and_exact_row_mapping_and_sh_inheritance(self):
        source = fixture()
        saved = copy.deepcopy(source)
        result = refine_adaptive_background_numpy(source,np.array([1],np.uint32),CoarseToFineConfig(),100,refine_state,np)
        self.assertEqual((result["selected"],result["replaced"],result["children"],result["added"]),(1,1,8,7))
        np.testing.assert_array_equal(result["source_rows"][:3],[0,2,3])
        np.testing.assert_array_equal(result["source_rows"][3:],-1)
        for name,value in source["model_state"].items():
            np.testing.assert_array_equal(value,saved["model_state"][name])
            np.testing.assert_array_equal(result["model_state"][name].reshape(11,-1)[:3],value.reshape(4,-1)[[0,2,3]])
        colors = result["model_state"]["radiance_sh"].reshape(11,27)[3:]
        np.testing.assert_array_equal(colors,np.tile(source["model_state"]["radiance_sh"].reshape(4,27)[1],(8,1)))
        self.assertEqual(result["model_metadata"], source["model_metadata"])
        validate_model_state(result["model_state"],result["model_metadata"])

    def test_unrepresentable_parent_retains_geometry_and_adam_mapping(self):
        source = fixture()
        source["model_state"]["log_scales"].reshape(4,3)[1] = np.log(1e-8)
        result = refine_adaptive_background_numpy(source,np.array([1],np.uint32),CoarseToFineConfig(),4,refine_state,np)
        self.assertEqual((result["replaced"],result["added"]),(0,0))
        np.testing.assert_array_equal(result["source_rows"],np.arange(4))
        for name,value in source["model_state"].items():
            np.testing.assert_array_equal(result["model_state"][name],value)

    def test_empty_parent_inside_nonempty_fitting_group_is_not_lost(self):
        source = fixture()
        source["model_state"]["cell_indices"][2] = adaptive_encode_numpy(3,1,4,np)
        source["model_state"]["log_scales"].reshape(4,3)[2] = np.log(1e-8)
        result = refine_adaptive_background_numpy(source,np.array([1,2],np.uint32),CoarseToFineConfig(),100,refine_state,np)
        self.assertEqual((result["selected"],result["replaced"],result["children"],result["added"]),(2,1,8,7))
        np.testing.assert_array_equal(result["source_rows"][:3],[0,2,3])
        for name,value in source["model_state"].items():
            np.testing.assert_array_equal(result["model_state"][name].reshape(11,-1)[1],value.reshape(4,-1)[2])

    def test_capacity_refusal_and_foreground_refinement_preserve_background(self):
        source = fixture()
        with self.assertRaisesRegex(RuntimeError,"max_active_voxels"):
            refine_adaptive_background_numpy(source,np.array([1],np.uint32),CoarseToFineConfig(),4,refine_state,np)
        result = refine_state(source,CoarseToFineConfig(),100)
        self.assertEqual(result["model_metadata"]["resolution"],8)
        np.testing.assert_array_equal(result["refinement_background_rows"],[1,2,3])
        for name,value in source["model_state"].items():
            np.testing.assert_array_equal(result["model_state"][name].reshape(-1,value.size//4)[:3],value.reshape(4,-1)[1:])


class AdaptiveCheckpointTests(unittest.TestCase):
    def test_configured_depth_guards_metadata_leaves_and_configuration(self):
        source = fixture()
        state,metadata = source["model_state"],source["model_metadata"]
        validate_model_state(state, metadata)  # Missing legacy bound defaults to 9.
        for maximum in (-1,10,True,"8",8.5):
            with self.subTest(maximum=maximum), self.assertRaisesRegex(ValueError,"background_max_depth"):
                validate_model_state(state,dict(metadata,background_max_depth=maximum))
        with self.assertRaisesRegex(ValueError,"leaf depth"):
            validate_model_state(state,dict(metadata,background_max_depth=8))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"adaptive.npz"
            save_checkpoint(path,state,dict(metadata,background_max_depth=9),
                config={"data":{"background_max_resolution":1024}})
            original=path.read_bytes()
            with self.assertRaisesRegex(ValueError,"background_max_depth"):
                save_checkpoint(path,state,dict(metadata,background_max_depth=9),
                    config={"data":{"background_max_resolution":512}})
            streamed={name:ArraySource(value.shape,value.dtype,
                lambda start,count,value=value:value[start:start+count]) for name,value in state.items()}
            with patch("reconstruction_export.CHUNK_BYTES",8), self.assertRaisesRegex(ValueError,"leaf depth"):
                save_checkpoint(path,streamed,dict(metadata,background_max_depth=8))
            self.assertEqual(path.read_bytes(),original)

    def test_uint64_save_restore_and_cross_chunk_ancestor_overlap(self):
        source = fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"adaptive.npz"
            def streamed(state):
                return {name:ArraySource(value.shape,value.dtype,
                    lambda start,count,value=value:value[start:start+count]) for name,value in state.items()}
            with patch("reconstruction_export.CHUNK_BYTES",8):
                save_checkpoint(path, model_metadata=source["model_metadata"],model_state=streamed(source["model_state"]))
            loaded = load_checkpoint(path)
            for name,value in source["model_state"].items():
                np.testing.assert_array_equal(loaded["model_state"][name],value)
            original = path.read_bytes()
            coarse,depth,morton = adaptive_decode_numpy(source["model_state"]["cell_indices"][1:2],2,np)
            source["model_state"]["cell_indices"][3] = adaptive_encode_numpy(coarse,depth+1,morton*8,np)[0]
            with patch("reconstruction_export.CHUNK_BYTES",8), self.assertRaisesRegex(ValueError,"overlap"):
                save_checkpoint(path,model_metadata=source["model_metadata"],model_state=streamed(source["model_state"]))
            self.assertEqual(path.read_bytes(),original)
            self.assertEqual([item.name for item in Path(directory).iterdir()],[path.name])


if __name__ == "__main__":
    unittest.main(verbosity=2)
