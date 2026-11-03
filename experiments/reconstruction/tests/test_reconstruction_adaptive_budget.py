"""CPU checks for adaptive transition headroom and explicit launch limits."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruction_memory import plan_refinement_memory, estimate_spatial_index_bytes
from train import TrainConfig, validate


class BudgetTests(unittest.TestCase):
    def test_old_and_new_parameters_alone_do_not_cover_adam_remap(self):
        request = dict(stage="adaptive_background_commit", exact_count=True,
            active_voxels=4100000, old_voxels=4000000, allocation_voxels=4100000,
            existing_child_count=0, background_child_count=120000,
            model_metadata=dict(sh_count=9, opacity_sh_count=1))
        plan = plan_refinement_memory(request, gpu={"memory_mib": 32768},
            live_memory={"free_bytes": 2*1024**3, "total_bytes": 32*1024**3},
            device_reserve_mib=0, workspace_reserve_mib=1024)
        self.assertFalse(plan["supported"])
        self.assertGreater(plan["additional_required_bytes"], 2*1024**3)

    def test_mixed_tree_bound_allows_background_finer_than_small_core(self):
        self.assertGreater(estimate_spatial_index_bytes(10000, 4,
            background_shell_levels=5, background_adaptive=True), 0)

    def test_initial_budget_cannot_exceed_global_cap(self):
        config = TrainConfig()
        config.training.rgb_only = True
        config.data.scene_mode = "unbounded"
        config.data.background_refinement = "adaptive"
        config.topology.max_active_voxels = 100
        with self.assertRaisesRegex(ValueError, "total active leaf cap"):
            validate(config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
