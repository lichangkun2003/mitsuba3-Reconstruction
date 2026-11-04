"""Small CUDA end-to-end test of adaptive shells, core refinement and resume."""

from pathlib import Path
import argparse
import json
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_reconstruction_colmap_training import write_scene
from reconstruction_checkpoint import load_checkpoint
from reconstruction_world_shell import ADAPTIVE_TAG, WORLD_SHELL_VERSION, validate_adaptive_cells_numpy


def run(output):
    import numpy as np
    dataset = output / "dataset"
    write_scene(dataset, np)
    full = output / "full"
    config = {
        "data": {"dataset": str(dataset), "scene_mode": "unbounded", "background_shell_levels": 2,
            "background_refinement": "adaptive", "background_initial_resolution": 2,
            "background_initial_leaf_budget": 500, "background_initial_pixel_threshold": 8,
            "background_max_resolution": 16, "background_resolution_mode": "core"},
        "model": {"grid_resolution": 4, "sh_count": 9, "opacity_sh_count": 1},
        "render": {"max_hits": 32},
        "training": {"rgb_only": True, "num_epochs": 8, "output": str(full),
            "profile_views": 0, "profile_interval_epochs": 0, "gpu_reference_cache_mib": 0,
            "reference_cache_mib": 16, "sh_degree_progression": True, "sh_degree_weights": [8, 0, 2]},
        "schedule": {"enabled": True, "start_resolution": 4, "target_resolution": 8, "level_weights": [1, 1]},
        "topology": {"enable_growth": True, "use_rgb_growth": True, "use_growth_evidence": False,
            "growth_warmup_epochs": 0, "growth_fine_warmup_epochs": 0, "growth_freeze_last_epochs": 0,
            "growth_interval": 2, "growth_wait_epochs": 0, "growth_protection_epochs": 0,
            "enable_deletion": True, "use_rgb_deletion": True, "use_low_opacity_deletion": True,
            "deletion_warmup_epochs": 0, "deletion_protection_epochs": 0, "deletion_interval": 2,
            "deletion_opacity_threshold": .05, "background_refine_interval": 1,
            "background_refine_max_new_leaves": 140, "background_refine_pixel_threshold": 1,
            "background_refine_fraction": .1, "background_max_active_voxels": 3000,
            "background_growth_max_candidates": 1000},
        "gradient": {"dssim_loss_weight": .2, "distortion_loss_weight": .1},
        "evaluation": {"spp": 1, "train_max_views": 0, "test_max_views": 0}}
    path = output / "config.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    train = Path(__file__).resolve().parents[1] / "train.py"

    def launch(name, arguments):
        with (output / (name + ".log")).open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, "-u", "-I", str(train), *arguments], stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{name} failed: {output / (name + '.log')}")

    launch("full", ["--config", str(path)])
    final = load_checkpoint(full / "model.npz")
    metadata, state = final["model_metadata"], final["model_state"]
    assert metadata["background_refinement"] == "adaptive" and metadata["resolution"] == 8
    assert metadata["background_resolution_mode"] == "core"
    assert metadata["background_max_depth"] == 3
    assert metadata["world_shell_version"] == WORLD_SHELL_VERSION
    validate_adaptive_cells_numpy(state["cell_indices"], 8, 2, np)
    assert np.any(state["cell_indices"] >= ADAPTIVE_TAG)
    history = [json.loads(line) for line in (full / "training.jsonl").read_text(encoding="utf-8").splitlines()]
    assert any(item["topology"].get("background_refined") for item in history)
    assert any(item["topology"].get("refined") for item in history)
    report = json.loads((full / "training-report.json").read_text(encoding="utf-8"))
    assert report["training_status"] == "passed" and report["evaluation"]["status"] == "passed"

    resumed = output / "resumed"
    launch("resume", ["--resume", str(full / "before-refine-0004-epoch-00004.npz"), "--output-dir", str(resumed)])
    recovered = load_checkpoint(resumed / "model.npz")
    for name, value in state.items():
        np.testing.assert_allclose(recovered["model_state"][name], value, rtol=2e-5, atol=2e-5, err_msg=name)
    for group, values in final["optimizer_state"]["groups"].items():
        for name, value in values.items():
            np.testing.assert_allclose(recovered["optimizer_state"]["groups"][group][name], value,
                                       rtol=2e-5, atol=2e-5, err_msg=f"Adam {group}/{name}")
    return dict(status="passed", output=str(output), checks=["adaptive mixed leaves", "background-only splits",
        "foreground refinement", "RGB grow/prune integration", "all-view evaluation", "model and Adam resume"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="adaptive-training-", dir=root))
    print(output, flush=True)
    report = run(output)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
