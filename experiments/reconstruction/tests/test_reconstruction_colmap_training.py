"""Small CUDA regression for COLMAP initialization, refinement, resume and evaluation.

Generates a nine-view 16x12 dataset outside the Git checkout. No project dataset
is changed. Run with the source Python environment after building Mitsuba.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preview import DataConfig, RenderConfig, RuntimeConfig
from reconstruction_checkpoint import load_checkpoint
from reconstruction_common import REPOSITORY, WORKSPACE, load_views, workspace_path


def write_scene(dataset, np):
    from PIL import Image

    model = dataset / "sparse" / "0"
    model.mkdir(parents=True)
    images = dataset / "images"
    images.mkdir()
    with (model / "cameras.bin").open("wb") as stream:
        stream.write(struct.pack("<QiiQQdddd", 1, 13, 1, 32, 24, 14., 17., 17.5, 10.5))
    with (model / "images.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", 9))
        for index in (8, 3, 0, 7, 1, 6, 2, 5, 4):
            center_x = (index - 4) * .15
            # Identity COLMAP rotation: camera looks along +Z from z=-2.
            stream.write(struct.pack("<idddddddi", 100 + index, 1., 0., 0., 0.,
                                     -center_x, 0., 2., 13))
            stream.write(f"view-{index:02d}.png".encode("ascii") + b"\0")
            stream.write(struct.pack("<Q", 0))
            rgb = np.empty((12, 16, 3), np.uint8)
            rgb[:, :, 0] = np.arange(16, dtype=np.uint8)[None, :] * 9 + 30
            rgb[:, :, 1] = np.arange(12, dtype=np.uint8)[:, None] * 10 + index * 3
            rgb[:, :, 2] = 90 + index * 7
            Image.fromarray(rgb).save(images / f"view-{index:02d}.png")
    points = [(-.05, .02, -1.7), (.15, .04, -1.73), (0., 0., 4.), (.3, -.2, 6.)]
    with (model / "points3D.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", len(points)))
        for index, point in enumerate(points):
            stream.write(struct.pack("<QdddBBBdQ", 400 + index, *point, 150, 90, 180, .1, 0))
    return points


def run_check(args, output):
    import numpy as np
    from PIL import Image

    dataset = output / "dataset"
    write_scene(dataset, np)
    full, resumed = output / "full", output / "resumed"
    config = {
        "data": {"dataset": str(dataset), "background_final_layers": 4},
        "model": {"grid_resolution": 4, "opacity_sh_count": 1},
        "render": {"width": 93, "height": 97, "spp": 1, "max_hits": 2},
        "runtime": {"build_dir": args.build_dir, "gpu": args.gpu},
        "training": {"rgb_only": True, "num_epochs": 3, "chunk_size": 0,
                     "save_count": 3, "profile_views": 0, "profile_interval_epochs": 0,
                     "output": str(full)},
        "schedule": {"enabled": True, "start_resolution": 4, "target_resolution": 16,
                     "level_weights": [1, 1, 1]},
        "topology": {"enable_growth": False, "enable_deletion": False},
        "evaluation": {"spp": 1, "max_views": None},
    }
    config_path = output / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    entry = Path(__file__).resolve().parents[1] / "train.py"

    def run(command, name):
        result = subprocess.run(command, text=True, capture_output=True)
        (output / f"{name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {output / (name + '.log')}:\n"
                               f"{result.stdout}\n{result.stderr}")

    run([sys.executable, "-u", "-I", str(entry), "--config", str(config_path)], "full")
    first = load_checkpoint(full / "epoch-00001.npz")
    final = load_checkpoint(full / "model.npz")
    report = json.loads((full / "training-report.json").read_text(encoding="utf-8"))
    assert report["training_status"] == report["evaluation"]["status"] == "passed"
    assert report["initial_render"]["resolution"] == 6
    assert report["initial_render"]["view_index"] == 1  # Global index 0 belongs to the held-out split.
    assert (report["initial_render"]["width"], report["initial_render"]["height"]) == (16, 12)
    assert final["model_metadata"]["resolution"] == 24
    assert final["model_metadata"]["core_resolution"] == 16
    assert final["model_metadata"]["scene_mapping"] == "radial_seed_shell"
    assert "background_refinement" not in final["model_metadata"]
    assert "background_layers" not in final["model_metadata"]
    assert tuple(final["config"]["render"]["background"]) == (0., 0., 0.)
    assert first["training_state"]["global_step"] == 7
    assert final["training_state"]["global_step"] == 21
    assert final["training_state"]["completed_epochs"] == 3
    np.testing.assert_array_equal(final["model_metadata"]["grid_min"], [-1.5] * 3)
    np.testing.assert_array_equal(final["model_metadata"]["grid_max"], [1.5] * 3)
    initialization = final["config"]["data"]["initialization"]
    assert first["config"]["data"]["initialization"] == initialization
    np.testing.assert_allclose(initialization["center"], [0., 0., -2.], atol=1e-15)
    np.testing.assert_allclose(initialization["radius"], .66, atol=1e-15)
    assert initialization["shell_width"] == .5
    assert initialization["shell_layers"] == 4
    assert initialization["version"] == 4
    assert initialization["background_refinement"] == "preserve_physical_width"
    assert initialization["background_final_layers"] == 4
    assert initialization["initial_background_layers"] == 1
    assert initialization["background_opacity"] == .5
    assert not (dataset / "init_points.ply").exists()
    history = [json.loads(line) for line in (full / "training.jsonl").read_text().splitlines()]
    assert [row["training_resolution"] for row in history] == [6, 12, 24]
    assert history[0]["topology"]["refined"] is True
    assert history[1]["topology"]["refined"] is True
    assert final["model_state"]["radiance_sh"].size > 0
    assert not np.array_equal(first["model_state"]["radiance_sh"], final["model_state"]["radiance_sh"])

    evaluation = json.loads((full / "evaluation-report.json").read_text(encoding="utf-8"))
    assert evaluation["status"] == "passed"
    assert (evaluation["render"]["width"], evaluation["render"]["height"]) == (16, 12)
    for split, expected in (("train", list(range(1, 8))), ("test", [0, 8])):
        result = evaluation["splits"][split]
        assert result["aggregate"]["view_count"] == len(expected)
        assert [view["index"] for view in result["views"]] == expected
        for view in result["views"]:
            assert Path(view["reference"]).name == f"view-{view['index']:02d}.png"
            with Image.open(view["prediction"]) as image:
                assert image.size == (16, 12)
    data = DataConfig(**final["config"]["data"])
    render = RenderConfig(**final["config"]["render"])
    train_views = load_views(dataset, data.cameras, render, np, data=data)
    assert [view["index"] for view in train_views] == list(range(1, 8))
    for view in train_views:
        camera = view["camera"]
        np.testing.assert_allclose([camera[key] for key in ("fx", "fy", "cx", "cy")],
                                   [7., 8.5, 8.75, 5.25], atol=0, rtol=0)
        assert (camera["width"], camera["height"]) == (16, 12)

    run([sys.executable, "-u", "-I", str(entry), "--resume", str(full / "epoch-00001.npz"),
         "--output-dir", str(resumed), "--build-dir", args.build_dir, "--gpu", str(args.gpu)], "resume")
    recovered = load_checkpoint(resumed / "model.npz")
    assert recovered["model_metadata"] == final["model_metadata"]
    assert recovered["config"]["data"]["initialization"] == initialization
    assert recovered["training_state"]["rng_state"] == final["training_state"]["rng_state"]
    assert recovered["training_state"]["global_step"] == 21
    for key, value in final["model_state"].items():
        np.testing.assert_allclose(recovered["model_state"][key], value, atol=2e-6, rtol=2e-6,
                                   err_msg=f"resumed model {key}")
    for group, parameters in final["optimizer_state"]["groups"].items():
        for key, value in parameters.items():
            np.testing.assert_allclose(recovered["optimizer_state"]["groups"][group][key], value,
                                       atol=2e-6, rtol=2e-6, err_msg=f"resumed Adam {group}/{key}")
    resume_evaluation = json.loads((resumed / "evaluation-report.json").read_text(encoding="utf-8"))
    assert resume_evaluation["status"] == "passed"
    for split, count in (("train", 7), ("test", 2)):
        assert resume_evaluation["splits"][split]["aggregate"]["view_count"] == count
    return {"status": "passed", "output": str(output), "checks": [
        "COLMAP binary initialization without PLY", "fixed camera normalization and thin shell",
        "fixed physical shell: layers 1 -> 2 -> 4 and grid 6 -> 12 -> 24",
        "sorted filename split: 7 train / 2 test",
        "full scaled fx/fy/cx/cy and 16x12 reference alignment", "all-view train/test evaluation",
        "resume preserves model, Adam, RNG and initialization metadata"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/mitsuba/colmap-training-tests")
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    root = workspace_path(args.output_dir, WORKSPACE)
    if root.is_relative_to(REPOSITORY):
        raise ValueError("Test artifacts must be outside the Git checkout")
    root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="colmap-regression-", dir=root))
    print(f"COLMAP training regression: {output}", flush=True)
    try:
        report = run_check(args, output)
    except Exception:
        report = {"status": "failed", "output": str(output), "error": traceback.format_exc()}
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        raise
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
