"""Small CUDA evaluation of all synthetic/COLMAP test views and real PNG scores.

Generated 16x12 fixtures only; no training or existing experiment is changed.
"""

from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from PIL import Image

import evaluate
from preview import DataConfig, RenderConfig, RuntimeConfig
from reconstruction_checkpoint import save_checkpoint
from reconstruction_dataset import prepare_colmap
from test_reconstruction_evaluation_datasets import synthetic_fixture, colmap_fixture


def run(root, runtime):
    checks = []
    for kind in ("synthetic", "colmap"):
        dataset = root / ("synthetic" if kind == "synthetic" else "bonsai")
        data = DataConfig(dataset=str(dataset), reference_compositing="srgb")
        if kind == "synthetic":
            synthetic_fixture(dataset)
            expected_test = list(range(23))
        else:
            colmap_fixture(dataset, count=97)
            data.images = "images_2"
            data.initialization = prepare_colmap(dataset, images_dir=data.images,
                                                load_points=False)["initialization"]
            expected_test = list(range(0, 97, 8))
        model = root / f"{kind}-model.npz"
        state = {
            "cell_indices": np.array([0], np.uint32),
            "centers": np.array([.5, .5, .5], np.float32),
            "log_scales": np.log(np.array([.6, .7, .8], np.float32)),
            "rotations": np.array([1., 0., 0., 0.], np.float32),
            "radiance_sh": np.array([.4, .7, .2], np.float32),
            "opacity_sh": np.array([1.], np.float32),
        }
        metadata = dict(resolution=1, grid_min=[-1.] * 3, grid_max=[1.] * 3,
                        sh_count=1, opacity_sh_count=1)
        save_checkpoint(model, state, metadata,
                        config={"data": asdict(data), "render": asdict(RenderConfig(width=16, height=12))})
        output = root / f"{kind}-evaluation"
        config = evaluate.EvaluationConfig(checkpoint=str(model), output_directory=str(output),
                                           spp=1, runtime=runtime)
        assert evaluate.run_evaluation(config) == 0
        report = json.loads((output / "evaluation-report.json").read_text(encoding="utf-8"))
        assert report["actual_variant"] == "cuda_ad_rgb"
        assert report["metric_protocol"]["primary_metric"] == "psnr_srgb_png"
        assert [view["index"] for view in report["splits"]["test"]["views"]] == expected_test
        assert report["splits"]["train"]["evaluated_view_count"] == 10
        for split in ("train", "test"):
            records = report["splits"][split]["views"]
            scores = []
            for record in records:
                with Image.open(record["prediction"]) as png:
                    prediction = np.array(png, np.float64) / 255
                    assert png.mode == "RGB" and png.size == (16, 12)
                with Image.open(record["reference_png"]) as png:
                    reference = np.array(png, np.float64) / 255
                if kind == "colmap":
                    with Image.open(record["reference"]) as source:
                        np.testing.assert_array_equal(reference * 255, np.array(source, np.float64))
                mse = float(np.mean((prediction - reference) ** 2))
                score = -10 * np.log10(mse)
                np.testing.assert_allclose(record["psnr_srgb_png"], score, atol=1e-12, rtol=0)
                scores.append(score)
            np.testing.assert_allclose(report["splits"][split]["aggregate"]["psnr_srgb_png"],
                                       np.mean(scores), atol=1e-12, rtol=0)
        checks.append({"dataset": kind, "train_views": 10, "test_views": len(expected_test),
                       "variant": report["actual_variant"], "saved_png_metric_recomputed": "passed"})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="mitsuba-png-evaluation-") as temporary:
        result = run(Path(temporary), RuntimeConfig(build_dir=args.build_dir, gpu=args.gpu))
    print(json.dumps({"status": "passed", "checks": result}), flush=True)


if __name__ == "__main__":
    main()
