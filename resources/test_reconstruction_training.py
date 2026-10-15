"""Small CUDA regression: gradients, resume and automatic train/test evaluation.

Only generated 16x16 images and point clouds are used. No project dataset is
rendered. Run explicitly with the source environment after compiling C++.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconstruction_common import (WORKSPACE, REPOSITORY, initialize_runtime, model_metadata,
                                   restore_model, workspace_path, render_view)
from reconstruction_preview import RuntimeConfig, RenderConfig
from reconstruction_checkpoint import load_checkpoint, save_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/mitsuba/training-tests")
    parser.add_argument("--build-dir", default=RuntimeConfig().build_dir)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    output = workspace_path(args.output_dir, WORKSPACE)
    if output.is_relative_to(REPOSITORY):
        raise ValueError("Test artifacts must be outside the Git checkout")
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "checks": []}
    try:
        mi, dr, np, handles = initialize_runtime(RuntimeConfig(args.build_dir, args.gpu), output, report)
        y00 = 1 / (2 * math.sqrt(math.pi))
        state = {"cell_indices": np.array([0], np.uint32),
                 "centers": np.array([.5, .5, .5], np.float32),
                 "log_scales": np.log(np.array([.35, .27, .3], np.float32)),
                 "rotations": np.array([1, 0, 0, 0], np.float32),
                 "radiance_sh": np.zeros(27, np.float32),
                 "opacity_sh": np.zeros(9, np.float32)}
        state["radiance_sh"][:3] = np.array([.2, .3, .4]) / y00
        state["opacity_sh"][0] = .3 / y00
        payload = {"model_state": state, "model_metadata": {"resolution": 1,
                   "grid_min": [0., 0., 0.], "grid_max": [1., 1., 1.], "sh_count": 9, "opacity_sh_count": 9}}
        model = restore_model(payload, mi)
        ray = mi.Ray3f(mi.Point3f(.5, .5, -1), mi.Vector3f(0, 0, 1))
        ray.maxt = 3.
        reference_rgb, reference_alpha, background = mi.Color3f(.6, .4, .2), mi.Float(.8), mi.Color3f(1)
        gradients = model.loss_and_grad(ray, reference_rgb, reference_alpha, background, mi.Float(1), {})
        target = np.array([.6, .4, .2]) * .8 + .2

        def objective():
            rgb, alpha, _ = model.trace(ray, background)
            dr.eval(rgb, alpha)
            color = np.array([float(rgb.x[0]), float(rgb.y[0]), float(rgb.z[0])])
            return float(np.sum((color-target)**2) + .5 * (float(alpha[0])-.8)**2)

        maximum_error = 0.
        for name in ("radiance_sh", "opacity_sh"):
            analytic = np.asarray(gradients["gradients"][name]).copy()
            for index in range(len(state[name])):
                plus, minus = state[name].copy(), state[name].copy()
                epsilon = .002
                plus[index] += epsilon
                minus[index] -= epsilon
                model.set_parameters({name: mi.Float(plus)})
                positive = objective()
                model.set_parameters({name: mi.Float(minus)})
                negative = objective()
                model.set_parameters({name: mi.Float(state[name])})
                numerical = (positive-negative)/(2*epsilon)
                maximum_error = max(maximum_error, float(abs(numerical-analytic[index])))
                np.testing.assert_allclose(analytic[index], numerical, atol=5e-5, rtol=3e-3,
                                           err_msg=f"appearance derivative {name}[{index}]")
        report["checks"].append({"appearance_finite_differences": 36, "max_absolute_error": maximum_error})
        path = save_checkpoint(output / "roundtrip.npz", state, model_metadata(model))
        restored = restore_model(load_checkpoint(path), mi)
        for first, second in zip(model.trace(ray, background), restored.trace(ray, background)):
            np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
        report["checks"].append("checkpoint native forward roundtrip")
        empty = {"model_state": {key: value[:0] for key, value in state.items()},
                 "model_metadata": payload["model_metadata"]}
        empty_model = restore_model(empty, mi)
        empty_result = empty_model.loss_and_grad(ray, reference_rgb, reference_alpha, background, mi.Float(1), {})
        np.testing.assert_allclose(np.asarray(empty_result["raw_rgb"]), 1.)
        np.testing.assert_allclose(np.asarray(empty_result["raw_alpha"]), 0.)
        assert all(dr.width(value) == 0 for value in empty_result["gradients"].values())
        report["checks"].append("empty reconstruction preserves background with zero-sized gradients")

        camera = {"position": [.5, .5, -2.], "forward": [0., 0., 1.],
                  "right": [1., 0., 0.], "up": [0., 1., 0.], "camera_angle_x": .7}
        render = RenderConfig(width=16, height=16, background=(1., 1., 1.))
        a = render_view(model, render, camera, 3, 42, 16, mi, dr, np)
        b = render_view(model, render, camera, 3, 42, 256, mi, dr, np)
        for first, second in zip(a, b):
            np.testing.assert_allclose(first, second, atol=2e-6, rtol=2e-6)
        report["checks"].append("multi-spp rays invariant to chunk size")

        dataset = output / "dataset"
        (dataset / "train").mkdir(parents=True, exist_ok=True)
        (dataset / "test").mkdir(parents=True, exist_ok=True)
        (dataset / "init_points.ply").write_text(
            "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n0 0 0 100 140 180\n", encoding="ascii")
        rgba = np.zeros((16, 16, 4), dtype=np.float32)
        rgba[4:12, 4:12, :3] = [.6, .2, .1]
        rgba[4:12, 4:12, 3] = 1.
        rgba[4, 4:12, 3] = .5  # Fractional-alpha pixels exercise optional contour history.
        # NeRF camera at (0,0,3), looking -Z, becomes Falcor (0,3,0), looking -Y.
        matrix = [[1,0,0,0], [0,1,0,0], [0,0,1,3], [0,0,0,1]]
        for split in ("train", "test"):
            frames = []
            for index in range(2):
                image = mi.Bitmap(rgba).convert(mi.Bitmap.PixelFormat.RGBA, mi.Struct.Type.UInt8, True)
                image.write(str(dataset / split / f"r_{index}.png"))
                frames.append({"file_path": f"./{split}/r_{index}", "transform_matrix": matrix})
            (dataset / f"transforms_{split}.json").write_text(json.dumps({"camera_angle_x": .7, "frames": frames}), encoding="utf-8")
        # Retain prior artifacts; each invocation gets an isolated output run.
        import tempfile
        run = Path(tempfile.mkdtemp(prefix="epoch-regression-", dir=output))
        config = {"data": {"dataset": str(dataset)},
                  "model": {"grid_resolution": 4, "bounds_min": [-1.,-1.,-1.], "bounds_max": [1.,1.,1.]},
                  "render": {"width": 16, "height": 16, "spp": 1},
                  "runtime": {"build_dir": args.build_dir, "gpu": args.gpu},
                  "training": {"num_epochs": 3, "chunk_size": 0, "profile_views": 2,
                               "save_count": 3, "output": str(run / "full")},
                  "evaluation": {"max_views": 1},  # Exercise the default 64 spp on a subset.
                  "schedule": {"enabled": False},
                  "optimizer": {"opacity_warmup_epochs": 0, "opacity_ramp_epochs": 0,
                                "lr_center": 0., "lr_log_scale": 0., "lr_rotation": 0.},
                  "topology": {"enable_deletion": False, "enable_growth": False}}
        config_path = run / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        entry = Path(__file__).with_name("reconstruction_train.py")

        def run_process(command, name, *, expect_success=True):
            result = subprocess.run(command, text=True, capture_output=True)
            (run / f"{name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
            if expect_success and result.returncode:
                raise RuntimeError(f"{name} failed:\n{result.stdout}\n{result.stderr}")
            if not expect_success and not result.returncode:
                raise AssertionError(f"{name} unexpectedly succeeded")
            return result

        def check_automatic_evaluation(directory, *, spp=64, count=1, max_views=1):
            evaluation = json.loads((directory / "evaluation-report.json").read_text(encoding="utf-8"))
            assert evaluation["status"] == "passed"
            assert evaluation["render"]["spp"] == spp
            assert evaluation["requested_config"]["max_views"] == max_views
            assert evaluation["checkpoint"]["global_step"] == 6
            for split in ("train", "test"):
                assert evaluation["splits"][split]["aggregate"]["view_count"] == count
                assert [view["index"] for view in evaluation["splits"][split]["views"]] == list(range(count))
                assert {path.name for path in (directory / split).glob("*.png")} == {
                    f"{index:04d}.png" for index in range(count)}
                assert (directory / split / "metrics.json").is_file()
            saved_evaluation = json.loads((directory / "evaluation-config.json").read_text(encoding="utf-8"))
            assert saved_evaluation["spp"] == spp
            assert saved_evaluation["max_views"] == max_views

        run_process([sys.executable, "-I", str(entry), "--config", str(config_path)], "full")
        first = load_checkpoint(run / "full" / "epoch-00001.npz")
        final = load_checkpoint(run / "full" / "model.npz")
        assert first["training_state"]["global_step"] == 2
        assert final["training_state"]["global_step"] == 6
        assert final["training_state"]["completed_epochs"] == 3
        assert final["model_metadata"]["resolution"] == 4
        assert {path.name for path in (run / "full").glob("*.npz")} == {"epoch-00001.npz", "epoch-00002.npz", "model.npz"}
        full_report = json.loads((run / "full" / "training-report.json").read_text(encoding="utf-8"))
        assert full_report["reconstructed_model"]["completed_epochs"] == 3
        assert full_report["reconstructed_model"]["global_step"] == 6
        assert full_report["training_status"] == "passed"
        assert full_report["evaluation"]["status"] == "passed"
        assert Path(full_report["reconstructed_model"]["path"]) == run / "full" / "model.npz"
        assert full_report["training_protocol"]["batches_per_view"] == 1
        profiles = [json.loads(line) for line in Path(full_report["profile_file"]).read_text(encoding="utf-8").splitlines()]
        assert len(profiles) == 2
        assert [item["progress"] for item in profiles] == [1, 2]
        assert all(item["jit_kernels"] > 0 and item["wall_seconds"] > 0 for item in profiles)
        assert "progress 2/2, camera " in (run / "full.log").read_text(encoding="utf-8")
        report["checks"].append("full-image training, completed-view progress and optional kernel profile")
        assert len((run / "full" / "training.jsonl").read_text(encoding="utf-8").splitlines()) == 3
        report["checks"].append("three epochs save two intermediate checkpoints and final model without duplicate archives")
        check_automatic_evaluation(run / "full")
        saved_config = json.loads((run / "full" / "config.json").read_text(encoding="utf-8"))
        assert saved_config["render"]["spp"] == final["config"]["render"]["spp"] == 1
        assert saved_config["training"]["num_epochs"] == 3
        assert final["config"]["evaluation"]["spp"] == 64
        assert final["config"]["evaluation"]["max_views"] == 1
        report["checks"].append("automatic 64-spp evaluation selects the first view of each split without limiting training or replacing training config")
        if np.array_equal(first["model_state"]["radiance_sh"], final["model_state"]["radiance_sh"]):
            raise AssertionError("Full-epoch training did not change radiance parameters")
        run_process([sys.executable, "-I", str(entry), "--resume", str(run / "full" / "epoch-00001.npz"),
                     "--output-dir", str(run / "resumed"), "--build-dir", args.build_dir, "--gpu", str(args.gpu),
                     "--skip-evaluation"], "resume")
        resumed = load_checkpoint(run / "resumed" / "model.npz")
        assert {path.name for path in (run / "resumed").glob("*.npz")} == {"epoch-00002.npz", "model.npz"}
        for key in final["model_state"]:
            np.testing.assert_allclose(resumed["model_state"][key], final["model_state"][key], atol=2e-6, rtol=2e-6)
        for name, group in final["optimizer_state"]["groups"].items():
            for key, value in group.items():
                np.testing.assert_allclose(resumed["optimizer_state"]["groups"][name][key], value, atol=2e-6, rtol=2e-6)
        assert resumed["training_state"]["rng_state"] == final["training_state"]["rng_state"]
        assert not (run / "resumed" / "evaluation-report.json").exists()
        resume_report = json.loads((run / "resumed" / "training-report.json").read_text(encoding="utf-8"))
        assert resume_report["training_status"] == "passed"
        assert resume_report["evaluation"]["status"] == "skipped"
        report["checks"].append("two-view/three-epoch updates and epoch-boundary resume equivalent")
        run_process([sys.executable, "-I", str(entry), "--resume", str(run / "full" / "model.npz"),
                     "--output-dir", str(run / "already-finished" / "training"), "--build-dir", args.build_dir,
                     "--gpu", str(args.gpu), "--eval-spp", "2", "--eval-max-views", "10"], "already-finished")
        assert not list((run / "already-finished" / "training").glob("*.npz"))
        assert {path.relative_to(run / "already-finished").as_posix()
                for path in (run / "already-finished").rglob("*.npz")} == {"model.npz"}
        completed_export = load_checkpoint(run / "already-finished" / "model.npz")
        assert completed_export["training_state"]["global_step"] == 6
        for key in final["model_state"]:
            np.testing.assert_array_equal(completed_export["model_state"][key], final["model_state"][key])
        check_automatic_evaluation(run / "already-finished", spp=2, count=2, max_views=10)
        assert not (run / "already-finished" / "config.json").exists()
        exported_config = json.loads((run / "already-finished" / "training" / "config.json").read_text(encoding="utf-8"))
        assert exported_config["render"]["spp"] == 1
        report["checks"].append("completed resume exports model.npz then evaluates; CLI limits larger than a split select all available views")

        # A missing evaluation split must not invalidate or remove the trained model.
        failure_config = run / "missing-test.json"
        failure_config.write_text(json.dumps({"evaluation": {"test_cameras": "missing-test.json", "spp": 1}}),
                                  encoding="utf-8")
        run_process([sys.executable, "-I", str(entry), "--resume", str(run / "full" / "model.npz"),
                     "--config", str(failure_config), "--output-dir", str(run / "evaluation-failed" / "training"),
                     "--build-dir", args.build_dir, "--gpu", str(args.gpu)],
                    "evaluation-failed", expect_success=False)
        failure_report = json.loads((run / "evaluation-failed" / "training" / "training-report.json").read_text(encoding="utf-8"))
        assert failure_report["training_status"] == "passed"
        assert failure_report["evaluation"]["status"] == "failed"
        failed_model = load_checkpoint(run / "evaluation-failed" / "model.npz")
        assert failed_model["training_state"]["global_step"] == 6
        for key in final["model_state"]:
            np.testing.assert_array_equal(failed_model["model_state"][key], final["model_state"][key])
        report["checks"].append("automatic evaluation failure exits nonzero while preserving the completed model")

        config["training"]["output"] = str(run / "coarse")
        config["schedule"] = {"enabled": True, "start_resolution": 4, "target_resolution": 8,
                              "enable_coarse_growth": False, "save_each_level": True}
        config["contour"] = {"enabled": True}
        config["evaluation"]["enabled"] = False
        config_path.write_text(json.dumps(config), encoding="utf-8")
        run_process([sys.executable, "-I", str(entry), "--config", str(config_path)], "coarse")
        coarse = load_checkpoint(run / "coarse" / "model.npz")
        assert coarse["model_metadata"]["resolution"] == 8
        assert {path.name for path in (run / "coarse").glob("*.npz")} == {"epoch-00001.npz", "epoch-00002.npz", "model.npz"}
        after_refine = load_checkpoint(run / "coarse" / "epoch-00001.npz")
        assert after_refine["model_metadata"]["resolution"] == 8
        assert after_refine["training_state"]["contour_state"] == {}
        assert not after_refine["training_state"]["pending_epoch_transition"]
        # New saves follow the epoch's topology transition. The training loader
        # separately retains compatibility with legacy pending-transition files.
        run_process([sys.executable, "-I", str(entry), "--resume", str(run / "coarse" / "epoch-00001.npz"),
                     "--output-dir", str(run / "coarse-resumed"), "--build-dir", args.build_dir,
                     "--gpu", str(args.gpu)], "coarse-resume")
        coarse_resumed = load_checkpoint(run / "coarse-resumed" / "model.npz")
        assert {path.name for path in (run / "coarse-resumed").glob("*.npz")} == {"epoch-00002.npz", "model.npz"}
        for key in coarse["model_state"]:
            np.testing.assert_allclose(coarse_resumed["model_state"][key], coarse["model_state"][key], atol=2e-6, rtol=2e-6)
        assert coarse_resumed["training_state"]["global_step"] == 6
        for view_id, history in coarse["training_state"]["contour_state"].items():
            np.testing.assert_allclose(coarse_resumed["training_state"]["contour_state"][view_id]["mean_alpha"],
                                       history["mean_alpha"], atol=2e-6)
        report["checks"].append("coarse-to-fine transition and post-refinement resume equivalent without extra level archives")
        run_process([sys.executable, "-I", str(Path(__file__).with_name("reconstruction_evaluate.py")),
                     "--checkpoint", str(run / "full" / "model.npz"), "--split", "both", "--spp", "2",
                     "--output-dir", str(run / "evaluation"), "--build-dir", args.build_dir,
                     "--gpu", str(args.gpu)], "evaluation")
        evaluation = json.loads((run / "evaluation" / "evaluation-report.json").read_text(encoding="utf-8"))
        assert evaluation["status"] == "passed"
        assert all(evaluation["splits"][split]["aggregate"]["view_count"] == 2 for split in ("train", "test"))
        report["checks"].append("independent train/test 2-spp evaluation and PSNR reports")
        report.update(status="passed", integration_output=str(run))
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        (output / "training-tests.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["checks"]), flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
