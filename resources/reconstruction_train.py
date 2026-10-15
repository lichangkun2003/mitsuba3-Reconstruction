"""Train the Falcor NoLightTransport voxel model: every view, 1 spp, one update.

All relative paths use the parent of this checkout. Edit CONFIG, provide a JSON
configuration, or override the main options on the command line.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reconstruction_preview import DataConfig, ModelConfig, RenderConfig, RuntimeConfig, PreviewConfig, validate_config
from reconstruction_common import (REPOSITORY, WORKSPACE, ContourSampler, generate_ray_batch,
    initialize_runtime, load_views, model_metadata, read_reference, restore_model, workspace_path, file_sha256)
from reconstruction_checkpoint import load_checkpoint, save_checkpoint
from reconstruction_experiment import allocate_training_output, resumed_training_output
from reconstruction_optimizer import FalcorOptimizer, OptimizerConfig
from reconstruction_topology import TopologyManager, TopologySettings, CoarseToFineConfig


@dataclass
class TrainingSettings:
    num_epochs: int = 1000
    chunk_size: int = 16384
    seed: int = 42
    shuffle: bool = True
    save_count: int = 3  # Total snapshots, including the final model.npz.
    output: str | None = None  # None: automatically allocate a daily experiment.


@dataclass
class GradientSettings:
    alpha_loss_weight: float = .5
    geometry_tau_voxel_fraction: float = .12
    geometry_grad_clamp: float = 5
    alpha_geometry_weight: float = .1
    background_carve_weight: float = .01
    geometry_two_sided_boundary: bool = False
    geometry_boundary_band_tau: float = 1
    geometry_transition_only: bool = True
    geometry_hit_expansion_band_tau: float = 0
    min_evidence_transmittance: float = .05
    min_removal_loss_delta: float = 1e-4
    max_candidates: int = 16


@dataclass
class ContourSettings:
    enabled: bool = False
    uniform_fraction: float = .5
    history_rate: float = .3


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    render: RenderConfig = field(default_factory=lambda: RenderConfig(background=(1., 1., 1.)))
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    gradient: GradientSettings = field(default_factory=GradientSettings)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    topology: TopologySettings = field(default_factory=TopologySettings)
    schedule: CoarseToFineConfig = field(default_factory=CoarseToFineConfig)
    contour: ContourSettings = field(default_factory=ContourSettings)


# ==================== 训练实验配置：常用参数在这里修改 ====================
# 路径相对于 mitsuba 仓库的父目录；切换场景只需修改下面的 dataset。
CONFIG = TrainConfig()
CONFIG.data.dataset = "datasets/lego"
CONFIG.data.point_cloud = "init_points.ply"
CONFIG.data.cameras = "transforms_train.json"

CONFIG.render.width = 800
CONFIG.render.height = 800
CONFIG.render.spp = 1  # 训练固定每视角 1 spp；多 spp 出图使用评价脚本。

CONFIG.schedule.enabled = True
CONFIG.schedule.start_resolution = 256
CONFIG.schedule.target_resolution = 512
CONFIG.training.num_epochs = 1000  # 整个实验总轮数，每轮遍历全部训练视角。
CONFIG.training.save_count = 3  # 包含最终 model.npz，共保存三次。
CONFIG.training.output = None  # 自动创建 outputs/场景-MM.DD-E编号/training。
# 需要手动指定目录时，将 None 改为路径字符串，或传 --output-dir。
# 高级参数使用 CONFIG.optimizer / gradient / topology / contour / runtime。
# ============================ 配置区结束 ============================


def update(config, values):
    if not isinstance(values, dict):
        raise ValueError("Config must be a dictionary of parameter groups")
    groups = {f.name for f in fields(config)}
    if set(values) - groups:
        raise ValueError(f"Unknown configuration groups: {set(values) - groups}")
    for name, entries in values.items():
        if name == "training" and isinstance(entries, dict):
            # Old configs/checkpoints remain loadable; interval-based saving
            # no longer controls the number of reconstruction snapshots.
            entries = {key: value for key, value in entries.items() if key != "save_every_epochs"}
        group = getattr(config, name)
        allowed = {f.name for f in fields(group)}
        if not isinstance(entries, dict) or set(entries) - allowed:
            raise ValueError(f"Unknown or malformed configuration fields in {name}")
        for key, value in entries.items():
            setattr(group, key, value)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--resume", help="Resume a completed-epoch checkpoint")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--dataset")
    parser.add_argument("--output-dir", help="Override automatic outputs/SCENE-MM.DD-Eindex/training; on resume the default reuses the checkpoint's experiment")
    parser.add_argument("--build-dir")
    parser.add_argument("--gpu", type=int, help="Local physical index; ignored under Slurm")
    parser.add_argument("--chunk-size", type=int)
    return parser.parse_args()


def apply_arguments(config, args, values):
    update(config, values)
    for value, group, key in ((args.epochs, "training", "num_epochs"),
        (args.dataset, "data", "dataset"), (args.output_dir, "training", "output"),
        (args.build_dir, "runtime", "build_dir"), (args.gpu, "runtime", "gpu"),
        (args.chunk_size, "training", "chunk_size")):
        if value is not None:
            setattr(getattr(config, group), key, value)
    config.schedule.total_epochs = config.training.num_epochs


def validate(config):
    validate_config(PreviewConfig(data=config.data, model=config.model, render=config.render, runtime=config.runtime))
    if config.render.spp != 1:
        raise ValueError("Training requires exactly 1 spp per view; use reconstruction_evaluate.py for multi-spp images")
    if list(config.render.background) != [1., 1., 1.]:
        raise ValueError("Falcor's training reference is composited onto white; training requires background [1,1,1]. Evaluation background is configurable.")
    for name in ("num_epochs", "chunk_size", "save_count"):
        value = getattr(config.training, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"training.{name} must be a positive integer")
    if config.training.seed < 0:
        raise ValueError("training.seed must be nonnegative")
    for name, value in asdict(config.gradient).items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid gradient.{name}: {value}")
    if config.gradient.geometry_tau_voxel_fraction <= 0:
        raise ValueError("geometry_tau_voxel_fraction must be positive")


def settings(config):
    result = asdict(config.gradient)
    result.update(foreground_alpha_min=config.topology.foreground_alpha_min,
        background_alpha_max=config.topology.background_alpha_max,
        collect_deletion_evidence=config.topology.enable_deletion,
        transmittance_threshold=config.render.transmittance_threshold,
        max_hits=config.render.max_hits, entry_offset_fraction=config.render.entry_offset_fraction)
    return result


def train_view(model, view, rgba, config, sampler, rng, mi, dr, np):
    """Chunk rays for memory, accumulating a full view before one Adam update."""
    count = config.render.width * config.render.height
    offsets, inverse_pdf, token = sampler.sample(view["index"], rgba[..., 3], rng)
    reference = rgba.reshape(-1, 4)
    raw_alpha = np.zeros(count, dtype=np.float32)
    estimated_alpha = np.zeros(count, dtype=np.float32)
    accumulator = {"gradients": {}}
    total_loss = 0.
    for begin in range(0, count, config.training.chunk_size):
        end = min(begin + config.training.chunk_size, count)
        ray = generate_ray_batch(config.render, view["camera"],
            np.arange(begin, end, dtype=np.uint32), offsets[begin:end], mi, dr)
        ref = reference[begin:end]
        result = model.loss_and_grad(ray, mi.Color3f(mi.Float(ref[:, 0]), mi.Float(ref[:, 1]), mi.Float(ref[:, 2])),
            mi.Float(ref[:, 3]), mi.Color3f(config.render.background), mi.Float(inverse_pdf[begin:end]), settings(config))
        total_loss += float(dr.sum(result["loss"])[0])
        raw_alpha[begin:end] = np.asarray(result["raw_alpha"])
        estimated_alpha[begin:end] = np.asarray(result["alpha"])
        for key, value in result["gradients"].items():
            accumulator["gradients"][key] = accumulator["gradients"].get(key, 0) + value
        for key in ("appearance_valid", "geometry_valid", "background_carve_valid"):
            accumulator[key] = accumulator.get(key, 0) + result[key]
        for key in ("foreground_evidence", "background_evidence"):
            accumulator[key] = dr.maximum(accumulator.get(key, 0), result[key])
        # Materialize per-chunk sums; never retain all image path records/graphs.
        dr.eval(*accumulator["gradients"].values(), *[value for key, value in accumulator.items() if key != "gradients"])
    dr.sync_thread()
    finite = math.isfinite(total_loss) and np.isfinite(raw_alpha).all()
    for name, gradient in accumulator["gradients"].items():
        if not bool(dr.all(dr.isfinite(gradient))[0]):
            raise RuntimeError(f"Nonfinite {name} gradient at training view {view['index']}")
    if not finite:
        raise RuntimeError(f"Nonfinite loss/alpha at training view {view['index']}")
    sampler.observe(token, raw_alpha)
    return accumulator, total_loss / count, estimated_alpha.reshape(config.render.height, config.render.width)


def dataset_signature(views, dataset, camera_file):
    return {"cameras": file_sha256(dataset / camera_file),
            "references": [file_sha256(Path(view["reference_path"])) for view in views]}


def checkpoint_epochs(num_epochs: int, save_count: int) -> tuple[int, ...]:
    """Evenly space completed-epoch saves; the final model counts as one save."""
    for value in (num_epochs, save_count):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("Epoch and checkpoint counts must be positive integers")
    count = min(num_epochs, save_count)
    return tuple((index * num_epochs + count - 1) // count for index in range(1, count + 1))


def final_model_path(output: Path) -> Path:
    """Keep the finished reconstruction beside train/test evaluation results."""
    experiment = output.parent if output.name == "training" else output
    return experiment / "model.npz"


def prepare_training_output(config, resume_path=None, checkpoint_config=None):
    """Resolve and record one output directory before initializing the GPU."""
    requested = config.training.output
    if requested is None:
        if resume_path is not None:
            output = resumed_training_output(resume_path, checkpoint_config or {}).resolve()
        else:
            if workspace_path("outputs", WORKSPACE).is_relative_to(REPOSITORY):
                raise ValueError("Keep checkpoints, logs and caches outside the Git checkout")
            output = allocate_training_output(config.data.dataset, WORKSPACE)
    elif not isinstance(requested, str) or not requested.strip():
        raise ValueError("training.output must be null/None for automatic naming or a nonempty path")
    else:
        output = workspace_path(requested, WORKSPACE)
    model_path = final_model_path(output)
    if output.is_relative_to(REPOSITORY) or model_path.is_relative_to(REPOSITORY):
        raise ValueError("Keep checkpoints, logs and caches outside the Git checkout")
    if resume_path is None and (any(path.exists() for path in (
            output / "latest.npz", output / "training.jsonl", model_path))
            or any(output.glob("epoch-*.npz"))):
        raise ValueError("Training output already contains a run; choose --output-dir or resume its checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    # Save a concrete, portable path, never the automatic naming sentinel.
    config.training.output = (output.relative_to(WORKSPACE).as_posix()
                              if output.is_relative_to(WORKSPACE) else str(output))
    return output


def validate_existing_model(path: Path, resumed_checkpoint: dict) -> None:
    """A resume must not replace another experiment's exported reconstruction."""
    if not path.exists():
        return
    existing = load_checkpoint(path)
    previous = resumed_checkpoint["config"]
    for group in ("data", "model", "render", "gradient", "optimizer", "topology", "schedule", "contour"):
        if existing["config"].get(group) != previous.get(group):
            raise ValueError(f"Exported model belongs to a different {group} configuration: {path}; choose a new experiment directory")
    for key in ("num_epochs", "seed", "shuffle"):
        if existing["config"].get("training", {}).get(key) != previous.get("training", {}).get(key):
            raise ValueError(f"Exported model has different training.{key}: {path}; choose a new experiment directory")
    signature = resumed_checkpoint["training_state"].get("dataset_signature")
    if signature is None or existing["training_state"].get("dataset_signature") != signature:
        raise ValueError(f"Exported model belongs to a different training dataset: {path}; choose a new experiment directory")


def main():
    args = parse_arguments()
    overrides = json.loads(workspace_path(args.config, WORKSPACE).read_text(encoding="utf-8")) if args.config else {}
    config = TrainConfig()
    update(config, asdict(CONFIG))
    apply_arguments(config, args, overrides)
    resume_path = workspace_path(args.resume, WORKSPACE) if args.resume else None
    checkpoint = load_checkpoint(resume_path) if resume_path is not None else None
    output = prepare_training_output(config, resume_path, checkpoint["config"] if checkpoint else None)
    model_path = final_model_path(output)
    print("Training output:", output, flush=True)
    print("Final reconstructed model:", model_path, flush=True)
    report = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Falcor NoLightTransport 1-spp full-view training; explicit geometry surrogate gradients",
              "falcor_reference_commit": "398f634ec788bbb2e86a6f4d0ab78d41158d9905",
              "training_protocol": {"spp": 1, "reference_background": [1, 1, 1],
                  "gradient_reduction": "sum over pixels, no hit-count normalization",
                  "reported_loss": "mean over pixels, then mean over training views",
                  "actual_sampling": "per-pixel uniform, optionally contour-guided; frozen for a view"}}
    try:
        validate(config)
        mi, dr, np, handles = initialize_runtime(config.runtime, output, report)
        if checkpoint:
            validate_existing_model(model_path, checkpoint)
            # Runtime and output are host-specific; all algorithm settings resume from the checkpoint.
            runtime, training_output = config.runtime, config.training.output
            update(config, checkpoint["config"])
            config.runtime, config.training.output = runtime, training_output
            apply_arguments(config, args, overrides)
            # JSON may request automatic naming with null; keep the already
            # selected location instead of restoring null or a stale host path.
            config.training.output = training_output
            validate(config)
            previous = checkpoint["config"]
            current = asdict(config)
            for group in ("model", "render", "gradient", "optimizer", "topology", "contour"):
                if current[group] != previous[group]:
                    raise ValueError(f"Cannot change {group} while resuming; start a new run for a different algorithm")
            for key in ("seed", "shuffle", "save_count"):
                previous_value = previous["training"].get(key, 3) if key == "save_count" else previous["training"][key]
                if current["training"][key] != previous_value:
                    raise ValueError(f"Cannot change training.{key} while resuming")
            # Changing the total budget changes the coarse-to-fine schedule retroactively.
            if current["schedule"] != previous["schedule"]:
                raise ValueError("Resume must preserve its original resolution schedule and epoch budget")
            existing_config = output / "config.json"
            if existing_config.exists():
                existing = json.loads(existing_config.read_text(encoding="utf-8"))
                for group in ("data", "model", "render", "gradient", "optimizer", "topology", "schedule", "contour"):
                    if existing.get(group) != previous[group]:
                        raise ValueError("Resume output belongs to a different configuration; choose a new --output-dir")
        report["requested_config"] = asdict(config)
        save_epochs = checkpoint_epochs(config.training.num_epochs, config.training.save_count)
        report["checkpoint_epochs"] = list(save_epochs)
        (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
        dataset = workspace_path(config.data.dataset, WORKSPACE)
        views = load_views(dataset, config.data.cameras, config.render, np)
        signature = dataset_signature(views, dataset, config.data.cameras)
        rng = np.random.default_rng(config.training.seed)
        sampler = ContourSampler(np, **asdict(config.contour))
        completed, global_step = 0, 0
        if checkpoint:
            state = checkpoint["training_state"]
            if state["dataset_signature"] != signature:
                raise ValueError("Training cameras/reference images differ from the checkpoint")
            model = restore_model(checkpoint, mi)
            completed, global_step = state["completed_epochs"], state["global_step"]
            if (type(completed) is not int or not 0 <= completed <= config.training.num_epochs
                    or type(global_step) is not int or global_step != completed * len(views)):
                raise ValueError("Checkpoint epoch/update count does not match a complete traversal of the training views")
            rng.bit_generator.state = state["rng_state"]
            sampler.load_state_dict(state["contour_state"])
        else:
            bounds_min = np.asarray(config.model.bounds_min)
            bounds_max = np.asarray(config.model.bounds_max)
            center, half = (bounds_min + bounds_max) / 2, (bounds_max - bounds_min) / 2 * config.model.bounds_padding
            resolution = config.schedule.start_resolution if config.schedule.enabled else config.model.grid_resolution
            model = mi.VoxelReconstruction(str(dataset / config.data.point_cloud), resolution,
                (center-half).tolist(), (center+half).tolist(), config.model.opacity_threshold,
                config.model.sh_count, config.model.opacity_sh_count)
        optimizer = FalcorOptimizer(model, config.optimizer)
        topology = TopologyManager(model, len(views), settings=config.topology,
            schedule=replace(config.schedule, save_each_level=False),
            opacity_warmup=config.optimizer.opacity_warmup_epochs, opacity_ramp=config.optimizer.opacity_ramp_epochs)
        if checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            topology.load_state_dict(checkpoint["training_state"]["topology_state"])
            # Loading legacy lifecycle state restores its schedule too. Avoid
            # making additional pre-refinement snapshots under the new policy.
            topology.schedule.save_each_level = False
            if checkpoint["training_state"].get("pending_epoch_transition", False):
                replacement, transition = topology.finish_epoch(model, optimizer, completed)
                if replacement is not None:
                    model = restore_model(replacement, mi)
                if transition.get("refined", False):
                    sampler = ContourSampler(np, **asdict(config.contour))
                topology.before_refine_checkpoint = None

        def save(path, epochs):
            dr.eval(*model.snapshot().values())
            dr.sync_thread()
            arrays = {key: np.array(value, copy=True) for key, value in model.snapshot().items()}
            metadata = model_metadata(model)
            optimizer_state, topology_state = optimizer.state_dict(), topology.state_dict()
            save_checkpoint(path, arrays, metadata,
                training_state={"completed_epochs": epochs, "global_step": global_step,
                    "rng_state": rng.bit_generator.state, "contour_state": sampler.state_dict(),
                    "topology_state": topology_state, "dataset_signature": signature,
                    "pending_epoch_transition": False},
                optimizer_state=optimizer_state, config=asdict(config), provenance=report["provenance"])
            report["checkpoint"] = str(path)

        report.update(view_count=len(views), resumed=checkpoint is not None)
        # A restart writes a new history segment, preserving any later entries
        # from the interrupted run instead of appending ambiguous duplicate epochs.
        history_name = ("training-resume-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".jsonl"
                        if checkpoint else "training.jsonl")
        report["history_file"] = str(output / history_name)
        for epoch in range(completed, config.training.num_epochs):
            order = rng.permutation(len(views)) if config.training.shuffle else np.arange(len(views))
            epoch_loss = 0.
            started = time.perf_counter()
            for index in order:
                view = views[int(index)]
                rgba = read_reference(view["reference_path"], config.render.width, config.render.height, mi, np)
                gradients, loss, raw_alpha = train_view(model, view, rgba, config, sampler, rng, mi, dr, np)
                topology.observe_view(model, gradients, view_id=int(index), epoch=epoch,
                    camera=view["camera"], reference_alpha=rgba[..., 3], rendered_alpha=raw_alpha)
                optimizer.step(model, gradients, sample_count=1, epoch=epoch, coarse_to_fine=config.schedule.enabled)
                global_step += 1
                epoch_loss += loss
                print(f"Epoch {epoch+1}/{config.training.num_epochs}, view {int(index)+1}/{len(views)}, loss={loss:.8g}", flush=True)
            new_state, event = topology.finish_epoch(model, optimizer, epoch+1)
            topology.before_refine_checkpoint = None
            if new_state is not None:
                model = restore_model(new_state, mi)
            if event.get("refined", False):
                # Falcor resets contour proposals whenever grid resolution changes.
                sampler = ContourSampler(np, **asdict(config.contour))
            entry = {"epoch": epoch+1, "global_step": global_step, "mean_loss": epoch_loss / len(views),
                     "seconds": time.perf_counter()-started, "resolution": int(model.resolution),
                     "active_voxels": int(model.active_voxel_count), "topology": event}
            with (output / history_name).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry) + "\n")
            report["last_epoch"] = entry
            print(json.dumps(entry), flush=True)
            if epoch+1 in save_epochs and epoch+1 < config.training.num_epochs:
                save(output / f"epoch-{epoch+1:05d}.npz", epoch+1)
        # Publish only the completed reconstruction, using the same portable,
        # atomic checkpoint format understood by independent evaluation scripts.
        save(model_path, config.training.num_epochs)
        print("Saved reconstructed voxel model:", model_path, flush=True)
        report.update(status="passed", completed_epochs=config.training.num_epochs, global_step=global_step,
                      checkpoint=str(model_path),
                      reconstructed_model={"path": str(model_path), "sha256": file_sha256(model_path),
                          "completed_epochs": config.training.num_epochs, "global_step": global_step})
    except Exception:
        report.update(status="failed", error=traceback.format_exc())
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "training-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
