"""Train voxel reconstruction: every view, 1 spp, one update.

All relative paths use the parent of this checkout. Edit CONFIG, provide a JSON
configuration, or override the main options on the command line.
"""

from __future__ import annotations

import argparse
import builtins
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
import gc
import io
import json
import math
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preview import DataConfig, ModelConfig, RenderConfig, RuntimeConfig, PreviewConfig, validate_config
from reconstruction_common import (REPOSITORY, WORKSPACE, ContourSampler, UniformSampler, ReferenceImageCache,
    generate_ray_batch, prepare_camera, initialize_runtime, load_views, model_metadata,
    restore_model, workspace_path, file_sha256, render_view, write_png)
from reconstruction_checkpoint import load_checkpoint, save_checkpoint
from reconstruction_color import validate_reference_compositing
from reconstruction_experiment import allocate_training_output, resumed_training_output
from reconstruction_optimizer import FalcorOptimizer, OptimizerConfig
from reconstruction_performance import PerformanceRecorder
from reconstruction_progress import ProgressLine
from reconstruction_reference import GPUReferenceCache
from reconstruction_replay import NativeLossReplay
from reconstruction_growth_rgb import RGB_GROWTH_EVIDENCE_VERSION
from reconstruction_topology import TopologyManager, TopologySettings, CoarseToFineConfig, level_budgets


@dataclass
class TrainingSettings:
    rgb_only: bool = False  # Legacy schema default; CONFIG enables RGB-only phase 1.
    num_epochs: int = 1000
    chunk_size: int = 0  # 0: submit the whole image; positive: explicit ray batches.
    profile_views: int = 0  # Record JIT/cache/device timings for the first N views.
    profile_interval_epochs: int = 50  # Sample one view periodically and on entering each resolution; 0 disables.
    profile_stages: bool = True  # Synchronize native phase boundaries only in sampled views.
    reference_cache_mib: int = 1024  # Decoded float32 RGBA in host RAM; 0 disables caching.
    gpu_reference_cache_mib: int = 1024  # Float32 reference planes on the GPU; 0 disables.
    freeze_native: bool = True  # Replay native GPU work between topology changes.
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
    max_growth_hits: int = 8  # Bounded hypothetical hits per ray; overflow abstains.


@dataclass
class ContourSettings:
    enabled: bool = False
    uniform_fraction: float = .5
    history_rate: float = .3


@dataclass
class EvaluationSettings:
    enabled: bool = True
    spp: int = 64
    max_views: int | None = 10  # First N cameras of each split; None: all views.
    chunk_size: int = 65536
    seed: int = 42
    test_cameras: str = "transforms_test.json"


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
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)


# ==================== 训练实验配置：常用参数在这里修改 ====================
# 路径相对于 mitsuba 仓库的父目录；切换场景只需修改下面的 dataset。
CONFIG = TrainConfig()
CONFIG.data.dataset = "datasets/lego"
CONFIG.data.point_cloud = "init_points.ply"
CONFIG.data.cameras = "transforms_train.json"
CONFIG.data.reference_compositing = "srgb"  # 参考 PNG 先合成背景，再解码；渲染与损失仍在线性空间。

CONFIG.render.width = 800
CONFIG.render.height = 800
CONFIG.render.spp = 1  # 训练固定每视角 1 spp；多 spp 出图使用评价脚本。
CONFIG.training.rgb_only = True  # 参考 alpha 仅用于读图时合成 RGB，不进入训练或生长投票。
CONFIG.model.opacity_sh_count = 1  # 仅优化 DC logit；opacity = sigmoid(Y0 * b0)。
CONFIG.gradient.alpha_loss_weight = 0.
CONFIG.gradient.alpha_geometry_weight = 0.
CONFIG.gradient.background_carve_weight = 0.
CONFIG.gradient.geometry_transition_only = False  # hit/miss 均保留正负 RGB 几何信号。
CONFIG.gradient.geometry_two_sided_boundary = False  # 保留原 sigmoid，不叠加额外带宽门控。
CONFIG.contour.enabled = False  # 所有像素均匀采样，不读取 alpha 轮廓标签。
CONFIG.topology.enable_growth = True
CONFIG.topology.enable_deletion = True
CONFIG.topology.use_rgb_deletion = True  # 复用正常反向合成，按移除单体素的 RGB 收益投票。
CONFIG.topology.use_low_opacity_deletion = True  # 补充清理持续近透明体素，不要求正 RGB 删除收益。
CONFIG.topology.deletion_opacity_threshold = .005  # 实际 sigmoid(Y0*b0) 阈值；作为首轮对照起点。
CONFIG.topology.deletion_opacity_confirmation_windows = 2  # 连续两个删除检查窗口低于阈值才清理。
CONFIG.topology.deletion_warmup_epochs = 20  # 每个层级先优化 20 轮，再按周期收集一轮全部视角。
CONFIG.topology.deletion_interval = 10
CONFIG.topology.deletion_protection_epochs = 20  # 新生体素先优化 20 轮。
CONFIG.topology.deletion_freeze_last_epochs = 100  # 末尾停止删除，配合生长冻结稳定优化。
CONFIG.topology.deletion_rgb_confirmation_windows = 2  # 连续两个证据窗口达标才删除。
CONFIG.topology.deletion_rgb_min_support_views = 3
CONFIG.topology.deletion_rgb_max_conflict_views = 1  # 一个可靠反对视角即可阻止本窗口删除。
CONFIG.topology.deletion_rgb_gain_epsilon = 1e-7
CONFIG.topology.deletion_rgb_min_mean_gain = 1e-7
CONFIG.topology.use_growth_evidence = False
CONFIG.topology.use_rgb_growth = True  # 复用训练前向的 RGB 收益票，不额外渲染。
CONFIG.topology.growth_rgb_normalization = "observed"  # 有效命中射线均值，再对可靠观测视角求均值。
CONFIG.topology.growth_warmup_epochs = 20  # 最粗层级先优化 20 轮，再按生长周期检查。
CONFIG.topology.growth_fine_warmup_epochs = 20  # 每个后续层级重新 warmup。
CONFIG.topology.growth_freeze_last_epochs = 100  # 整个训练的最后 100 轮停止生长，继续优化已有体素；0 关闭。
CONFIG.topology.growth_rgb_min_support_views = 3
CONFIG.topology.growth_rgb_max_conflict_views = 2  # 达到 2 个反对视角则拒绝。
CONFIG.topology.growth_rgb_gain_epsilon = 1e-7  # 有效命中射线的平均 RGB 收益票阈值（保留数值，单位已变）。
CONFIG.topology.growth_rgb_min_mean_gain = 1e-7  # 可靠观测视角的带符号平均收益门槛。
CONFIG.gradient.max_growth_hits = 8  # 单条射线的候选格记录容量；同格多父提案共享记录，溢出则弃权。
CONFIG.schedule.enable_coarse_growth = True  # 分层训练的生长开关；分裂规则保持不变。
CONFIG.training.chunk_size = 0  # 0 = 整图提交；显存不足时可显式改为 65536 等。
CONFIG.training.profile_views = 0  # 设为 3 可记录前三个视角的编译、缓存和 GPU 耗时。
CONFIG.training.profile_interval_epochs = 50  # 每 50 轮及首次进入各分辨率时抽样记录内核耗时；0 关闭。
CONFIG.training.profile_stages = True  # 仅在抽样视角细分前向、命中反向和候选梯度；会增加诊断开销。
CONFIG.training.reference_cache_mib = 1024  # 内存缓存参考图，避免每轮重新解码；0 关闭。
CONFIG.training.gpu_reference_cache_mib = 1024  # 显存缓存参考图；0 恢复逐视角上传。
CONFIG.training.freeze_native = True  # 重放原生计算；拓扑变化后重新记录，分段诊断时旁路。

CONFIG.schedule.enabled = True
CONFIG.schedule.start_resolution = 256
CONFIG.schedule.target_resolution = 512
CONFIG.schedule.level_weights = (6, 4)  # 粗、细层级轮数比例；1000 轮分成 600 / 400。
CONFIG.training.num_epochs = 1000  # 整个实验总轮数，每轮遍历全部训练视角。
CONFIG.training.save_count = 3  # 包含最终 model.npz，共保存三次。
CONFIG.training.output = None  # 自动创建 outputs/场景-MM.DD-E编号/training。
# 需要手动指定目录时，将 None 改为路径字符串，或传 --output-dir。
CONFIG.evaluation.enabled = True  # 最终 model.npz 保存后，自动评价训练集和测试集。
CONFIG.evaluation.spp = 64  # 仅用于评价出图，不改变训练的 1 spp。
CONFIG.evaluation.max_views = 10  # 各取相机文件中的前 10 个视角；None 表示全部。
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
    parser.add_argument("--rgb-only", action=argparse.BooleanOptionalAction, default=None,
                        help="Use RGB-only training (CONFIG default); requires DC opacity and disabled reference-alpha priors")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--stop-after-epochs", type=int,
                        help="Stop after N additional complete epochs, saving a resumable checkpoint without changing the full schedule or publishing model.npz")
    parser.add_argument("--dataset")
    parser.add_argument("--output-dir", help="Override automatic outputs/SCENE-MM.DD-Eindex/training; on resume the default reuses the checkpoint's experiment")
    parser.add_argument("--build-dir")
    parser.add_argument("--gpu", type=int, help="Local physical index; ignored under Slurm")
    parser.add_argument("--chunk-size", type=int, help="0 (default): full image; positive: maximum rays per batch")
    parser.add_argument("--profile-views", type=int, help="Record kernel/cache timings for the first N processed views")
    parser.add_argument("--profile-interval-epochs", type=int, help="Sample one view every N epochs and on entering a resolution (default: 50); 0 disables automatic samples")
    parser.add_argument("--profile-stages", action=argparse.BooleanOptionalAction, default=None,
                        help="Synchronize and time native phases in sampled views; --no-profile-stages keeps only aggregate kernel profiling")
    parser.add_argument("--reference-cache-mib", type=int, help="RAM budget for decoded reference images in MiB (default: 1024); 0 disables caching")
    parser.add_argument("--gpu-reference-cache-mib", type=int, help="GPU reference-image cache budget in MiB (default: 1024); 0 disables caching")
    parser.add_argument("--freeze-native", action=argparse.BooleanOptionalAction, default=None,
                        help="Replay native GPU work without repeated tracing (default: enabled); stage diagnostics run directly")
    parser.add_argument("--skip-evaluation", action="store_true", help="Save the final model without automatic train/test evaluation")
    parser.add_argument("--eval-spp", type=int, help="Samples per pixel for final evaluation (default: 64)")
    parser.add_argument("--eval-max-views", type=int, help="Evaluate the first N cameras of each split (default: 10)")
    return parser.parse_args()


def apply_arguments(config, args, values):
    update(config, values)
    for value, group, key in ((args.epochs, "training", "num_epochs"),
        (args.dataset, "data", "dataset"), (args.output_dir, "training", "output"),
        (args.build_dir, "runtime", "build_dir"), (args.gpu, "runtime", "gpu"),
        (args.chunk_size, "training", "chunk_size"),
        (args.profile_views, "training", "profile_views"),
        (args.profile_interval_epochs, "training", "profile_interval_epochs"),
        (args.profile_stages, "training", "profile_stages"),
        (args.reference_cache_mib, "training", "reference_cache_mib"),
        (args.gpu_reference_cache_mib, "training", "gpu_reference_cache_mib"),
        (args.freeze_native, "training", "freeze_native"),
        (args.rgb_only, "training", "rgb_only"),
        (args.eval_spp, "evaluation", "spp"),
        (args.eval_max_views, "evaluation", "max_views")):
        if value is not None:
            setattr(getattr(config, group), key, value)
    if args.skip_evaluation:
        config.evaluation.enabled = False
    if (not config.training.rgb_only and
            "use_rgb_growth" not in values.get("topology", {})):
        # Explicit legacy supervision must not inherit the current RGB-only
        # experiment's growth mode. An explicit incompatible JSON still errors.
        config.topology.use_rgb_growth = False
    if (not config.training.rgb_only and
            "use_rgb_deletion" not in values.get("topology", {})):
        config.topology.use_rgb_deletion = False
    if (not config.training.rgb_only and
            "use_low_opacity_deletion" not in values.get("topology", {})):
        config.topology.use_low_opacity_deletion = False
    if (not config.training.rgb_only and
            "reference_compositing" not in values.get("data", {})):
        config.data.reference_compositing = "linear"
    config.schedule.total_epochs = config.training.num_epochs


def validate(config):
    validate_config(PreviewConfig(data=config.data, model=config.model, render=config.render, runtime=config.runtime))
    if config.schedule.enabled:
        level_budgets(config.schedule.start_resolution, config.schedule.target_resolution,
                      config.schedule.total_epochs, config.schedule.level_weights)
    if config.render.spp != 1:
        raise ValueError("Training requires exactly 1 spp per view; use evaluate.py for multi-spp images")
    if list(config.render.background) != [1., 1., 1.]:
        raise ValueError("Falcor's training reference is composited onto white; training requires background [1,1,1]. Evaluation background is configurable.")
    if not isinstance(config.training.rgb_only, bool):
        raise ValueError("training.rgb_only must be a boolean")
    if not config.training.rgb_only and config.data.reference_compositing != "linear":
        raise ValueError("data.reference_compositing='srgb' requires RGB-only training; "
                         "legacy alpha-supervised training uses linear compositing")
    if config.topology.use_rgb_growth and not config.training.rgb_only:
        raise ValueError("RGB growth evidence requires training.rgb_only=true")
    if not isinstance(config.topology.use_rgb_deletion, bool):
        raise ValueError("topology.use_rgb_deletion must be a boolean")
    if not isinstance(config.topology.use_low_opacity_deletion, bool):
        raise ValueError("topology.use_low_opacity_deletion must be a boolean")
    if config.topology.use_rgb_deletion and not config.training.rgb_only:
        raise ValueError("RGB deletion evidence requires training.rgb_only=true")
    for name in ("deletion_warmup_epochs", "deletion_protection_epochs", "deletion_freeze_last_epochs"):
        value = getattr(config.topology, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"topology.{name} must be a nonnegative integer")
    for name in ("deletion_interval", "deletion_rgb_confirmation_windows",
                 "deletion_rgb_min_support_views", "deletion_rgb_max_conflict_views",
                 "deletion_opacity_confirmation_windows"):
        value = getattr(config.topology, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"topology.{name} must be a positive integer")
    for name in ("deletion_rgb_gain_epsilon", "deletion_rgb_min_mean_gain"):
        value = getattr(config.topology, name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"topology.{name} must be finite and nonnegative")
    threshold = config.topology.deletion_opacity_threshold
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or not 0 < threshold < 1):
        raise ValueError("topology.deletion_opacity_threshold must be finite and strictly between 0 and 1")
    if config.topology.growth_rgb_normalization not in ("observed", "full_image"):
        raise ValueError("topology.growth_rgb_normalization must be 'observed' or 'full_image'")
    if (type(config.topology.growth_freeze_last_epochs) is not int or
            config.topology.growth_freeze_last_epochs < 0):
        raise ValueError("topology.growth_freeze_last_epochs must be a nonnegative integer")
    if config.training.rgb_only:
        if config.model.opacity_sh_count != 1:
            raise ValueError("RGB-only training requires model.opacity_sh_count=1 (DC logit only)")
        disabled = {
            "gradient.alpha_loss_weight": config.gradient.alpha_loss_weight,
            "gradient.alpha_geometry_weight": config.gradient.alpha_geometry_weight,
            "gradient.background_carve_weight": config.gradient.background_carve_weight,
            "gradient.geometry_transition_only": config.gradient.geometry_transition_only,
            "gradient.geometry_two_sided_boundary": config.gradient.geometry_two_sided_boundary,
            "contour.enabled": config.contour.enabled,
            "topology.use_growth_evidence": config.topology.use_growth_evidence,
        }
        enabled = [name for name, value in disabled.items() if value != 0]
        if enabled:
            raise ValueError("RGB-only training requires these settings to be zero/false: " + ", ".join(enabled))
        if config.topology.enable_deletion and not config.topology.use_rgb_deletion:
            raise ValueError("RGB-only deletion requires topology.use_rgb_deletion=true; legacy alpha deletion is unavailable")
        if not config.topology.use_rgb_growth and (config.topology.enable_growth or config.schedule.enable_coarse_growth):
            raise ValueError("RGB-only growth requires topology.use_rgb_growth=true; legacy alpha growth is unavailable")
    if (config.topology.enable_deletion and config.topology.use_low_opacity_deletion
            and config.model.opacity_sh_count != 1):
        raise ValueError("Low-opacity deletion requires model.opacity_sh_count=1 (DC logit only)")
    if (isinstance(config.gradient.max_growth_hits, bool) or not isinstance(config.gradient.max_growth_hits, int)
            or not 1 <= config.gradient.max_growth_hits <= 32):
        raise ValueError("gradient.max_growth_hits must be an integer in [1,32]")
    for name in ("num_epochs", "save_count"):
        value = getattr(config.training, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"training.{name} must be a positive integer")
    for name in ("chunk_size", "profile_views", "profile_interval_epochs", "reference_cache_mib", "gpu_reference_cache_mib"):
        value = getattr(config.training, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"training.{name} must be a nonnegative integer")
    if config.training.seed < 0:
        raise ValueError("training.seed must be nonnegative")
    if not isinstance(config.training.profile_stages, bool):
        raise ValueError("training.profile_stages must be a boolean")
    if not isinstance(config.training.freeze_native, bool):
        raise ValueError("training.freeze_native must be a boolean")
    for name, value in asdict(config.gradient).items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid gradient.{name}: {value}")
    if config.gradient.geometry_tau_voxel_fraction <= 0:
        raise ValueError("geometry_tau_voxel_fraction must be positive")
    if not isinstance(config.evaluation.enabled, bool):
        raise ValueError("evaluation.enabled must be a boolean")
    from evaluate import validate_config as validate_evaluation
    validate_evaluation(final_evaluation_config(config, Path("model.npz")))


def final_evaluation_config(config, model_path):
    """Use the saved model, current runtime and experiment directory for evaluation."""
    from evaluate import EvaluationConfig
    return EvaluationConfig(checkpoint=str(model_path), dataset=config.data.dataset,
        splits=("train", "test"), cameras={"train": config.data.cameras, "test": config.evaluation.test_cameras},
        spp=config.evaluation.spp, max_views=config.evaluation.max_views,
        seed=config.evaluation.seed, chunk_size=config.evaluation.chunk_size,
        output_directory=str(model_path.parent), runtime=config.runtime)


def save_initial_training_view(config, model, views, model_path, mi, dr, np, *, resumed=False):
    """Render the fresh point-cloud initialization without touching training RNG."""
    path = model_path.parent / "train" / "0000-initial.png"
    if resumed:
        # The original initialization is unavailable in a trained checkpoint.
        # Preserve an existing image; never relabel a restored model as initial.
        return dict(status="skipped", reason="resumed_checkpoint", image=str(path),
                    existing_image=path.is_file())
    if not config.evaluation.enabled:
        return dict(status="skipped", reason="evaluation_disabled")
    if path.exists():
        raise ValueError(f"Initial image already exists: {path}; choose a new experiment directory")
    return _save_training_view_zero(config, model, views, path, mi, dr, np,
                                    completed_epochs=0, global_step=0)


def is_coarse_training_end(topology, completed_epochs):
    """Capture the original coarse level only, before its refinement."""
    return (topology.schedule.enabled
            and topology.resolution == topology.schedule.start_resolution
            and topology.resolution < topology.schedule.target_resolution
            and completed_epochs >= topology.level_start_epoch + topology.level_budget)


def save_coarse_training_view(config, model, views, model_path, mi, dr, np, *,
                             completed_epochs, global_step):
    """Render the optimized coarse model, never a replacement fine model."""
    path = model_path.parent / "train" / "0000-coarse.png"
    if not config.evaluation.enabled:
        return dict(status="skipped", reason="evaluation_disabled")
    if path.exists():
        # A checkpoint preceding the split can revisit this boundary on resume.
        # Preserve the original coarse-stage comparison image.
        return dict(status="skipped", reason="existing_image", image=str(path),
                    existing_image=True)
    return _save_training_view_zero(config, model, views, path, mi, dr, np,
                                    completed_epochs=completed_epochs, global_step=global_step)


def _save_training_view_zero(config, model, views, path, mi, dr, np, *,
                             completed_epochs, global_step):
    """Use evaluation camera/seed settings without consuming training RNG."""
    view = next(view for view in views if view["index"] == 0)
    # Match evaluate.render_evaluation_record's train camera 0 sample sequence.
    seed = int(np.random.SeedSequence([config.evaluation.seed, 0, 0]).generate_state(1)[0])
    started = time.perf_counter()
    rgb, _ = render_view(model, config.render, view["camera"], config.evaluation.spp,
                         seed, config.evaluation.chunk_size, mi, dr, np)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_png(path, rgb, mi, np)
    return dict(status="passed", image=str(path), view_index=0,
                completed_epochs=completed_epochs, global_step=global_step,
                spp=config.evaluation.spp, seed=seed,
                width=config.render.width, height=config.render.height,
                resolution=int(model.resolution), active_voxels=int(model.active_voxel_count),
                seconds=time.perf_counter()-started)


def make_sampler(config, np):
    return UniformSampler(np) if config.training.rgb_only else ContourSampler(np, **asdict(config.contour))


def validate_resume_supervision(config, checkpoint):
    """Never silently reinterpret a legacy checkpoint as a new RGB experiment."""
    previous = checkpoint["config"].get("training", {}).get("rgb_only", False)
    if config.training.rgb_only != previous:
        raise ValueError("Cannot change training.rgb_only while resuming. Start a new experiment "
                         "for RGB-only phase 1; legacy checkpoints remain available for evaluation "
                         "or explicit --no-rgb-only resume.")


def resolve_resume_reference_compositing(config, checkpoint, overrides):
    """Keep a saved target protocol, including the pre-field linear default."""
    saved = checkpoint["config"].get("data", {}).get("reference_compositing", "linear")
    validate_reference_compositing(saved)
    requested = overrides.get("data", {}).get("reference_compositing", saved)
    validate_reference_compositing(requested)
    if requested != saved:
        raise ValueError("Cannot change data.reference_compositing while resuming; "
                         "start a new experiment for different reference targets")
    config.data.reference_compositing = saved


def settings(config):
    result = asdict(config.gradient)
    result.update(rgb_only=config.training.rgb_only,
        foreground_alpha_min=config.topology.foreground_alpha_min,
        background_alpha_max=config.topology.background_alpha_max,
        collect_deletion_evidence=config.topology.enable_deletion and not config.topology.use_rgb_deletion,
        collect_rgb_deletion_evidence=config.topology.enable_deletion and config.topology.use_rgb_deletion,
        transmittance_threshold=config.render.transmittance_threshold,
        max_hits=config.render.max_hits, entry_offset_fraction=config.render.entry_offset_fraction)
    return result


def effective_chunk_size(config):
    count = config.render.width * config.render.height
    return min(config.training.chunk_size or count, count)


def shard_pixel_range(count, rank, world_size):
    """Partition global pixel IDs exactly once, including uneven image sizes."""
    if any(isinstance(value, bool) or not isinstance(value, int)
           for value in (count, rank, world_size)):
        raise ValueError("Pixel count, rank and world size must be integers")
    if world_size < 1 or count < world_size or not 0 <= rank < world_size:
        raise ValueError("Each GPU requires a nonempty pixel shard and a valid rank")
    return count * rank // world_size, count * (rank + 1) // world_size


def compute_view_shard(model, view, rgba, config, offsets, inverse_pdf, token, mi, dr, *,
                       pixel_begin=0, pixel_end=None, need_rendered_alpha=False,
                       collect_deletion_evidence=None, stage_profile=None,
                       gpu_reference=None, native_replay=None, zero_alpha=False,
                       collect_growth_evidence=False):
    """Compute a pixel interval from an already sampled full view, without readback.

    Return the accumulator, unnormalized loss sum, finite flag, and optional
    full-sized alpha buffers. Partial buffers are zero outside the shard so a
    SUM reduction reconstructs the original image without changing its pixels.
    Sampling, contour observation and Adam updates are deliberately outside.
    """
    count = config.render.width * config.render.height
    pixel_end = count if pixel_end is None else pixel_end
    if (any(isinstance(value, bool) or not isinstance(value, int)
            for value in (pixel_begin, pixel_end))
            or not 0 <= pixel_begin < pixel_end <= count):
        raise ValueError("Pixel shard must be a nonempty interval within the full image")
    chunk_size = effective_chunk_size(config)
    channels = 3 if config.training.rgb_only else 4
    if rgba.shape != (config.render.height, config.render.width, channels):
        raise ValueError(f"Training reference must have shape (height, width, {channels}); "
                         "RGB-only targets must be composited before training")
    if config.training.rgb_only and token is not None:
        raise ValueError("RGB-only training requires uniform sampling without alpha contour history")
    reference = rgba.reshape(-1, channels)
    allocate = dr.zeros if zero_alpha or pixel_begin != 0 or pixel_end != count else dr.empty
    raw_alpha = allocate(mi.Float, count) if token is not None else None
    estimated_alpha = allocate(mi.Float, count) if need_rendered_alpha else None
    accumulator = {"gradients": {}}
    total_loss, finite = mi.Float(0), mi.Bool(True)
    camera = prepare_camera(config.render, view["camera"], mi, dr)
    native_settings = settings(config)
    native_settings["collect_growth_evidence"] = bool(collect_growth_evidence)
    if stage_profile is not None:
        native_settings["profile_stages"] = True
    if collect_deletion_evidence is not None:
        # The topology schedule decides whether these counters have a consumer.
        # Disabling unused counters does not disable loss/appearance/geometry terms.
        native_settings["collect_deletion_evidence"] = bool(collect_deletion_evidence) and not config.topology.use_rgb_deletion
        native_settings["collect_rgb_deletion_evidence"] = bool(collect_deletion_evidence) and config.topology.use_rgb_deletion
    rgb_deletion_requested = native_settings["collect_rgb_deletion_evidence"]
    for begin in range(pixel_begin, pixel_end, chunk_size):
        end = min(begin + chunk_size, pixel_end)
        indices = dr.arange(mi.UInt32, end - begin) + dr.opaque(mi.UInt32, begin)
        ray = generate_ray_batch(config.render, view["camera"],
            indices, offsets[begin:end], mi, dr, prepared_camera=camera)
        if gpu_reference is None:
            ref = reference[begin:end]
            reference_rgb = mi.Color3f(*(mi.Float(ref[:, channel]) for channel in range(3)))
            # The native legacy signature retains this argument. In RGB-only
            # mode it is a constant placeholder, never a reference mask.
            reference_alpha = mi.Float(1.) if config.training.rgb_only else mi.Float(ref[:, 3])
        else:
            if config.training.rgb_only and not getattr(gpu_reference, "rgb_only", False):
                raise ValueError("RGB-only training requires a precomposited RGB device reference")
            reference_rgb, reference_alpha = gpu_reference.batch(begin, end)
        # Without contour proposals the sampler returns exactly unit weights.
        # Preserve CPU RNG consumption and offsets; avoid uploading a constant array.
        pdf = mi.Float(inverse_pdf[begin:end]) if token is not None else dr.opaque(mi.Float, 1.)
        arguments = (ray, reference_rgb, reference_alpha, mi.Color3f(config.render.background), pdf, native_settings)
        result = (native_replay(model, *arguments) if native_replay is not None
                  else model.loss_and_grad(*arguments))
        if stage_profile is not None:
            if "stage_profile" not in result:
                raise RuntimeError("Native stage profiling is unavailable; rebuild this checkout's C++ bindings")
            for phase, measurements in result["stage_profile"].items():
                totals = stage_profile.setdefault(phase, {})
                for name, value in measurements.items():
                    value = float(value) if name == "wall_seconds" or name.endswith("_ms") else int(value)
                    totals[name] = totals.get(name, 0) + value
        total_loss += dr.sum(result["loss"])
        finite &= dr.all(dr.isfinite(result["raw_alpha"])) & dr.all(dr.isfinite(result["alpha"]))
        for name, target in (("raw_alpha", raw_alpha), ("alpha", estimated_alpha)):
            if target is not None:
                # An opaque offset avoids a different scatter kernel per batch.
                indices = dr.arange(mi.UInt32, end - begin) + dr.opaque(mi.UInt32, begin)
                dr.scatter(target, result[name], indices)
        for key, value in result["gradients"].items():
            accumulator["gradients"][key] = accumulator["gradients"].get(key, 0) + value
        for key in ("appearance_valid", "geometry_valid", "background_carve_valid"):
            accumulator[key] = accumulator.get(key, 0) + result[key]
        for key in ("foreground_evidence", "background_evidence"):
            accumulator[key] = dr.maximum(accumulator.get(key, 0), result[key])
        if collect_growth_evidence:
            for key in ("growth_gain", "growth_observed", "growth_unknown"):
                if key not in result:
                    raise RuntimeError("RGB growth evidence is unavailable; rebuild this source checkout")
                accumulator[key] = accumulator.get(key, 0) + result[key]
            finite &= dr.all(dr.isfinite(result["growth_gain"]))
        if rgb_deletion_requested:
            for key in ("deletion_gain", "deletion_observed", "deletion_unknown"):
                if key not in result:
                    raise RuntimeError("RGB deletion evidence is unavailable; rebuild this source checkout")
                accumulator[key] = accumulator.get(key, 0) + result[key]
            finite &= dr.all(dr.isfinite(result["deletion_gain"]))
        # Materialize per-chunk sums; never retain all image path records/graphs.
        dr.eval(total_loss, finite, *[x for x in (raw_alpha, estimated_alpha) if x is not None],
                *accumulator["gradients"].values(),
                *[value for key, value in accumulator.items() if key != "gradients"])
    for gradient in accumulator["gradients"].values():
        finite &= dr.all(dr.isfinite(gradient))
    if "growth_gain" in accumulator:
        finite &= dr.all(dr.isfinite(accumulator["growth_gain"]))
    if "deletion_gain" in accumulator:
        finite &= dr.all(dr.isfinite(accumulator["deletion_gain"]))
    finite &= dr.isfinite(total_loss)
    return accumulator, total_loss, finite, raw_alpha, estimated_alpha


def finalize_view(partial, view, config, sampler, token, mi, dr, np, *, alpha_on_device=False):
    """Validate a globally reduced view and observe its contour history once."""
    accumulator, total_loss, finite, raw_alpha, estimated_alpha = partial
    count = config.render.width * config.render.height
    # Read loss and the combined validity flag together, rather than performing
    # a separate host synchronization for every parameter group and ray batch.
    summary = dr.zeros(mi.Float, 2)
    dr.scatter(summary, total_loss / count, 0)
    dr.scatter(summary, mi.Float(finite), 1)
    dr.eval(summary)
    host_summary = np.asarray(summary)
    mean_loss = float(host_summary[0])
    if not bool(host_summary[1]) or not math.isfinite(mean_loss):
        raise RuntimeError(f"Nonfinite loss/alpha/gradient at training view {view['index']}")
    if mean_loss < 0:
        raise RuntimeError(f"Negative squared-error loss at training view {view['index']}: {mean_loss}; "
                           "check GPU reduction/readback correctness before continuing")
    if raw_alpha is not None:
        sampler.observe(token, np.asarray(raw_alpha))
    # Topology voting consumes the device image directly. Keep the host return
    # available for diagnostic callers and older scripts that inspect pixels.
    alpha = estimated_alpha
    if alpha is not None and not alpha_on_device:
        alpha = np.asarray(alpha).reshape(config.render.height, config.render.width)
    return accumulator, mean_loss, alpha


def train_view(model, view, rgba, config, sampler, rng, mi, dr, np, *, need_rendered_alpha=False,
               collect_deletion_evidence=None, stage_profile=None, gpu_reference=None,
               native_replay=None, collective=None, alpha_on_device=False,
               collect_growth_evidence=False):
    """Compute one full-view gradient, optionally over synchronized pixel shards.

    All ranks consume the same full-view sampling stream and contour history.
    A collective combines gradient/validity SUMs, evidence MAXs and disjoint
    alpha buffers before the single logical Adam update performed by the caller.
    The ordinary path retains one small loss/finite readback per view. Contour
    history still reads its image; production topology keeps alpha on device.
    """
    count = config.render.width * config.render.height
    begin, end = ((0, count) if collective is None else
                  shard_pixel_range(count, collective.rank, collective.world_size))
    offsets, inverse_pdf, token = sampler.sample(
        view["index"], rgba if config.training.rgb_only else rgba[..., 3], rng)
    partial = compute_view_shard(model, view, rgba, config, offsets, inverse_pdf, token, mi, dr,
        pixel_begin=begin, pixel_end=end, need_rendered_alpha=need_rendered_alpha,
        collect_deletion_evidence=collect_deletion_evidence, stage_profile=stage_profile,
        gpu_reference=gpu_reference, native_replay=native_replay, zero_alpha=collective is not None,
        collect_growth_evidence=collect_growth_evidence)
    if collective is not None:
        partial = collective.reduce_view(*partial)
        accumulator, total_loss, finite, raw_alpha, estimated_alpha = partial
        # Finite local values can overflow when summed across ranks. Reject the
        # global result before any replica changes its optimizer or model state.
        for gradient in accumulator["gradients"].values():
            finite &= dr.all(dr.isfinite(gradient))
        if "growth_gain" in accumulator:
            finite &= dr.all(dr.isfinite(accumulator["growth_gain"]))
        if "deletion_gain" in accumulator:
            finite &= dr.all(dr.isfinite(accumulator["deletion_gain"]))
        finite &= dr.isfinite(total_loss)
        partial = accumulator, total_loss, finite, raw_alpha, estimated_alpha
    return finalize_view(partial, view, config, sampler, token, mi, dr, np,
                         alpha_on_device=alpha_on_device)


def kernel_profile(history):
    """Separate JIT compilation from device work; all Dr.Jit times are ms."""
    entries = list(history)
    jit = [entry for entry in entries if entry.type.name == "JIT"]
    return {"operations": len(entries), "jit_kernels": len(jit),
            "cache_hits": sum(bool(entry.cache_hit) for entry in jit),
            "cache_misses": sum(not entry.cache_hit for entry in jit),
            "codegen_ms": sum(entry.codegen_time for entry in entries),
            "backend_compile_ms": sum(entry.backend_time for entry in entries),
            "device_ms": sum(entry.execution_time for entry in entries),
            "jit_hashes": sorted({str(entry.hash) for entry in jit})}


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
            output / "latest.npz", output / "training.jsonl", model_path,
            model_path.parent / "train" / "0000-initial.png",
            model_path.parent / "train" / "0000-coarse.png"))
            or any(output.glob("epoch-*.npz"))):
        raise ValueError("Training output already contains a run; choose --output-dir or resume its checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    # Save a concrete, portable path, never the automatic naming sentinel.
    config.training.output = (output.relative_to(WORKSPACE).as_posix()
                              if output.is_relative_to(WORKSPACE) else str(output))
    return output


def normalized_algorithm_group(name, values):
    """Compare legacy saves with explicit defaults, without changing algorithms."""
    if name == "data" and values is not None:
        return {"reference_compositing": "linear", **values}
    defaults = {"topology": TopologySettings, "gradient": GradientSettings}
    if values is not None and name in defaults:
        return {**asdict(defaults[name]()), **values}
    return values


def validate_existing_model(path: Path, resumed_checkpoint: dict) -> None:
    """A resume must not replace another experiment's exported reconstruction."""
    if not path.exists():
        return
    existing = load_checkpoint(path)
    previous = resumed_checkpoint["config"]
    for group in ("data", "model", "render", "gradient", "optimizer", "topology", "schedule", "contour"):
        if (normalized_algorithm_group(group, existing["config"].get(group))
                != normalized_algorithm_group(group, previous.get(group))):
            raise ValueError(f"Exported model belongs to a different {group} configuration: {path}; choose a new experiment directory")
    for key in ("num_epochs", "seed", "shuffle"):
        if existing["config"].get("training", {}).get(key) != previous.get("training", {}).get(key):
            raise ValueError(f"Exported model has different training.{key}: {path}; choose a new experiment directory")
    if (existing["config"].get("training", {}).get("rgb_only", False)
            != previous.get("training", {}).get("rgb_only", False)):
        raise ValueError(f"Exported model has different training.rgb_only: {path}; choose a new experiment directory")
    signature = resumed_checkpoint["training_state"].get("dataset_signature")
    if signature is None or existing["training_state"].get("dataset_signature") != signature:
        raise ValueError(f"Exported model belongs to a different training dataset: {path}; choose a new experiment directory")


def run_training(distributed=None):
    """Finish and save training before releasing its GPU state for evaluation."""
    run_started = time.perf_counter()
    primary = distributed is None or distributed.rank == 0
    emit = builtins.print if primary else lambda *a, **k: None
    args = parse_arguments()
    if args.stop_after_epochs is not None and args.stop_after_epochs < 1:
        raise ValueError("--stop-after-epochs must be positive")
    overrides = json.loads(workspace_path(args.config, WORKSPACE).read_text(encoding="utf-8")) if args.config else {}
    config = TrainConfig()
    update(config, asdict(CONFIG))
    apply_arguments(config, args, overrides)
    resume_path = workspace_path(args.resume, WORKSPACE) if args.resume else None
    checkpoint = load_checkpoint(resume_path) if resume_path is not None else None
    if checkpoint is not None:
        validate_resume_supervision(config, checkpoint)
        resolve_resume_reference_compositing(config, checkpoint, overrides)
    output = prepare_training_output(config, resume_path, checkpoint["config"] if checkpoint else None) if primary else None
    if distributed is not None:
        location, config.training.output = distributed.broadcast(
            (str(output), config.training.output) if primary else None)
        output = Path(location)
    model_path = final_model_path(output)
    emit("Training output:", output, flush=True)
    emit("Final reconstructed model:", model_path, flush=True)
    report = {"status": "running", "training_status": "running", "evaluation": {"status": "not_started"},
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Falcor NoLightTransport 1-spp full-view training; explicit geometry surrogate gradients",
              "falcor_reference_commit": "398f634ec788bbb2e86a6f4d0ab78d41158d9905",
              "training_protocol": {"spp": 1, "reference_background": [1, 1, 1],
                  "gradient_reduction": "sum over pixels, no hit-count normalization",
                  "reported_loss": "mean over pixels, then mean over training views",
                  "actual_sampling": "per-pixel uniform, optionally contour-guided; frozen for a view"}}
    progress_line = ProgressLine() if primary else ProgressLine(stream=io.StringIO())
    reference_cache = None
    gpu_reference_cache = None
    native_replay = None
    collective = None
    topology = None
    history_name = ("training-resume-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + ".jsonl"
                    if checkpoint else "training.jsonl")
    performance_name = ("performance-summary-" + Path(history_name).stem.removeprefix("training-") + ".json"
                        if checkpoint else "performance-summary.json")
    performance = PerformanceRecorder(output / performance_name,
        {"history_file": str(output / history_name), "resumed_from": str(resume_path) if resume_path else None},
        started_at=run_started)
    report["performance_file"] = str(output / performance_name)
    publish_performance = performance.write if primary else lambda *a, **k: None
    training_loop_started = None
    emit("Performance summary:", report["performance_file"], flush=True)
    try:
        validate(config)
        mi, dr, np, handles = initialize_runtime(config.runtime, output, report,
            cuda_device=distributed.rank if distributed else None,
            cpu_threads=distributed.cpus_per_rank if distributed else None)
        if config.training.rgb_only and getattr(mi.VoxelReconstruction, "rgb_only_version", 0) < 1:
            raise RuntimeError("RGB-only training requires the new native implementation; "
                               "incrementally rebuild this Mitsuba source checkout")
        if distributed is not None:
            from reconstruction_collective import NcclCollective
            collective = NcclCollective(distributed.rank, distributed.world_size,
                                         distributed.connection, mi, dr, np)
            report["distributed"] = dict(world_size=distributed.world_size, rank=distributed.rank,
                sharding="contiguous global pixel intervals of the same view",
                gradient_reduction="SUM; one logical Adam update per view; no GPU-count averaging",
                communication=collective.statistics())
            performance.metadata["distributed"] = report["distributed"]
            distributed.heartbeat("runtime_initialized", force=True)
        performance.metadata.update({key: report.get(key) for key in (
            "gpu", "cpu_threads", "python_executable", "mitsuba_version", "drjit_version", "actual_variant",
            "mitsuba_file", "drjit_file", "numpy_version", "provenance")})
        if checkpoint:
            if primary:
                validate_existing_model(model_path, checkpoint)
            # Runtime and output are host-specific; all algorithm settings resume from the checkpoint.
            runtime, training_output = config.runtime, config.training.output
            # Fields introduced after a checkpoint must use their legacy
            # dataclass defaults, not the latest experiment's CONFIG switches.
            config.data = DataConfig()
            config.topology = TopologySettings()
            config.gradient = GradientSettings()
            update(config, checkpoint["config"])
            # The field did not exist in legacy checkpoints. Do not inherit a
            # newer CONFIG default when loading their algorithm settings.
            config.training.rgb_only = checkpoint["config"]["training"].get("rgb_only", False)
            config.runtime, config.training.output = runtime, training_output
            apply_arguments(config, args, overrides)
            validate_resume_supervision(config, checkpoint)
            resolve_resume_reference_compositing(config, checkpoint, overrides)
            # JSON may request automatic naming with null; keep the already
            # selected location instead of restoring null or a stale host path.
            config.training.output = training_output
            validate(config)
            previous = checkpoint["config"]
            current = asdict(config)
            for group in ("model", "render", "gradient", "optimizer", "topology", "contour"):
                previous_group = normalized_algorithm_group(group, previous[group])
                if current[group] != previous_group:
                    raise ValueError(f"Cannot change {group} while resuming; start a new run for a different algorithm")
            for key in ("seed", "shuffle", "save_count"):
                previous_value = previous["training"].get(key, 3) if key == "save_count" else previous["training"][key]
                if current["training"][key] != previous_value:
                    raise ValueError(f"Cannot change training.{key} while resuming")
            # Changing the total budget changes the coarse-to-fine schedule retroactively.
            if current["schedule"] != previous["schedule"]:
                raise ValueError("Resume must preserve its original resolution schedule and epoch budget")
            existing_config = output / "config.json"
            if primary and existing_config.exists():
                existing = json.loads(existing_config.read_text(encoding="utf-8"))
                # Checkpoints preserve tuples, whereas config.json stores JSON
                # arrays. Compare their serialized forms. Only the writer checks
                # this file so a peer cannot race its replacement below.
                previous_json = json.loads(json.dumps(previous))
                for group in ("data", "model", "render", "gradient", "optimizer", "topology", "schedule", "contour"):
                    if (normalized_algorithm_group(group, existing.get(group))
                            != normalized_algorithm_group(group, previous_json[group])):
                        raise ValueError("Resume output belongs to a different configuration; choose a new --output-dir")
        rgb_growth_enabled = config.topology.use_rgb_growth and config.topology.enable_growth and (
            not config.schedule.enabled or config.schedule.enable_coarse_growth)
        if rgb_growth_enabled and getattr(mi.VoxelReconstruction, "rgb_growth_version", 0) < RGB_GROWTH_EVIDENCE_VERSION:
            raise RuntimeError("Multi-parent RGB growth evidence requires native rgb_growth_version >= 3; "
                               "incrementally rebuild this checkout")
        rgb_deletion_enabled = config.topology.use_rgb_deletion and config.topology.enable_deletion
        low_opacity_deletion_enabled = config.topology.use_low_opacity_deletion and config.topology.enable_deletion
        if rgb_deletion_enabled and getattr(mi.VoxelReconstruction, "rgb_deletion_version", 0) < 1:
            raise RuntimeError("RGB deletion evidence requires native rgb_deletion_version >= 1; "
                               "incrementally rebuild this checkout")
        report["requested_config"] = asdict(config)
        report["scope"] = ("RGB-only: DC opacity, bidirectional geometry surrogate, refinement"
                           + (" and one-pass RGB growth" if rgb_growth_enabled else "")
                           + (" and RGB deletion from existing reverse paths" if rgb_deletion_enabled else "")
                           + (" and persistent low-opacity pruning" if low_opacity_deletion_enabled else "")
                           if config.training.rgb_only else "Legacy Falcor NoLightTransport alpha-supervised training")
        report["training_protocol"].update(
            rgb_only=config.training.rgb_only,
            render_color_space="linear",
            loss_color_space="linear",
            reference_compositing=config.data.reference_compositing,
            reference_preprocessing=(
                "sRGB-encoded sources: composite RGB with background in sRGB, then decode RGB to linear; "
                "linear sources: composite in linear; retain only RGB"
                if config.training.rgb_only and config.data.reference_compositing == "srgb" else
                "RGBA decoded to linear, then composited onto white; only RGB retained"
                                     if config.training.rgb_only else "linear RGBA; composed inside native training"),
            actual_sampling="per-pixel uniform" if config.training.rgb_only else "per-pixel uniform or contour-guided",
            growth_evidence=("same-DDA independent parent proposals; per-cell winner after signed 1-spp RGB votes; no confirmation render"
                             if rgb_growth_enabled else "disabled" if config.training.rgb_only else "legacy alpha evidence"),
            growth_normalization=config.topology.growth_rgb_normalization if rgb_growth_enabled else None,
            growth_evidence_version=(mi.VoxelReconstruction.rgb_growth_version if rgb_growth_enabled else None),
            growth_unknown_policy="exclude unknown rays; retain reliable rays from the same view"
                                  if rgb_growth_enabled else None,
            growth_freeze_last_epochs=config.topology.growth_freeze_last_epochs,
            growth_freeze_start_epoch=(max(1, config.training.num_epochs - config.topology.growth_freeze_last_epochs + 1)
                                       if config.topology.growth_freeze_last_epochs else None),
            deletion_evidence=("single-removal RGB gain from complete normal training paths; no extra trace"
                               if rgb_deletion_enabled else "disabled" if config.training.rgb_only else "legacy alpha evidence"),
            deletion_evidence_version=(mi.VoxelReconstruction.rgb_deletion_version if rgb_deletion_enabled else None),
            deletion_normalization="reliable-hit mean per view, then reliable-view mean" if rgb_deletion_enabled else None,
            deletion_unknown_policy="RGB evidence: unknown rays abstain individually; unobserved gives no removal vote" if rgb_deletion_enabled else None,
            deletion_confirmation_windows=config.topology.deletion_rgb_confirmation_windows if rgb_deletion_enabled else None,
            low_opacity_deletion={
                "enabled": low_opacity_deletion_enabled,
                "opacity": "sigmoid(Y0 * b0); model DC opacity, not reference alpha",
                "threshold": config.topology.deletion_opacity_threshold,
                "confirmation_windows": config.topology.deletion_opacity_confirmation_windows,
                "check_interval_epochs": config.topology.deletion_interval,
                "policy": "consecutive epoch-end parameter checks; OR with RGB deletion; no extra trace",
                "readback": "only DC opacity parameters at scheduled checks",
                "loss_guarantee": "heuristic pruning; no guarantee of joint RGB loss decrease"},
            deletion_freeze_last_epochs=config.topology.deletion_freeze_last_epochs,
            deletion_freeze_start_epoch=(max(1, config.training.num_epochs - config.topology.deletion_freeze_last_epochs + 1)
                                         if config.topology.deletion_freeze_last_epochs else None),
            deletion_batch_policy="independent online votes; not a guarantee of joint loss decrease" if rgb_deletion_enabled else None,
            reported_loss="mean pixel RGB squared-error sum over channels, then mean over views"
                          if config.training.rgb_only else "mean RGB plus alpha loss over pixels and views")
        performance.metadata["config"] = asdict(config)
        batch_size = effective_chunk_size(config)
        ray_count = config.render.width * config.render.height
        report["training_protocol"].update(rays_per_view=ray_count, rays_per_batch=batch_size,
            batches_per_view=(ray_count + batch_size - 1) // batch_size,
            alpha_readback=("none during RGB-only training" if config.training.rgb_only else
                            "only contour history; scheduled growth evidence consumes device alpha"),
            camera_parameters="opaque runtime inputs",
            gradients_stage_scope="forward paths, loss, hit/candidate gradients, reductions and readbacks",
            native_stage_profiling="synchronized sampled views only; bypasses frozen replay" if config.training.profile_stages else "disabled",
            freeze_native=config.training.freeze_native,
            gpu_reference_cache_mib=config.training.gpu_reference_cache_mib)
        emit(f"Training rays: {ray_count} per view, {batch_size} per batch, "
              f"{report['training_protocol']['batches_per_view']} batch(es) per update", flush=True)
        save_epochs = checkpoint_epochs(config.training.num_epochs, config.training.save_count)
        report["checkpoint_epochs"] = list(save_epochs)
        if primary:
            (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
        dataset = workspace_path(config.data.dataset, WORKSPACE)
        emit("Loading training views and checking reference files:", dataset, flush=True)
        views = load_views(dataset, config.data.cameras, config.render, np)
        signature = dataset_signature(views, dataset, config.data.cameras)
        reference_cache = ReferenceImageCache(config.render.width, config.render.height, mi, np,
                                             config.training.reference_cache_mib,
                                             rgb_only=config.training.rgb_only, background=config.render.background,
                                             reference_compositing=config.data.reference_compositing)
        gpu_reference_cache = GPUReferenceCache(config.render.width, config.render.height, mi, dr, np,
                                                config.training.gpu_reference_cache_mib,
                                                rgb_only=config.training.rgb_only)
        native_replay = NativeLossReplay(dr, enabled=config.training.freeze_native)
        emit(f"Reference image cache: {config.training.reference_cache_mib} MiB RAM, "
              f"{config.training.gpu_reference_cache_mib} MiB GPU, "
              "loaded on first use; dataset must remain unchanged during training", flush=True)
        emit("Native GPU replay:", "enabled" if config.training.freeze_native else "disabled", flush=True)
        emit("Reference compositing:", config.data.reference_compositing,
             "| rendering and RGB loss: linear", flush=True)
        emit("Training supervision:", ("RGB only; DC opacity; "
             + "refinement" + (" + RGB growth" if rgb_growth_enabled else "")
             + (" + RGB deletion" if rgb_deletion_enabled else "")) if config.training.rgb_only
             else "legacy RGB + reference alpha", flush=True)
        if rgb_growth_enabled:
            emit("RGB growth normalization:", config.topology.growth_rgb_normalization, flush=True)
            emit("RGB growth warmup (epochs per level):", config.topology.growth_warmup_epochs,
                 "coarse,", config.topology.growth_fine_warmup_epochs,
                 "fine; no extra forward render", flush=True)
            if config.topology.growth_freeze_last_epochs:
                emit("RGB growth final freeze:", config.topology.growth_freeze_last_epochs,
                     "epochs; no growth from epoch", report["training_protocol"]["growth_freeze_start_epoch"],
                     "through", config.training.num_epochs, flush=True)
        if rgb_deletion_enabled:
            emit("RGB deletion:", config.topology.deletion_warmup_epochs, "warmup epochs per level;",
                 config.topology.deletion_interval, "epoch interval;",
                 config.topology.deletion_rgb_confirmation_windows, "consecutive evidence windows;",
                 config.topology.deletion_protection_epochs, "newborn protection epochs", flush=True)
            if config.topology.deletion_freeze_last_epochs:
                emit("RGB deletion final freeze: epochs", report["training_protocol"]["deletion_freeze_start_epoch"],
                     "through", config.training.num_epochs, flush=True)
        if low_opacity_deletion_enabled:
            emit("Low-opacity deletion: opacity <", config.topology.deletion_opacity_threshold,
                 "at", config.topology.deletion_opacity_confirmation_windows, "consecutive checks;",
                 config.topology.deletion_interval, "epoch interval;",
                 config.topology.deletion_warmup_epochs, "warmup epochs per level;",
                 config.topology.deletion_protection_epochs, "newborn protection epochs", flush=True)
        rng = np.random.default_rng(config.training.seed)
        sampler = make_sampler(config, np)
        completed, global_step = 0, 0
        if checkpoint:
            emit("Restoring voxel model and training state", flush=True)
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
            emit("Initializing voxel model from point cloud:", dataset / config.data.point_cloud, flush=True)
            bounds_min = np.asarray(config.model.bounds_min)
            bounds_max = np.asarray(config.model.bounds_max)
            center, half = (bounds_min + bounds_max) / 2, (bounds_max - bounds_min) / 2 * config.model.bounds_padding
            resolution = config.schedule.start_resolution if config.schedule.enabled else config.model.grid_resolution
            model = mi.VoxelReconstruction(str(dataset / config.data.point_cloud), resolution,
                (center-half).tolist(), (center+half).tolist(), config.model.opacity_threshold,
                config.model.sh_count, config.model.opacity_sh_count)
        if config.training.rgb_only and int(model.opacity_sh_count) != 1:
            raise ValueError("RGB-only model/checkpoint must contain exactly one opacity coefficient per voxel")
        optimizer = FalcorOptimizer(model, config.optimizer)
        topology = TopologyManager(model, len(views), settings=config.topology,
            schedule=replace(config.schedule, save_each_level=False),
            opacity_warmup=config.optimizer.opacity_warmup_epochs, opacity_ramp=config.optimizer.opacity_ramp_epochs)
        coarse_path = model_path.parent / "train" / "0000-coarse.png"
        if not config.evaluation.enabled:
            report["coarse_render"] = dict(status="skipped", reason="evaluation_disabled")
        elif (not config.schedule.enabled
              or config.schedule.start_resolution == config.schedule.target_resolution):
            report["coarse_render"] = dict(status="skipped", reason="no_coarse_level")
        elif int(model.resolution) != config.schedule.start_resolution:
            report["coarse_render"] = dict(status="skipped", reason="coarse_stage_already_completed",
                                          image=str(coarse_path), existing_image=coarse_path.is_file())
        else:
            report["coarse_render"] = dict(status="pending", image=str(coarse_path))

        def capture_coarse_image(completed_epochs):
            if not primary or not is_coarse_training_end(topology, completed_epochs):
                return None
            if config.evaluation.enabled:
                progress_line.finish()
                emit(f"Rendering coarse voxel model before refinement: train view 0, "
                     f"{config.evaluation.spp} spp", flush=True)
            record = save_coarse_training_view(
                config, model, views, model_path, mi, dr, np,
                completed_epochs=completed_epochs, global_step=global_step)
            report["coarse_render"] = record
            if record["status"] == "passed":
                emit("Coarse voxel image:", record["image"], flush=True)
            return record

        if checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            topology.load_state_dict(checkpoint["training_state"]["topology_state"])
            # Loading legacy lifecycle state restores its schedule too. Avoid
            # making additional pre-refinement snapshots under the new policy.
            topology.schedule.save_each_level = False
            if primary and checkpoint["training_state"].get("pending_epoch_transition", False):
                capture_coarse_image(completed)
                replacement, transition = topology.finish_epoch(model, optimizer, completed)
                if replacement is not None:
                    model = restore_model(replacement, mi)
                if transition.get("refined", False):
                    sampler = make_sampler(config, np)
                topology.before_refine_checkpoint = None

        emit("Training epochs by resolution:", json.dumps(topology.budgets), flush=True)

        if distributed is not None:
            # Establish a canonical model/Adam/RNG before the first collective,
            # including legacy checkpoints whose epoch transition was pending.
            initial = None
            if primary:
                initial = dict(model_state={key: np.asarray(value).copy() for key, value in model.snapshot().items()},
                    model_metadata=model_metadata(model), optimizer_state=optimizer.state_dict(),
                    rng_state=rng.bit_generator.state, contour_state=sampler.state_dict(),
                    dataset_signature=signature, config=asdict(config), global_step=global_step)
            initial = distributed.broadcast(initial)
            if initial["dataset_signature"] != signature or initial["config"] != asdict(config):
                raise ValueError("GPU workers have different datasets or training configurations")
            if initial["global_step"] != global_step:
                raise ValueError("GPU workers restored different training steps")
            if not primary:
                model = restore_model(initial, mi)
                optimizer = FalcorOptimizer(model, config.optimizer)
                optimizer.load_state_dict(initial["optimizer_state"])
                rng.bit_generator.state = initial["rng_state"]
                sampler.load_state_dict(initial["contour_state"])
            del initial
            distributed.barrier()
            distributed.heartbeat("replicas_initialized", force=True)

        if primary:
            if checkpoint is None and config.evaluation.enabled:
                emit(f"Rendering initial voxel model: train view 0, {config.evaluation.spp} spp", flush=True)
            report["initial_render"] = save_initial_training_view(
                config, model, views, model_path, mi, dr, np, resumed=checkpoint is not None)
            if report["initial_render"]["status"] == "passed":
                emit("Initial voxel image:", report["initial_render"]["image"], flush=True)
        if distributed is not None:
            # Only rank 0 renders this one image; peers wait before any update.
            report["initial_render"] = distributed.broadcast(report.get("initial_render"))
            distributed.heartbeat("initial_render_finished", force=True)

        def save(path, epochs):
            if not primary:
                return
            save_started = time.perf_counter()
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
            performance.record_checkpoint(path, time.perf_counter()-save_started)

        report.update(view_count=len(views), resumed=checkpoint is not None)
        # A restart writes a new history segment, preserving any later entries
        # from the interrupted run instead of appending ambiguous duplicate epochs.
        report["history_file"] = str(output / history_name)
        profile_path = output / ("profile-" + Path(history_name).stem + ".jsonl")
        if config.training.profile_views or config.training.profile_interval_epochs:
            report["profile_file"] = str(profile_path)
            performance.metadata["profile_file"] = str(profile_path)
            emit("Kernel profile log:", profile_path, flush=True)
        processed_views = 0
        last_training_resolution = None
        performance.set_phase_seconds("setup", time.perf_counter()-run_started)
        publish_performance("running")
        training_loop_started = time.perf_counter()
        emit("Starting training; first kernels may require JIT compilation", flush=True)
        end_epoch = (min(config.training.num_epochs, completed + args.stop_after_epochs)
                     if args.stop_after_epochs is not None else config.training.num_epochs)
        for epoch in range(completed, end_epoch):
            started = time.perf_counter()
            training_resolution = int(model.resolution)
            active_voxels_before = int(model.active_voxel_count)
            entering_resolution = last_training_resolution != training_resolution
            growth_prepare_started = time.perf_counter()
            previous_growth_count = int(getattr(model, "growth_candidate_count", 0))
            if primary:
                rgb_growth_requested = False
                if config.topology.use_rgb_growth:
                    if previous_growth_count or topology._growth_due(epoch + 1):
                        native_replay.clear()
                    rgb_growth_requested = topology.prepare_rgb_growth(model, epoch)
                payload = topology.pending_growth["candidate_state"] if rgb_growth_requested else None
                flags = (topology.needs_rendered_alpha(epoch), topology.collect_deletion(epoch),
                         rgb_growth_requested, payload)
            else:
                flags = None
            if distributed is not None:
                flags = distributed.broadcast(flags)
            alpha_growth_requested, deletion_evidence_collected, rgb_growth_requested, growth_payload = flags
            if not primary and (previous_growth_count or rgb_growth_requested):
                native_replay.clear()
                if rgb_growth_requested:
                    model.set_growth_candidates({key: mi.UInt32(value) if key == "cell_indices" else mi.Float(value)
                                                 for key, value in growth_payload.items()})
                else:
                    model.clear_growth_candidates()
            growth_prepare_seconds = time.perf_counter() - growth_prepare_started
            growth_evidence_requested = alpha_growth_requested or rgb_growth_requested
            interval = config.training.profile_interval_epochs
            order = rng.permutation(len(views)) if config.training.shuffle else np.arange(len(views))
            epoch_loss = 0.
            stage_totals = dict(reference=0., gradients=0., topology_observation=0., optimizer=0.)
            communication_before = collective.statistics() if collective is not None else None
            progress_line.update(f"Epoch {epoch+1}/{config.training.num_epochs} | view 0/{len(views)} (0%) "
                                 f"| grid={int(model.resolution)}", force=True)
            for progress, index in enumerate(order, 1):
                view = views[int(index)]
                view_started = time.perf_counter()
                rgba = reference_cache.get(view["reference_path"])
                gpu_reference = gpu_reference_cache.get(view["reference_path"], rgba)
                reference_done = time.perf_counter()
                reasons = []
                if processed_views < config.training.profile_views:
                    reasons.append("initial_views")
                if interval and progress == 1:
                    if entering_resolution:
                        reasons.append("enter_resolution")
                    if (epoch + 1) % interval == 0:
                        reasons.append("periodic_epoch")
                profiling = bool(reasons)
                native_stages = {} if profiling and config.training.profile_stages else None
                # Sparse kernel-history samples add instrumentation. Wall timings
                # below include tracing, compilation, transfers and waits.
                with dr.kernel_history() if profiling else nullcontext() as history:
                    gradients, loss, rendered_alpha = train_view(model, view, rgba, config, sampler, rng, mi, dr, np,
                        need_rendered_alpha=alpha_growth_requested,
                        collect_deletion_evidence=deletion_evidence_collected,
                        collect_growth_evidence=rgb_growth_requested,
                        stage_profile=native_stages, gpu_reference=gpu_reference,
                        native_replay=native_replay, collective=collective, alpha_on_device=True)
                    gradients_done = time.perf_counter()
                    if primary and rgb_growth_requested:
                        topology.observe_rgb_growth(model, gradients, view_id=int(index), epoch=epoch,
                                                    pixel_count=ray_count)
                    if primary and rgb_deletion_enabled and deletion_evidence_collected:
                        topology.observe_rgb_deletion(model, gradients, view_id=int(index), epoch=epoch,
                                                      pixel_count=ray_count)
                    if primary and not config.training.rgb_only:
                        topology.observe_view(model, gradients, view_id=int(index), epoch=epoch,
                            camera=view["camera"],
                            reference_alpha=gpu_reference.alpha if gpu_reference is not None else rgba[..., 3],
                            rendered_alpha=rendered_alpha,
                            image_shape=(config.render.height, config.render.width))
                    topology_done = time.perf_counter()
                    optimizer.step(model, gradients, sample_count=1, epoch=epoch, coarse_to_fine=config.schedule.enabled)
                    optimizer_done = time.perf_counter()
                stage = dict(reference=reference_done-view_started, gradients=gradients_done-reference_done,
                             topology_observation=topology_done-gradients_done, optimizer=optimizer_done-topology_done)
                for key, duration in stage.items():
                    stage_totals[key] += duration
                processed_views += 1
                global_step += 1
                if distributed is not None:
                    distributed.heartbeat(f"epoch {epoch+1} view {progress}")
                epoch_loss += loss
                progress_line.update(f"Epoch {epoch+1}/{config.training.num_epochs} "
                    f"| view {progress}/{len(views)} ({100*progress/len(views):.0f}%) "
                    f"| cam {int(index)+1} | loss={loss:.6f} | {optimizer_done-view_started:.2f}s/view")
                if profiling and primary:
                    profile = dict(epoch=epoch+1, progress=progress, camera=int(index)+1,
                                   training_resolution=training_resolution, active_voxels=int(model.active_voxel_count),
                                   sample_reasons=reasons, growth_evidence_requested=growth_evidence_requested,
                                   rgb_deletion_evidence_requested=rgb_deletion_enabled and deletion_evidence_collected,
                                   native_execution=("direct_stage_profile" if native_stages is not None else
                                                     "direct_empty_model" if config.training.freeze_native and not model.active_voxel_count else
                                                     "frozen" if config.training.freeze_native else "direct"),
                                   native_replay=native_replay.statistics(),
                                   wall_seconds=optimizer_done-view_started, stage_seconds=stage,
                                   **kernel_profile(history))
                    if native_stages is not None:
                        profile["native_stages"] = native_stages
                        profile["native_stage_timing"] = "synchronized boundaries; instrumented timings, not normal throughput"
                    with profile_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(profile) + "\n")
                    performance.record_profile(profile)
            progress_line.update(f"Epoch {epoch+1}/{config.training.num_epochs} "
                                 f"| view {len(views)}/{len(views)} (100%) | updating topology...", force=True)
            coarse_render = capture_coarse_image(epoch+1)
            topology_started = time.perf_counter()
            new_state, event = topology.finish_epoch(model, optimizer, epoch+1) if primary else (None, None)
            topology.before_refine_checkpoint = None
            if distributed is not None:
                transition = distributed.broadcast((new_state, event,
                    optimizer.state_dict() if new_state is not None else None,
                    coarse_render) if primary else None)
                new_state, event, new_optimizer, coarse_render = transition
                if coarse_render is not None:
                    report["coarse_render"] = coarse_render
                    distributed.heartbeat("coarse_render_finished", force=True)
            if new_state is not None:
                # Drop recordings retaining the old occupancy/parameter buffers,
                # including same-size replacements and transitions after pruning.
                native_replay.clear()
                model = restore_model(new_state, mi)
                if not primary:
                    optimizer = FalcorOptimizer(model, config.optimizer)
                    optimizer.load_state_dict(new_optimizer)
            if event.get("refined", False):
                # Falcor resets contour proposals whenever grid resolution changes.
                sampler = make_sampler(config, np)
            entry = {"epoch": epoch+1, "global_step": global_step, "mean_loss": epoch_loss / len(views),
                     "rgb_only": config.training.rgb_only,
                     "seconds": time.perf_counter()-started, "resolution": int(model.resolution),
                     "training_resolution": training_resolution, "active_voxels_before": active_voxels_before,
                     "growth_evidence_requested": growth_evidence_requested,
                     "rgb_growth_evidence_requested": rgb_growth_requested,
                     "deletion_evidence_collected": deletion_evidence_collected,
                     "rgb_deletion_evidence_requested": rgb_deletion_enabled and deletion_evidence_collected,
                     "active_voxels": int(model.active_voxel_count), "topology": event,
                     "reference_cache": reference_cache.statistics(),
                     "gpu_reference_cache": gpu_reference_cache.statistics(),
                     "native_replay": native_replay.statistics(),
                     "topology_execution": topology.statistics() if primary else None,
                     "view_stage_seconds": stage_totals,
                     "epoch_growth_prepare_seconds": growth_prepare_seconds,
                     "epoch_topology_seconds": time.perf_counter()-topology_started}
            if coarse_render is not None:
                entry["coarse_render"] = coarse_render
                entry["epoch_coarse_render_seconds"] = coarse_render.get("seconds", 0.)
            if collective is not None:
                communication_after = collective.statistics()
                entry["collective"] = communication_after
                entry["epoch_communication_seconds"] = communication_after["seconds"] - communication_before["seconds"]
                entry["epoch_communication_bytes"] = communication_after["payload_bytes"] - communication_before["payload_bytes"]
            if primary:
                with (output / history_name).open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(entry) + "\n")
            report["last_epoch"] = entry
            performance.record_epoch(entry)
            last_training_resolution = training_resolution
            progress_line.finish(f"Epoch {epoch+1}/{config.training.num_epochs} done "
                f"| loss={entry['mean_loss']:.6f} | {entry['seconds']:.2f}s "
                f"| grid={entry['resolution']} | voxels={entry['active_voxels']}")
            if epoch+1 in save_epochs and epoch+1 < config.training.num_epochs:
                save(output / f"epoch-{epoch+1:05d}.npz", epoch+1)
                emit("Saved checkpoint:", output / f"epoch-{epoch+1:05d}.npz", flush=True)
            performance.set_phase_seconds("training_loop", time.perf_counter()-training_loop_started)
            publish_performance("running", training_status="running",
                              gpu_reference_cache=entry["gpu_reference_cache"], native_replay=entry["native_replay"],
                              topology_execution=entry["topology_execution"])
        performance.set_phase_seconds("training_loop", time.perf_counter()-training_loop_started)
        training_loop_started = None
        if end_epoch < config.training.num_epochs:
            stopped_checkpoint = output / f"epoch-{end_epoch:05d}.npz"
            if end_epoch not in save_epochs:
                save(stopped_checkpoint, end_epoch)
            if distributed is not None:
                distributed.barrier()
            report.update(status="stopped", training_status="stopped", completed_epochs=end_epoch,
                          global_step=global_step, checkpoint=str(stopped_checkpoint),
                          evaluation={"status": "skipped", "reason": "bounded training trial"})
            emit("Stopped at a complete epoch; resume checkpoint:", stopped_checkpoint, flush=True)
            return config, model_path, output, report, performance
        # Publish only the completed reconstruction, using the same portable,
        # atomic checkpoint format understood by independent evaluation scripts.
        export_started = time.perf_counter()
        save(model_path, config.training.num_epochs)
        emit("Saved reconstructed voxel model:", model_path, flush=True)
        if distributed is not None:
            distributed.barrier()
        report.update(status="running" if config.evaluation.enabled and primary else "passed", training_status="passed",
                      evaluation={"status": "pending" if config.evaluation.enabled else "skipped"},
                      completed_epochs=config.training.num_epochs, global_step=global_step,
                      checkpoint=str(model_path),
                      reconstructed_model={"path": str(model_path), "sha256": file_sha256(model_path),
                          "completed_epochs": config.training.num_epochs, "global_step": global_step})
        performance.set_phase_seconds("final_export", time.perf_counter()-export_started)
    except KeyboardInterrupt:
        progress_line.finish()
        report.update(status="interrupted", training_status="interrupted")
        raise
    except Exception:
        progress_line.finish()
        report.update(status="failed", training_status="failed", error=traceback.format_exc())
        emit(report["error"], file=sys.stderr, flush=True)
        if distributed is not None:
            raise
    finally:
        progress_line.finish()
        if reference_cache is not None:
            report["reference_cache"] = reference_cache.statistics()
        if gpu_reference_cache is not None:
            report["gpu_reference_cache"] = gpu_reference_cache.statistics()
            gpu_reference_cache.clear()
        if native_replay is not None:
            report["native_replay"] = native_replay.statistics()
            native_replay.clear()
        if topology is not None and primary:
            report["topology_execution"] = topology.statistics()
        if collective is not None:
            report["collective"] = collective.statistics()
            collective.close(abort=report["training_status"] not in ("passed", "stopped"))
        if training_loop_started is not None:
            performance.set_phase_seconds("training_loop", time.perf_counter()-training_loop_started)
        report["training_finished_utc"] = datetime.now(timezone.utc).isoformat()
        if report["status"] != "running":
            report["finished_utc"] = report["training_finished_utc"]
        if primary:
            (output / "training-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
        publish_performance(report["status"], training_status=report["training_status"],
            reference_cache=report.get("reference_cache"), gpu_reference_cache=report.get("gpu_reference_cache"),
            native_replay=report.get("native_replay"), collective=report.get("collective"),
            topology_execution=report.get("topology_execution"),
            training_report=str(output / "training-report.json"))
    return config, model_path, output, report, performance


def main(distributed=None):
    primary = distributed is None or distributed.rank == 0
    emit = builtins.print if primary else lambda *a, **k: None
    config, model_path, output, report, performance = run_training(distributed)
    if report["training_status"] == "stopped":
        return 0
    if report["training_status"] != "passed":
        return 1
    if not config.evaluation.enabled:
        return 0
    evaluation = final_evaluation_config(config, model_path)
    evaluation_report = model_path.parent / "evaluation-report.json"
    report["evaluation"] = {"status": "running", "spp": evaluation.spp,
        "max_views_per_split": evaluation.max_views, "splits": list(evaluation.splits),
        "gpu_workers": distributed.world_size if distributed else 1,
        "output": evaluation.output_directory, "report": str(evaluation_report)}
    if primary:
        (output / "training-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    limit = "all views" if evaluation.max_views is None else f"first {evaluation.max_views} views per split"
    emit(f"Starting final train/test evaluation: {evaluation.spp} spp, {limit}"
         + (f", {distributed.world_size} GPUs by view" if distributed else ""), flush=True)
    evaluation_started = time.perf_counter()
    try:
        # run_training's model, optimizer and gradients are out of scope. Reclaim
        # their allocations before restoring the published checkpoint for evaluation.
        gc.collect()
        import drjit as dr
        dr.sync_thread()
        dr.flush_malloc_cache()
        from evaluate import run_evaluation
        if distributed is not None:
            distributed.heartbeat("evaluation_start", force=True)
        result = run_evaluation(evaluation, progress_callback=distributed.heartbeat if distributed else None,
                                distributed=distributed)
        report["evaluation"]["status"] = "passed" if result == 0 else "failed"
        report["status"] = "passed" if result == 0 else "failed"
        if result != 0:
            emit(f"Training completed and model is saved at {model_path}; "
                  f"evaluation failed. See {evaluation_report}", file=sys.stderr, flush=True)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        report["evaluation"]["status"] = "interrupted"
        raise
    except Exception:
        report["status"] = "failed"
        report["evaluation"].update(status="failed", error=traceback.format_exc())
        emit(report["evaluation"]["error"], file=sys.stderr, flush=True)
        emit(f"Training completed; the saved model is preserved at {model_path}", file=sys.stderr, flush=True)
        if distributed is not None:
            raise
    finally:
        performance.set_phase_seconds("evaluation", time.perf_counter()-evaluation_started)
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        if primary:
            (output / "training-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
            performance.write(report["status"], training_status=report["training_status"],
                evaluation=report["evaluation"], reference_cache=report.get("reference_cache"),
                gpu_reference_cache=report.get("gpu_reference_cache"), native_replay=report.get("native_replay"),
                collective=report.get("collective"),
                topology_execution=report.get("topology_execution"),
                training_report=str(output / "training-report.json"), evaluation_report=str(evaluation_report))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
