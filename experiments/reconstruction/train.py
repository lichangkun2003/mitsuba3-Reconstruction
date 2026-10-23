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

from preview import (DataConfig, ModelConfig, RenderConfig, RuntimeConfig, PreviewConfig,
                     validate_config, validate_traversal, native_traversal_settings,
                     scene_background_protocol)
from reconstruction_common import (REPOSITORY, WORKSPACE, ContourSampler, UniformSampler, ReferenceImageCache,
    generate_ray_batch, prepare_camera, initialize_runtime, load_views, model_metadata,
    prepare_dataset, initialize_model,
    restore_model, workspace_path, file_sha256, render_view, write_png)
from reconstruction_checkpoint import load_checkpoint, save_checkpoint
from reconstruction_color import validate_reference_compositing
from reconstruction_experiment import (allocate_training_output, resumed_training_output,
                                       finalize_training_output, relocate_experiment_paths)
from reconstruction_optimizer import FalcorOptimizer, OptimizerConfig
from reconstruction_checks import DeferredViewChecks
from reconstruction_performance import PerformanceRecorder
from reconstruction_progress import ProgressLine
from reconstruction_reference import GPUReferenceCache
from reconstruction_replay import NativeLossReplay
from reconstruction_growth_rgb import RGB_GROWTH_EVIDENCE_VERSION
from reconstruction_topology import (TopologyManager, TopologySettings, CoarseToFineConfig,
                                     level_budgets, validate_growth_opacity)


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
    sync_view_checks: bool = False  # Diagnostic: read loss/validity after every view.
    seed: int = 42
    shuffle: bool = True
    save_count: int = 3  # Total snapshots, including the final model.npz.
    output: str | None = None  # None: automatically allocate a daily experiment.


@dataclass
class GradientSettings:
    alpha_loss_weight: float = .5
    distortion_loss_weight: float = 0.  # Optional hard-hit depth concentration; 0 preserves the baseline.
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
    train_max_views: int | None = 10  # First N training cameras; None or 0: all.
    test_max_views: int | None = None  # Evaluate the complete held-out test split.
    max_views: int | None = None  # Legacy explicit override of both split limits; 0: all.
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
CONFIG.data.dataset = "datasets/bicycle"
CONFIG.data.point_cloud = "init_points.ply"
CONFIG.data.cameras = "transforms_train.json"
CONFIG.data.reference_compositing = "srgb"  # 参考 PNG 先合成背景，再解码；渲染与损失仍在线性空间。
CONFIG.data.scene_mode = "auto"  # 合成场景：有界白色背景；COLMAP：无界远景层、无白色背景尾项。
CONFIG.data.background_layers = 8  # 初始化远景层数；独立于 render.max_hits，升层时随体素细分。
CONFIG.data.background_opacity = 0.5  # 远景新生体素 opacity；允许后续优化和删除。

CONFIG.render.width = 800
CONFIG.render.height = 800
CONFIG.render.spp = 1  # 训练固定每视角 1 spp；多 spp 出图使用评价脚本。
CONFIG.render.traversal = "octree"  # 默认八叉占用树跳过空区域；"dda" 仅用于对照。
CONFIG.training.rgb_only = True  # 参考 alpha 仅用于读图时合成 RGB，不进入训练或生长投票。
CONFIG.model.opacity_sh_count = 1  # 仅优化 DC logit；opacity = sigmoid(Y0 * b0)。
CONFIG.gradient.alpha_loss_weight = 0.
CONFIG.gradient.alpha_geometry_weight = 0.
CONFIG.gradient.background_carve_weight = 0.
CONFIG.gradient.distortion_loss_weight = 0.  # 显式使用 --distortion-weight 0.01 开启深度集中正则；0 为原基线。
CONFIG.gradient.geometry_transition_only = False  # hit/miss 均保留正负 RGB 几何信号。
CONFIG.gradient.geometry_two_sided_boundary = False  # 保留原 sigmoid，不叠加额外带宽门控。
CONFIG.contour.enabled = False  # 所有像素均匀采样，不读取 alpha 轮廓标签。
CONFIG.topology.enable_growth = True
CONFIG.topology.enable_deletion = True
CONFIG.topology.use_rgb_deletion = True  # 复用正常反向合成，按移除单体素的 RGB 收益投票。
CONFIG.topology.use_low_opacity_deletion = True  # 补充清理持续低不透明度体素，不要求正 RGB 删除收益。
CONFIG.topology.deletion_opacity_threshold = .1  # 实际 sigmoid(Y0*b0) < 0.1；仍需成熟保护和连续窗口确认。
CONFIG.topology.deletion_opacity_confirmation_windows = 2  # 连续两个删除检查窗口低于阈值才清理。
CONFIG.topology.deletion_warmup_epochs = 20  # 每个层级先优化 20 轮，再按周期收集一轮全部视角。
CONFIG.topology.deletion_interval = 10
CONFIG.topology.deletion_protection_epochs = 20  # 新生体素先优化 20 轮。
CONFIG.topology.deletion_freeze_last_epochs = 0  # 不在末尾冻结删除，继续按周期清理至训练结束。
CONFIG.topology.deletion_rgb_confirmation_windows = 2  # 连续两个证据窗口达标才删除。
CONFIG.topology.deletion_rgb_min_support_views = 3
CONFIG.topology.deletion_rgb_max_conflict_views = 1  # 一个可靠反对视角即可阻止本窗口删除。
CONFIG.topology.deletion_rgb_gain_epsilon = 1e-7
CONFIG.topology.deletion_rgb_min_mean_gain = 1e-7
CONFIG.topology.use_growth_evidence = False
CONFIG.topology.use_rgb_growth = True  # 复用训练前向的 RGB 收益票，不额外渲染。
CONFIG.topology.growth_inherit_opacity = True
CONFIG.topology.growth_opacity_scale = .8  # 出生 opacity 继承父体素概率值的 80%，不是缩放 logit。
CONFIG.topology.growth_initial_opacity = .3  # 继承模式的出生上限：min(0.3, 0.8 * parent_opacity)。
CONFIG.topology.growth_rgb_normalization = "observed"  # 有效命中射线均值，再对可靠观测视角求均值。
CONFIG.topology.growth_warmup_epochs = 20  # 最粗层级先优化 20 轮，再按生长周期检查。
CONFIG.topology.growth_fine_warmup_epochs = 20  # 每个后续层级重新 warmup。
CONFIG.topology.growth_freeze_last_epochs = 50  # 最后 50 轮停止生长，继续优化和删除；0 关闭生长冻结。
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
CONFIG.training.reference_cache_mib = 4096  # 内存缓存上限；bicycle 全部 RGB 参考约 1967 MiB，避免每轮重新解码。
CONFIG.training.gpu_reference_cache_mib = 4096  # 显存缓存上限，按需分配；容纳全部参考图以避免反复上传，0 关闭。
CONFIG.training.freeze_native = True  # 重放原生计算；拓扑变化后重新记录，分段诊断时旁路。

CONFIG.schedule.enabled = True
CONFIG.schedule.start_resolution = 256
CONFIG.schedule.target_resolution = 512
CONFIG.schedule.level_weights = (6, 4)  # 粗、细层级轮数比例；1500 轮分成 900 / 600。
CONFIG.training.num_epochs = 1500  # 整个实验总轮数，每轮遍历全部训练视角。
CONFIG.training.save_count = 3  # 包含最终 model.npz，共保存三次。
CONFIG.training.output = None  # 自动创建 outputs/场景-MM.DD-E编号/training。
# 需要手动指定目录时，将 None 改为路径字符串，或传 --output-dir。
CONFIG.evaluation.enabled = True  # 最终 model.npz 保存后，自动评价训练集和测试集。
CONFIG.evaluation.spp = 64  # 仅用于评价出图，不改变训练的 1 spp。
CONFIG.evaluation.train_max_views = 10  # 训练集前 10 个视角。
CONFIG.evaluation.test_max_views = None  # 测试集全部视角，适用于合成场景和 COLMAP/Mip-NeRF 360。
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
    parser.add_argument("--distortion-weight", type=float,
                        help="Nonnegative hard-hit distortion loss weight (default: 0, disabled); try 0.01 for an ablation")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--stop-after-epochs", type=int,
                        help="Stop after N additional complete epochs, saving a resumable checkpoint without changing the full schedule or publishing model.npz")
    parser.add_argument("--dataset")
    parser.add_argument("--scene-mode", choices=("auto", "bounded", "unbounded"),
                        help="auto: bounded synthetic / unbounded COLMAP; override for a new scene")
    parser.add_argument("--background-layers", type=int, help="Initial radial shell width in voxels (default: 8)")
    parser.add_argument("--background-opacity", type=float, help="Initial far-field voxel opacity (default: 0.5)")
    parser.add_argument("--output-dir", help="Override automatic outputs/SCENE-MM.DD-Eindex/training; on resume the default reuses the checkpoint's experiment")
    parser.add_argument("--build-dir")
    parser.add_argument("--gpu", type=int, help="Local physical index; ignored under Slurm")
    parser.add_argument("--chunk-size", type=int, help="0 (default): full image; positive: maximum rays per batch")
    parser.add_argument("--traversal", choices=("octree", "dda"),
                        help="Spatial traversal (default: octree); dda is the comparison path")
    parser.add_argument("--profile-views", type=int, help="Record kernel/cache timings for the first N processed views")
    parser.add_argument("--profile-interval-epochs", type=int, help="Sample one view every N epochs and on entering a resolution (default: 50); 0 disables automatic samples")
    parser.add_argument("--profile-stages", action=argparse.BooleanOptionalAction, default=None,
                        help="Synchronize and time native phases in sampled views; --no-profile-stages keeps only aggregate kernel profiling")
    parser.add_argument("--reference-cache-mib", type=int, help=f"RAM budget for decoded reference images in MiB (default: {CONFIG.training.reference_cache_mib}); 0 disables caching")
    parser.add_argument("--gpu-reference-cache-mib", type=int, help=f"GPU reference-image cache budget in MiB (default: {CONFIG.training.gpu_reference_cache_mib}); 0 disables caching")
    parser.add_argument("--freeze-native", action=argparse.BooleanOptionalAction, default=None,
                        help="Replay native GPU work without repeated tracing (default: enabled); stage diagnostics run directly")
    parser.add_argument("--sync-view-checks", action=argparse.BooleanOptionalAction, default=None,
                        help="Diagnostic per-view loss/validity readback; default keeps guards on GPU and reports at epoch/profile boundaries")
    parser.add_argument("--gpu-topology", action=argparse.BooleanOptionalAction, default=None,
                        help="Build RGB growth candidates and update topology on GPU (default); --no-gpu-topology selects the NumPy reference")
    parser.add_argument("--skip-evaluation", action="store_true", help="Save the final model without automatic train/test evaluation")
    parser.add_argument("--eval-spp", type=int, help="Samples per pixel for final evaluation (default: 64)")
    parser.add_argument("--eval-train-max-views", type=int,
                        help="Evaluate the first N training cameras (default: 10); 0 evaluates all")
    parser.add_argument("--eval-test-max-views", type=int,
                        help="Evaluate the first N held-out test cameras (default: all); 0 evaluates all")
    parser.add_argument("--eval-max-views", type=int,
                        help="Deprecated: override BOTH train/test limits; 0 evaluates all. Prefer the split-specific options")
    return parser.parse_args()


def apply_arguments(config, args, values):
    update(config, values)
    for value, group, key in ((args.epochs, "training", "num_epochs"),
        (args.dataset, "data", "dataset"), (args.output_dir, "training", "output"),
        (getattr(args, "scene_mode", None), "data", "scene_mode"),
        (getattr(args, "background_layers", None), "data", "background_layers"),
        (getattr(args, "background_opacity", None), "data", "background_opacity"),
        (args.build_dir, "runtime", "build_dir"), (args.gpu, "runtime", "gpu"),
        (args.chunk_size, "training", "chunk_size"),
        (getattr(args, "traversal", None), "render", "traversal"),
        (args.profile_views, "training", "profile_views"),
        (args.profile_interval_epochs, "training", "profile_interval_epochs"),
        (args.profile_stages, "training", "profile_stages"),
        (args.reference_cache_mib, "training", "reference_cache_mib"),
        (args.gpu_reference_cache_mib, "training", "gpu_reference_cache_mib"),
        (args.freeze_native, "training", "freeze_native"),
        (getattr(args, "sync_view_checks", None), "training", "sync_view_checks"),
        (getattr(args, "gpu_topology", None), "topology", "gpu_topology"),
        (args.rgb_only, "training", "rgb_only"),
        (getattr(args, "distortion_weight", None), "gradient", "distortion_loss_weight"),
        (args.eval_spp, "evaluation", "spp"),
        (getattr(args, "eval_train_max_views", None), "evaluation", "train_max_views"),
        (getattr(args, "eval_test_max_views", None), "evaluation", "test_max_views"),
        (args.eval_max_views, "evaluation", "max_views")):
        if value is not None:
            setattr(getattr(config, group), key, value)
    if args.skip_evaluation:
        config.evaluation.enabled = False
    if config.evaluation.enabled and config.evaluation.max_views is not None:
        print("Deprecated evaluation.max_views / --eval-max-views explicitly overrides BOTH "
              "train and test limits, including split-specific options. Remove it (or set "
              "evaluation.max_views=null) to use train_max_views=10 and test_max_views=null "
              "for full test-set evaluation.", file=sys.stderr)
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
    if (not config.training.rgb_only and not getattr(args, "resume", None) and
            "growth_inherit_opacity" not in values.get("topology", {})):
        config.topology.growth_inherit_opacity = False
        if "growth_initial_opacity" not in values.get("topology", {}):
            config.topology.growth_initial_opacity = TopologySettings().growth_initial_opacity
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
    # prepare_dataset resolves the ray tail and reference composite together:
    # white for bounded scenes, black for the explicit unbounded background.
    if not isinstance(config.training.rgb_only, bool):
        raise ValueError("training.rgb_only must be a boolean")
    if not config.training.rgb_only and config.data.reference_compositing != "linear":
        raise ValueError("data.reference_compositing='srgb' requires RGB-only training; "
                         "legacy alpha-supervised training uses linear compositing")
    if config.topology.use_rgb_growth and not config.training.rgb_only:
        raise ValueError("RGB growth evidence requires training.rgb_only=true")
    if not isinstance(config.topology.use_rgb_deletion, bool):
        raise ValueError("topology.use_rgb_deletion must be a boolean")
    if not isinstance(config.topology.gpu_topology, bool):
        raise ValueError("topology.gpu_topology must be a boolean")
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
    validate_growth_opacity(config.topology, config.model.opacity_sh_count)
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
    if not isinstance(config.training.sync_view_checks, bool):
        raise ValueError("training.sync_view_checks must be a boolean")
    if isinstance(config.gradient.distortion_loss_weight, bool):
        raise ValueError("gradient.distortion_loss_weight must be finite and nonnegative")
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
        train_max_views=config.evaluation.train_max_views, test_max_views=config.evaluation.test_max_views,
        seed=config.evaluation.seed, chunk_size=config.evaluation.chunk_size,
        traversal=config.render.traversal,
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
    core = getattr(topology, "core_resolution", topology.resolution)
    return (topology.schedule.enabled
            and core == topology.schedule.start_resolution
            and core < topology.schedule.target_resolution
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
    # COLMAP holds out global filename index 0. Select the first training view,
    # and keep its global index so the final evaluation uses the same samples.
    view = views[0]
    view_index = int(view["index"])
    seed = int(np.random.SeedSequence([config.evaluation.seed, 0, view_index]).generate_state(1)[0])
    started = time.perf_counter()
    rgb, _ = render_view(model, config.render, view["camera"], config.evaluation.spp,
                         seed, config.evaluation.chunk_size, mi, dr, np)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_png(path, rgb, mi, np)
    return dict(status="passed", image=str(path), view_index=view_index,
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
    if config.gradient.distortion_loss_weight == 0:
        # The disabled path stays usable with existing native builds and keeps
        # its original replay key. The native default is also zero.
        result.pop("distortion_loss_weight")
    result.update(rgb_only=config.training.rgb_only,
        foreground_alpha_min=config.topology.foreground_alpha_min,
        background_alpha_max=config.topology.background_alpha_max,
        collect_deletion_evidence=config.topology.enable_deletion and not config.topology.use_rgb_deletion,
        collect_rgb_deletion_evidence=config.topology.enable_deletion and config.topology.use_rgb_deletion,
        transmittance_threshold=config.render.transmittance_threshold,
        max_hits=config.render.max_hits, entry_offset_fraction=config.render.entry_offset_fraction,
        use_octree=config.render.traversal == "octree")
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


def host_topology_payload(value, dr, np):
    """Materialize only at the existing cross-process topology boundary.

    Single-GPU lifecycle arrays remain on device. WorkerContext's control pipe
    cannot serialize CUDA allocations, so two-GPU topology transfers stay CPU
    messages; ordinary gradient reductions remain NCCL device collectives.
    """
    if dr.is_array_v(type(value)):
        return np.array(value, copy=True)
    if isinstance(value, dict):
        return {key: host_topology_payload(item, dr, np) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(host_topology_payload(item, dr, np) for item in value)
    if isinstance(value, list):
        return [host_topology_payload(item, dr, np) for item in value]
    return value


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
    distortion_enabled = config.gradient.distortion_loss_weight > 0
    if distortion_enabled:
        accumulator.update(rgb_loss_sum=mi.Float(0), distortion_loss_sum=mi.Float(0))
    total_loss, finite = mi.Float(0), mi.Bool(True)
    camera = prepare_camera(config.render, view["camera"], mi, dr)
    native_settings = native_traversal_settings(model, settings(config))
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
        if distortion_enabled:
            for name in ("rgb_loss", "distortion_loss"):
                if name not in result:
                    raise RuntimeError("Distortion loss components are unavailable; incrementally rebuild this source checkout")
                accumulator[name + "_sum"] += dr.sum(result[name])
                finite &= dr.all(dr.isfinite(result[name]))
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
    if distortion_enabled:
        finite &= dr.isfinite(accumulator["rgb_loss_sum"]) & dr.isfinite(accumulator["distortion_loss_sum"])
    return accumulator, total_loss, finite, raw_alpha, estimated_alpha


def finalize_view(partial, view, config, sampler, token, mi, dr, np, *, alpha_on_device=False,
                  loss_components=None, deferred_checks=None):
    """Validate a globally reduced view and observe its contour history once."""
    accumulator, total_loss, finite, raw_alpha, estimated_alpha = partial
    count = config.render.width * config.render.height
    # Read loss and the combined validity flag together, rather than performing
    # a separate host synchronization for every parameter group and ray batch.
    distortion_enabled = config.gradient.distortion_loss_weight > 0
    if deferred_checks is not None:
        # Four scalar statistics per view; no image/parameter readback. A device
        # guard blocks invalid updates immediately, even before host reporting.
        mean_loss = total_loss / count
        rgb = accumulator.pop("rgb_loss_sum") / count if distortion_enabled else mean_loss
        distortion = accumulator.pop("distortion_loss_sum") / count if distortion_enabled else mi.Float(0.)
        finite &= dr.isfinite(mean_loss) & dr.isfinite(rgb) & dr.isfinite(distortion)
        summary = dr.concat([mean_loss, mi.Float(finite), rgb, distortion])
        deferred_checks.add(view["index"], summary)
        accumulator["_update_allowed"] = finite & (mean_loss >= 0)
        if token is not None:
            # Legacy contour history consumes a host image; preserve its original
            # validation and observation ordering when explicitly requested.
            deferred_checks.read(rgb_only=config.training.rgb_only,
                                 distortion_weight=config.gradient.distortion_loss_weight)
        if raw_alpha is not None:
            sampler.observe(token, np.asarray(raw_alpha))
        alpha = estimated_alpha
        if alpha is not None and not alpha_on_device:
            alpha = np.asarray(alpha).reshape(config.render.height, config.render.width)
        return accumulator, mean_loss, alpha
    summary = dr.zeros(mi.Float, 4 if distortion_enabled else 2)
    dr.scatter(summary, total_loss / count, 0)
    dr.scatter(summary, mi.Float(finite), 1)
    if distortion_enabled:
        dr.scatter(summary, accumulator.pop("rgb_loss_sum") / count, 2)
        dr.scatter(summary, accumulator.pop("distortion_loss_sum") / count, 3)
    dr.eval(summary)
    host_summary = np.asarray(summary)
    mean_loss = float(host_summary[0])
    if not bool(host_summary[1]) or not math.isfinite(mean_loss):
        raise RuntimeError(f"Nonfinite loss/alpha/gradient at training view {view['index']}")
    if mean_loss < 0:
        raise RuntimeError(f"Negative squared-error loss at training view {view['index']}: {mean_loss}; "
                           "check GPU reduction/readback correctness before continuing")
    if loss_components is not None:
        # Distortion is deliberately not evaluated at weight zero. In RGB-only
        # mode the baseline total is already the pure RGB loss, without another
        # reduction or host transfer. Legacy alpha loss cannot be separated here.
        rgb_loss = float(host_summary[2]) if distortion_enabled else mean_loss if config.training.rgb_only else None
        distortion_loss = float(host_summary[3]) if distortion_enabled else 0.
        if (rgb_loss is not None and not math.isfinite(rgb_loss)) or not math.isfinite(distortion_loss):
            raise RuntimeError(f"Nonfinite loss component at training view {view['index']}")
        loss_components.update(rgb_loss=rgb_loss, distortion_loss=distortion_loss,
                               weighted_distortion_loss=config.gradient.distortion_loss_weight * distortion_loss)
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
               collect_growth_evidence=False, loss_components=None, deferred_checks=None):
    """Compute one full-view gradient, optionally over synchronized pixel shards.

    All ranks consume the same full-view sampling stream and contour history.
    A collective combines gradient/validity SUMs, evidence MAXs and disjoint
    alpha buffers before the single logical Adam update performed by the caller.
    Direct callers retain one small loss/finite readback per view. The training
    loop supplies deferred_checks to report at epoch/profile boundaries. Contour
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
        if config.gradient.distortion_loss_weight > 0:
            finite &= dr.isfinite(accumulator["rgb_loss_sum"]) & dr.isfinite(accumulator["distortion_loss_sum"])
        partial = accumulator, total_loss, finite, raw_alpha, estimated_alpha
    return finalize_view(partial, view, config, sampler, token, mi, dr, np,
                         alpha_on_device=alpha_on_device, loss_components=loss_components,
                         deferred_checks=deferred_checks)


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


def dataset_signature(views, dataset, camera_file, initialization=None):
    if initialization is None:
        cameras = file_sha256(dataset / camera_file)
    else:
        source = dataset / initialization["source_model"]
        cameras = {name: file_sha256(source / name) for name in ("cameras.bin", "images.bin")}
    signature = {"cameras": cameras,
                 "references": [file_sha256(Path(view["reference_path"])) for view in views]}
    if initialization is not None:
        signature["initialization"] = initialization
    return signature


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
    if name == "render" and values is not None:
        # Traversal is an acceleration choice and can change on resume; it does
        # not change leaf shading, gradients, candidate evidence, or sampling.
        return {key: value for key, value in values.items() if key != "traversal"}
    if name == "data" and values is not None:
        # Omitted/None COLMAP options are identical for historical synthetic runs.
        return {"reference_compositing": "linear", "scene_mode": "auto",
            "background_layers": 8, "background_opacity": .5, **{
            key: value for key, value in values.items()
            if key not in ("images", "initialization") or value is not None}}
    defaults = {"topology": TopologySettings, "gradient": GradientSettings}
    if values is not None and name in defaults:
        normalized = {**asdict(defaults[name]()), **values}
        if name == "topology":
            # CPU and GPU execute the same topology policy; the implementation
            # backend may be switched for a resumed diagnostic comparison.
            normalized.pop("gpu_topology", None)
        return normalized
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


def resumed_reconstruction_time(checkpoint):
    """Recover elapsed active wall time without including time between runs."""
    if checkpoint is None:
        return 0., True
    state = checkpoint["training_state"]
    if "reconstruction_seconds" not in state:
        return 0., state.get("completed_epochs", 0) == 0
    seconds = state["reconstruction_seconds"]
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds) or seconds < 0):
        raise ValueError("Checkpoint reconstruction_seconds must be finite and nonnegative")
    complete = state.get("reconstruction_time_complete", True)
    if not isinstance(complete, bool):
        raise ValueError("Checkpoint reconstruction_time_complete must be a boolean")
    return float(seconds), complete


def finalize_experiment_output(config, model_path, output, report, performance, distributed=None):
    """Name completed experiments only after training closes its output handles."""
    if report["training_status"] != "passed" or report.get("trained_epochs_this_run", 0) == 0:
        return model_path, output
    primary = distributed is None or distributed.rank == 0
    old_root = model_path.parent
    if distributed is not None:
        distributed.barrier()
    result = None
    if primary:
        seconds = report["reconstruction_seconds"]
        renamed = output
        naming = dict(status="unchanged", original=str(old_root), path=str(old_root),
                      seconds=seconds, minutes=max(1, math.ceil(seconds / 60)))
        try:
            renamed = finalize_training_output(output, seconds, WORKSPACE)
            if renamed != output:
                naming.update(status="renamed", path=str(final_model_path(renamed).parent))
        except OSError as error:
            # A cosmetic rename must never overwrite another result or discard
            # a successfully saved reconstruction.
            naming.update(status="failed", error=str(error))
            print(f"Experiment directory rename skipped: {error}", file=sys.stderr, flush=True)
        portable = (renamed.relative_to(WORKSPACE).as_posix()
                    if renamed.is_relative_to(WORKSPACE) else str(renamed))
        result = str(final_model_path(renamed)), str(renamed), portable, naming
    if distributed is not None:
        result = distributed.broadcast(result)
    new_model, new_output, portable, naming = result
    model_path, output = Path(new_model), Path(new_output)
    new_root = model_path.parent
    if new_root != old_root:
        relocated = relocate_experiment_paths(report, old_root, new_root, WORKSPACE)
        report.clear()
        report.update(relocated)
        performance.path = Path(relocate_experiment_paths(
            str(performance.path), old_root, new_root, WORKSPACE))
        for name in ("metadata", "checkpoints", "epochs", "profiles"):
            setattr(performance, name, relocate_experiment_paths(
                getattr(performance, name), old_root, new_root, WORKSPACE))
    config.training.output = portable
    report["experiment_directory"] = naming
    performance.metadata["reconstruction_seconds"] = report["reconstruction_seconds"]
    performance.metadata["reconstruction_time_complete"] = report.get("reconstruction_time_complete", True)
    if primary:
        # Model/checkpoint bytes and append-only history remain immutable.
        # Resume locates them by their actual path, as after a machine transfer.
        (output / "config.json").write_text(
            json.dumps(asdict(config), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (output / "training-report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        performance.write(report["status"], training_status=report["training_status"],
            reference_cache=report.get("reference_cache"), gpu_reference_cache=report.get("gpu_reference_cache"),
            native_replay=report.get("native_replay"), collective=report.get("collective"),
            topology_execution=report.get("topology_execution"),
            training_report=str(output / "training-report.json"))
        if naming["status"] == "renamed":
            print("Completed experiment:", new_root, flush=True)
            print("Reconstruction time:", f"{naming['minutes']}min (before final evaluation)", flush=True)
            if not report.get("reconstruction_time_complete", True):
                print("Timing excludes unrecorded work before the legacy checkpoint.", flush=True)
    return model_path, output


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
    prior_seconds, time_complete = resumed_reconstruction_time(checkpoint)
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
              "trained_epochs_this_run": 0,
              "reconstruction_time_complete": time_complete,
              "reconstruction_time_scope": "Active wall time through final model export, before final evaluation; "
                                           "resume adds time recorded at its checkpoint and excludes downtime.",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Falcor NoLightTransport 1-spp full-view training; explicit geometry surrogate gradients",
              "falcor_reference_commit": "398f634ec788bbb2e86a6f4d0ab78d41158d9905",
              "training_protocol": {"spp": 1, "reference_background": list(config.render.background),
                  "gradient_reduction": "sum over pixels, no hit-count normalization",
                  "reported_loss": "mean over pixels, then mean over training views",
                  "actual_sampling": "per-pixel uniform, optionally contour-guided; frozen for a view"}}
    progress_line = ProgressLine() if primary else ProgressLine(stream=io.StringIO())
    reference_cache = None
    gpu_reference_cache = None
    native_replay = None
    collective = None
    topology = None
    model = None
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
            config.render = RenderConfig()
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
                if normalized_algorithm_group(group, current[group]) != previous_group:
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
        validate_traversal(config.render, mi, report)
        performance.metadata["traversal"] = report["traversal"]
        rgb_growth_enabled = config.topology.use_rgb_growth and config.topology.enable_growth and (
            not config.schedule.enabled or config.schedule.enable_coarse_growth)
        distortion_enabled = config.gradient.distortion_loss_weight > 0
        if distortion_enabled and getattr(mi.VoxelReconstruction, "distortion_version", 0) < 1:
            raise RuntimeError("Distortion loss requires native distortion_version >= 1; "
                               "incrementally rebuild this Mitsuba source checkout")
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
                           + (" and hard-hit distortion loss" if distortion_enabled else "")
                           if config.training.rgb_only else "Legacy Falcor NoLightTransport alpha-supervised training")
        report["training_protocol"].update(
            rgb_only=config.training.rgb_only,
            render_color_space="linear",
            traversal=config.render.traversal,
            loss_color_space="linear",
            distortion={
                "enabled": distortion_enabled,
                "weight": config.gradient.distortion_loss_weight,
                "native_version": getattr(mi.VoxelReconstruction, "distortion_version", 0),
                "formula": "2 * sum(i<j, w_i*w_j*abs(s_i-s_j)); w_i=T_i*opacity_i",
                "depth": "hard-hit ellipsoid quadratic-minimum depth, relative to ray-grid entry, divided by grid diagonal",
                "samples": "actual recorded hard hits only; no background sample or interval-width term",
                "estimator": "one inverse_pdf factor; sum gradients, mean loss over all view pixels",
                "topology_evidence": "RGB-only insertion/removal gains remain unchanged",
                "zero_weight": "not computed; logged distortion values are zero, not measurements",
            },
            reference_compositing=config.data.reference_compositing,
            reference_preprocessing=(
                "sRGB-encoded sources: composite RGB with background in sRGB, then decode RGB to linear; "
                "linear sources: composite in linear; retain only RGB"
                if config.training.rgb_only and config.data.reference_compositing == "srgb" else
                "RGBA decoded to linear, then composited onto white; only RGB retained"
                                     if config.training.rgb_only else "linear RGBA; composed inside native training"),
            actual_sampling="per-pixel uniform" if config.training.rgb_only else "per-pixel uniform or contour-guided",
            growth_evidence=("same-traversal independent parent proposals; per-cell winner after signed 1-spp RGB votes; no confirmation render"
                             if rgb_growth_enabled else "disabled" if config.training.rgb_only else "legacy alpha evidence"),
            growth_normalization=config.topology.growth_rgb_normalization if rgb_growth_enabled else None,
            growth_birth_opacity={
                "mode": "parent_scaled" if config.topology.growth_inherit_opacity else "fixed",
                "parent_scale": config.topology.growth_opacity_scale if config.topology.growth_inherit_opacity else None,
                "birth_cap" if config.topology.growth_inherit_opacity else "fixed_value":
                    config.topology.growth_initial_opacity,
            },
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
            reported_loss=("mean pixel RGB squared-error sum over channels, then mean over views"
                           if config.training.rgb_only else "mean RGB plus alpha loss over pixels and views")
                          + (" + weight * mean hard-hit distortion" if distortion_enabled else ""))
        dataset = workspace_path(config.data.dataset, WORKSPACE)
        emit("Loading training views and checking reference files:", dataset, flush=True)
        prepared = prepare_dataset(config, np, load_points=checkpoint is None)
        report["scene_background"] = scene_background_protocol(config.data, config.render)
        report["training_protocol"]["reference_background"] = list(config.render.background)
        views = (prepared["views"] if prepared is not None else
                 load_views(dataset, config.data.cameras, config.render, np))
        signature = dataset_signature(views, dataset, config.data.cameras, config.data.initialization)
        if prepared is not None:
            if config.data.cameras not in ("train", "transforms_train.json"):
                raise ValueError("COLMAP training must use the train split; held-out views are reserved for evaluation")
            report["initialization"] = config.data.initialization
            report["resolved"]["point_cloud"] = prepared["point_source"]
            report["effective_grid"] = {key: prepared[key] for key in
                ("resolution", "core_resolution", "grid_min", "grid_max")}
            emit("COLMAP initialization:", json.dumps(config.data.initialization), flush=True)
            emit(f"Reference images: {config.render.width} x {config.render.height}; "
                 f"{len(views)} training views; grid {prepared['resolution']} "
                 f"(core {prepared['core_resolution']})", flush=True)
            from evaluate import evaluation_view_limits
            emit("Final evaluation view limits:",
                 json.dumps(evaluation_view_limits(final_evaluation_config(config, model_path))), flush=True)
        performance.metadata["config"] = asdict(config)
        batch_size = effective_chunk_size(config)
        ray_count = config.render.width * config.render.height
        report["training_protocol"].update(rays_per_view=ray_count, rays_per_batch=batch_size,
            batches_per_view=(ray_count + batch_size - 1) // batch_size,
            alpha_readback=("none during RGB-only training" if config.training.rgb_only else
                            "only contour history; scheduled growth evidence consumes device alpha"),
            camera_parameters="opaque runtime inputs",
            gradients_stage_scope="forward paths, loss, hit/candidate gradients and reductions; deferred mode records host submission time",
            validation_readback="each view" if config.training.sync_view_checks else "epoch and sampled-profile boundaries; GPU-latched update guard",
            sync_view_checks=config.training.sync_view_checks,
            optimizer_execution="sparse valid-voxel Adam; compact native transaction",
            optimizer_compaction_sync="appearance/geometry list lengths; includes prior queued work",
            native_stage_profiling="synchronized sampled views only; bypasses frozen replay" if config.training.profile_stages else "disabled",
            freeze_native=config.training.freeze_native,
            gpu_reference_cache_mib=config.training.gpu_reference_cache_mib)
        emit(f"Training rays: {ray_count} per view, {batch_size} per batch, "
              f"{report['training_protocol']['batches_per_view']} batch(es) per update", flush=True)
        save_epochs = checkpoint_epochs(config.training.num_epochs, config.training.save_count)
        report["checkpoint_epochs"] = list(save_epochs)
        if primary:
            (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
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
        emit("Training validity checks:", "synchronous per view" if config.training.sync_view_checks
             else "GPU guarded; loss/validity read at epoch and sampled-profile boundaries", flush=True)
        emit("Reference compositing:", config.data.reference_compositing,
             "| rendering and RGB loss: linear", flush=True)
        emit("Distortion loss:", f"weight={config.gradient.distortion_loss_weight:g}; "
             + ("hard-hit pairwise depth concentration" if distortion_enabled else "disabled"), flush=True)
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
            emit("Initializing voxel model:", "COLMAP normalized points" if prepared is not None
                 else str(dataset / config.data.point_cloud), flush=True)
            model = initialize_model(config, dataset, mi, np, prepared=prepared)
        del prepared
        if config.training.rgb_only and int(model.opacity_sh_count) != 1:
            raise ValueError("RGB-only model/checkpoint must contain exactly one opacity coefficient per voxel")
        optimizer = FalcorOptimizer(model, config.optimizer)
        if not callable(getattr(model, "set_parameters_sparse", None)):
            raise RuntimeError("Sparse Adam requires a rebuilt native set_parameters_sparse binding; "
                               "incrementally rebuild this checkout before training")
        topology = TopologyManager(model, len(views), settings=config.topology,
            schedule=replace(config.schedule, save_each_level=False),
            opacity_warmup=config.optimizer.opacity_warmup_epochs, opacity_ramp=config.optimizer.opacity_ramp_epochs)
        coarse_path = model_path.parent / "train" / "0000-coarse.png"
        if not config.evaluation.enabled:
            report["coarse_render"] = dict(status="skipped", reason="evaluation_disabled")
        elif (not config.schedule.enabled
              or config.schedule.start_resolution == config.schedule.target_resolution):
            report["coarse_render"] = dict(status="skipped", reason="no_coarse_level")
        elif int(getattr(model, "core_resolution", model.resolution)) != config.schedule.start_resolution:
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

        emit("Training epochs by core resolution:", json.dumps(topology.budgets), flush=True)

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

        if callable(getattr(model, "statistics", None)):
            # Native statistics contain host metadata (including tree allocation)
            # only; this does not transfer voxel buffers or synchronize the GPU.
            report["initial_model_statistics"] = dict(model.statistics())
        if hasattr(model, "background_initialization"):
            report["background_initialization"] = dict(model.background_initialization)

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
                    # Sample before serialization; a resumed checkpoint cannot
                    # recover the time spent writing that same checkpoint.
                    "reconstruction_seconds": prior_seconds + time.perf_counter() - run_started,
                    "reconstruction_time_complete": time_complete,
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
            growth_prepare_stage_seconds = {}
            previous_growth_count = int(getattr(model, "growth_candidate_count", 0))
            if primary:
                rgb_growth_requested = False
                if config.topology.use_rgb_growth:
                    if previous_growth_count or topology._growth_due(epoch + 1):
                        native_replay.clear()
                    rgb_growth_requested = topology.prepare_rgb_growth(model, epoch)
                    growth_prepare_stage_seconds = dict(topology.growth_prepare_stage_seconds)
                payload = topology.pending_growth["candidate_state"] if rgb_growth_requested else None
                flags = (topology.needs_rendered_alpha(epoch), topology.collect_deletion(epoch),
                         rgb_growth_requested, payload)
            else:
                flags = None
            if distributed is not None:
                if primary:
                    flags = host_topology_payload(flags, dr, np)
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
            epoch_rgb_loss = 0. if config.training.rgb_only or distortion_enabled else None
            epoch_distortion_loss = 0.
            deferred_checks = (None if config.training.sync_view_checks else DeferredViewChecks(mi, dr, np))
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
                loss_components = {}
                # Sparse kernel-history samples add instrumentation. Wall timings
                # below include tracing, compilation, transfers and waits.
                with dr.kernel_history() if profiling else nullcontext() as history:
                    gradients, loss, rendered_alpha = train_view(model, view, rgba, config, sampler, rng, mi, dr, np,
                        need_rendered_alpha=alpha_growth_requested,
                        collect_deletion_evidence=deletion_evidence_collected,
                        collect_growth_evidence=rgb_growth_requested,
                        stage_profile=native_stages, gpu_reference=gpu_reference,
                        native_replay=native_replay, collective=collective, alpha_on_device=True,
                        loss_components=loss_components, deferred_checks=deferred_checks)
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
                    update_result = optimizer.step(model, gradients, sample_count=1, epoch=epoch,
                        coarse_to_fine=config.schedule.enabled,
                        update_allowed=gradients.get("_update_allowed"))
                    if deferred_checks is not None:
                        deferred_checks.updated(update_result["accepted"])
                        if profiling:
                            # Sparse diagnostics are explicit synchronization
                            # boundaries; normal views never fetch loss scalars.
                            measurements = deferred_checks.read(rgb_only=config.training.rgb_only,
                                distortion_weight=config.gradient.distortion_loss_weight)[-1]
                            loss = measurements.pop("loss")
                            loss_components.update(measurements)
                    optimizer_done = time.perf_counter()
                stage = dict(reference=reference_done-view_started, gradients=gradients_done-reference_done,
                             topology_observation=topology_done-gradients_done, optimizer=optimizer_done-topology_done)
                for key, duration in stage.items():
                    stage_totals[key] += duration
                processed_views += 1
                global_step += 1
                if distributed is not None:
                    distributed.heartbeat(f"epoch {epoch+1} view {progress}")
                if deferred_checks is None:
                    epoch_loss += loss
                    if epoch_rgb_loss is not None:
                        epoch_rgb_loss += loss_components["rgb_loss"]
                    epoch_distortion_loss += loss_components["distortion_loss"]
                loss_text = f"loss={loss:.6f}" if deferred_checks is None or profiling else "loss at epoch end"
                progress_line.update(f"Epoch {epoch+1}/{config.training.num_epochs} "
                    f"| view {progress}/{len(views)} ({100*progress/len(views):.0f}%) "
                    f"| cam {int(index)+1} | {loss_text} | {optimizer_done-view_started:.2f}s/submit")
                if profiling and primary:
                    profile = dict(epoch=epoch+1, progress=progress, camera=int(index)+1,
                                   training_resolution=training_resolution, active_voxels=int(model.active_voxel_count),
                                   sample_reasons=reasons, growth_evidence_requested=growth_evidence_requested,
                                   rgb_deletion_evidence_requested=rgb_deletion_enabled and deletion_evidence_collected,
                                   native_execution=("direct_stage_profile" if native_stages is not None else
                                                     "direct_empty_model" if config.training.freeze_native and not model.active_voxel_count else
                                                     "frozen" if config.training.freeze_native else "direct"),
                                   native_replay=native_replay.statistics(),
                                   loss=loss, **loss_components,
                                   wall_seconds=optimizer_done-view_started, stage_seconds=stage,
                                   **kernel_profile(history))
                    if native_stages is not None:
                        profile["native_stages"] = native_stages
                        profile["native_stage_timing"] = "synchronized boundaries; instrumented timings, not normal throughput"
                    with profile_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(profile) + "\n")
                    performance.record_profile(profile)
            validation_started = time.perf_counter()
            if deferred_checks is not None:
                measurements = deferred_checks.read(rgb_only=config.training.rgb_only,
                    distortion_weight=config.gradient.distortion_loss_weight)
                # Sum host doubles in view order, exactly as the original per-view
                # path. Topology/save cannot consume an unvalidated epoch.
                epoch_loss = sum(value["loss"] for value in measurements)
                if epoch_rgb_loss is not None:
                    epoch_rgb_loss = sum(value["rgb_loss"] for value in measurements)
                epoch_distortion_loss = sum(value["distortion_loss"] for value in measurements)
            validation_seconds = time.perf_counter() - validation_started
            progress_line.update(f"Epoch {epoch+1}/{config.training.num_epochs} "
                                 f"| view {len(views)}/{len(views)} (100%) | updating topology...", force=True)
            coarse_render = capture_coarse_image(epoch+1)
            topology_started = time.perf_counter()
            new_state, event = topology.finish_epoch(model, optimizer, epoch+1) if primary else (None, None)
            topology.before_refine_checkpoint = None
            if distributed is not None:
                transition = distributed.broadcast((host_topology_payload(new_state, dr, np), event,
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
                     "mean_rgb_loss": epoch_rgb_loss / len(views) if epoch_rgb_loss is not None else None,
                     "mean_distortion_loss": epoch_distortion_loss / len(views),
                     "mean_weighted_distortion_loss": config.gradient.distortion_loss_weight * epoch_distortion_loss / len(views),
                     "distortion_loss_weight": config.gradient.distortion_loss_weight,
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
                     "view_stage_timing": "host submission (asynchronous except sampled diagnostics)" if deferred_checks is not None else "synchronized per-view checks",
                     "epoch_validation_seconds": validation_seconds,
                     "epoch_growth_prepare_seconds": growth_prepare_seconds,
                     "epoch_growth_prepare_stage_seconds": growth_prepare_stage_seconds,
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
            report["trained_epochs_this_run"] = epoch + 1 - completed
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
        if callable(getattr(model, "statistics", None)):
            report["final_model_statistics"] = dict(model.statistics())
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
        report["reconstruction_seconds"] = prior_seconds + time.perf_counter() - run_started
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
    model_path, output = finalize_experiment_output(
        config, model_path, output, report, performance, distributed)
    if not config.evaluation.enabled:
        return 0
    evaluation = final_evaluation_config(config, model_path)
    from evaluate import evaluation_view_limits
    view_limits = evaluation_view_limits(evaluation)
    evaluation_report = model_path.parent / "evaluation-report.json"
    report["evaluation"] = {"status": "running", "spp": evaluation.spp,
        "max_views_per_split": evaluation.max_views, "splits": list(evaluation.splits),
        "view_limits": view_limits, "score_metric": "psnr_srgb_png",
        "gpu_workers": distributed.world_size if distributed else 1,
        "output": evaluation.output_directory, "report": str(evaluation_report)}
    if primary:
        (output / "training-report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    limit = ", ".join(f"{split}: all views" if count is None else f"{split}: first {count} views"
                      for split, count in view_limits.items())
    emit(f"Starting final train/test evaluation: {evaluation.spp} spp, {limit}, metric=psnr_srgb_png"
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
