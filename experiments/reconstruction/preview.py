"""Preview Falcor-style voxel initialization with the explicitly built CUDA module.

Edit CONFIG below, supply --config JSON, or use command-line overrides. All
relative paths are resolved from the parent of this source checkout, regardless
of the terminal's working directory. This is a forward preview, not training.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback

# Python -I excludes the script directory; only explicitly add our own sources.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reconstruction_color import reference_from_bitmap, validate_reference_compositing

REPOSITORY = Path(__file__).resolve().parents[2]
WORKSPACE = REPOSITORY.parent


@dataclass
class DataConfig:
    dataset: str = "datasets/lego"
    point_cloud: str = "init_points.ply"  # Relative to dataset.
    cameras: str = "transforms_train.json"  # Relative to dataset.
    camera_index: int = 0
    reference_compositing: str = "linear"  # Missing field in old checkpoints keeps the old convention.


@dataclass
class ModelConfig:
    grid_resolution: int = 256
    bounds_min: tuple[float, float, float] = (-1.6, -1.6, -1.6)
    bounds_max: tuple[float, float, float] = (1.6, 1.6, 1.6)
    bounds_padding: float = 1.02  # Falcor's actual grid is [-1.632, 1.632]^3.
    opacity_threshold: float = 0.05
    sh_count: int = 9  # Coefficient counts, not SH degrees.
    opacity_sh_count: int = 9


@dataclass
class RenderConfig:
    width: int = 800
    height: int = 800
    spp: int = 1
    seed: int = 42
    pixel_sampling: str = "shared_jitter"  # Or "center" for fixed-pixel comparisons.
    near_clip: float = 0.1
    far_clip: float = 1000.0
    entry_offset_fraction: float = 1e-6  # Falcor's near-to-far segment entry offset.
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)
    transmittance_threshold: float = 0.01
    max_hits: int = 16
    traversal: str = "octree"  # "dda" retains the original per-cell reference traversal.


@dataclass
class OutputConfig:
    directory: str = "outputs/mitsuba/reconstruction-preview"
    save_model: bool = False


@dataclass
class RuntimeConfig:
    build_dir: str = (Path(REPOSITORY.name) / "build" / "Release" if os.name == "nt"
                      else Path(REPOSITORY.name) / "build").as_posix()
    gpu: int = 0  # Physical nvidia-smi index locally; ignored inside Slurm jobs.


@dataclass
class PreviewConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


# This is the single user-editable configuration entry point.
CONFIG = PreviewConfig()
CONFIG.data.reference_compositing = "srgb"  # Composite encoded PNG colors, then decode for linear rendering.


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_provenance(repository: Path) -> dict:
    """Record the current checkout separately from the loaded build artifacts."""
    result = {"git": {}, "source_files_sha256": {}}
    try:
        head = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain=v1", "--untracked-files=normal"],
            check=True, capture_output=True, text=True, timeout=30,
        ).stdout
        result["git"] = {"head": head, "dirty": bool(status.strip()), "status": status.splitlines()}
    except (OSError, subprocess.SubprocessError) as error:
        result["git"] = {"head": None, "dirty": None, "error": str(error)}
    for relative in (
        "include/mitsuba/render/reconstruction.h",
        "src/render/reconstruction.cpp",
        "src/render/reconstruction_pointcloud.h",
        "src/render/reconstruction_pointcloud.cpp",
        "src/render/python/reconstruction_v.cpp",
        "src/python/main_v.cpp",
        "src/render/CMakeLists.txt",
        "src/render/python/CMakeLists.txt",
        "experiments/reconstruction/preview.py",
        "experiments/reconstruction/reconstruction_color.py",
        "experiments/reconstruction/tests/test_reconstruction.py",
    ):
        try:
            result["source_files_sha256"][relative] = file_sha256(repository / relative)
        except OSError as error:
            result.setdefault("source_file_errors", {})[relative] = str(error)
    result["note"] = "Source hashes describe the checkout at launch; rebuild after C++ edits before comparing results."
    return result


def workspace_path(value: str | Path, workspace: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else workspace / path).resolve()


def update_config(config: PreviewConfig, values: dict) -> None:
    allowed_groups = {item.name for item in fields(config)}
    if not isinstance(values, dict) or set(values) - allowed_groups:
        raise ValueError("Configuration must contain only data/model/render/output/runtime groups")
    for group_name, entries in values.items():
        group = getattr(config, group_name)
        allowed = {item.name for item in fields(group)}
        if not isinstance(entries, dict) or set(entries) - allowed:
            raise ValueError(f"Unknown or malformed configuration fields in {group_name}")
        for name, value in entries.items():
            setattr(group, name, value)


def parse_config(workspace: Path) -> PreviewConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON config, relative to the workspace parent")
    parser.add_argument("--dataset")
    parser.add_argument("--point-cloud")
    parser.add_argument("--cameras")
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--reference-compositing", choices=("linear", "srgb"))
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--spp", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--grid-resolution", type=int)
    parser.add_argument("--pixel-sampling", choices=("shared_jitter", "center"))
    parser.add_argument("--traversal", choices=("octree", "dda"),
                        help="Spatial traversal (default: octree); dda is the comparison path")
    parser.add_argument("--output-dir")
    parser.add_argument("--build-dir")
    parser.add_argument("--gpu", type=int, help="Local nvidia-smi index; ignored inside Slurm jobs")
    parser.add_argument("--save-model", action="store_true", default=None)
    args = parser.parse_args()
    config = PreviewConfig()
    update_config(config, asdict(CONFIG))
    if args.config:
        update_config(config, json.loads(workspace_path(args.config, workspace).read_text(encoding="utf-8")))
    mapping = {
        "dataset": ("data", "dataset"), "point_cloud": ("data", "point_cloud"),
        "cameras": ("data", "cameras"), "camera_index": ("data", "camera_index"),
        "reference_compositing": ("data", "reference_compositing"),
        "grid_resolution": ("model", "grid_resolution"),
        "output_dir": ("output", "directory"), "save_model": ("output", "save_model"),
        "build_dir": ("runtime", "build_dir"), "gpu": ("runtime", "gpu"),
    }
    for name in ("width", "height", "spp", "seed", "pixel_sampling", "traversal"):
        mapping[name] = ("render", name)
    for argument, (group, name) in mapping.items():
        value = getattr(args, argument)
        if value is not None:
            setattr(getattr(config, group), name, value)
    return config


def validate_config(config: PreviewConfig) -> None:
    validate_reference_compositing(config.data.reference_compositing)
    if config.render.traversal not in ("octree", "dda"):
        raise ValueError("render.traversal must be octree or dda")
    for name in ("width", "height", "spp", "max_hits"):
        value = getattr(config.render, name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"render.{name} must be a positive integer")
    if config.data.camera_index < 0 or config.runtime.gpu < 0 or config.render.seed < 0:
        raise ValueError("Camera index, GPU index, and seed must be nonnegative")
    if config.render.pixel_sampling not in ("shared_jitter", "center"):
        raise ValueError("pixel_sampling must be shared_jitter or center")
    if not math.isfinite(config.render.near_clip) or config.render.near_clip <= 0:
        raise ValueError("near_clip must be finite and positive")
    if not math.isfinite(config.render.far_clip) or config.render.far_clip <= config.render.near_clip:
        raise ValueError("far_clip must be finite and larger than near_clip")
    if not math.isfinite(config.render.entry_offset_fraction) or config.render.entry_offset_fraction < 0:
        raise ValueError("entry_offset_fraction must be finite and nonnegative")
    if not 0 <= config.render.transmittance_threshold < 1:
        raise ValueError("transmittance_threshold must be in [0, 1)")
    model = config.model
    if not isinstance(model.grid_resolution, int) or not 1 <= model.grid_resolution <= 1024:
        raise ValueError("grid_resolution must be an integer in [1, 1024]")
    if not math.isfinite(model.bounds_padding) or model.bounds_padding <= 0:
        raise ValueError("bounds_padding must be finite and positive")
    for name, value in (("bounds_min", model.bounds_min), ("bounds_max", model.bounds_max),
                        ("background", config.render.background)):
        if len(value) != 3 or not all(math.isfinite(x) for x in value):
            raise ValueError(f"{name} must contain three finite values")
    if any(lo >= hi for lo, hi in zip(model.bounds_min, model.bounds_max)):
        raise ValueError("Each bounds_min coordinate must be smaller than bounds_max")
    if not 0 < model.opacity_threshold < 1:
        raise ValueError("opacity_threshold must be in (0, 1)")
    if any(not isinstance(n, int) or not 1 <= n <= 16 for n in (model.sh_count, model.opacity_sh_count)):
        raise ValueError("SH coefficient counts must be integers in [1, 16]")


def validate_traversal(render, mi, report=None):
    """Reject a stale build instead of silently using DDA for an octree request."""
    if render.traversal not in ("octree", "dda"):
        raise ValueError("render.traversal must be octree or dda")
    version = int(getattr(mi.VoxelReconstruction, "octree_version", 0))
    if render.traversal == "octree" and version < 1:
        raise RuntimeError("Octree traversal requires native octree_version >= 1; "
                           "incrementally rebuild this Mitsuba source checkout")
    if report is not None:
        report["traversal"] = {"mode": render.traversal, "native_octree_version": version}
        print("Spatial traversal:", render.traversal, flush=True)


def trace_model(model, ray, background, render):
    """Pass the selected traversal explicitly; keep old DDA builds comparable."""
    arguments = (ray, background, render.transmittance_threshold,
                 render.max_hits, render.entry_offset_fraction)
    if getattr(model, "octree_version", 0) >= 1:
        return model.trace(*arguments, render.traversal == "octree")
    if render.traversal != "dda":
        raise RuntimeError("Octree traversal is unavailable; rebuild this source VoxelReconstruction module")
    return model.trace(*arguments)


def native_traversal_settings(model, options):
    """Adapt an explicit DDA request to older native settings dictionaries."""
    if getattr(model, "octree_version", 0) >= 1 or "use_octree" not in options:
        return options
    if options["use_octree"]:
        raise RuntimeError("Octree traversal is unavailable; rebuild this source VoxelReconstruction module")
    # Older builds reject unknown options, including a false acceleration flag.
    # Only remove that flag, and never mutate caller-owned replay/config input.
    return {name: value for name, value in options.items() if name != "use_octree"}


def prepare_device_environment(gpu: int, report: dict) -> None:
    """Keep Slurm's CUDA namespace intact; only local runs select a physical GPU."""
    if "SLURM_JOB_ID" in os.environ or "SLURM_JOBID" in os.environ:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        tokens = [token.strip() for token in visible.split(",")] if visible is not None else []
        report["gpu"] = {
            "selection": "slurm", "cuda_visible_devices": visible,
            "visible_device_tokens": tokens, "local_gpu_index_ignored": gpu,
            "slurm": {name: os.environ[name] for name in (
                "SLURM_JOB_ID", "SLURM_JOBID", "SLURM_STEP_ID", "SLURM_JOB_GPUS",
                "SLURM_STEP_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_CPUS_PER_TASK",
            ) if name in os.environ},
        }
        # CUDA accepts numeric ordinals, GPU UUIDs (including unique prefixes),
        # modern MIG UUIDs, and legacy MIG-GPU-<UUID>/<GI>/<CI> identifiers.
        uuid = r"[0-9a-fA-F]+(?:-[0-9a-fA-F]+)*"
        device_pattern = rf"(?:[0-9]+|GPU-{uuid}|MIG-{uuid}|MIG-GPU-{uuid}/[0-9]+/[0-9]+)"
        if not tokens or any(re.fullmatch(device_pattern, token) is None for token in tokens):
            raise RuntimeError(
                "Slurm job requires a valid GPU allocation in CUDA_VISIBLE_DEVICES; "
                f"got {visible!r}. Run inside a GPU-allocated job step. "
                "The scheduler environment will not be changed and local GPU selection is disabled."
            )
        cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if cpus is not None and (re.fullmatch(r"[0-9]+", cpus) is None or int(cpus) < 1):
            raise RuntimeError(f"SLURM_CPUS_PER_TASK must be a positive integer; got {cpus!r}")
        report["cpu_threads"] = {
            "source": "SLURM_CPUS_PER_TASK" if cpus is not None else "slurm_default",
            "requested": int(cpus) if cpus is not None else 1,
        }
        return

    query = subprocess.run(
        ["nvidia-smi", f"--id={gpu}",
         "--query-gpu=index,name,uuid,driver_version,memory.total", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=30,
    )
    rows = list(csv.reader(query.stdout.strip().splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 5:
        raise RuntimeError(f"Unexpected nvidia-smi output: {query.stdout!r}")
    report["gpu"] = dict(zip(("index", "name", "uuid", "driver_version", "memory_mib"), rows[0]))
    os.environ["CUDA_VISIBLE_DEVICES"] = report["gpu"]["uuid"]
    report["gpu"].update(selection="local", cuda_visible_devices=report["gpu"]["uuid"])


def configure_drjit_threads(dr, report: dict) -> None:
    """Set the Slurm CPU budget before importing Mitsuba or initializing a model."""
    if report["gpu"]["selection"] == "slurm":
        budget = report["cpu_threads"]["requested"]
        dr.set_thread_count(budget)
        actual = dr.thread_count()
        report["cpu_threads"]["actual"] = actual
        if not 1 <= actual <= budget:
            raise RuntimeError(f"Dr.Jit CPU thread count {actual} exceeds or invalidates Slurm budget {budget}")
    else:
        report["cpu_threads"] = {"source": "local_default", "actual": dr.thread_count()}
    print("Dr.Jit CPU threads:", json.dumps(report["cpu_threads"]), flush=True)


def verify_cuda_device(array, report: dict) -> None:
    """Check CUDA residency using process-visible ordinals, not physical GPU IDs."""
    device = list(array.__dlpack_device__())
    report["dlpack_device"] = device
    visible_count = (len(report["gpu"]["visible_device_tokens"])
                     if report["gpu"]["selection"] == "slurm" else 1)
    if len(device) != 2 or device[0] != 2 or not 0 <= device[1] < visible_count:
        raise RuntimeError(
            f"Expected a CUDA device with process-visible index in [0, {visible_count}), got {device}"
        )
    report["gpu"]["process_visible_device_index"] = device[1]


def prepare_runtime(config: PreviewConfig, workspace: Path, output: Path, report: dict) -> list:
    runtime = workspace_path(config.runtime.build_dir, workspace)
    package_root = runtime / "python"
    for package in ("mitsuba", "drjit"):
        if not (package_root / package / "__init__.py").is_file():
            raise RuntimeError(f"Source build is incomplete: {package_root / package}")
    sys.path.insert(0, str(package_root))
    handles = []
    if os.name == "nt":
        for directory in (runtime, package_root / "drjit", package_root / "mitsuba"):
            handles.append(os.add_dll_directory(str(directory)))
    prepare_device_environment(config.runtime.gpu, report)
    cache_root = workspace / ".cache" / "mitsuba"
    for variable, name in (("DRJIT_CACHE_DIR", "drjit"), ("CUDA_CACHE_PATH", "cuda"),
                           ("OPTIX_CACHE_PATH", "optix")):
        cache = cache_root / name
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(cache)
    report["resolved"]["build_dir"] = str(runtime)
    report["resolved"]["cache_dir"] = str(cache_root)
    print("GPU:", json.dumps(report["gpu"]), flush=True)
    return handles


def load_camera(config: PreviewConfig, dataset: Path, np) -> tuple[dict, Path]:
    camera_path = (dataset / config.data.cameras).resolve()
    document = json.loads(camera_path.read_text(encoding="utf-8"))
    angle = float(document["camera_angle_x"])
    if not math.isfinite(angle) or not 0 < angle < math.pi:
        raise ValueError("camera_angle_x must be a finite horizontal FOV in (0, pi)")
    frames = document["frames"]
    if not 0 <= config.data.camera_index < len(frames):
        raise ValueError(f"camera_index must be in [0, {len(frames) - 1}]")
    frame = frames[config.data.camera_index]
    matrix = np.asarray(frame["transform_matrix"], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 4x4 camera-to-world transform")
    convert = np.array(((1., 0., 0.), (0., 0., 1.), (0., -1., 0.)))
    position = convert @ matrix[:3, 3]
    forward = convert @ -matrix[:3, 2]
    up = convert @ matrix[:3, 1]
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    if not np.isfinite(np.stack((position, forward, right, up))).all():
        raise ValueError("Degenerate camera basis")
    reference = dataset / frame["file_path"]
    if not reference.suffix:
        reference = reference.with_suffix(".png")
    if not reference.is_file():
        raise FileNotFoundError(reference)
    return {
        "camera_file": str(camera_path), "frame_count": len(frames),
        "camera_index": config.data.camera_index, "camera_angle_x": angle,
        "position": position.tolist(), "right": right.tolist(),
        "up": up.tolist(), "forward": forward.tolist(),
        "nerf_to_falcor": "(x, y, z) -> (x, z, -y)",
        "near_clip": config.render.near_clip,
        "far_clip": config.render.far_clip,
        "entry_offset_fraction": config.render.entry_offset_fraction,
        "ray_origin": "near plane; camera_position + near_clip * unnormalized_direction",
        "sampling": config.render.pixel_sampling,
        "rng": "NumPy PCG64; not Falcor UniformSampleGenerator; use center for fixed rays",
    }, reference.resolve()


def generate_rays(config: RenderConfig, camera: dict, offset, mi, dr):
    index = dr.arange(mi.UInt32, config.width * config.height)
    u = (mi.Float(index % config.width) + float(offset[0])) / config.width
    v = (mi.Float(index // config.width) + float(offset[1])) / config.height
    tan_x = math.tan(camera["camera_angle_x"] / 2)
    tan_y = tan_x * config.height / config.width
    unnormalized = (mi.Vector3f(camera["forward"])
                    + ((2 * u - 1) * tan_x) * mi.Vector3f(camera["right"])
                    + ((1 - 2 * v) * tan_y) * mi.Vector3f(camera["up"]))
    origin = mi.Point3f(camera["position"]) + config.near_clip * unnormalized
    ray = mi.Ray3f(origin, dr.normalize(unnormalized))
    ray.maxt = (config.far_clip - config.near_clip) * dr.norm(unnormalized)
    return ray


def write_png(path: Path, pixels, mi, np, srgb: bool = True) -> None:
    bitmap = mi.Bitmap(np.ascontiguousarray(pixels, dtype=np.float32))
    pixel_format = mi.Bitmap.PixelFormat.Y if pixels.ndim == 2 else mi.Bitmap.PixelFormat.RGB
    bitmap.convert(pixel_format, mi.Struct.Type.UInt8, srgb).write(str(path))
    if path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
        raise RuntimeError(f"Invalid PNG output: {path}")
    if list(mi.Bitmap(str(path)).size()) != [pixels.shape[1], pixels.shape[0]]:
        raise RuntimeError(f"PNG dimensions differ: {path}")


def main() -> int:
    repository, workspace = REPOSITORY, WORKSPACE
    config = parse_config(workspace)
    output = workspace_path(config.output.directory, workspace)
    if output.is_relative_to(repository):
        raise ValueError("Keep preview outputs and runtime caches outside the Git checkout")
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "preview-report.json"
    report = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Source CUDA forward initialization preview; no optimization, gradients, or Falcor pixel equivalence verified",
        "python_executable": sys.executable, "python_version": platform.python_version(),
        "python_prefix": sys.prefix, "platform": platform.platform(),
        "requested_variant": "cuda_ad_rgb", "requested_config": asdict(config),
        "resolved": {"workspace": str(workspace), "output": str(output)},
    }
    try:
        validate_config(config)
        report["provenance"] = source_provenance(repository)
        (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
        print("Python executable:", sys.executable, flush=True)
        print("Python version:", platform.python_version(), flush=True)
        dll_handles = prepare_runtime(config, workspace, output, report)  # Keep handles alive.
        import drjit as dr
        package_root = Path(report["resolved"]["build_dir"]) / "python"
        if not Path(dr.__file__).resolve().is_relative_to(package_root):
            raise RuntimeError(f"Source build required; imported {dr.__file__}")
        configure_drjit_threads(dr, report)
        import mitsuba as mi
        import numpy as np

        for module in (mi, dr):
            if not Path(module.__file__).resolve().is_relative_to(package_root):
                raise RuntimeError(f"Source build required; imported {module.__file__}")
        report.update({"mitsuba_version": mi.__version__, "drjit_version": dr.__version__,
                       "numpy_version": np.__version__, "mitsuba_file": mi.__file__, "drjit_file": dr.__file__})
        print("Mitsuba:", mi.__version__, mi.__file__, flush=True)
        print("Dr.Jit:", dr.__version__, dr.__file__, flush=True)
        mi.set_variant("cuda_ad_rgb")
        report["actual_variant"] = mi.variant()
        variant_module = sys.modules.get("mitsuba." + mi.variant())
        variant_file = getattr(variant_module, "__file__", None)
        if variant_file:
            report["provenance"]["variant_extension"] = {
                "path": str(Path(variant_file).resolve()),
                "sha256": file_sha256(Path(variant_file)),
            }
        else:
            report["provenance"]["variant_extension"] = {
                "path": None, "note": "Loaded variant extension file could not be confirmed from sys.modules"
            }
        print("Actual variant:", mi.variant(), flush=True)
        if mi.variant() != "cuda_ad_rgb" or dr.backend_v(mi.Float) != dr.JitBackend.CUDA:
            raise RuntimeError("CUDA variant/backend verification failed")
        if not hasattr(mi, "VoxelReconstruction"):
            raise RuntimeError("This source build lacks VoxelReconstruction; rebuild the current checkout")
        validate_traversal(config.render, mi, report)

        dataset = workspace_path(config.data.dataset, workspace)
        point_cloud = (dataset / config.data.point_cloud).resolve()
        if not point_cloud.is_file():
            raise FileNotFoundError(point_cloud)
        report["resolved"].update(dataset=str(dataset), point_cloud=str(point_cloud))
        camera, reference_path = load_camera(config, dataset, np)
        report["camera"] = camera
        report["resolved"]["reference"] = str(reference_path)
        lo, hi = np.asarray(config.model.bounds_min), np.asarray(config.model.bounds_max)
        middle, half_extent = (lo + hi) / 2, (hi - lo) / 2 * config.model.bounds_padding
        grid_min, grid_max = (middle - half_extent).tolist(), (middle + half_extent).tolist()
        report["effective_grid"] = {"min": grid_min, "max": grid_max, "resolution": config.model.grid_resolution}
        print("Point cloud:", point_cloud, flush=True)
        started = time.perf_counter()
        model = mi.VoxelReconstruction(
            str(point_cloud), config.model.grid_resolution, grid_min, grid_max,
            config.model.opacity_threshold, config.model.sh_count, config.model.opacity_sh_count,
        )
        dr.sync_thread()
        report["initialization_seconds"] = time.perf_counter() - started
        report["model_statistics"] = model.statistics()
        report["active_voxel_count"] = int(model.active_voxel_count)
        print("Model:", json.dumps(report["model_statistics"]), flush=True)
        if model.active_voxel_count == 0:
            raise RuntimeError("Initialization produced no occupied voxels")

        started = time.perf_counter()
        random = np.random.default_rng(config.render.seed)
        rgb_sum, alpha_sum, hits_sum = mi.Color3f(0), mi.Float(0), mi.UInt32(0)
        offsets = []
        for sample in range(config.render.spp):
            offset = random.random(2) if config.render.pixel_sampling == "shared_jitter" else (.5, .5)
            offsets.append([float(value) for value in offset])
            ray = generate_rays(config.render, camera, offset, mi, dr)
            rgb, alpha, hits = trace_model(model, ray, mi.Color3f(config.render.background), config.render)
            rgb_sum += rgb
            alpha_sum += alpha
            hits_sum += hits
            dr.eval(rgb_sum, alpha_sum, hits_sum)
            dr.sync_thread()
            print(f"Rendered sample {sample + 1}/{config.render.spp}", flush=True)
        rgb, alpha = rgb_sum / config.render.spp, alpha_sum / config.render.spp
        dr.eval(rgb, alpha, hits_sum)
        dr.sync_thread()
        report["render_seconds_including_jit"] = time.perf_counter() - started
        verify_cuda_device(rgb.x, report)
        height, width = config.render.height, config.render.width
        rgb_np = np.stack([np.asarray(rgb.x), np.asarray(rgb.y), np.asarray(rgb.z)], axis=-1).reshape(height, width, 3)
        alpha_np = np.asarray(alpha).reshape(height, width)
        hits_np = np.asarray(hits_sum).reshape(height, width)
        finite = bool(np.isfinite(rgb_np).all() and np.isfinite(alpha_np).all())
        report["checks"] = {
            "shape": list(rgb_np.shape), "all_finite": finite,
            "nonblack": bool(np.any(rgb_np > 0)), "pixels_with_hits": int(np.count_nonzero(hits_np)),
            "pixels_with_alpha": int(np.count_nonzero(alpha_np > 0)),
            "alpha_min": float(alpha_np.min()), "alpha_max": float(alpha_np.max()),
            "rgb_min": float(rgb_np.min()), "rgb_max": float(rgb_np.max()),
            "max_summed_hits": int(hits_np.max()),
        }
        if not finite or np.any(alpha_np < -1e-6) or np.any(alpha_np > 1 + 1e-6):
            raise RuntimeError("Output contains nonfinite values or alpha outside [0, 1]")
        if not np.any(hits_np > 0) or not np.any(alpha_np > 0):
            raise RuntimeError("Selected camera has no visible initialized geometry: inspect dataset/bounds/camera")
        # Black radiance is valid for Gaussian PLY initialization; test alpha/hits instead.
        background = np.asarray(config.render.background, dtype=np.float32)
        foreground = rgb_np - (1 - alpha_np[..., None]) * background
        white = foreground + (1 - alpha_np[..., None])
        write_png(output / "preview.png", rgb_np, mi, np)
        write_png(output / "preview-white.png", white, mi, np)
        write_png(output / "alpha.png", alpha_np, mi, np, srgb=False)

        reference_bitmap = mi.Bitmap(str(reference_path))
        reference = reference_from_bitmap(reference_bitmap, mi, np, path=reference_path)
        reference_white = reference_from_bitmap(
            reference_bitmap, mi, np, rgb_only=True, background=(1, 1, 1),
            reference_compositing=config.data.reference_compositing, path=reference_path)
        write_png(output / "reference-white.png", reference_white, mi, np)
        report["reference_shape"] = list(reference.shape)
        report["reference_resampled"] = False
        report["reference_color_protocol"] = {
            "requested_compositing": config.data.reference_compositing,
            "source_srgb": bool(reference_bitmap.srgb_gamma()),
            "effective_compositing": (
                "srgb" if config.data.reference_compositing == "srgb"
                and reference_bitmap.srgb_gamma() else "linear"),
            "target_space": "linear", "background_linear": [1, 1, 1],
            "alpha_usage": "reference compositing only",
            "linear_source_policy": "linear compositing; preserve EXR/HDR values",
        }
        np.savez_compressed(output / "render.npz", rgb_linear=rgb_np, alpha=alpha_np,
                            hit_count_sum=hits_np, rgb_white_linear=white,
                            pixel_offsets=np.asarray(offsets), reference_rgba_linear=reference,
                            reference_white_linear=reference_white)
        if config.output.save_model:
            snapshot = model.snapshot()
            dr.eval(*snapshot.values())
            dr.sync_thread()
            np.savez_compressed(output / "initial-model.npz", **{name: np.asarray(value) for name, value in snapshot.items()})
            report["model_snapshot_layout"] = "AoS flat arrays; rotations wxyz; SH coefficient counts in config"
        report["pixel_offsets"] = offsets
        output_names = ["preview.png", "preview-white.png", "alpha.png", "reference-white.png", "render.npz", "config.json"]
        if config.output.save_model:
            output_names.append("initial-model.npz")
        report["files"] = {
            name: {"path": str(output / name), "sha256": file_sha256(output / name)}
            for name in output_names
        }
        report["status"] = "passed"
        print("PASS: source CUDA initialization preview; finite RGB/alpha and visible hits.", flush=True)
        print("Preview:", output / "preview-white.png", flush=True)
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("Report:", report_path, flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
