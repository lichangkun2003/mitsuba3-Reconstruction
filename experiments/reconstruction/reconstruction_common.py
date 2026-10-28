"""Shared source-runtime, camera and rendering utilities for reconstruction."""

from __future__ import annotations

from collections import OrderedDict
import json
import math
from pathlib import Path
import sys

from reconstruction_color import reference_from_bitmap, validate_reference_compositing
from reconstruction_sh_schedule import restore_sh_state, sh_model_metadata

from preview import (
    REPOSITORY, WORKSPACE, PreviewConfig, RenderConfig, RuntimeConfig, configure_drjit_threads,
    file_sha256, load_camera, prepare_runtime, source_provenance,
    verify_cuda_device, workspace_path, write_png, trace_model,
)

def initialize_runtime(runtime: RuntimeConfig, output: Path, report: dict, *,
                       cuda_device=None, cpu_threads=None):
    """Use only this checkout's build, respecting the local/Slurm device policy."""
    report.setdefault("resolved", {}).update(workspace=str(WORKSPACE), output=str(output))
    report.update(python_executable=sys.executable, requested_variant="cuda_ad_rgb")
    report["provenance"] = source_provenance(REPOSITORY)
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        report["provenance"]["source_files_sha256"][path.relative_to(REPOSITORY).as_posix()] = file_sha256(path)
    handles = prepare_runtime(PreviewConfig(runtime=runtime), WORKSPACE, output, report)
    package_root = Path(report["resolved"]["build_dir"]) / "python"
    import drjit as dr
    if not Path(dr.__file__).resolve().is_relative_to(package_root):
        raise RuntimeError(f"Source build required; imported {dr.__file__}")
    configure_drjit_threads(dr, report)
    if cpu_threads is not None:
        allocation = report["cpu_threads"].get("requested", dr.thread_count())
        if type(cpu_threads) is not int or not 1 <= cpu_threads <= allocation:
            raise ValueError("Worker CPU budget must fit within the Slurm task allocation")
        dr.set_thread_count(cpu_threads)
        report["cpu_threads"].update(worker_budget=cpu_threads, actual=dr.thread_count())
        if not 1 <= report["cpu_threads"]["actual"] <= cpu_threads:
            raise RuntimeError("Dr.Jit worker thread count exceeds its share of the Slurm CPU allocation")
        print("Dr.Jit worker CPU threads:", json.dumps(report["cpu_threads"]), flush=True)
    import mitsuba as mi
    import numpy as np
    if not Path(mi.__file__).resolve().is_relative_to(package_root):
        raise RuntimeError(f"Source build required; imported {mi.__file__}")
    mi.set_variant("cuda_ad_rgb")
    if cuda_device is not None:
        if not hasattr(mi, "_reconstruction_cuda_context"):
            raise RuntimeError("Rebuild this checkout: multi-GPU CUDA device bindings are missing")
        device = mi._reconstruction_cuda_context(cuda_device)
        report["worker_cuda"] = {name: device[name] for name in ("device", "raw_device", "device_count")}
    if mi.variant() != "cuda_ad_rgb" or dr.backend_v(mi.Float) != dr.JitBackend.CUDA:
        raise RuntimeError("CUDA variant/backend verification failed")
    if not hasattr(mi, "VoxelReconstruction") or not hasattr(mi.VoxelReconstruction, "from_state"):
        raise RuntimeError("Rebuild this checkout: VoxelReconstruction training/checkpoint bindings are missing")
    report.update(mitsuba_version=mi.__version__, drjit_version=dr.__version__,
                  numpy_version=np.__version__, mitsuba_file=mi.__file__, drjit_file=dr.__file__,
                  actual_variant=mi.variant())
    variant_file = getattr(sys.modules.get("mitsuba." + mi.variant()), "__file__", None)
    if variant_file:
        report["provenance"]["variant_extension"] = {
            "path": str(Path(variant_file).resolve()), "sha256": file_sha256(Path(variant_file))}
    probe = dr.zeros(mi.Float, 1)
    dr.eval(probe)
    dr.sync_thread()
    verify_cuda_device(probe, report)
    print("Python:", sys.executable, flush=True)
    print("Mitsuba:", mi.__version__, mi.__file__, flush=True)
    print("Dr.Jit:", dr.__version__, dr.__file__, flush=True)
    print("Actual variant:", mi.variant(), flush=True)
    return mi, dr, np, handles


def _colmap_split(cameras_file):
    names = {"train": "train", "test": "test", "transforms_train.json": "train",
             "transforms_test.json": "test"}
    if cameras_file not in names:
        raise ValueError(f"Unknown COLMAP split: {cameras_file}; use train or test")
    return names[cameras_file]


def _colmap_views(prepared, render):
    """Keep rays and reference pixels in the same selected image resolution."""
    if (render.width, render.height) != (prepared["width"], prepared["height"]):
        raise ValueError("COLMAP reference image dimensions differ from the saved render size; "
                         "use the original data.images directory")
    for view in prepared["views"]:
        view["camera"].update(
            sampling="per_pixel_uniform",
            rng="NumPy PCG64; distribution matches uniform Falcor samples, bit sequence differs",
            near_clip=render.near_clip, far_clip=render.far_clip,
            entry_offset_fraction=render.entry_offset_fraction,
            ray_origin="near plane; camera_position + near_clip * unnormalized_direction")
        view["camera"]["projective_unbounded"] = (
            prepared["initialization"].get("scene_mapping") == "projective_shell")
    return prepared["views"]


def prepare_dataset(config, np, *, load_points=True):
    """Resolve the 360 initializer; synthetic initialization stays unchanged.

    This runs before batch sizes, caches or config files are created. Images are
    used at their selected on-disk resolution, with matching full intrinsics.
    """
    from reconstruction_dataset import detect_dataset, prepare_colmap
    from preview import resolve_scene_background
    dataset = workspace_path(config.data.dataset, WORKSPACE)
    kind = detect_dataset(dataset, config.data.cameras)
    mode = resolve_scene_background(config, kind)
    if kind == "synthetic":
        if mode == "unbounded":
            raise ValueError("Unbounded shell initialization currently requires COLMAP camera normalization")
        if config.data.initialization is not None:
            raise ValueError("COLMAP initialization metadata cannot be used with synthetic cameras")
        return None
    schedule = getattr(config, "schedule", None)
    start = schedule.start_resolution if schedule and schedule.enabled else config.model.grid_resolution
    target = schedule.target_resolution if schedule and schedule.enabled else start
    options = {}
    if mode == "legacy":
        options["shell_layers"] = config.data.initialization["shell_layers"]
    else:
        options.update(scene_mode=mode, background_opacity=config.data.background_opacity)
        saved = config.data.initialization
        if mode == "unbounded" and (saved is None or saved.get("version") in (4, 5)):
            if config.data.background_layers is not None:
                raise ValueError("New unbounded runs use data.background_final_layers / --background-final-layers; "
                                 "background_layers is only for legacy v2/v3 initializations")
            options["background_final_layers"] = config.data.background_final_layers
        else:
            if mode == "unbounded" and config.data.background_final_layers is not None:
                raise ValueError("Saved v2/v3 initialization must keep its original background_layers; "
                                 "start a new run to use background_final_layers")
            options["background_layers"] = config.data.background_layers
    prepared = prepare_colmap(dataset, split=_colmap_split(config.data.cameras), images_dir=config.data.images,
        point_cloud=config.data.point_cloud, initialization=config.data.initialization,
        start_core_resolution=start, target_core_resolution=target,
        load_points=load_points, **options)
    config.data.initialization = prepared["initialization"]
    config.render.width, config.render.height = prepared["width"], prepared["height"]
    _colmap_views(prepared, config.render)
    return prepared


def initialize_model(config, dataset, mi, np, *, prepared=None):
    if prepared is not None:
        if not hasattr(mi.VoxelReconstruction, "from_points"):
            raise RuntimeError("Rebuild this checkout: in-memory point-cloud initialization is missing")
        mapping = prepared["initialization"].get("scene_mapping")
        if mapping == "projective_shell" and not hasattr(mi.VoxelReconstruction, "set_scene_mapping"):
            raise RuntimeError("Rebuild this checkout: projective scene traversal is missing")
        model = mi.VoxelReconstruction.from_points(
            np.ascontiguousarray(prepared["positions"], dtype=np.float32).reshape(-1),
            np.ascontiguousarray(prepared["colors"], dtype=np.float32).reshape(-1),
            prepared["resolution"], prepared["grid_min"], prepared["grid_max"],
            config.model.sh_count, config.model.opacity_sh_count, prepared["core_resolution"])
        if mapping in ("radial_seed_shell", "projective_shell"):
            import drjit as dr
            from reconstruction_background import initialize_background_model
            options = {"mapping": mapping} if mapping == "projective_shell" else {}
            return initialize_background_model(model, mi, dr, np,
                opacity=prepared["initialization"]["background_opacity"],
                refinement=prepared["initialization"].get("background_refinement"),
                layers=prepared["initialization"].get("initial_background_layers",
                                                    prepared["initialization"].get("background_layers")),
                **options)
        return model
    lo, hi = np.asarray(config.model.bounds_min), np.asarray(config.model.bounds_max)
    center, half = (lo + hi) / 2, (hi - lo) / 2 * config.model.bounds_padding
    schedule = getattr(config, "schedule", None)
    resolution = schedule.start_resolution if schedule and schedule.enabled else config.model.grid_resolution
    return mi.VoxelReconstruction(str(dataset / config.data.point_cloud), resolution,
        (center-half).tolist(), (center+half).tolist(), config.model.opacity_threshold,
        config.model.sh_count, config.model.opacity_sh_count)


def load_views(dataset: Path, cameras_file: str, render: RenderConfig, np, *,
               data=None, split=None) -> list[dict]:
    from reconstruction_dataset import detect_dataset, prepare_colmap
    if detect_dataset(dataset, cameras_file) == "colmap":
        initialization = getattr(data, "initialization", None)
        if initialization is None:
            raise ValueError("COLMAP views require the model's saved initialization metadata")
        if split is None:
            split = _colmap_split(cameras_file)
        prepared = prepare_colmap(dataset, split=split, images_dir=data.images,
            initialization=initialization,
            start_core_resolution=initialization["start_core_resolution"],
            target_core_resolution=initialization["target_core_resolution"],
            shell_layers=initialization["shell_layers"], load_points=False)
        return _colmap_views(prepared, render)
    document = json.loads((dataset / cameras_file).read_text(encoding="utf-8"))
    frames = document.get("frames", [])
    if not frames:
        raise ValueError(f"Camera split is empty: {cameras_file}")
    config = PreviewConfig(render=render)
    config.data.cameras = cameras_file
    views = []
    for index in range(len(frames)):
        config.data.camera_index = index
        camera, reference = load_camera(config, dataset, np)
        camera["sampling"] = "per_pixel_uniform"
        camera["rng"] = "NumPy PCG64; distribution matches uniform Falcor samples, bit sequence differs"
        camera.update(width=render.width, height=render.height)
        views.append({"index": index, "camera": camera, "reference_path": str(reference)})
    return views


def read_reference(path, width, height, mi, np, *, rgb_only=False, background=(1, 1, 1),
                   reference_compositing="linear"):
    """Read linear targets with the selected reference compositing convention."""
    return reference_from_bitmap(
        mi.Bitmap(str(path)), mi, np, rgb_only=rgb_only, background=background,
        reference_compositing=reference_compositing, width=width, height=height, path=path)


class ReferenceImageCache:
    """Bounded RAM cache of linear targets; RGB-only entries retain no alpha."""

    def __init__(self, width, height, mi, np, capacity_mib=1024, *,
                 rgb_only=False, background=(1, 1, 1), reference_compositing="linear"):
        if isinstance(capacity_mib, bool) or not isinstance(capacity_mib, int) or capacity_mib < 0:
            raise ValueError("reference_cache_mib must be a nonnegative integer")
        self.width, self.height, self.mi, self.np = width, height, mi, np
        self.rgb_only, self.background = rgb_only, background
        validate_reference_compositing(reference_compositing)
        self.reference_compositing = reference_compositing
        self.capacity_bytes = capacity_mib * 1024 * 1024
        self.resident_bytes = 0
        self.hits = self.misses = self.evictions = 0
        self._images = OrderedDict()

    def get(self, path):
        if self.capacity_bytes == 0:
            self.misses += 1
            return read_reference(path, self.width, self.height, self.mi, self.np,
                                  rgb_only=self.rgb_only, background=self.background,
                                  reference_compositing=self.reference_compositing)
        key = (str(Path(path).resolve()), self.width, self.height, self.rgb_only,
               tuple(self.background), self.reference_compositing)
        if key in self._images:
            self.hits += 1
            self._images.move_to_end(key)
            return self._images[key]
        self.misses += 1
        rgba = read_reference(key[0], self.width, self.height, self.mi, self.np,
                              rgb_only=self.rgb_only, background=self.background,
                              reference_compositing=self.reference_compositing)
        if rgba.nbytes <= self.capacity_bytes:
            while self.resident_bytes + rgba.nbytes > self.capacity_bytes:
                _, old = self._images.popitem(last=False)
                self.resident_bytes -= old.nbytes
                self.evictions += 1
            # Sharing cached buffers must never let a consumer change a later epoch's target.
            rgba.setflags(write=False)
            self._images[key] = rgba
            self.resident_bytes += rgba.nbytes
        return rgba

    def statistics(self):
        return {"capacity_bytes": self.capacity_bytes, "resident_bytes": self.resident_bytes,
                "images": len(self._images), "hits": self.hits, "misses": self.misses,
                "evictions": self.evictions}


def prepare_camera(render, camera, mi, dr):
    """Upload camera values as runtime inputs so views can share JIT kernels.

    Literal camera arrays let Dr.Jit specialize the kernel for each viewpoint.
    Opaque arrays retain the same float32 values, but store them in device memory
    instead of embedding them in generated code. Keep these separately from the
    serializable camera dictionary and reuse them for every batch of a view.
    """
    values = {
        "position": dr.opaque(mi.Point3f, camera["position"]),
        "forward": dr.opaque(mi.Vector3f, camera["forward"]),
        "right": dr.opaque(mi.Vector3f, camera["right"]),
        "up": dr.opaque(mi.Vector3f, camera["up"]),
    }
    if "fx" in camera:
        values.update({key: dr.opaque(mi.Float, camera[key]) for key in ("fx", "fy", "cx", "cy")})
    else:
        tan_x = math.tan(camera["camera_angle_x"] / 2)
        tan_y = tan_x * render.height / render.width
        values.update(tan_x=dr.opaque(mi.Float, tan_x), tan_y=dr.opaque(mi.Float, tan_y))
    return values


def generate_ray_batch(render, camera, indices, offsets, mi, dr, *, prepared_camera=None):
    values = (prepare_camera(render, camera, mi, dr)
              if prepared_camera is None else prepared_camera)
    index = mi.UInt32(indices)
    u = (mi.Float(index % render.width) + mi.Float(offsets[:, 0])) / render.width
    v = (mi.Float(index // render.width) + mi.Float(offsets[:, 1])) / render.height
    if "fx" in values:
        horizontal = (u * render.width - values["cx"]) / values["fx"]
        vertical = (values["cy"] - v * render.height) / values["fy"]
    else:
        horizontal = (2 * u - 1) * values["tan_x"]
        vertical = (1 - 2 * v) * values["tan_y"]
    direction = values["forward"] + horizontal * values["right"] + vertical * values["up"]
    ray = mi.Ray3f(values["position"] + render.near_clip * direction,
                  dr.normalize(direction))
    ray.maxt = (float("inf") if camera.get("projective_unbounded", False) else
                (render.far_clip - render.near_clip) * dr.norm(direction))
    return ray


def render_view(model, render, camera, spp, seed, chunk_size, mi, dr, np):
    if spp < 1 or chunk_size < 1:
        raise ValueError("spp and chunk_size must be positive")
    count = render.width * render.height
    # Keep the original per-pixel, sample-ordered float64 sums on the device.
    # The persistent buffers occupy 32 bytes/pixel, independently of spp;
    # tracing and its temporary buffers still operate on at most chunk_size rays.
    Float64 = dr.float64_array_t(dr.detached_t(mi.Float))
    chunks = []
    for begin in range(0, count, chunk_size):
        end = min(begin + chunk_size, count)
        # Opaque zero also prevents the first add being simplified to a copy
        # (which could otherwise preserve a negative zero, unlike NumPy's +0).
        zero = dr.opaque(Float64, 0, end - begin)
        chunks.append((begin, end, [zero] * 4))
    rng = np.random.default_rng(seed)
    prepared_camera = prepare_camera(render, camera, mi, dr)
    for sample in range(spp):
        # Draw the entire sample before chunking so chunk size cannot change rays.
        offsets = rng.random((count, 2)).astype(np.float32)
        np.minimum(offsets, np.nextafter(np.float32(1), np.float32(0)), out=offsets)
        for begin, end, sums in chunks:
            ray = generate_ray_batch(render, camera, np.arange(begin, end, dtype=np.uint32),
                                     offsets[begin:end], mi, dr, prepared_camera=prepared_camera)
            rgb, alpha, hits = trace_model(model, ray, mi.Color3f(render.background), render)
            for channel, value in enumerate((rgb.x, rgb.y, rgb.z, alpha)):
                sums[channel] = sums[channel] + Float64(dr.detach(value))
            # Evaluate every batch to bound graph lifetime, but let consecutive
            # kernels execute asynchronously on the same ordered GPU stream.
            # Each add is evaluated before the next sample's add, so neither a
            # parallel reduction nor a reassociation changes the summation order.
            dr.eval(*sums, hits)
    if chunks:
        totals = [dr.concat([sums[channel] for _, _, sums in chunks])
                  for channel in range(4)]
        dr.eval(*totals)
        dr.sync_thread()
        rgb_sum = np.stack([np.asarray(value) for value in totals[:3]], axis=-1)
        alpha_sum = np.asarray(totals[3])
    else:
        rgb_sum = np.zeros((count, 3), dtype=np.float64)
        alpha_sum = np.zeros(count, dtype=np.float64)
    # Retain NumPy's final float64 division and float32 conversion exactly.
    # Only these four finished image channels are transferred to the CPU.
    rgb = (rgb_sum / spp).astype(np.float32).reshape(render.height, render.width, 3)
    alpha = (alpha_sum / spp).astype(np.float32).reshape(render.height, render.width)
    if not np.isfinite(rgb).all() or not np.isfinite(alpha).all():
        raise RuntimeError("Nonfinite rendered output")
    return rgb, alpha


def model_metadata(model) -> dict:
    metadata = {"resolution": int(model.resolution), "grid_min": list(model.grid_min),
            "grid_max": list(model.grid_max), "sh_count": int(model.sh_count),
            "opacity_sh_count": int(model.opacity_sh_count)}
    metadata.update(sh_model_metadata(model))
    core = int(getattr(model, "core_resolution", model.resolution))
    if core != metadata["resolution"]:
        metadata["core_resolution"] = core
    mapping = getattr(model, "scene_mapping", "bounded")
    if mapping in ("radial_seed_shell", "projective_shell"):
        metadata["scene_mapping"] = mapping
        if hasattr(model, "background_refinement"):
            metadata["background_refinement"] = model.background_refinement
            metadata["background_layers"] = int(model.background_layers)
    return metadata


def restore_model(checkpoint, mi):
    metadata = checkpoint["model_metadata"]
    state = {key: (mi.UInt32(value) if key == "cell_indices" else mi.Float(value))
             for key, value in checkpoint["model_state"].items()}
    options = ({"core_resolution": metadata["core_resolution"]} if "core_resolution" in metadata else {})
    model = mi.VoxelReconstruction.from_state(metadata["resolution"], metadata["grid_min"],
        metadata["grid_max"], state, metadata["sh_count"], metadata["opacity_sh_count"], **options)
    restore_sh_state(model, metadata)
    mapping = metadata.get("scene_mapping", "bounded")
    if mapping in ("radial_seed_shell", "projective_shell"):
        model.scene_mapping = mapping
        if mapping == "projective_shell":
            if not hasattr(model, "set_scene_mapping"):
                raise RuntimeError("Rebuild this checkout: projective scene traversal is missing")
            model.set_scene_mapping(mapping)
        if "background_refinement" in metadata:
            model.background_refinement = metadata["background_refinement"]
            model.background_layers = metadata["background_layers"]
    return model


class UniformSampler:
    """One uniform ray per pixel, independent of target values or alpha."""

    def __init__(self, np):
        self.np = np
        self.enabled = False

    def sample(self, view_id, reference_rgb, rng):
        np = self.np
        if len(reference_rgb.shape) != 3 or reference_rgb.shape[-1] != 3:
            raise ValueError("UniformSampler requires an RGB reference of shape (height, width, 3)")
        count = reference_rgb.shape[0] * reference_rgb.shape[1]
        offsets = rng.random((count, 2)).astype(np.float32)
        np.minimum(offsets, np.nextafter(np.float32(1), np.float32(0)), out=offsets)
        return offsets, np.ones(count, dtype=np.float32), None

    def observe(self, token, raw_alpha):
        if token is not None:
            raise ValueError("Uniform RGB sampling cannot consume contour history")

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        if state != {}:
            raise ValueError("Uniform RGB sampling cannot restore nonempty contour history")


class ContourSampler:
    """Falcor's frozen 4x4 proposal and raw-alpha EMA, specialized to 1 spp."""

    def __init__(self, np, enabled=False, uniform_fraction=.5, history_rate=.3):
        if not .05 <= uniform_fraction <= 1 or not 0 < history_rate <= 1:
            raise ValueError("Invalid contour sampling mixture/history rate")
        self.np, self.enabled = np, enabled
        self.uniform_fraction, self.history_rate = uniform_fraction, history_rate
        self.views = {}

    def sample(self, view_id, reference_alpha, rng):
        np = self.np
        alpha = reference_alpha.reshape(-1)
        offsets = rng.random((len(alpha), 2)).astype(np.float32)
        np.minimum(offsets, np.nextafter(np.float32(1), np.float32(0)), out=offsets)
        inverse_pdf = np.ones(len(alpha), dtype=np.float32)
        if not self.enabled:
            return offsets, inverse_pdf, None
        key = str(view_id)
        if key not in self.views:
            pixels = np.flatnonzero((alpha > 0) & (alpha < 1))
            self.views[key] = {"pixels": pixels, "mean_alpha": np.repeat(alpha[pixels, None], 16, axis=1)}
        history = self.views[key]
        pixels = history["pixels"]
        if not len(pixels):
            return offsets, inverse_pdf, None
        weights = np.maximum(history["mean_alpha"], 0)
        total = weights.sum(axis=1, keepdims=True)
        probability = np.where(total > 0,
            self.uniform_fraction / 16 + (1 - self.uniform_fraction) * weights / np.maximum(total, 1e-30), 1 / 16)
        random = rng.random((len(pixels), 3))
        bins = np.minimum((random[:, :1] >= np.cumsum(probability, axis=1)).sum(axis=1), 15)
        offsets[pixels, 0] = (bins % 4 + random[:, 1]) / 4
        offsets[pixels, 1] = (bins // 4 + random[:, 2]) / 4
        np.minimum(offsets, np.nextafter(np.float32(1), np.float32(0)), out=offsets)
        inverse_pdf[pixels] = 1 / (16 * probability[np.arange(len(pixels)), bins])
        return offsets, inverse_pdf, (key, bins)

    def observe(self, token, raw_alpha):
        if token is None:
            return
        key, bins = token
        history = self.views[key]
        rows = self.np.arange(len(bins))
        old = history["mean_alpha"][rows, bins]
        history["mean_alpha"][rows, bins] = old + self.history_rate * (raw_alpha.reshape(-1)[history["pixels"]] - old)

    def state_dict(self):
        return self.views

    def load_state_dict(self, state):
        self.views = state
