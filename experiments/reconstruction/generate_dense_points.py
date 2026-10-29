"""Create Mip-NeRF 360 init_points.ply from its supplied COLMAP sparse/0 model.

Edit SCENE below, or pass --scene. Uses only NumPy, Pillow and the COLMAP CLI;
Mitsuba does not need to be built. Intermediate work resumes in dense_init/.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconstruction_colmap_dense import inspect_dense_inputs, prepare_dense_model


# Change only SCENE to generate another standard Mip-NeRF 360 scene.
SCENE = "bonsai"
WORKSPACE = Path(__file__).resolve().parents[3]  # reconstruction/
DATASETS_ROOT = WORKSPACE / "datasets"
COLMAP = WORKSPACE.parent / "colmap-x64-windows-cuda" / "COLMAP.bat"
IMAGE_DIRECTORY = None  # None: outdoor images_4, indoor images_2.
SPLIT = "train"  # Same sorted-filename every-eighth holdout as train.py; "all" is explicit.
GPU_INDEX = "0"

OUTDOOR = {"bicycle", "flowers", "garden", "stump", "treehill"}
INDOOR = {"room", "counter", "kitchen", "bonsai"}


def choose_images(dataset, requested=None):
    if requested is not None:
        path = Path(requested)
        path = path if path.is_absolute() else dataset / path
    elif dataset.name.lower() in OUTDOOR:
        path = dataset / "images_4"
    elif dataset.name.lower() in INDOOR:
        path = dataset / "images_2"
    else:
        raise ValueError(f"Unknown 360 scene {dataset.name!r}; specify --images explicitly")
    if not path.is_dir():
        raise FileNotFoundError(f"Selected image directory is missing: {path}")
    return path.absolute()


def colmap_executable(path):
    """Use the Windows distribution's EXE with the same environment as COLMAP.bat.

    Passing argument lists directly avoids shell quoting problems with spaces
    and special characters in scene paths. No interactive GUI is launched.
    """
    requested = Path(path).expanduser()
    located = str(requested) if requested.is_file() else shutil.which(str(path))
    if not located:
        raise FileNotFoundError(f"COLMAP not found: {path}; edit COLMAP or pass --colmap")
    executable = Path(located).absolute()
    env = dict(os.environ)
    if executable.suffix.lower() in (".bat", ".cmd"):
        root = executable.parent
        executable = root / "bin" / "colmap.exe"
        if not executable.is_file():
            raise FileNotFoundError(f"Expected the COLMAP distribution executable: {executable}")
        env["QT_PLUGIN_PATH"] = str(root / "plugins") + os.pathsep + env.get("QT_PLUGIN_PATH", "")
    env["PATH"] = str(executable.parent) + os.pathsep + env.get("PATH", "")
    return executable, env


def file_identity(path, *, digest=False):
    stat = path.stat()
    result = dict(path=str(path.absolute()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if digest:
        with path.open("rb") as stream:
            result["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def ply_vertex_count(path):
    """Require a nonempty complete COLMAP binary XYZ/normal/RGB PLY."""
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY point cloud: {path}")
        count, properties, binary = 0, [], False
        sizes = {b"float": 4, b"float32": 4, b"double": 8, b"float64": 8,
                 b"uchar": 1, b"uint8": 1}
        for _ in range(128):
            parts = stream.readline().split()
            if not parts:
                raise ValueError(f"Incomplete PLY header: {path}")
            if parts[:1] == [b"format"]:
                binary = parts[1] == b"binary_little_endian"
            elif parts[:2] == [b"element", b"vertex"]:
                count = int(parts[2])
            elif parts[:1] == [b"property"]:
                if len(parts) != 3 or parts[1] not in sizes:
                    raise ValueError(f"Unsupported PLY property: {parts}")
                properties.append((parts[2], sizes[parts[1]]))
            elif parts == [b"end_header"]:
                break
        else:
            raise ValueError(f"PLY header is too long: {path}")
        if not binary or count <= 0 or not {b"x", b"y", b"z", b"red", b"green", b"blue"} <= {
                name for name, _ in properties}:
            raise ValueError(f"Fusion did not produce nonempty binary XYZ/RGB points: {path}")
        expected = count * sum(size for _, size in properties)
        if path.stat().st_size - stream.tell() < expected:
            raise ValueError(f"Truncated fused PLY: {path}")
    return count


def run_command(command, env, log_path):
    print("\n" + subprocess.list2cmdline([str(x) for x in command]), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\nCOMMAND: " + subprocess.list2cmdline([str(x) for x in command]) + "\n")
        with subprocess.Popen([str(x) for x in command], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace") as process:
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except KeyboardInterrupt:
                # Ctrl-C is also delivered to COLMAP in the same console.
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait()
                raise
        if code:
            raise RuntimeError(f"COLMAP exited with code {code}; see {log_path}")


def commands_for(args, executable, images, work):
    dense = work / "dense"
    return [
        ("undistort", [executable, "image_undistorter", "--image_path", images,
            "--input_path", work / "scaled_sparse", "--output_path", dense,
            "--output_type", "COLMAP", "--max_image_size", str(args.max_image_size),
            "--num_patch_match_src_images", str(args.source_images),
            "--num_threads", str(args.num_threads)]),
        ("patch_match", [executable, "patch_match_stereo", "--workspace_path", dense,
            "--workspace_format", "COLMAP", "--PatchMatchStereo.geom_consistency", "1",
            "--PatchMatchStereo.gpu_index", args.gpu,
            "--PatchMatchStereo.max_image_size", str(args.max_image_size),
            "--PatchMatchStereo.num_iterations", str(args.iterations),
            "--PatchMatchStereo.cache_size", str(args.cache_size),
            "--PatchMatchStereo.num_threads", str(args.num_threads)]),
        ("fusion", [executable, "stereo_fusion", "--workspace_path", dense,
            "--workspace_format", "COLMAP", "--input_type", "geometric",
            "--output_path", work / "fused.ply", "--StereoFusion.use_cache", "1",
            "--StereoFusion.cache_size", str(args.cache_size),
            "--StereoFusion.max_image_size", str(args.max_image_size),
            "--StereoFusion.num_threads", str(args.num_threads)]),
    ]


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=SCENE, help="Scene name beneath datasets; default: %(default)s")
    parser.add_argument("--dataset", type=Path, help="Override the full scene path")
    parser.add_argument("--colmap", default=COLMAP)
    parser.add_argument("--images", default=IMAGE_DIRECTORY, help="Override images_2/images_4 selection")
    parser.add_argument("--split", choices=("train", "all"), default=SPLIT)
    parser.add_argument("--gpu", default=GPU_INDEX, help="COLMAP GPU indices, e.g. 0 or 0,1")
    parser.add_argument("--workspace", type=Path, help="Intermediate directory; default: SCENE/dense_init/IMAGES-SPLIT")
    parser.add_argument("--output", type=Path, help="Output PLY; default: SCENE/init_points.ply")
    parser.add_argument("--max-image-size", type=int, default=-1, help="-1 keeps selected images' full resolution")
    parser.add_argument("--source-images", type=int, default=20, help="PatchMatch source views per reference")
    parser.add_argument("--iterations", type=int, default=5, help="PatchMatch iterations")
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--cache-size", type=float, default=4., help="CPU RAM cache in GiB for each COLMAP stage")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print commands without writing or computing depth")
    parser.add_argument("--prepare-only", action="store_true", help="Prepare scaled sparse model and undistorted workspace; defer depth/fusion")
    parser.add_argument("--overwrite", action="store_true", help="Back up an existing output PLY before replacing it on success")
    args = parser.parse_args()
    if args.max_image_size != -1 and args.max_image_size < 1:
        parser.error("--max-image-size must be -1 or positive")
    if (min(args.source_images, args.iterations, args.num_threads, args.cache_size) <= 0
            or not math.isfinite(args.cache_size)):
        parser.error("source-images, iterations, num-threads and cache-size must be positive")
    if Path(args.scene).name != args.scene or args.scene in (".", ".."):
        parser.error("--scene must be a scene name; use --dataset for a full path")
    return args


def generate(args):
    dataset = (args.dataset or DATASETS_ROOT / args.scene).expanduser().absolute()
    images = choose_images(dataset, args.images)
    source = dataset / "sparse" / "0"
    info = inspect_dense_inputs(source, images, args.split)
    executable, env = colmap_executable(args.colmap)
    work = (args.workspace or dataset / "dense_init" / f"{images.name}-{args.split}").absolute()
    output = (args.output or dataset / "init_points.ply").absolute()
    if output.suffix.lower() != ".ply":
        raise ValueError("Output filename must end in .ply")
    protected = [source.resolve(), images.resolve()]
    if any(work.resolve() == p or work.resolve().is_relative_to(p) or p.is_relative_to(work.resolve())
           or output.resolve().is_relative_to(p) for p in protected):
        raise ValueError("Work/output paths must not overwrite source images or sparse/0")
    if output.resolve() == (work / "fused.ply").resolve():
        raise ValueError("Output must be distinct from the intermediate fused.ply")
    commands = commands_for(args, executable, images, work)
    print(f"Scene: {dataset}\nImages: {images}\nCameras: {source}\n"
          f"Views: {info['selected_image_count']}/{info['registered_images']} ({args.split})\n"
          f"Workspace: {work}\nOutput: {output}", flush=True)
    for camera in info["cameras"]:
        print(f"Camera {camera['id']}: {camera['source_width']}x{camera['source_height']} -> "
              f"{camera['width']}x{camera['height']}", flush=True)
    if args.dry_run:
        for _, command in commands:
            print(subprocess.list2cmdline([str(x) for x in command]))
        return
    if output.exists() and not args.overwrite and not args.prepare_only:
        raise FileExistsError(f"Output already exists: {output}. Use --overwrite to preserve a backup and replace it.")
    identity = dict(version=1, input=info, colmap=file_identity(executable),
                    sparse=[file_identity(source / name, digest=True)
                            for name in ("cameras.bin", "images.bin", "points3D.bin")],
                    images=[file_identity(images / image['name']) for image in info['selected_images']],
                    commands=[[str(x) for x in command] for _, command in commands])
    manifest_path = work / "dense-init.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["identity"] != identity:
            raise ValueError(f"Inputs/settings changed since this workspace was created: {work}. "
                             "Use another --workspace to keep existing depth maps separate.")
    else:
        if work.exists() and any(work.iterdir()):
            raise FileExistsError(f"Unrecognized nonempty workspace: {work}; choose another --workspace")
        work.mkdir(parents=True, exist_ok=True)
        manifest = dict(identity=identity, stages={}, status="prepared")
        write_json(manifest_path, manifest)
    started = time.perf_counter()
    try:
        scaled = work / "scaled_sparse"
        if not scaled.is_dir():
            print("Preparing camera scales, training views and sparse tracks...", flush=True)
            manifest["sparse_preparation"] = prepare_dense_model(source, images, scaled, args.split)
            write_json(manifest_path, manifest)
        (work / "dense").mkdir(exist_ok=True)
        for stage, command in commands:
            if args.prepare_only and stage != "undistort":
                break
            if stage in manifest["stages"]:
                print(f"Reusing completed {stage} stage", flush=True)
                continue
            manifest.update(status="running", current_stage=stage)
            write_json(manifest_path, manifest)
            stage_started = time.perf_counter()
            run_command(command, env, work / f"{stage}.log")
            if stage == "fusion":
                ply_vertex_count(work / "fused.ply")
            manifest["stages"][stage] = dict(seconds=time.perf_counter() - stage_started)
            write_json(manifest_path, manifest)
        if args.prepare_only:
            manifest["status"] = "prepared"
            print("Prepared. Rerun without --prepare-only to compute and fuse dense depth maps.", flush=True)
        else:
            count = ply_vertex_count(work / "fused.ply")
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                if not args.overwrite:
                    raise FileExistsError(output)
                backup = output.with_name(output.stem + ".backup-" +
                    datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".ply")
                shutil.copy2(output, backup)
                manifest["backup"] = str(backup)
                print(f"Previous point cloud backed up: {backup}", flush=True)
            with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".dense-points-", suffix=".ply", delete=False) as stream:
                pending = Path(stream.name)
            try:
                shutil.copyfile(work / "fused.ply", pending)
                pending.replace(output)
            finally:
                pending.unlink(missing_ok=True)
            manifest.update(status="complete", output=str(output), vertex_count=count,
                            coordinate_system="original COLMAP world coordinates; no normalization or contraction")
            print(f"\nSaved {count:,} points: {output}", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=str(error))
        raise
    finally:
        manifest["last_run_seconds"] = time.perf_counter() - started
        write_json(manifest_path, manifest)


def main():
    try:
        generate(parse_arguments())
    except KeyboardInterrupt:
        print("\nInterrupted; rerun the same command to resume.", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Dense reconstruction failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
