"""Real COLMAP CUDA smoke test using a tiny known textured plane.

This test writes only a temporary synthetic scene, never a project dataset.
It verifies world coordinates, the scaled-camera MVS pipeline, and compatibility
with the reconstruction initializer's colored PLY reader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruction_dataset import read_ply_points


def write_plane_scene(dataset):
    """Nine cameras see a plane at world z=3; model pixels are twice the images."""
    dataset = Path(dataset)
    model = dataset / "sparse" / "0"
    images = dataset / "images_2"
    model.mkdir(parents=True)
    images.mkdir()
    width, height, focal, depth = 192, 128, 160., 3.
    rng = np.random.default_rng(31)
    texture = Image.fromarray(rng.integers(10, 246, (height, 256, 3), dtype=np.uint8))
    texture = texture.filter(ImageFilter.GaussianBlur(.65))
    camera_centers = [(index - 4) * .15 for index in range(9)]
    for index in range(9):
        # Camera translations produce integer shifts in both resolutions.
        start = index * 8
        crop = texture.crop((start, 0, start + width, height))
        crop.resize((width // 2, height // 2), Image.Resampling.BOX).save(
            images / f"view-{index:02d}.png")

    points = [(x, y, depth) for y in np.linspace(-.75, .75, 11)
              for x in np.linspace(-.9, .9, 13)]
    with (model / "cameras.bin").open("wb") as stream:
        stream.write(struct.pack("<QiiQQdddd", 1, 7, 1, width, height,
                                 focal, focal, width / 2, height / 2))
    with (model / "images.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", len(camera_centers)))
        for index, center in enumerate(camera_centers):
            stream.write(struct.pack("<idddddddi", 100 + index, 1., 0., 0., 0.,
                                     -center, 0., 0., 7))
            stream.write(f"view-{index:02d}.png".encode("ascii") + b"\0")
            stream.write(struct.pack("<Q", len(points)))
            for point_index, (x, y, z) in enumerate(points):
                stream.write(struct.pack("<ddq", focal * (x - center) / z + width / 2,
                                         focal * y / z + height / 2, 1000 + point_index))
    with (model / "points3D.bin").open("wb") as stream:
        stream.write(struct.pack("<Q", len(points)))
        for point_index, (x, y, z) in enumerate(points):
            stream.write(struct.pack("<QdddBBBdQ", 1000 + point_index, x, y, z,
                                     128, 145, 160, .01, len(camera_centers)))
            for image_index in range(len(camera_centers)):
                stream.write(struct.pack("<ii", 100 + image_index, point_index))
    return dataset


def source_hashes(dataset):
    return {str(path.relative_to(dataset)): hashlib.sha256(path.read_bytes()).hexdigest()
            for pattern in ("sparse/0/*.bin", "images_2/*.png") for path in dataset.glob(pattern)}


def run_check(root, colmap, gpu):
    root = Path(root)
    dataset = write_plane_scene(root / "dataset")
    original = source_hashes(dataset)
    destination = root / "init_points.ply"
    workspace = root / "dense workspace"
    script = Path(__file__).resolve().parents[1] / "generate_dense_points.py"
    command = [sys.executable, "-u", "-I", str(script), "--dataset", str(dataset),
               "--images", "images_2", "--split", "all", "--colmap", str(colmap),
               "--workspace", str(workspace), "--output", str(destination),
               "--source-images", "4", "--iterations", "2", "--max-image-size", "-1",
               "--cache-size", "1", "--num-threads", "2", "--gpu", str(gpu)]

    def run(options, name, expected=0):
        result = subprocess.run(command + options, text=True, encoding="utf-8", errors="replace",
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        path = root / f"{name}.log"
        path.write_text(result.stdout, encoding="utf-8")
        if result.returncode != expected:
            raise RuntimeError(f"Generator failed ({result.returncode}); see {path}:\n{result.stdout}")
        return result.stdout

    run(["--dry-run"], "dry-run")
    assert not workspace.exists() and not destination.exists()
    run(["--prepare-only"], "prepare")
    assert not destination.exists()
    manifest_path = workspace / "dense-init.json"
    prepared = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert prepared["status"] == "prepared" and set(prepared["stages"]) == {"undistort"}
    run([], "run")
    positions, colors = read_ply_points(destination)
    assert len(positions) > 100, len(positions)
    assert colors.shape == positions.shape
    assert np.isfinite(positions).all() and np.isfinite(colors).all()
    assert np.all((colors >= 0) & (colors <= 1))
    depth_error = np.abs(positions[:, 2] - 3.)
    assert np.median(depth_error) < .05, np.quantile(depth_error, [.5, .9, .99])
    assert np.mean(depth_error < .15) > .9, np.quantile(depth_error, [.5, .9, .99])
    assert np.ptp(positions[:, 0]) > 1 and np.ptp(positions[:, 1]) > 1
    output_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
    run([], "existing-output", expected=1)
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == output_hash
    logs_before = {str(path): path.stat().st_mtime_ns for path in workspace.glob("*.log")}
    reused = run(["--overwrite"], "overwrite")
    assert "Reusing completed" in reused
    assert {str(path): path.stat().st_mtime_ns for path in workspace.glob("*.log")} == logs_before
    backups = list(destination.parent.glob("init_points.backup-*.ply"))
    assert len(backups) == 1
    assert hashlib.sha256(backups[0].read_bytes()).hexdigest() == output_hash
    assert hashlib.sha256(destination.read_bytes()).hexdigest() == output_hash
    assert source_hashes(dataset) == original
    return {"status": "passed", "output": str(root), "point_count": len(positions),
            "median_world_depth_error": float(np.median(depth_error)),
            "checks": ["real COLMAP undistortion, CUDA PatchMatch and fusion",
                       "downsampled images with scaled sparse cameras/observations",
                       "colored normal-bearing PLY readable by reconstruction initialization",
                       "original COLMAP world coordinates preserved",
                       "source camera, image, sparse point and RGB files unchanged",
                       "dry-run writes nothing; prepare-only resumes into dense stages",
                       "existing output protected; overwrite backs up and reuses complete stages"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    from generate_dense_points import COLMAP
    parser.add_argument("--colmap", default=COLMAP)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix="colmap-dense-smoke-", dir=args.output_dir))
    print(f"COLMAP dense smoke artifacts: {root}", flush=True)
    report = run_check(root, args.colmap, args.gpu)
    (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
