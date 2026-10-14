"""Render a small Cornell box on CUDA and record the actual runtime environment.

This tests the imported Mitsuba build. A wheel does not include C++ edits in
this checkout. No CPU fallback is allowed.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository.parent / "outputs" / "mitsuba" / "local-smoke",
    )
    parser.add_argument("--gpu", type=int, default=0, help="nvidia-smi GPU index")
    parser.add_argument(
        "--build-dir", type=Path,
        help="Require a source-build runtime directory, e.g. build/Release on Windows",
    )
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "gpu-smoke.json"
    image_path = output / "cornell-64x64.png"
    report = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "python_prefix": sys.prefix,
        "python_base_prefix": sys.base_prefix,
        "platform": platform.platform(),
        "requested_variant": "cuda_ad_rgb",
        "width": 64,
        "height": 64,
        "spp": 4,
        "seed": 42,
        "max_depth": 4,
        "scope": "Imported Mitsuba build only; no validation of checkout C++ changes.",
    }
    print("Python executable:", sys.executable, flush=True)
    print("Python version:", platform.python_version(), flush=True)
    try:
        dll_directories = []  # Keep Windows DLL search handles alive during the test.
        if args.build_dir is not None:
            runtime = args.build_dir.resolve()
            python_directory = runtime / "python"
            for package in ("mitsuba", "drjit"):
                if not (python_directory / package / "__init__.py").is_file():
                    raise RuntimeError(f"Source build is incomplete: {python_directory / package}")
            sys.path.insert(0, str(python_directory))
            if os.name == "nt":
                for directory in (runtime, python_directory / "drjit", python_directory / "mitsuba"):
                    dll_directories.append(os.add_dll_directory(str(directory)))
            report["build_directory"] = str(runtime)
            report["scope"] = "Source build loaded explicitly; basic forward CUDA rendering only."

        # Restrict this process to one known GPU before either library is imported.
        query = subprocess.run(
            [
                "nvidia-smi", f"--id={args.gpu}",
                "--query-gpu=index,name,uuid,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True, capture_output=True, text=True, timeout=30,
        )
        rows = list(csv.reader(query.stdout.strip().splitlines(), skipinitialspace=True))
        if len(rows) != 1 or len(rows[0]) != 5:
            raise RuntimeError(f"Unexpected nvidia-smi result: {query.stdout!r}")
        report["gpu"] = dict(zip(
            ("index", "name", "uuid", "driver_version", "memory_mib"), rows[0]
        ))
        os.environ["CUDA_VISIBLE_DEVICES"] = report["gpu"]["uuid"]
        # All runtime caches stay alongside the outputs, outside the checkout.
        for variable, dirname in (
            ("DRJIT_CACHE_DIR", "drjit"),
            ("CUDA_CACHE_PATH", "cuda"),
            ("OPTIX_CACHE_PATH", "optix"),
        ):
            cache = output / "cache" / dirname
            cache.mkdir(parents=True, exist_ok=True)
            os.environ[variable] = str(cache)
        print("GPU:", json.dumps(report["gpu"]), flush=True)

        import drjit as dr
        import mitsuba as mi

        report.update({
            "mitsuba_version": mi.__version__,
            "drjit_version": dr.__version__,
            "mitsuba_file": mi.__file__,
            "drjit_file": dr.__file__,
            "numpy_version": version("numpy"),
        })
        print("Mitsuba:", mi.__version__, flush=True)
        print("Dr.Jit:", dr.__version__, flush=True)
        print("NumPy:", report["numpy_version"], flush=True)
        print("mitsuba.__file__:", mi.__file__, flush=True)
        print("drjit.__file__:", dr.__file__, flush=True)
        if args.build_dir is not None:
            for module in (mi, dr):
                if not Path(module.__file__).resolve().is_relative_to(python_directory):
                    raise RuntimeError(f"Expected source build, imported {module.__file__}")
            print("Source-build import paths verified.", flush=True)
        mi.set_variant("cuda_ad_rgb")
        report["actual_variant"] = mi.variant()
        print("Actual variant:", mi.variant(), flush=True)
        if mi.variant() != "cuda_ad_rgb" or dr.backend_v(mi.Float) != dr.JitBackend.CUDA:
            raise RuntimeError("CUDA variant/backend verification failed")

        description = mi.cornell_box()
        description["integrator"]["max_depth"] = report["max_depth"]
        description["sensor"]["film"].update(width=64, height=64, pixel_format="rgb")
        description["sensor"]["sampler"]["sample_count"] = report["spp"]
        scene = mi.load_dict(description)
        started = time.perf_counter()
        image = mi.render(scene, spp=report["spp"], seed=report["seed"])
        dr.eval(image)
        dr.sync_thread()
        report["render_seconds_including_jit"] = time.perf_counter() - started
        report["shape"] = list(image.shape)
        # DLPack device type 2 denotes CUDA, independently of the variant name.
        report["dlpack_device"] = list(image.array.__dlpack_device__())
        report["all_finite"] = bool(dr.all(dr.isfinite(image.array)))
        report["nonblack"] = bool(dr.any(image.array > 0))
        if report["shape"] != [64, 64, 3]:
            raise RuntimeError(f"Unexpected image shape: {report['shape']}")
        if report["dlpack_device"] != [2, 0]:
            raise RuntimeError(f"Unexpected image device: {report['dlpack_device']}")
        if not report["all_finite"] or not report["nonblack"]:
            raise RuntimeError("Image contains non-finite values or is entirely black")
        report["minimum"] = float(dr.min(image.array)[0])
        report["maximum"] = float(dr.max(image.array)[0])
        report["mean"] = float(dr.mean(image.array)[0])
        bitmap = mi.Bitmap(image).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, True)
        bitmap.write(str(image_path))  # Synchronous write.
        if image_path.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            raise RuntimeError("Output is not a PNG file")
        saved = mi.Bitmap(str(image_path))
        if list(saved.size()) != [64, 64]:
            raise RuntimeError("Saved PNG has an unexpected size")
        report["png"] = str(image_path)
        report["png_sha256"] = hashlib.sha256(image_path.read_bytes()).hexdigest()
        report["status"] = "passed"
        print("PASS: CUDA image is 64x64x3, finite, and nonblack.", flush=True)
        print("PNG:", image_path, flush=True)
    except Exception:
        report["status"] = "failed"
        report["error"] = traceback.format_exc()
        print(report["error"], file=sys.stderr, flush=True)
    finally:
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print("Report:", report_path, flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
