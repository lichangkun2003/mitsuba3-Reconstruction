"""Sequentially generate missing init_points.ply files for installed 360 scenes.

Uses generate_dense_points.py unchanged for each scene, including training-only
views, image scales, external caches and resumable COLMAP stages. Existing valid
point clouds are skipped. Original images and sparse models remain read-only.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_dense_points as single


SCENES = single.OUTDOOR | single.INDOOR


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
        epilog="Other options are forwarded to the single-scene generator, e.g. "
               "--gpu 0 --num-threads 8 --colmap PATH --fusion-use-cache. "
               "Rerun the same command to skip completed scenes and resume unfinished stages.")
    parser.add_argument("--datasets-root", type=Path, default=single.DATASETS_ROOT,
                        help="Dataset parent directory; default: %(default)s")
    parser.add_argument("--scenes", nargs="+", choices=sorted(SCENES),
                        help="Optionally restrict the batch to these scene names")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate inputs and print COLMAP commands without generating or writing files")
    batch, forwarded = parser.parse_known_args(argv)
    # Resolve all options with the original parser, so validation and defaults
    # do not drift. Per-scene destinations and replacement are not batch options.
    options = single.parse_arguments(forwarded)
    if (options.dataset is not None or options.workspace is not None or options.output is not None
            or options.overwrite or options.scene != single.SCENE
            or any("--scene".startswith(arg.split("=", 1)[0])
                   for arg in forwarded if arg.startswith("--"))):
        parser.error("Use --datasets-root/--scenes; --scene, --dataset, --workspace, --output "
                     "and --overwrite are not supported in batch mode")
    options.dry_run = batch.dry_run or options.dry_run
    return batch, options


def discover_scenes(root, names=None):
    root = root.expanduser().absolute()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    if names is not None:
        result = [root / name for name in sorted(set(names))]
        missing = [path.name for path in result if not path.is_dir()]
        if missing:
            raise FileNotFoundError("Requested scene directories are missing: " + ", ".join(missing))
        return result
    return sorted((path for path in root.iterdir()
                   if path.is_dir() and path.name.lower() in SCENES), key=lambda path: path.name.lower())


def run_batch(datasets, options):
    if not datasets:
        print("No installed Mip-NeRF 360 scenes found.", flush=True)
        return 0
    print("360 scene order: " + " -> ".join(path.name for path in datasets), flush=True)
    completed, skipped, failures = [], [], []
    for index, dataset in enumerate(datasets, 1):
        output = dataset / "init_points.ply"
        print(f"\n[{index}/{len(datasets)}] {dataset.name}", flush=True)
        try:
            if output.exists():
                # Never silently overwrite an incomplete or unrelated file.
                # A failed atomic generation normally leaves no final PLY.
                count = single.ply_vertex_count(output)
                print(f"SKIP: {output} ({count:,} points)", flush=True)
                skipped.append(dataset.name)
                continue
            args = copy.copy(options)
            args.dataset, args.scene = dataset, dataset.name
            # COLMAP already runs in separate child processes for each stage.
            # Calling its existing wrapper preserves Ctrl-C cleanup and its
            # manifest; the batch never initializes a GPU runtime itself.
            single.generate(args)
            if not args.dry_run and not args.prepare_only:
                single.ply_vertex_count(output)
            completed.append(dataset.name)
        except KeyboardInterrupt:
            print("\nBatch interrupted. Rerun to skip completed scenes and resume the remaining ones.",
                  file=sys.stderr, flush=True)
            return 130
        except Exception as error:
            failures.append((dataset.name, str(error)))
            print(f"FAILED: {dataset.name}: {error}", file=sys.stderr, flush=True)
    action = "Validated" if options.dry_run else "Prepared" if options.prepare_only else "Generated"
    print(f"\n{action}: {len(completed)} | skipped: {len(skipped)} | failed: {len(failures)}", flush=True)
    for scene, error in failures:
        print(f"  {scene}: {error}", file=sys.stderr, flush=True)
    return 1 if failures else 0


def main(argv=None):
    batch, options = parse_arguments(argv)
    try:
        return run_batch(discover_scenes(batch.datasets_root, batch.scenes), options)
    except (OSError, ValueError) as error:
        print(f"Dense batch failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
