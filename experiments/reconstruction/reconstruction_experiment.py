"""Allocate daily experiment directories without importing a rendering backend."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
import math
from numbers import Real
import os
from pathlib import Path
import re


# Use the same experiment date on Windows and HPC, regardless of host timezone.
EXPERIMENT_TIMEZONE = timezone(timedelta(hours=4), "Asia/Dubai")


@contextmanager
def _allocation_lock(workspace: Path):
    """Hold a process lock while scanning and reserving a global daily index."""
    path = workspace / ".cache" / "mitsuba" / "experiment-index.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the file: unlinking a lock file can give competing processes different
    # lock objects. The OS releases the lock even if a training process crashes.
    with path.open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def allocate_training_output(dataset: str, workspace: Path, *, day: date | None = None) -> Path:
    """Reserve outputs/<scene>-MM.DD-E<index>/training across all scenes."""
    # Use the configured directory name, not a dataset symlink's target name.
    scene = Path(dataset.replace("\\", "/")).name
    if (scene in ("", ".", "..") or scene.endswith((".", " "))
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', scene)):
        raise ValueError("data.dataset must end in a valid scene directory name")
    stamp = (day if day is not None else datetime.now(EXPERIMENT_TIMEZONE).date()).strftime("%m.%d")
    pattern = re.compile(r".+-" + re.escape(stamp) + r"-E([0-9]+)(?:-[0-9]+min)?")
    root = (workspace / "outputs").resolve()
    with _allocation_lock(workspace):
        root.mkdir(parents=True, exist_ok=True)
        indices = [int(match[1]) for path in root.iterdir()
                   if path.is_dir() and (match := pattern.fullmatch(path.name))]
        index = max(indices, default=-1) + 1
        while True:
            experiment = root / f"{scene}-{stamp}-E{index}"
            try:
                experiment.mkdir()
            except FileExistsError:
                # Also avoid a same-name file or a directory created manually.
                index += 1
                continue
            output = experiment / "training"
            output.mkdir()
            return output


def finalize_training_output(output: Path, seconds: float, workspace: Path) -> Path:
    """Add elapsed whole minutes to an automatically named experiment directory.

    Only direct children of the workspace outputs directory participate. Custom
    output directories keep their names, and existing destinations are never
    replaced. The allocation lock also prevents another allocator from reusing
    an experiment number while its directory is being renamed.
    """
    if (isinstance(seconds, bool) or not isinstance(seconds, Real)
            or not math.isfinite(seconds) or seconds < 0):
        raise ValueError("Training duration must be a finite nonnegative number of seconds")
    minutes = max(1, math.ceil(seconds / 60))
    output = Path(output)
    source = output.parent
    match = re.fullmatch(r"(.+-[0-9]{2}\.[0-9]{2}-E[0-9]+)(?:-[0-9]+min)?",
                         source.name)
    root = (workspace / "outputs").resolve()
    if output.name != "training" or match is None or source.parent.resolve() != root:
        return output
    if source.is_symlink() or output.is_symlink():
        raise ValueError("Cannot rename an experiment or training directory that is a symlink")
    source = source.resolve()
    resolved_output = output.resolve()
    if source.parent != root or resolved_output.parent != source:
        raise ValueError("Experiment rename must stay within the workspace outputs directory")
    destination = root / f"{match[1]}-{minutes}min"
    with _allocation_lock(workspace):
        if not source.is_dir() or not resolved_output.is_dir():
            raise FileNotFoundError(f"Training output directory does not exist: {output}")
        if source == destination:
            return resolved_output
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Experiment destination already exists: {destination}")
        if destination.resolve().parent != root:
            raise ValueError("Experiment destination must stay within the workspace outputs directory")
        source.rename(destination)
    return destination / "training"


def relocate_experiment_paths(value, old_root: Path, new_root: Path, workspace: Path):
    """Rebase path strings in metadata without changing unrelated text or paths.

    Absolute paths and paths relative to ``workspace`` retain their respective
    form. Only complete path prefixes match; a similar name, a historical path
    on another machine, or a path appearing inside prose is left untouched.
    """
    def normalized(path):
        return str(path).replace("\\", "/").rstrip("/")

    old = normalized(old_root)
    new = normalized(new_root)
    base = normalized(workspace)
    windows = os.name == "nt" or bool(re.match(r"^[A-Za-z]:/", old)) or old.startswith("//")

    def equal(left, right):
        return left.casefold() == right.casefold() if windows else left == right

    def suffix(text, prefix):
        if equal(text, prefix):
            return ""
        if len(text) > len(prefix) and text[len(prefix)] == "/" and equal(text[:len(prefix)], prefix):
            return text[len(prefix):]
        return None

    replacements = [(old, new)]
    old_relative = suffix(old, base)
    new_relative = suffix(new, base)
    if old_relative is not None and new_relative is not None:
        replacements.append((old_relative.lstrip("/"), new_relative.lstrip("/")))

    def visit(item):
        if isinstance(item, dict):
            return {key: visit(entry) for key, entry in item.items()}
        if isinstance(item, list):
            return [visit(entry) for entry in item]
        if isinstance(item, tuple):
            return tuple(visit(entry) for entry in item)
        if not isinstance(item, str):
            return item
        text = item.replace("\\", "/")
        # A parent traversal can refer to something outside the experiment even
        # when its spelling starts with the old directory. Leave it untouched.
        if ".." in text.split("/"):
            return item
        for previous, replacement in replacements:
            tail = suffix(text, previous)
            if tail is not None:
                result = replacement + tail
                return result.replace("/", "\\") if "\\" in item and "/" not in item else result
        return item

    return visit(value)


def resumed_training_output(checkpoint_path: Path, checkpoint_config: dict) -> Path:
    """Reuse a checkpoint's current location, including after copying to HPC."""
    parent = checkpoint_path.parent
    previous = checkpoint_config.get("training", {}).get("output")
    nested = isinstance(previous, str) and Path(previous.replace("\\", "/")).name == "training"
    if checkpoint_path.name == "model.npz" and (nested or (parent / "training").is_dir()):
        return parent / "training"
    # Epoch snapshots and legacy flat model/latest snapshots live in the output.
    return parent
