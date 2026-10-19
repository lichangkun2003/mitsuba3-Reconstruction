"""Allocate daily experiment directories without importing a rendering backend."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
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
    pattern = re.compile(r".+-" + re.escape(stamp) + r"-E([0-9]+)")
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


def resumed_training_output(checkpoint_path: Path, checkpoint_config: dict) -> Path:
    """Reuse a checkpoint's current location, including after copying to HPC."""
    parent = checkpoint_path.parent
    previous = checkpoint_config.get("training", {}).get("output")
    nested = isinstance(previous, str) and Path(previous.replace("\\", "/")).name == "training"
    if checkpoint_path.name == "model.npz" and (nested or (parent / "training").is_dir()):
        return parent / "training"
    # Epoch snapshots and legacy flat model/latest snapshots live in the output.
    return parent
