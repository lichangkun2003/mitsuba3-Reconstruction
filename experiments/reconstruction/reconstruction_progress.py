"""Small terminal progress display without external dependencies."""

from __future__ import annotations

import os
import shutil
import sys
import time


class ProgressLine:
    """Refresh one terminal line; write only explicit summaries in redirected logs."""

    def __init__(self, stream=None, interval: float = 0.15):
        self.stream = sys.stdout if stream is None else stream
        self.interval = interval
        self._terminal = bool(getattr(self.stream, "isatty", lambda: False)())
        self._active = False
        self._width = 0
        self._last_update = float("-inf")

    def _columns(self):
        try:
            columns = os.get_terminal_size(self.stream.fileno()).columns
        except (AttributeError, OSError, ValueError):
            columns = shutil.get_terminal_size(fallback=(80, 24)).columns
        # Leave one unused column to avoid terminal autowrap at the right edge.
        return max(0, columns - 1)

    def _write_line(self, message):
        columns = self._columns()
        message = str(message).replace("\r", " ").replace("\n", " ")
        if len(message) > columns:
            message = (message[:columns - 3] + "...") if columns >= 3 else message[:columns]
        width = min(columns, max(self._width, len(message)))
        self.stream.write("\r" + message + " " * (width - len(message)))
        self._width = width
        self._active = True

    def update(self, message, force=False):
        """Display progress in terminals, throttled unless force=True."""
        if not self._terminal:
            return
        now = time.monotonic()
        if not force and now - self._last_update < self.interval:
            return
        self._write_line(message)
        self.stream.flush()
        self._last_update = now

    def finish(self, message=None):
        """End the active line, optionally replacing it with a final summary."""
        if message is not None:
            if self._terminal:
                self._write_line(message)
            else:
                self.stream.write(str(message))
            self.stream.write("\n")
            self.stream.flush()
        elif self._active:
            self.stream.write("\n")
            self.stream.flush()
        self._active = False
        self._width = 0
        self._last_update = float("-inf")
