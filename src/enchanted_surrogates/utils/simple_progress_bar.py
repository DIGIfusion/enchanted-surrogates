# Paste this class into your module (replaces the previous SimpleProgressBar).
import sys
import shutil
import time
import ctypes
import os
from datetime import datetime, timedelta

class SimpleProgressBar:
    """
    Dependency-free terminal progress bar with:
    - optional header/description
    - percentage and optional fraction
    - clock-based ETA and expected total duration
    - keeps final status block visible by default
    - auto-finalize when reaching total if auto_finish=True
    """

    def __init__(
        self,
        total=None,
        width=None,
        file=None,
        enable_ansi=True,
        header=None,
        description=None,
        show_percent=True,
        show_fraction=True,
        show_time=True,
        auto_finish=True,
    ):
        self.total = total
        self.current = 0
        self.file = file or sys.stderr
        self.width = width
        self._last_lines = 0
        self.header = header
        self.description = description
        self.show_percent = show_percent
        self.show_fraction = show_fraction
        self.show_time = show_time
        self.auto_finish = auto_finish

        self._start = None
        self._last_update_time = None
        self._finished = False

        self._ansi = enable_ansi and os.name != "nt"
        if enable_ansi and os.name == "nt":
            self._ansi = self._enable_windows_ansi()

    def _enable_windows_ansi(self):
        try:
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_uint()
            if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                return False
            ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
            new_mode = mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
            return bool(kernel32.SetConsoleMode(handle, new_mode))
        except Exception:
            return False

    def _term_width(self):
        if self.width:
            return self.width
        try:
            return shutil.get_terminal_size(fallback=(80, 20)).columns
        except Exception:
            return 80

    def _clear_prev(self):
        """
        Robustly clear the previously printed block using single CSI moves.
        Works only when ANSI sequences are supported. If not, falls back to
        safe rewriting with spaces.
        """
        if self._last_lines <= 0:
            return
        term_w = self._term_width()

        if self._ansi:
            # Move cursor up exactly _last_lines lines (if currently at start of last line),
            # then clear each line and move down; finally return cursor to first line.
            # Use CSI {n}A to move up n, CSI 2K to erase entire line.
            # If self._last_lines is N, move up N-1 to reach first line (we're at start of last line).
            ups = max(0, self._last_lines - 1)
            if ups:
                self.file.write(f"\033[{ups}A")
            for i in range(self._last_lines):
                # erase line and move cursor down (except after last line)
                self.file.write("\033[2K")
                if i < self._last_lines - 1:
                    self.file.write("\n")
            # move cursor back up to the first line, ready for overwriting
            if self._last_lines > 1:
                self.file.write(f"\033[{self._last_lines - 1}A")
        else:
            # Non-ANSI fallback: overwrite lines top-to-bottom with spaces.
            for _ in range(self._last_lines - 1):
                self.file.write("\r" + " " * term_w + "\n")
            self.file.write("\r" + " " * term_w + "\r")
        self.file.flush()

    def _format_ddhhmmss(self, dt):
        if isinstance(dt, timedelta):
            total_seconds = int(round(dt.total_seconds()))
            if total_seconds < 0:
                total_seconds = 0
            days, rem = divmod(total_seconds, 86400)
            hours, rem = divmod(rem, 3600)
            minutes, seconds = divmod(rem, 60)
            return f"{days:02d}-{hours:02d}:{minutes:02d}:{seconds:02d}"
        elif isinstance(dt, datetime):
            return dt.strftime("%d-%H:%M:%S")
        else:
            return "??-??:??:??"

    def _render_bar(self, pct, available_width):
        bar_space = max(10, available_width - 2)
        filled = int(bar_space * pct)
        if filled >= bar_space:
            return "[" + "=" * bar_space + "]"
        return "[" + "=" * filled + ">" + " " * max(0, bar_space - filled - 1) + "]"

    def _build_time_info(self):
        started_dt = datetime.now() if self._start is None else datetime.fromtimestamp(self._start)
        last_update_dt = None if self._last_update_time is None else datetime.fromtimestamp(self._last_update_time)

        if self._start is None or self.current <= 0:
            elapsed_td = timedelta(0)
        else:
            elapsed_td = timedelta(seconds=(time.time() - self._start))

        eta_dt = None
        expected_total_td = None
        if self.show_time and self.total and self.current > 0:
            mean_per_update = elapsed_td.total_seconds() / float(self.current)
            remaining = max(0, self.total - self.current)
            remaining_seconds = mean_per_update * remaining
            eta_dt = datetime.now() + timedelta(seconds=remaining_seconds)
            expected_total_td = elapsed_td + timedelta(seconds=remaining_seconds)

        return elapsed_td, eta_dt, expected_total_td, started_dt, last_update_dt

    def _wrap_or_truncate(self, text, width):
        if text is None:
            return None
        text = str(text)
        if len(text) <= width:
            return text
        if width > 3:
            return text[: max(0, width - 3)] + "..."
        return text[:width]

    def _build_lines(self):
        term_w = self._term_width()

        elapsed_td, eta_dt, expected_total_td, started_dt, last_update_dt = self._build_time_info()
        started_str = self._format_ddhhmmss(started_dt) if started_dt else "??-??:??:??"
        last_str = self._format_ddhhmmss(last_update_dt) if last_update_dt else "??-??:??:??"
        eta_str = self._format_ddhhmmss(eta_dt) if eta_dt else "??-??:??:??"
        expected_total_str = self._format_ddhhmmss(expected_total_td) if expected_total_td else "??-??:??:??"

        status = (
            f"Started {started_str}, Last update {last_str}, "
            f"ETA {eta_str}, Expected total duration {expected_total_str}"
        )
        status = self._wrap_or_truncate(status, term_w)

        if self.total:
            pct = max(0.0, min(1.0, self.current / float(self.total)))
        else:
            pct = 0.0

        pct_text = ""
        if self.total and self.show_percent:
            pct_text = f"{int(pct * 100):3d}% "

        fraction_text = ""
        if self.total and self.show_fraction:
            fraction_text = f" {self.current}/{self.total}"

        elapsed_compact = self._format_ddhhmmss(elapsed_td)
        time_suffix = f"  Elapsed {elapsed_compact}" if self.show_time else ""

        reserved = len(pct_text) + len(fraction_text) + len(time_suffix)
        avail_for_bar = max(10, term_w - reserved - 1)
        bar = self._render_bar(pct, avail_for_bar)
        bar_line = (pct_text + bar + fraction_text + time_suffix).rstrip()
        bar_line = self._wrap_or_truncate(bar_line, term_w)

        lines = []
        if self.header is not None:
            lines.append(self._wrap_or_truncate(self.header, term_w))
        lines.append(bar_line)
        if self.description is not None:
            lines.append(self._wrap_or_truncate(self.description, term_w))
        lines.append(status)
        return lines

    def _finalize_after_complete(self):
        """Non-recursive finalization called when reaching total in update()."""
        if self._finished:
            return
        # Move cursor below the printed block so new output appears after it.
        if self._ansi:
            self.file.write("\n")
        else:
            # Non-ANSI: we likely already printed with newlines; ensure trailing newline
            self.file.write("\n")
        self.file.flush()
        self._finished = True

    def update(self, n=1, header=None, description=None):
        """
        Increment progress by n and redraw. If auto_finish True and total reached,
        finalizes automatically (non-recursive).
        """
        if self._finished:
            return  # no-op after finalization

        now = time.time()
        if self._start is None:
            self._start = now
        self._last_update_time = now

        self.current += n
        if header is not None:
            self.header = header
        if description is not None:
            self.description = description

        # clamp current if there's a total
        if self.total is not None and self.current > self.total:
            self.current = self.total

        new_lines = self._build_lines()
        self._clear_prev()

        term_w = self._term_width()
        # Print all lines (each ends with newline). After printing, move cursor up one line
        # so the next update overwrites the status line in-place.
        for line in new_lines:
            padded = line + " " * max(0, term_w - len(line))
            self.file.write(padded + "\n")
        if self._ansi:
            # Move up one line (to the start of the last printed line) and carriage return.
            self.file.write("\033[1A\r")
        else:
            # Non-ANSI: nothing to do; we've already printed the lines and cursor is below block.
            pass
        self.file.flush()
        self._last_lines = len(new_lines)

        # If we reached the total and auto_finish is enabled, finalize now (non-recursive).
        if self.auto_finish and self.total is not None and self.current >= self.total:
            self._finalize_after_complete()

    def finish(self, newline=True, keep_status=True):
        """
        Explicit finish. Safe to call even if auto-finished earlier.
        - newline: ensure newline after block (default True)
        - keep_status: if True (default), leave final block visible; if False, clear it
        """
        if self._finished:
            # already finalized by auto-finish; ensure a newline below if requested
            if newline:
                self.file.write("\n")
                self.file.flush()
            return

        if self.total:
            if self._start is None:
                self._start = time.time()
            self.current = self.total
            # render final block without triggering recursive auto-finalize
            # temporarily suppress auto_finish during this update
            saved_auto = self.auto_finish
            self.auto_finish = False
            try:
                self.update(0)
            finally:
                self.auto_finish = saved_auto

        if keep_status:
            # move cursor below the block
            if self._ansi:
                self.file.write("\n")
            self.file.flush()
            self._finished = True
        else:
            self._clear_prev()
            if newline:
                if self._ansi:
                    self.file.write("\r\033[K\n")
                else:
                    self.file.write("\n")
                self.file.flush()
            self._last_lines = 0
            self._finished = False


if __name__ == "__main__":
    # quick demo
    p = SimpleProgressBar(
        total=20,
        header="Downloading package",
        description="Starting...",
        show_percent=True,
        show_fraction=True,
        show_time=True,
    )
    for i in range(20):
        time.sleep(0.05 + (i % 3) * 1)
        # update description and advance
        p.update(1, description=f"Processed {i+1}/20")
    # p.finish()
    print('REST OF CODE, OTHER OUTPUT TO STDERR')
