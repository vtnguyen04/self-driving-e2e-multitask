# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
import sys
import time
from functools import lru_cache
from typing import IO, Any


@lru_cache(maxsize=1)
def is_noninteractive_console() -> bool:
    """Check for known non-interactive console environments."""
    return "GITHUB_ACTIONS" in os.environ or "RUNPOD_POD_ID" in os.environ


class TQDM:
    """Lightweight zero-dependency progress bar for NeuroPilot (adapted from Ultralytics).

    Provides clean, rich-style progress bars suitable for various environments.
    """

    # Constants
    MIN_RATE_CALC_INTERVAL = 0.01  # Minimum time interval for rate calculation
    RATE_SMOOTHING_FACTOR = 0.3  # Factor for exponential smoothing of rates
    MAX_SMOOTHED_RATE = 1000000  # Maximum rate to apply smoothing to
    NONINTERACTIVE_MIN_INTERVAL = 60.0  # Minimum interval for non-interactive environments

    def __init__(
        self,
        iterable: Any = None,
        desc: str | None = None,
        total: int | None = None,
        leave: bool = True,
        file: IO[str] | None = None,
        mininterval: float = 0.1,
        disable: bool | None = None,
        unit: str = "it",
        unit_scale: bool = True,
        unit_divisor: int = 1000,
        bar_format: str | None = None,  # kept for API compatibility; not used for formatting
        initial: int = 0,
        ncols: int | None = None,
        **kwargs,
    ) -> None:
        """Initialize the TQDM progress bar."""
        # Disable if not verbose
        if disable is None:
            disable = False

        self.iterable = iterable
        self.desc = desc or ""
        self.total = total or (len(iterable) if hasattr(iterable, "__len__") else None) or None
        self.disable = disable
        self.unit = unit
        self.unit_scale = unit_scale
        self.unit_divisor = unit_divisor
        self.leave = leave
        self.noninteractive = is_noninteractive_console()
        self.mininterval = max(mininterval, self.NONINTERACTIVE_MIN_INTERVAL) if self.noninteractive else mininterval
        self.initial = initial
        self.ncols = ncols

        self.bar_format = bar_format
        self.file = file or sys.stdout

        # Internal state
        self.n = self.initial
        self.last_print_n = self.initial
        self.last_print_t = time.time()
        self.start_t = time.time()
        self.last_rate = 0.0
        self.closed = False
        self.is_bytes = unit_scale and unit in {"B", "bytes"}
        self.scales = (
            [(1073741824, "GB/s"), (1048576, "MB/s"), (1024, "KB/s")]
            if self.is_bytes
            else [(1e9, f"G{self.unit}/s"), (1e6, f"M{self.unit}/s"), (1e3, f"K{self.unit}/s")]
        )

        if not self.disable and self.total and not self.noninteractive:
            self._display()

    def _format_rate(self, rate: float) -> str:
        """Format rate with units, switching between it/s and s/it for readability."""
        if rate <= 0:
            return ""

        inv_rate = 1 / rate if rate else None
        if inv_rate and inv_rate > 1:
            return f"{inv_rate:.1f}s/B" if self.is_bytes else f"{inv_rate:.1f}s/{self.unit}"

        fallback = f"{rate:.1f}B/s" if self.is_bytes else f"{rate:.1f}{self.unit}/s"
        return next((f"{rate / t:.1f}{u}" for t, u in self.scales if rate >= t), fallback)

    def _format_num(self, num: int | float) -> str:
        """Format number with optional unit scaling."""
        if not self.unit_scale or not self.is_bytes:
            return str(num)

        for unit in ("", "K", "M", "G", "T"):
            if abs(num) < self.unit_divisor:
                return f"{num:3.1f}{unit}B" if unit else f"{num:.0f}B"
            num /= self.unit_divisor
        return f"{num:.1f}PB"

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format time duration."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}:{seconds % 60:02.0f}"
        else:
            h, m = int(seconds // 3600), int((seconds % 3600) // 60)
            return f"{h}:{m:02d}:{seconds % 60:02.0f}"

    def _generate_bar(self, width: int = 15) -> str:
        """Generate progress bar."""
        if self.total is None:
            return "━" * width if self.closed else "─" * width

        frac = min(1.0, self.n / self.total)
        filled = int(frac * width)
        bar = "━" * filled + "─" * (width - filled)
        if filled < width and frac * width - filled > 0.5:
            bar = f"{bar[:filled]}╸{bar[filled + 1 :]}"
        return bar

    def _should_update(self, dt: float, dn: int) -> bool:
        """Check if display should update."""
        if self.noninteractive:
            return False
        return (self.total is not None and self.n >= self.total) or (dt >= self.mininterval)

    def _get_ncols(self):
        """Get terminal columns."""
        try:
            import fcntl
            import termios
            import struct
            return struct.unpack('hh', fcntl.ioctl(sys.stderr.fileno(), termios.TIOCGWINSZ, '1234'))[1]
        except:
             try:
                 return os.get_terminal_size().columns
             except:
                 return 80

    def _display(self, final: bool = False) -> None:
        """Display progress bar."""
        if self.disable or (self.closed and not final):
            return

        current_time = time.time()
        dt = current_time - self.last_print_t
        dn = self.n - self.last_print_n

        if not final and not self._should_update(dt, dn):
            return

        if dt > self.MIN_RATE_CALC_INTERVAL:
            rate = dn / dt if dt else 0.0
            if rate < self.MAX_SMOOTHED_RATE:
                self.last_rate = self.RATE_SMOOTHING_FACTOR * rate + (1 - self.RATE_SMOOTHING_FACTOR) * self.last_rate
                rate = self.last_rate
        else:
            rate = self.last_rate

        if self.total and self.n >= self.total:
            overall_elapsed = current_time - self.start_t
            if overall_elapsed > 0:
                rate = self.n / overall_elapsed

        self.last_print_n = self.n
        self.last_print_t = current_time
        elapsed = current_time - self.start_t

        remaining_str = ""
        if self.total and 0 < self.n < self.total and elapsed > 0:
            est_rate = rate or (self.n / elapsed)
            remaining_str = f"<{self._format_time((self.total - self.n) / est_rate)}"

        if self.total:
            percent = (self.n / self.total) * 100
            n_str = self._format_num(self.n)
            t_str = self._format_num(self.total)
            if self.is_bytes and n_str[-2] == t_str[-2]:
                n_str = n_str.rstrip("KMGTPB")
        else:
            percent = 0.0
            n_str, t_str = self._format_num(self.n), "?"

        elapsed_str = self._format_time(elapsed)
        rate_str = self._format_rate(rate) or (self._format_rate(self.n / elapsed) if elapsed > 0 else "")

        bar = self._generate_bar()

        if self.total:
            if self.is_bytes and self.n >= self.total:
                progress_str = f"{self.desc}: {percent:.0f}% {bar} {t_str} {rate_str} {elapsed_str}"
            else:
                progress_str = (
                    f"{self.desc}: {percent:.0f}% {bar} {n_str}/{t_str} {rate_str} {elapsed_str}{remaining_str}"
                )
        else:
            progress_str = f"{self.desc}: {bar} {n_str} {rate_str} {elapsed_str}"

        # FIX: Ensure it doesn't exceed terminal width to avoid wrapping/duplication
        ncols_term = self._get_ncols()
        ncols_to_use = min(self.ncols, ncols_term) if self.ncols else ncols_term
        if len(progress_str) > ncols_to_use - 1:
            progress_str = progress_str[:ncols_to_use-4] + "..."

        try:
            if self.noninteractive:
                self.file.write(progress_str + "\n")
            else:
                self.file.write(f"\r\033[K{progress_str}")
            self.file.flush()
        except Exception:
            pass

    def update(self, n: int = 1) -> None:
        """Update progress by n steps."""
        if not self.disable and not self.closed:
            self.n += n
            self._display()

    def set_description(self, desc: str | None) -> None:
        """Set description."""
        self.desc = desc or ""
        if not self.disable:
            self._display()

    def set_postfix(self, **kwargs: Any) -> None:
        """Set postfix (appends to description)."""
        if kwargs:
            # Shorten values for display
            def format_val(v):
                if isinstance(v, float): return f"{v:.4g}"
                return str(v)
            postfix = ", ".join(f"{k}={format_val(v)}" for k, v in kwargs.items())
            base_desc = self.desc.split(" | ")[0] if " | " in self.desc else self.desc
            self.set_description(f"{base_desc} | {postfix}")

    def close(self) -> None:
        """Close progress bar."""
        if self.closed:
            return

        self.closed = True

        if not self.disable:
            if self.total and self.n >= self.total:
                self.n = self.total
                if self.n != self.last_print_n:
                    self._display(final=True)
            else:
                self._display(final=True)

            if self.leave:
                self.file.write("\n")
            else:
                self.file.write("\r\033[K")

            try:
                self.file.flush()
            except Exception:
                pass

    def __enter__(self) -> TQDM:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __iter__(self) -> Any:
        if self.iterable is None:
            raise TypeError("'NoneType' object is not iterable")
        try:
            for item in self.iterable:
                yield item
                self.update(1)
        finally:
            self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
