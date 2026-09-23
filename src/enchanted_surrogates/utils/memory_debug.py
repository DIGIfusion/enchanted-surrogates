"""
Lightweight RSS logging for diagnosing driver-process memory growth (see the
job 22222070 OOM investigation: a shared-allocation multi-method active-
learning run was killed after MaxRSS grew to ~33.5GB against a 32GB request,
and no single cause was found for the actual memory growth -- see
supervisor.py's and the *_time_aware_active_sampler.py's log_rss calls for
where this is wired in). Meant to stay in place permanently at DEBUG level so
a future recurrence can be diagnosed from logs alone, without needing to
reproduce it live first.

Usage: log_rss(log, "some_label") emits one DEBUG line with the *current
process's* RSS in MB, prefixed with the label, so grepping a run's log for
"RSS" gives a timeline of memory at each labeled checkpoint. Cheap enough
(a single /proc read) to call at every batch/checkpoint without meaningfully
affecting runtime, even at high call frequency.
"""
try:
    import psutil
    _PROCESS = psutil.Process()
except ImportError:
    _PROCESS = None


def log_rss(log, label: str) -> None:
    """
    Logs the current process's RSS (MB) at DEBUG level, tagged with label.
    No-ops silently if psutil isn't installed, so this is safe to call
    unconditionally without adding a hard dependency.
    """
    if _PROCESS is None:
        return
    rss_mb = _PROCESS.memory_info().rss / 1e6
    log.debug("RSS[%s] = %.1f MB", label, rss_mb)
