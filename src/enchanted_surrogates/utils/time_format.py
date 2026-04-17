import time

def time_format(seconds: int) -> str:
    """
    Convert a duration in seconds into 'Xd HH:MM:SS' format.
    Days are included only when nonzero.
    """
    days = seconds // 86400
    t = time.gmtime(seconds)

    if days > 0:
        return f"{days}d {t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}"
    else:
        return f"{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}"
