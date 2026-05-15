import time

def time_format(seconds: int) -> str:
    """
    Convert a duration in seconds into a human‑readable time string.

    The output is formatted as ``"Xd - HH:MM:SS"`` when the duration includes
    one or more full days. If the number of days is zero, the output is
    formatted as ``"HH:MM:SS"`` instead.

    Args:
        seconds (int): The total duration in seconds. Must be non‑negative.

    Returns:
        str: A formatted time string representing the duration. Days are
            included only when nonzero.

    Examples:
        >>> time_format(3661)
        '01:01:01'

        >>> time_format(90061)
        '1d - 01:01:01'
    """

    days = seconds // 86400
    t = time.gmtime(seconds)

    if days > 0:
        return f"{days}d - {t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}"
    else:
        return f"{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}"
