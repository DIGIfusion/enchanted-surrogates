def ascii_loading_bar(total, progress, bar_size=40):
    """
    Return a simple ASCII loading bar.
    total: total amount of work
    progress: completed amount
    bar_size: width of the bar in characters
    """
    if total <= 0:
        raise ValueError("total must be positive")

    ratio = max(0.0, min(1.0, progress / total))
    filled = int(ratio * bar_size)
    empty = bar_size - filled

    bar = "[" + "#" * filled + "-" * empty + f"] {ratio*100:5.1f}%"
    return bar
