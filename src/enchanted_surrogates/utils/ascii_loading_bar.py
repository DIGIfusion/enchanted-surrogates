import warnings

def ascii_loading_bar(total, progress, bar_size=40):
    """
    Return a simple ASCII loading bar.
    total: total amount of work
    progress: completed amount
    bar_size: width of the bar in characters
    """
    if total == 0 and progress == 0:
        total = 1
        progress = 0
    
    if total <= 0:
        warnings.warn(f"Total should be positive | Progress: {progress} | Total: {total}")
        return f"loading bar failed: Total should be positive | Progress: {progress} | Total: {total}" 
    if progress > total:
        warnings.warn(f"Progress must be less than or equal to total | Progress: {progress} | Total: {total}")
        return f"loading bar failed: Progress must be less than or equal to total | Progress: {progress} | Total: {total}"
    ratio = max(0.0, min(1.0, progress / total))
    filled = int(ratio * bar_size)
    empty = bar_size - filled

    bar = "[" + "#" * filled + "-" * empty + f"] {ratio*100:5.1f}%"
    return bar
