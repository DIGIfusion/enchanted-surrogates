import warnings

def ascii_loading_bar(total, progress, bar_size=40):
    """
    Generate an ASCII loading bar representing progress toward a total amount of work.

    Args:
        total (int or float): The total amount of work. Must be positive unless both
            `total` and `progress` are zero, in which case the function treats the
            total as 1 to avoid division by zero.
        progress (int or float): The amount of completed work. Must be less than or
            equal to `total`.
        bar_size (int, optional): The length of the bar in characters. Defaults to 40.

    Returns:
        str: A formatted ASCII progress bar, or an error message if inputs are invalid.
            The bar uses '#' for completed segments and '-' for remaining segments,
            followed by a percentage indicator.

    Raises:
        None: The function does not raise exceptions; instead, it returns descriptive
            error messages for invalid inputs.

    Examples:
        >>> ascii_loading_bar(100, 25)
        '[##########------------------------------]  25.0%'

        >>> ascii_loading_bar(0, 0)
        '[----------------------------------------]   0.0%'

        >>> ascii_loading_bar(10, 15)
        'loading bar failed: Progress must be less than or equal to total | Progress: 15 | Total: 10'
    """
    if total == 0 and progress == 0:
        total = 1
        progress = 0
    
    if total <= 0:
        return f"loading bar failed: Total should be positive | Progress: {progress} | Total: {total}" 
    if progress > total:
        return f"loading bar failed: Progress must be less than or equal to total | Progress: {progress} | Total: {total}"
    ratio = max(0.0, min(1.0, progress / total))
    filled = int(ratio * bar_size)
    empty = bar_size - filled

    bar = "[" + "#" * filled + "-" * empty + f"] {ratio*100:5.1f}%"
    return bar
