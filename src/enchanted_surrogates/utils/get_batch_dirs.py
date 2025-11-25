import re
import os
def get_batch_dirs(base_run_dir):
    """
    Scans `base_dir` for subdirectories whose names contain an integer,
    and returns a list of those subdirectory names sorted by the integer.

    Parameters:
    - base_dir (str): Path to the directory to scan.

    Returns:
    - List[str]: Sorted list of subdirectory names.
    """
    subdirs = []
    for name in os.listdir(base_run_dir):
        full_path = os.path.join(base_run_dir, name)
        if os.path.isdir(full_path):
            match = re.search(r'\d+', name)
            if match:
                subdirs.append((int(match.group()), name))

    # Sort by extracted integer
    sorted_subdirs = [name for _, name in sorted(subdirs)]
    sorted_subdirs = [os.path.join(base_run_dir, d) for d in sorted_subdirs]
    return sorted_subdirs
