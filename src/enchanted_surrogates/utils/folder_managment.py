import os
import subprocess
import tempfile
import logging
from datetime import datetime
from joblib import Parallel, delayed
import multiprocessing
import math
def setup_logger(base_run_dir):
    log_dir = os.path.join(base_run_dir, 'file_managment_loggs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"dir_size_{timestamp}.log")

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return log_path

def get_file_size(path):
    try:
        return os.path.getsize(path)
    except Exception:
        return 0

def collect_file_paths(root_dir):
    file_paths = []
    for dirpath, _, _ in os.walk(root_dir):
        try:
            with os.scandir(dirpath) as it:
                for entry in it:
                    if entry.is_file(follow_symlinks=False):
                        file_paths.append(entry.path)
        except Exception:
            continue
    return file_paths

def measure_directory_size(root_dir, unit='GB'):
    log_file = setup_logger(root_dir)

    file_paths = collect_file_paths(root_dir)
    num_cores = multiprocessing.cpu_count()

    logging.info(f"Scanning {len(file_paths):,} files using {num_cores} cores...")

    total_size_bytes = sum(
        Parallel(n_jobs=num_cores)(
            delayed(get_file_size)(path) for path in file_paths
        )
    )

    if unit == 'MB':
        total_size = total_size_bytes / (1024 ** 2)
    elif unit == 'GB':
        total_size = total_size_bytes / (1024 ** 3)
    else:
        total_size = total_size_bytes

    logging.info(f"Total size of '{root_dir}': {total_size:.2f} {unit}")
    print(f"Total size of '{root_dir}': {total_size:.2f} {unit}")
    print(f"Log saved to: {log_file}")
    return total_size


def create_ext4_image_from_directory(source_dir, image_size_gb=None):
    assert os.path.isdir(source_dir), f"Source directory '{source_dir}' does not exist"
    if not image_size_gb:
        image_size_gb = measure_directory_size(source_dir, unit='GB')
    
    image_path = os.path.join(os.path.dirname(source_dir), os.path.basename(source_dir)+'_ext4_mountable')
        
    image_size_bytes = image_size_gb * 1024 ** 3

    print(f"Creating {image_size_gb} GB ext4 image at '{image_path}'...")

    # Step 1: Create empty file
    subprocess.run([
        "dd", "if=/dev/zero", f"of={image_path}",
        "bs=1M", f"count={math.ceil(image_size_bytes // (1024 * 1024))}"
    ], check=True)

    # Step 2: Format as ext4
    subprocess.run(["mkfs.ext4", "-F", image_path], check=True)

    # Step 3: Mount the image (assuming user has permission)
    mount_point = tempfile.mkdtemp(prefix="ext4mnt_")
    subprocess.run(["mount", "-o", "loop", image_path, mount_point], check=True)

    try:
        # Step 4: Copy contents
        subprocess.run(["cp", "-a", f"{source_dir}/.", mount_point], check=True)
        print(f"Copied contents of '{source_dir}' into image.")
    finally:
        # Step 5: Unmount
        subprocess.run(["umount", mount_point], check=True)
        os.rmdir(mount_point)
        print(f"Unmounted image and cleaned up mount point.")

    print(f"Image creation complete: {image_path}")
    
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python folder_manamgment.py /path/to/folder")
    else:
        folder = sys.argv[1]
        create_ext4_image_from_directory(folder)


