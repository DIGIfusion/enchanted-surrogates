import os
import subprocess
import tempfile
import logging
from datetime import datetime
import shutil
from joblib import Parallel, delayed
import multiprocessing
import math

def enchanted_data_squash(source_dir, image_path=None, num_threads=None):
    print(f'USING SQUASHFS TO COMPRESS TO GENERATED DATA INTO A READ ONLY FORMAT. SEE README.txt IN: {source_dir}')
    assert os.path.isdir(source_dir), f"Source directory '{source_dir}' does not exist"

    # Default image path outside source_dir
    if not image_path:
        image_path = os.path.join(
            os.path.dirname(source_dir),
            os.path.basename(source_dir) + "_squashfs.img"
        )

    if not num_threads:
        num_threads = multiprocessing.cpu_count()

    print(f"Creating SquashFS image at '{image_path}' using {num_threads} threads...")

    # Step 1: Create SquashFS image
    subprocess.run([
        "mksquashfs", source_dir, image_path,
        "-noappend", "-comp", "xz", "-processors", str(num_threads), '-no-xattrs'
    ], check=True)

    print(f"SquashFS image creation complete: {image_path}")

    # Step 2: Remove original contents
    for entry in os.listdir(source_dir):
        entry_path = os.path.join(source_dir, entry)
        try:
            if os.path.isdir(entry_path):
                shutil.rmtree(entry_path)
            else:
                os.remove(entry_path)
        except Exception as e:
            print(f"Warning: Could not delete {entry_path}: {e}")

    # Step 3: Move image into source_dir
    local_image_path = os.path.join(source_dir, os.path.basename(image_path))
    shutil.move(image_path, local_image_path)

    # Step 4: Create mount point
    mount_point = os.path.join(source_dir, "enchanted_data_mount")
    os.makedirs(mount_point, exist_ok=True)

    # Step 5: Add README.txt with instructions
    readme_path = os.path.join(source_dir, "README.txt")
    with open(readme_path, "w") as f:
        f.write(
            f"Your original data has been archived into a compressed SquashFS image:\n"
            f"{os.path.basename(local_image_path)}\n\n"
            "To access your data:\n"
            "1. Ensure squashfs-tools are installed.\n"
            "2. Mount the image using:\n"
            f"   mount -t squashfs -o loop {os.path.basename(local_image_path)} enchanted_data_mount\n"
            "3. Your files will appear inside the 'enchanted_data_mount' directory.\n"
            "4. To unmount:\n"
            "   umount enchanted_data_mount\n\n"
            "Note: This is a read-only archive. To modify contents, extract the image using 'unsquashfs'.\n"
        )

    print(f"Image moved to '{local_image_path}'. Mount point created. README.txt saved.")

import os
import subprocess
import tempfile
import shutil
import math


import tempfile
import logging
from datetime import datetime
import shutil
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

    # Parallel size calculation
    total_size_bytes = sum(
        Parallel(n_jobs=num_cores)(
            delayed(get_file_size)(path) for path in file_paths
        )
    )

    # Convert to requested unit
    if unit == 'KB':
        total_size = total_size_bytes / 1024
    elif unit == 'MB':
        total_size = total_size_bytes / (1024 ** 2)
    elif unit == 'GB':
        total_size = total_size_bytes / (1024 ** 3)
    else:  # Default to bytes
        total_size = total_size_bytes

    logging.info(f"Total size of '{root_dir}': {total_size:.2f} {unit}")
    print(f"Total size of '{root_dir}': {total_size:.2f} {unit}")
    print(f"Log saved to: {log_file}")

    return total_size

def format_ext4_image_in_container(image_path, sif_path):
    """
    Formats the given image file as ext4 using a prebuilt SIF container that includes mkfs.ext4.

    Parameters:
    - image_path: Path to the blank image file to format.
    - sif_path: Path to the SIF container with mkfs.ext4 installed.
    """
    print(f"Formatting ext4 image using container: {sif_path}")
    try:
        subprocess.run([
            "singularity", "exec", sif_path,
            "mkfs.ext4", "-F", image_path
        ], check=True)
        print("Image formatted successfully.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to format ext4 image using container: {sif_path}") from e


def create_ext4_image_from_directory(source_dir, sif_path, image_size_gb=None):
    assert os.path.isdir(source_dir), f"Source directory '{source_dir}' does not exist"

    # Estimate image size if not provided
    if not image_size_gb:
        image_size_gb = measure_directory_size(source_dir, unit='GB') + 1  # Add buffer

    image_path = os.path.join(
        os.path.dirname(source_dir),
        os.path.basename(source_dir) + "_ext4.img"
    )
    image_size_bytes = int(image_size_gb * 1024 ** 3)

    print(f"Creating {image_size_gb:.2f} GB ext4 image at '{image_path}'...")

    # Step 1: Create blank image file
    subprocess.run([
        "dd", "if=/dev/zero", f"of={image_path}",
        "bs=1M", f"count={math.ceil(image_size_bytes / (1024 * 1024))}"
    ], check=True)

    # Step 2: Format as ext4 inside Singularity
    format_ext4_image_in_container(image_path, sif_path)

    print("ext4 image formatted successfully.")

    # Step 3: Mount image temporarily
    mount_point = tempfile.mkdtemp(prefix="ext4mnt_")
    subprocess.run(["mount", "-o", "loop", image_path, mount_point], check=True)

    try:
        # Step 4: Copy contents into image
        subprocess.run(["cp", "-a", f"{source_dir}/.", mount_point], check=True)
        print(f"Copied contents of '{source_dir}' into image.")
    finally:
        # Step 5: Unmount and clean up
        subprocess.run(["umount", mount_point], check=True)
        os.rmdir(mount_point)
        print("Unmounted image and cleaned up mount point.")

    # Step 6: Move image into source_dir
    local_image_path = os.path.join(source_dir, os.path.basename(image_path))
    shutil.move(image_path, local_image_path)

    # Step 7: Create mount point and README
    os.makedirs(os.path.join(source_dir, "enchanted_data_mount"), exist_ok=True)
    with open(os.path.join(source_dir, "README.txt"), "w") as f:
        f.write(
            f"Your original data has been archived into an ext4 image:\n"
            f"{os.path.basename(local_image_path)}\n\n"
            "To access your data:\n"
            "1. Ensure loop device support is available.\n"
            "2. Mount the image using:\n"
            f"   mount -o loop {os.path.basename(local_image_path)} enchanted_data_mount\n"
            "3. Your files will appear inside the 'enchanted_data_mount' directory.\n"
            "4. To unmount:\n"
            "   umount enchanted_data_mount\n\n"
            "Note: ext4 images are writable once mounted.\n"
        )

    print(f"Image moved to '{local_image_path}'. Mount point and README.txt created.")



if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python folder_manamgment.py /path/to/folder")
    else:
        folder = sys.argv[1]
        create_ext4_image_from_directory(folder)
        # enchanted_data_squash(folder)


