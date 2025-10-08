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
    print(f'USING SQUASHFS TO COMPRESS TO GENERATED DATA INTO A READ ONLY FORMAT. SEE README.txt IN: {executor.base_run_dir}')
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
        "-noappend", "-comp", "xz", "-processors", str(num_threads)
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


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python folder_manamgment.py /path/to/folder")
    else:
        folder = sys.argv[1]
        # create_ext4_image_from_directory(folder)
        enchanted_data_squash(folder)


