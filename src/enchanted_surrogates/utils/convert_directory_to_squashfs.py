import os
import shutil
import json
import fnmatch
import subprocess
import numpy as np  # kept for symmetry; not actually used here


def convert_directory_to_squashfs(source_dir, squashfs_name="archive.sqsh",
                                  skip_delete=None, skip_patterns=None):
    """
    Pack source_dir into a SquashFS file and optionally delete original files/dirs.

    Parameters
    - source_dir (str)
    - squashfs_name (str): name of the SquashFS file created inside source_dir
    - skip_delete (iterable[str] | None): exact filenames or relative paths to preserve
    - skip_patterns (iterable[str] | None): glob-style patterns matched against relative paths
    """
    print("PACKING DATA INTO SquashFS FILE")

    # Ensure mksquashfs is available
    if shutil.which("mksquashfs") is None:
        raise RuntimeError("mksquashfs not found on PATH. Please install squashfs-tools.")

    squashfs_path = os.path.join(source_dir, squashfs_name)
    skip_delete = set(skip_delete or [])
    skip_patterns = list(skip_patterns or [])

    # 1. Add notebook and README *before* creating the SquashFS archive
    create_jupyter_notebook(source_dir, squashfs_name)
    create_readme(source_dir, squashfs_name)

    # 2. Build the SquashFS image from the entire directory
    #    We run mksquashfs source_dir squashfs_path -noappend
    #    so that the archive is a snapshot of the directory at this moment.
    print(f"Running mksquashfs on '{source_dir}' -> '{squashfs_path}'")
    result = subprocess.run(
        ["mksquashfs", source_dir, squashfs_path, "-noappend"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        print("mksquashfs failed:")
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError("mksquashfs failed; see output above.")

    print("✅ SquashFS archive created successfully.")

    def should_preserve(rel_path):
        rel_path_posix = rel_path.replace(os.path.sep, "/").lstrip("/")
        if rel_path_posix in skip_delete:
            return True
        for pat in skip_patterns:
            if fnmatch.fnmatch(rel_path_posix, pat):
                return True
        return False

    # 3. Delete everything in source_dir except:
    #    - the SquashFS archive itself
    #    - anything matching skip_delete / skip_patterns
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        rel_item = item  # top-level relative path

        # Never remove the archive itself
        if os.path.abspath(item_path) == os.path.abspath(squashfs_path):
            continue

        if should_preserve(rel_item):
            print(f"⏭ Preserving top-level item: {rel_item}")
            continue

        if os.path.isdir(item_path):
            preserve_dir = False
            for root, dirs, files in os.walk(item_path):
                for name in dirs + files:
                    rel_child = os.path.relpath(os.path.join(root, name), source_dir)
                    rel_child = rel_child.replace(os.path.sep, "/")
                    if should_preserve(rel_child):
                        preserve_dir = True
                        break
                if preserve_dir:
                    break
            if preserve_dir:
                print(f"⏭ Preserving directory because it contains preserved item: {rel_item}")
                continue
            shutil.rmtree(item_path)
        else:
            os.remove(item_path)

    print(f"✅ Packed '{source_dir}' into '{squashfs_name}' and cleaned up originals.")


def create_jupyter_notebook(target_dir, squashfs_name):
    notebook_path = os.path.join(target_dir, "explore_hdf5.ipynb")
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 📁 Explore Your SquashFS Archive\n",
                    f"This notebook documents the archive `{squashfs_name}`.\n",
                    "\n",
                    "Note: SquashFS is a compressed, read-only filesystem image. "
                    "You typically mount it with `mount` or `squashfuse` to browse its contents.\n"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# This cell is a placeholder. In most workflows, you will mount the SquashFS\n",
                    "# image externally (e.g., on a cluster node) and then browse it via the OS.\n",
                    f"squashfs_path = '{squashfs_name}'",
                    "print('SquashFS archive:', squashfs_path)",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.8",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 2,
    }

    with open(notebook_path, "w") as f:
        json.dump(notebook_content, f)


def create_readme(target_dir, squashfs_name):
    readme_path = os.path.join(target_dir, "README.txt")
    with open(readme_path, "w") as f:
        f.write(
            f"""# SquashFS Archive Overview

Your original directory has been packed into a single SquashFS file: `{squashfs_name}`.

## What is SquashFS?

SquashFS is a compressed, read-only filesystem image format. It is commonly used in
HPC and container-like environments to distribute large, immutable datasets or software
trees in a compact form.

## How to Explore

1. Copy `{squashfs_name}` to a system with SquashFS support.
2. Create a mount point, for example:
   - `mkdir /mnt/mydata`
3. Mount the image (requires root or squashfuse):
   - With root: `mount -t squashfs -o loop {squashfs_name} /mnt/mydata`
   - With squashfuse: `squashfuse {squashfs_name} /mnt/mydata`
4. Browse `/mnt/mydata` as a normal directory.

When finished, unmount:
- `umount /mnt/mydata`  (or `fusermount -u /mnt/mydata` for FUSE)

Enjoy your compact, HPC-friendly SquashFS archive!
"""
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pack a directory into a SquashFS archive.")
    parser.add_argument("folder", help="Path to folder to pack")
    parser.add_argument(
        "--name",
        "-n",
        default="archive.sqsh",
        help="SquashFS filename to create in the folder",
    )
    parser.add_argument(
        "--preserve",
        "-p",
        nargs="*",
        default=[],
        help="Exact relative paths to preserve (top-level or subpaths)",
    )
    parser.add_argument(
        "--preserve-pattern",
        "-P",
        nargs="*",
        default=[],
        help="Glob patterns (relative paths) to preserve",
    )
    args = parser.parse_args()

    convert_directory_to_squashfs(
        args.folder,
        squashfs_name=args.name,
        skip_delete=args.preserve,
        skip_patterns=args.preserve_pattern,
    )
