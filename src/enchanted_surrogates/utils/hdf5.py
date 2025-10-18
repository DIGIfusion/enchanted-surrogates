import os
import h5py
import shutil
import json
import numpy as np
import fnmatch

def convert_directory_to_hdf5(source_dir, hdf5_name="archive.h5", skip_delete=None, skip_patterns=None):
    """
    Pack source_dir into an HDF5 file and optionally delete original files/dirs.
    
    Parameters
    - source_dir (str)
    - hdf5_name (str): name of the HDF5 file created inside source_dir
    - skip_delete (iterable[str] | None): exact filenames or relative paths to preserve (examples: "keep.txt", "subdir/keep.dat")
    - skip_patterns (iterable[str] | None): glob-style patterns matched against relative paths (examples: "*.log", "cache/*")
    """
    print('PACKING DATA INTO hdf5 FILE')
    hdf5_path = os.path.join(source_dir, hdf5_name)
    skip_delete = set(skip_delete or [])
    skip_patterns = list(skip_patterns or [])

    with h5py.File(hdf5_path, "w") as h5f:
        for root, _, files in os.walk(source_dir):
            rel_root = os.path.relpath(root, source_dir)
            if rel_root == '.':
                rel_root = ''
            group = h5f.require_group(rel_root)
            for file in files:
                file_path = os.path.join(root, file)
                # Skip the HDF5 file itself
                if os.path.abspath(file_path) == os.path.abspath(hdf5_path):
                    continue

                with open(file_path, "rb") as f:
                    data = f.read()

                # dataset path inside HDF5 uses posix-like relative path
                dataset_path = os.path.join(rel_root, file) if rel_root else file
                if dataset_path in h5f:
                    print(f"⚠️ Skipping duplicate: {dataset_path}")
                    continue

                try:
                    decoded = data.decode("utf-8")
                    group.create_dataset(file, data=decoded)
                    group[file].attrs["type"] = "text"
                except (UnicodeDecodeError, ValueError):
                    try:
                        group.create_dataset(file, data=np.frombuffer(data, dtype='uint8'))
                        group[file].attrs["type"] = "binary"
                    except Exception:
                        print(f"⚠️ PATH FAILURE: {dataset_path}")

    # Helper to decide whether to remove a path
    def should_preserve(rel_path):
        # Exact-name preserve
        if rel_path in skip_delete:
            return True
        # Pattern preserve (glob-style)
        for pat in skip_patterns:
            if fnmatch.fnmatch(rel_path, pat):
                return True
        return False

    ## Remove original files and folders except HDF5 and any preserved entries
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        rel_item = item  # top-level relative path
        if os.path.abspath(item_path) == os.path.abspath(hdf5_path):
            continue  # never remove the archive itself
        if should_preserve(rel_item):
            print(f"⏭ Preserving top-level item: {rel_item}")
            continue
        if os.path.isdir(item_path):
            # For directories, check if any child matches skip rules; if so, preserve entire dir
            preserve_dir = False
            for root, dirs, files in os.walk(item_path):
                for name in dirs + files:
                    rel_child = os.path.relpath(os.path.join(root, name), source_dir)
                    rel_child = rel_child.replace(os.path.sep, '/')
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

    # Add notebook and README
    create_jupyter_notebook(source_dir, hdf5_name)
    create_readme(source_dir, hdf5_name)

    print(f"✅ Packed '{source_dir}' into '{hdf5_name}' with notebook and README.")

# create_jupyter_notebook and create_readme unchanged (copy from original)
def create_jupyter_notebook(target_dir, hdf5_name):
    notebook_path = os.path.join(target_dir, "explore_hdf5.ipynb")
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 📁 Explore Your HDF5 Archive\n",
                    f"This notebook lets you browse the contents of `{hdf5_name}` interactively."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 🌳 Interactive HDF5 Tree Explorer with Preview",
                    "import h5py",
                    "import ipywidgets as widgets",
                    "from IPython.display import display",
                    "from functools import partial",
                    "",
                    f"hdf5_path = '{hdf5_name}'",
                    "h5f = h5py.File(hdf5_path, 'r')",
                    "",
                    "# Create a shared output box",
                    "preview = widgets.Output()",
                    "",
                    "def display_dataset(path):",
                    "    with preview:",
                    "        preview.clear_output()",
                    "        obj = h5f[path]",
                    "        print(f\"📄 Dataset: {path}\")",
                    "        dtype = obj.attrs.get(\"type\", \"unknown\")",
                    "        print(f\"Type: {dtype}, Shape: {obj.shape}\")",
                    "        if dtype == \"binary\":",
                    "            print(f\"Binary file — {len(obj[()])} bytes\")",
                    "        else:",
                    "            try:",
                    "                print(obj[()].decode(\"utf-8\"))",
                    "            except Exception as e:",
                    "                print(f\"Error decoding text: {e}\")",
                    "",
                    "def build_tree(group, path=\"\"):",
                    "    items = []",
                    "    for key in group:",
                    "        obj = group[key]",
                    "        full_path = f\"{path}/{key}\" if path else key",
                    "        if isinstance(obj, h5py.Group):",
                    "            sub_items = build_tree(obj, full_path)",
                    "            accordion = widgets.Accordion(children=[widgets.VBox(sub_items)])",
                    "            accordion.set_title(0, f\"📁 {key}/\")",
                    "            items.append(accordion)",
                    "        elif isinstance(obj, h5py.Dataset):",
                    "            dtype = obj.attrs.get(\"type\", \"unknown\")",
                    "            btn = widgets.Button(description=f\"📄 {key} — {dtype}\", layout=widgets.Layout(width='100%'))",
                    "            btn.on_click(partial(lambda b, p: display_dataset(p), p=full_path))",
                    "            items.append(btn)",
                    "    return items",
                    "",
                    "tree = widgets.VBox(build_tree(h5f))",
                    "display(tree, preview)"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.8"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open(notebook_path, "w") as f:
        json.dump(notebook_content, f)

def create_readme(target_dir, hdf5_name):
    readme_path = os.path.join(target_dir, "README.txt")
    with open(readme_path, "w") as f:
        f.write(
            f"""# HDF5 Archive Overview

Your original directory has been packed into a single HDF5 file: `{hdf5_name}`.

## What is HDF5?

HDF5 is a hierarchical data format that stores datasets and groups in a tree-like structure. It’s ideal for archiving large numbers of files in one compact, searchable file.

## How to Explore

1. Open the included Jupyter Notebook: `explore_hdf5.ipynb`
2. Run the cells to browse groups and datasets.
3. Preview any file by specifying its path inside the archive.

## Web Viewer

You can also explore your HDF5 file using the browser-based tool [myHDF5](https://myhdf5.hdfgroup.org/):
- Drag and drop `{hdf5_name}` into the viewer.
- Browse the structure and preview contents.

Enjoy your compact, HPC-friendly archive!
"""
        )

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description="Pack a directory into a single HDF5 archive.")
    parser.add_argument("folder", help="Path to folder to pack")
    parser.add_argument("--name", "-n", default="archive.h5", help="HDF5 filename to create in the folder")
    parser.add_argument("--preserve", "-p", nargs="*", default=[], help="Exact relative paths to preserve (top-level or subpaths)")
    parser.add_argument("--preserve-pattern", "-P", nargs="*", default=[], help="Glob patterns (relative paths) to preserve")
    args = parser.parse_args()

    convert_directory_to_hdf5(args.folder, hdf5_name=args.name, skip_delete=args.preserve, skip_patterns=args.preserve_pattern)
