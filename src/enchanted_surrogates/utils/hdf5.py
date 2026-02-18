import os
import h5py
import shutil
import json
import numpy as np
import fnmatch
import tempfile
        

def convert_directory_to_hdf5(source_dir, hdf5_name="archive.h5", skip_delete=None, skip_patterns=None):
    """
    Pack source_dir into an HDF5 file and optionally delete original files/dirs.

    Parameters
    - source_dir (str)
    - hdf5_name (str): name of the HDF5 file created inside source_dir
    - skip_delete (iterable[str] | None): exact filenames or relative paths to preserve
    - skip_patterns (iterable[str] | None): glob-style patterns matched against relative paths
    """
    print('PACKING DATA INTO hdf5 FILE')
    hdf5_path = os.path.join(source_dir, hdf5_name)
    skip_delete = set(skip_delete or [])
    skip_patterns = list(skip_patterns or [])

    with h5py.File(hdf5_path, "w") as h5f:
        for root, _, files in os.walk(source_dir):
            rel_root = os.path.relpath(root, source_dir)
            if rel_root == '.':
                rel_root = ''  # top-level
            # Normalize rel_root to posix style for consistent dataset paths and pattern matching
            rel_root_posix = rel_root.replace(os.path.sep, '/').lstrip('/')

            # Use the HDF5 root group for top-level, otherwise require/create a subgroup
            if rel_root_posix == '':
                group = h5f  # root group
            else:
                group = h5f.require_group(rel_root_posix)

            for file in files:
                file_path = os.path.join(root, file)
                # Skip the HDF5 file itself
                if os.path.abspath(file_path) == os.path.abspath(hdf5_path):
                    continue

                with open(file_path, "rb") as f:
                    data = f.read()

                # dataset path inside HDF5 uses posix-like relative path
                dataset_path = f"{rel_root_posix}/{file}" if rel_root_posix else file
                dataset_path = dataset_path.lstrip('/')

                if dataset_path in h5f:
                    print(f"⚠️ Skipping duplicate: {dataset_path}")
                    continue

                try:
                    decoded = data.decode("utf-8")
                    # create dataset under the current group using the plain filename
                    group.create_dataset(file, data=decoded)
                    group[file].attrs["type"] = "text"
                except (UnicodeDecodeError, ValueError):
                    try:
                        group.create_dataset(file, data=np.frombuffer(data, dtype='uint8'))
                        group[file].attrs["type"] = "binary"
                    except Exception:
                        print(f"⚠️ PATH FAILURE: {dataset_path}")

    def should_preserve(rel_path):
        rel_path_posix = rel_path.replace(os.path.sep, '/').lstrip('/')
        if rel_path_posix in skip_delete:
            return True
        for pat in skip_patterns:
            if fnmatch.fnmatch(rel_path_posix, pat):
                return True
        return False

    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        rel_item = item  # top-level relative path
        if os.path.abspath(item_path) == os.path.abspath(hdf5_path):
            continue  # never remove the archive itself
        if should_preserve(rel_item):
            print(f"⏭ Preserving top-level item: {rel_item}")
            continue
        if os.path.isdir(item_path):
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



# -------------------------------
# Worker helper (static, picklable)
# -------------------------------
class HDF5RunExtractor:

    @staticmethod
    def extract_group(h5_group, out_root):
        """Recursively extract an HDF5 group to a filesystem directory."""
        for key in h5_group:
            obj = h5_group[key]
            out_path = os.path.join(out_root, key)

            if isinstance(obj, h5py.Group):
                os.makedirs(out_path, exist_ok=True)
                HDF5RunExtractor.extract_group(obj, out_path)
                continue

            # Dataset
            dtype = obj.attrs.get("type", "binary")
            data = obj[()]

            if dtype == "text":
                if isinstance(data, bytes):
                    data = data.decode("utf-8", errors="replace")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(data)
            else:
                if isinstance(data, np.ndarray):
                    data = bytes(data)
                with open(out_path, "wb") as f:
                    f.write(data)

    # ---------------------------------------------------------
    # NEW: Extract only a list of files (no recursion needed)
    # ---------------------------------------------------------
    @staticmethod
    def extract_file_list(h5_group, out_root, file_list):
        """
        Extract only the files in file_list from this group.
        file_list is a list of filenames like ["fort.10", "input.dat"].
        """
        for key in h5_group:
            if key not in file_list:
                continue

            obj = h5_group[key]
            out_path = os.path.join(out_root, key)

            # If it's a dataset, extract it
            if isinstance(obj, h5py.Dataset):
                dtype = obj.attrs.get("type", "binary")
                data = obj[()]

                if dtype == "text":
                    if isinstance(data, bytes):
                        data = data.decode("utf-8", errors="replace")
                    with open(out_path, "w", encoding="utf-8") as f:
                        f.write(data)
                else:
                    if isinstance(data, np.ndarray):
                        data = bytes(data)
                    with open(out_path, "wb") as f:
                        f.write(data)

            # If it's a subgroup, recurse into it
            elif isinstance(obj, h5py.Group):
                subdir = os.path.join(out_root, key)
                os.makedirs(subdir, exist_ok=True)
                HDF5RunExtractor.extract_file_list(obj, subdir, file_list)

    # ---------------------------------------------------------
    # NEW: Extract only selected files into a temp directory
    # ---------------------------------------------------------
    @staticmethod
    def extract_files_to_temp(hdf5_path, group_path, file_list):
        """
        Extract only the files in file_list from the given group_path
        into a temporary directory.
        """
        tmp = tempfile.mkdtemp(prefix="gene_run_")
        with h5py.File(hdf5_path, "r") as h5f:
            group = h5f[group_path]
            HDF5RunExtractor.extract_file_list(group, tmp, file_list)
        return tmp

    @staticmethod
    def extract_to_temp(hdf5_path, group_path):
        """(Original) Extract entire group."""
        tmp = tempfile.mkdtemp(prefix="gene_run_")
        with h5py.File(hdf5_path, "r") as h5f:
            group = h5f[group_path]
            HDF5RunExtractor.extract_group(group, tmp)
        return tmp



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
