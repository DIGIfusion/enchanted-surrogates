import os
import h5py
import shutil
import json
import numpy as np

def convert_directory_to_hdf5(source_dir, hdf5_name="archive.h5"):
    hdf5_path = os.path.join(source_dir, hdf5_name)

    with h5py.File(hdf5_path, "w") as h5f:
        for root, _, files in os.walk(source_dir):
            rel_root = os.path.relpath(root, source_dir)
            group = h5f.require_group(rel_root)
            for file in files:
                file_path = os.path.join(root, file)
                if file_path == hdf5_path:
                    continue  # Skip the HDF5 file itself

                with open(file_path, "rb") as f:
                    data = f.read()

                dataset_path = os.path.join(rel_root, file)
                if dataset_path in h5f:
                    print(f"⚠️ Skipping duplicate: {dataset_path}")
                else:
                    try:
                        decoded = data.decode("utf-8")
                        group.create_dataset(file, data=decoded)
                        group[file].attrs["type"] = "text"
                    except (UnicodeDecodeError, ValueError):
                        try:
                            group.create_dataset(file, data=np.frombuffer(data, dtype='uint8'))
                            group[file].attrs["type"] = "binary"
                        except:
                            print(f"⚠️ PATH FAILURE: {dataset_path}")
                            


    ## Remove original files and folders
    for item in os.listdir(source_dir):
        item_path = os.path.join(source_dir, item)
        if item_path != hdf5_path:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)

    # Add notebook and README
    create_jupyter_notebook(source_dir, hdf5_name)
    create_readme(source_dir, hdf5_name)

    print(f"✅ Packed '{source_dir}' into '{hdf5_name}' with notebook and README.")

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
    if len(sys.argv) < 2:
        print("Usage: python hdf5.py /path/to/folder")
    else:
        folder = sys.argv[1]
        convert_directory_to_hdf5(folder)