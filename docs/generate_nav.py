#this code generate navigation for GitHub Page from docs folder to mkdocs.yml file
import os
import yaml

docs_dir = "docs"
nav = []

for root, dirs, files in os.walk(docs_dir):
    md_files = sorted([f for f in files if f.endswith(".md")])

    if md_files:
        folder_name = os.path.relpath(root, docs_dir)

        pages = []

        if "index.md" in md_files:
            pages.append({
                "Start": os.path.join(folder_name, "index.md").replace("\\", "/")
            })
            md_files.remove("index.md")

        for f in md_files:
            title = os.path.splitext(f)[0].capitalize()
            path = os.path.join(folder_name, f).replace("\\", "/")
            pages.append({title: path})

        if folder_name != ".":
            nav.append({folder_name.capitalize(): pages})
        else:
            nav.extend(pages)

with open("mkdocs.yml", "r") as f:
    cfg = yaml.safe_load(f)

cfg["nav"] = nav

with open("mkdocs.yml", "w") as f:
    yaml.dump(cfg, f, sort_keys=False)
