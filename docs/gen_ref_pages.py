from pathlib import Path
import ast
import mkdocs_gen_files
import sys

SRC_PATH = Path("src/enchanted_surrogates")
DOCS_PATH = Path("docs")

sys.path.insert(0, str(SRC_PATH.resolve()))

print("GEN_REF_PAGES STARTED")

def ensure_init_files():
    for path in SRC_PATH.rglob("*"):
        if path.is_dir() and path != SRC_PATH:
            init_file = path / "__init__.py"
            if not init_file.exists():
                init_file.touch()

ensure_init_files()

def find_existing_md(module_parts, base_name):
    direct_path = DOCS_PATH / Path(*module_parts).with_suffix(".md")
    if direct_path.exists():
        return direct_path

    root_path = DOCS_PATH / f"{base_name}.md"
    if root_path.exists():
        return root_path
    
    for md_file in DOCS_PATH.rglob(f"{base_name}.md"):
        return md_file
    
    full_name = "_".join(module_parts)
    full_name_path = DOCS_PATH / f"{full_name}.md"
    if full_name_path.exists():
        return full_name_path
    
    if len(module_parts) >= 2:
        last_two = "_".join(module_parts[-2:])
        last_two_path = DOCS_PATH / f"{last_two}.md"
        if last_two_path.exists():
            return last_two_path
    
    if len(module_parts) >= 2:
        nested_path = DOCS_PATH / module_parts[-2] / f"{base_name}.md"
        if nested_path.exists():
            return nested_path
    
    return None

def is_already_documented(md_path, module_name):
    if not md_path or not md_path.exists():
        return False
    
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    return f"::: {module_name}" in content


for path in SRC_PATH.rglob("*.py"):
    if path.name == "__init__.py":
        continue
    
    rel_path = path.relative_to(SRC_PATH)
    module_parts = rel_path.with_suffix("").parts
    module_name = ".".join(module_parts)
    base_name = path.stem
    
    # Check if docstring exist
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())
    if not ast.get_docstring(tree):
        continue
    
    try:
        __import__(module_name)
    except ImportError:
        continue
    
    # find existing md file
    existing_md = find_existing_md(module_parts, base_name)
    
    if existing_md:
        print("Found existing md:", existing_md)
        # Check if module already documented
        if is_already_documented(existing_md, module_name):
            continue
        
        with mkdocs_gen_files.open(existing_md.relative_to(DOCS_PATH), "a") as f:
            f.write(f"\n\n## Modul `{module_name}`\n\n")
            f.write(f"::: {module_name}\n")
    
    else:
        # Create new file if module not documented yet
        doc_path = rel_path.with_suffix(".md")
        full_doc_path = DOCS_PATH / doc_path
        
        full_doc_path.parent.mkdir(parents=True, exist_ok=True)
        
        with mkdocs_gen_files.open(full_doc_path.relative_to(DOCS_PATH), "w") as f:
            f.write(f"# {base_name}\n\n")
            f.write(f"::: {module_name}\n")
