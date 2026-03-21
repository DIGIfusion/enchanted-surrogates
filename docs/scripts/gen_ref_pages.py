from pathlib import Path
import ast
import mkdocs_gen_files
import sys
import importlib

SRC_PATH = Path("src/enchanted_surrogates")
DOCS_PATH = Path("docs")

sys.path.insert(0, str(SRC_PATH.resolve()))
print(f"[DEBUG] PYTHONPATH updated with: {SRC_PATH.resolve()}")

def ensure_init_files():
    """Creates missing __init__.py files in all subdirectories"""
    for path in SRC_PATH.rglob("*"):
        if path.is_dir() and path != SRC_PATH:
            init_file = path / "__init__.py"
            if not init_file.exists():
                init_file.touch()
                print(f"[DEBUG] Created missing __init__.py: {init_file}")
            else:
                print(f"[DEBUG] __init__.py already exists: {init_file}")

ensure_init_files()

def find_existing_md(module_parts, base_name):
    """Finds an existing Markdown file for the module, with multiple strategies"""
    direct_path = DOCS_PATH / Path(*module_parts).with_suffix(".md")
    if direct_path.exists():
        print(f"[DEBUG] Found MD via direct path: {direct_path}")
        return direct_path

    root_path = DOCS_PATH / f"{base_name}.md"
    if root_path.exists():
        print(f"[DEBUG] Found MD via base_name at root: {root_path}")
        return root_path
    
    for md_file in DOCS_PATH.rglob(f"{base_name}.md"):
        print(f"[DEBUG] Found MD via recursive search: {md_file}")
        return md_file
    
    full_name = "_".join(module_parts)
    full_name_path = DOCS_PATH / f"{full_name}.md"
    if full_name_path.exists():
        print(f"[DEBUG] Found MD via joined module parts: {full_name_path}")
        return full_name_path
    
    if len(module_parts) >= 2:
        last_two = "_".join(module_parts[-2:])
        last_two_path = DOCS_PATH / f"{last_two}.md"
        if last_two_path.exists():
            print(f"[DEBUG] Found MD via last two module parts: {last_two_path}")
            return last_two_path
    
    if len(module_parts) >= 2:
        nested_path = DOCS_PATH / module_parts[-2] / f"{base_name}.md"
        if nested_path.exists():
            print(f"[DEBUG] Found MD via nested folder structure: {nested_path}")
            return nested_path
    
    print(f"[DEBUG] No existing MD file found for module: {'.'.join(module_parts)}")
    return None

def is_already_documented(md_path, module_name):
    if not md_path or not md_path.exists():
        print(f"[DEBUG] MD path does not exist or is None: {md_path}")
        return False
    
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    already = f"::: {module_name}" in content
    print(f"[DEBUG] Module '{module_name}' already documented in {md_path}: {already}")
    return already


for path in SRC_PATH.rglob("*.py"):
    if path.name == "__init__.py":
        continue
    
    rel_path = path.relative_to(SRC_PATH)
    module_parts = rel_path.with_suffix("").parts
    module_name = ".".join(module_parts)
    base_name = path.stem

    print(f"\n[DEBUG] Processing file: {path}")
    print(f"[DEBUG] Module name: {module_name}, Base name: {base_name}")

    # Check if the module has a docstring
    try:
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        doc_exists = ast.get_docstring(tree) is not None
        print(f"[DEBUG] Module has docstring: {doc_exists}")
        if not doc_exists:
            continue
    except Exception as e:
        print(f"[DEBUG] Failed to parse {path}: {e}")
        continue  

    try:
        importlib.import_module(module_name)
        print(f"[DEBUG] Import OK: {module_name}")
    except ModuleNotFoundError as e:
        if e.name == module_name.split('.')[0]:
            print(f"[DEBUG] Import FAILED: {module_name} -> {e}")
            continue
        else:
            print(f"[DEBUG] Dependency missing (ignored): {module_name} -> {e}")

    # Find existing Markdown file
    existing_md = find_existing_md(module_parts, base_name)
    
    if existing_md:
        if is_already_documented(existing_md, module_name):
            print(f"[DEBUG] Module already documented, skipping: {module_name}")
            continue
        
        # Append to existing file
        print(f"[DEBUG] Appending documentation to existing MD: {existing_md}")
        with mkdocs_gen_files.open(existing_md.relative_to(DOCS_PATH), "a") as f:
            f.write(f"\n\n## Module `{module_name}`\n\n")
            f.write(f"::: {module_name}\n")
    
    else:
        # Create new Markdown file
        doc_path = rel_path.with_suffix(".md")
        full_doc_path = DOCS_PATH / doc_path
        full_doc_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[DEBUG] Creating new MD file: {full_doc_path}")

        with mkdocs_gen_files.open(full_doc_path.relative_to(DOCS_PATH), "w") as f:
            f.write(f"# {base_name}\n\n")
            f.write(f"::: {module_name}\n")

print("\n[DEBUG] Documentation generation finished.")