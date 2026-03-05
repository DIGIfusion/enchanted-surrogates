from pathlib import Path
import ast
import mkdocs_gen_files
import sys

print("===== GEN_REF_PAGES SCRIPT STARTED =====")

SRC_ROOT = Path("src")
SRC_PATH = SRC_ROOT / "enchanted_surrogates"
DOCS_PATH = Path("docs")

print("Current working directory:", Path.cwd())
print("SRC_ROOT:", SRC_ROOT.resolve())
print("SRC_PATH:", SRC_PATH.resolve())
print("DOCS_PATH:", DOCS_PATH.resolve())

print("SRC_ROOT exists:", SRC_ROOT.exists())
print("SRC_PATH exists:", SRC_PATH.exists())
print("DOCS_PATH exists:", DOCS_PATH.exists())

sys.path.insert(0, str(SRC_ROOT.resolve()))
print("PYTHONPATH UPDATED:", sys.path[0])

py_files = list(SRC_PATH.rglob("*.py"))
print("Total Python files found:", len(py_files))
print("First Python files:", py_files[:10])

md_files = list(DOCS_PATH.rglob("*.md"))
print("Total Markdown files found:", len(md_files))
print("First Markdown files:", md_files[:10])


def find_existing_md(module_parts, base_name):

    print("Searching md for module:", module_parts)

    direct_path = DOCS_PATH / Path(*module_parts).with_suffix(".md")
    if direct_path.exists():
        print("Found direct md:", direct_path)
        return direct_path

    root_path = DOCS_PATH / f"{base_name}.md"
    if root_path.exists():
        print("Found root md:", root_path)
        return root_path

    for md_file in DOCS_PATH.rglob(f"{base_name}.md"):
        print("Found recursive md:", md_file)
        return md_file

    print("No md found for:", base_name)
    return None


def is_already_documented(md_path, module_name):
    if not md_path.exists():
        return False

    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    if f"::: {module_name}" in content:
        print("Module already documented:", module_name)
        return True

    return False


for path in py_files:

    print("\n---")
    print("Processing file:", path)

    if path.name == "__init__.py":
        print("Skipping __init__.py")
        continue

    rel_path = path.relative_to(SRC_ROOT)
    module_parts = rel_path.with_suffix("").parts
    module_name = ".".join(module_parts)

    base_name = path.stem

    print("Module parts:", module_parts)
    print("Module name:", module_name)
    print("Base name:", base_name)

    try:
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except Exception as e:
        print("AST parse failed:", e)
        continue

    module_doc = ast.get_docstring(tree)
    print("Module docstring exists:", bool(module_doc))

    try:
        __import__(module_name)
        print("Import OK:", module_name)
    except ImportError as e:
        print("Import FAILED:", module_name, e)
        continue

    existing_md = find_existing_md(module_parts, base_name)

    print("Existing md result:", existing_md)

    if not existing_md:
        print("No md file found, skipping module")
        continue

    # check if already documented
    if is_already_documented(existing_md, module_name):
        continue

    relative_md = existing_md.relative_to(DOCS_PATH)
    print("Appending documentation to:", relative_md)

    try:
        with mkdocs_gen_files.open(relative_md, "a") as f:
            f.write(f"\n\n## Module `{module_name}`\n\n")
            f.write(f"::: {module_name}\n")
        print("Documentation block added for:", module_name)

    except Exception as e:
        print("Failed to write documentation:", e)


print("\n===== GEN_REF_PAGES SCRIPT FINISHED =====")