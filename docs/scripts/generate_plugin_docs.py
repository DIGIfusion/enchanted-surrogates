from pathlib import Path
import yaml

CONFIG_PATH = Path("plugins.yml")
DOCS_DIR = Path("docs/plugins")


def get_modules(src_path, base_module):
    modules = []

    for py_file in src_path.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue

        rel = py_file.relative_to(src_path).with_suffix("")
        module_path = ".".join(rel.parts)

        modules.append(f"{base_module}.{module_path}")

    return modules


def update_md_file(md_path: Path, modules: list[str]):
    content = md_path.read_text() if md_path.exists() else ""

    blocks = "\n\n".join(
    f"""::: {m}
    options:
      show_source: false
    """
    for m in modules)

    new_content = f"{content.strip()}\n\n{blocks}\n"

    md_path.write_text(new_content)
    print(f"[UPDATE] {md_path}")


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    for plugin in config["plugins"]:
        name = plugin["name"]
        module = plugin["module"].split(".")[0]
        src_path = Path("plugins") / name / plugin["src_path"] / module

        modules = get_modules(src_path, module)

        md_file = DOCS_DIR / f"{name}.md"
        update_md_file(md_file, modules)


if __name__ == "__main__":
    main()