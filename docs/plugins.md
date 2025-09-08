# Plugins

Plugins extend the functionality of the `enchanted-surrogates` package by adding support for specific simulation codes. Each plugin typically consists of a parser and a runner that handle the specifics of reading/writing input/output files and executing simulations for a particular code.

## How to install plugins

To install a plugin, you need to clone its repository and install it using pip. For example, to install the HELENA plugin, you can use the following commands:

```bash
git clone ...
pip install -e enchanted-plugin-helena
```


## How to use plugins

Once you have installed a plugin, you can use it with the `enchanted-surrogates` interface. You need to specify the parser and runner corresponding to the plugin in your configuration file. 


## How to develop plugins

### 📂 Project Structure

```
enchanted-plugin-codename/
├── src/
│  └── enchanted_plugin_codename/
│    ├── init.py # Package initializer (can be empty)
│    ├── codename_parser.py
│    └── codename_runner.py
├── tests/
│    └── test_plugin.py # Unit tests for the plugin
├── README.md
├── pyproject.toml
```

Note that the repository name should follow the format `enchanted-plugin-codename`, where `codename` is a unique identifier for your plugin. The package name inside `src/` should match this name but with **underscores** between the words instead of hyphens.


### 🛠️ Development Steps

1. **Set Up Your Environment**:
   - Create a virtual environment and activate it:
     ```bash
     python -m venv .venv
     source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
     ```
   - Clone the `enchanted-surrogates` repository to have access to the core package:
     ```bash
     git clone  --single-branch --branch develop-helena-plugin https://github.com/DIGIfusion/enchanted-surrogates.git
     cd enchanted-surrogates
     pip install -e .
     cd ..
     ```
     (Currnetly using a specific branch for HELENA plugin development, since the plugin support is not yet merged to main.)

2. **Create the Plugin Structure**:
   - Create a new directory for your plugin, e.g., `enchanted-plugin-codename`.
     ```bash
     mkdir enchanted-plugin-codename
     cd enchanted-plugin-codename
     git init  # Initialize a git repository here
     mkdir src tests
     cd src
     mkdir enchanted_plugin_codename
     ```
   - Inside this directory, create the necessary files and folders as shown in the project structure above.
3. **Dependencies and entrypoints**
    - Create a `pyproject.toml` file in the root of your plugin directory to manage dependencies and define entry points. Here is a basic example:
      ```toml
      [tool.poetry]
      name = "enchanted-plugin-codename"
      version = "0.1.0"
      description = "A plugin for enchanted-surrogates to support Codename simulations."
      readme = "README.md"
      authors = [
        {name = "", email = ""
        },
      ]
      license = {text = "MIT"}
      dependencies = [
        "enchanted_surrogates",
        "f90nml==1.4.4",
        "numpy",
        "fortranformat",
        "scipy",]
      requires-python = ">=3.11"

      [tool.setuptools.packages.find]
      where = ["src"]
      ```

   - Define entry points in `pyproject.toml` to register your parser and runner with the `enchanted-surrogates` framework. For example:
     ```toml
     [tool.poetry.plugins."enchanted_surrogates.parsers"]
     codename_parser = "enchanted_plugin_codename.helena_parser:CodenameParser"

     [tool.poetry.plugins."enchanted_surrogates.runners"]
     codename_runner = "enchanted_plugin_codename.helena_runner:CodenameRunner"
     ```
     These will allow the main package to discover and utilize your plugin seamlessly.

4. **Implement the Plugin**:
   - In `codename_parser.py`, implement the class CodenameParser with logic to read and write input/output files.
   - In `codename_runner.py`, implement the class CodenameRunner with logic to execute simulations.
   - Ensure your code adheres to the interfaces defined in the `enchanted-surrogates` package.
   - When the entrypoints and the logic is implemented, you can install your plugin in editable mode for development:
     ```bash
     cd enchanted-plugin-codename
     pip install -e .
     ```
     after which you can use your plugin with `enchanted-surrogates`.

5. **Testing**:
   - Write unit tests in `test_plugin.py` to validate the functionality of your parser and runner.
   - Use `pytest` to run your tests:
     ```bash
     pytest tests/
     ```
6. **Github Actions**:
   - Set up a GitHub Actions workflow to automate testing and deployment.


## Existing plugins

- HELENA: [enchanted-plugin-helena](https://github.com/DIGIfusion/enchanted-plugin-helena)
- MISHKA: [enchanted-plugin-mishka](https://github.com/DIGIfusion/enchanted-plugin-mishka)
  