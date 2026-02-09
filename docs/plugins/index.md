# Plugins

Plugins extend the functionality of the `enchanted-surrogates` package by adding support for specific simulation codes. Each plugin typically consists of a parser and a runner that handle the specifics of reading/writing input/output files and executing simulations for a particular code.

## How to install plugins

To install a plugin, you need to clone its repository and install it using pip. For example, you can use the following commands:

```bash
git clone https://github.com/DIGIfusion/enchanted-plugin-codename.git
pip install -e enchanted-plugin-codename/
```


## How to use plugins

Once you have installed a plugin, you can use it with the `enchanted-surrogates` interface. You need to specify the parser and runner corresponding to the plugin in your configuration file. 


## How to develop plugins

### 📂 Project Structure

Go to [enchanted-plugin-template](https://github.com/DIGIfusion/enchanted-plugin-template) and click "Use this template" to create a new repository for your plugin. 

Note that the repository name should follow the format `enchanted-plugin-template`, where `template` is a unique identifier for your plugin. The package name inside `src/` should match this name but with **underscores** between the words instead of hyphens.

The basic structure of the plugin should look like this:

```
enchanted-plugin-template/
├── src/
│  └── enchanted_plugin_template/
│    ├── __init__.py
│    ├── template_parser.py
│    └── template_runner.py
├── tests/
│    └── test_plugin.py
│    └── ...
├── README.md
├── pyproject.toml
```


The template contains basic units tests and a GitHub Actions workflow. Change the names and contents of the files to match your plugin.


### 🛠️ Development Steps

1. **Set Up Your Environment**:
   - Create a virtual environment and activate it:
     ```bash
     python -m venv .venv
     source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
     ```
   - Clone the `enchanted-surrogates` repository to have access to the core package:
     ```bash
     git clone https://github.com/DIGIfusion/enchanted-surrogates.git
     pip install -e enchanted-surrogates/
     ```

2. **Create the Plugin Repository**:
   - Go to [enchanted-plugin-template](https://github.com/DIGIfusion/enchanted-plugin-template) and click "Use this template" to create a new repository for your plugin. Name it following the format `enchanted-plugin-codename`, where `codename` is a unique identifier for your plugin.
   - Clone your new repository and install it in editable mode:
     ```bash
     git clone https://github.com/DIGIfusion/enchanted-plugin-codename.git
     pip install -e enchanted-plugin-codename/
     ```
   - Inside this directory, create/edit the necessary files.
  
3. **Dependencies and entrypoints**
    - Create a `pyproject.toml` file in the root of your plugin directory to manage dependencies and define entry points. Here is a basic example:
      ```toml
      [tool.poetry]
      name = "enchanted-plugin-codename"
      version = "0.1.0"
      description = "A plugin for enchanted-surrogates to support Codename simulations."
      readme = "README.md"
      authors = [{name = "", email = ""},]
      license = {text = "MIT"}
      dependencies = [
        "enchanted_surrogates",
        "f90nml==1.4.4",
        "numpy",
        "fortranformat",
        "scipy",]
      requires-python = ">=3.10"

      [tool.setuptools.packages.find]
      where = ["src"]
      ```

   - Define entry points in `pyproject.toml` to register your parser and runner with the `enchanted-surrogates` framework. For example:
     ```toml
     [tool.poetry.plugins."enchanted_surrogates.parsers"]
     codename_parser = "enchanted_plugin_codename.codename_parser:CodenameParser"

     [tool.poetry.plugins."enchanted_surrogates.runners"]
     codename_runner = "enchanted_plugin_codename.codename_runner:CodenameRunner"
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
   - Write unit tests in the `tests\` folder to validate the functionality of your parser and runner. The folder already contains some test files that should be edited to correctly test your plugin.
   - Use `pytest` to run your tests:
     ```bash
     pytest tests/ -sv
     ```
6. **Github Actions**:
   - The plugin template contains a GitHub Actions workflow setup to automate testing and deployment. Make sure the tests pass.

7. **Documentation**:
   - Update the `README.md` file with information about your plugin, including installation instructions, usage examples, and any other relevant details.
   - In case you want to add your plugin to the official documentation, create a pull request to the `docs/plugins/` in the `enchanted-surrogates` repository, following the format used for existing plugins.


## Existing plugins

The template for creating new plugins is publicly available at [enchanted-plugin-template](https://github.com/DIGIfusion/enchanted-plugin-template).

The existing plugins are private repositories, but you can request access by contacting DIGIfusion. The currently available plugins are:

- HELENA: [enchanted-plugin-helena](https://github.com/DIGIfusion/enchanted-plugin-helena)
- MISHKA: [enchanted-plugin-mishka](https://github.com/DIGIfusion/enchanted-plugin-mishka)
- CASTOR: [enchanted-plugin-castor](https://github.com/DIGIfusion/enchanted-plugin-castor)
- DREAM+SOFT: [enchanted-plugin-dream](https://github.com/DIGIfusion/enchanted-plugin-dream)
- GENE: [enchanted-plugin-gene](https://github.com/DIGIfusion/enchanted-plugin-gene)
  
