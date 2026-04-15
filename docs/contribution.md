# Contribution guidlines

We encourage contributions to the enchanted-surrogates package! Here are some
guidelines to help you get started. If you are interested in adding a new code
plugin, please refer to the [Plugins](./plugins/index.md) section for more
details.

- `main` branch is for stable code (releases)
- `develop` branch is for latest development code (merges from feature branches)
- `develop/{feature}` or `develop/{user}` for changes.
- `bug/{descriptive_name}` for bug fixes.
- Enable linting pre-commit hook (stops the commit if linting rules are violated) by
  running (Can be overridden with `git commit --no-verify` if needed):

  ```bash
  git config core.hooksPath .githooks
  ```



- **One feature or fix per pull request**. This ensures that changes are
  isolated and easier to review. Be respectful of your fellow developers and
  create small, focused pull requests.
- Use pull requests to merge branches. Delete branch after merge.
- Use `Issues` for bug reports, feature requests, etc.
- For longer term items to be integrated, e.g., Active Learning, suggest to use
  `Issues` followed by a branch.
- The configs folder in the source is to be kept for example config files and
  example cases that would be benefical to the wider community. Plugin-specific
  config files should be kept in the plugin repository.
- If any samplers, executors, or runners that are added require optional dependencies,
  please mark that clearly in the docstring.

---

## Coding Style Standards

The coding standard [PEP8](https://peps.python.org/pep-0008/) should be used.

---

## Testing

#### New samplers

In addition to any sampler-specific features, new samplers should be tested to

 - return values within sampler bounds

 - return correct number of samples specified by the batch size
 
 - respect sampler budget


#### Automated Testing at Pull Requests

The `tests` folder contains unit tests. These can be run manually by using the
command:

    pytest tests -v -s

and will also be automatically run by Github Actions at certain pushes and pull
requests. It is recommended to locally run the tests before making a commit.

If on HPC you must be using an interactive session with roughly 4 cores and at
least 500MB of memory.

<!-- **NB:** submodules are necessary to run the tests.  -->

#### Workflow tests

The `workflow_tests` folder contains larger workflow tests. These can be run
manually by using the command:

    pytest workflow_tests/automated_tests_no_HPC -v -s

#### Linting Tests

A Github Actions workflow is also used for running Ruff tests. These are
currently only testing for issues categorized as Errors or Fatal. Message
overview [here](https://docs.astral.sh/ruff/rules/). The list of enabled rules
can be found in pyproject.toml. To check the linting locally and get a full
overview of all possible issues, run:

For single file check:

    ruff /path/to/file.py

For all python files in $PWD:

    ruff $(find $PWD -name "*.py")

#### Machine Specific Tests

For now, no HPC specific tests are run as part of the automated testsing
procedure. So if you use enchanted surrogates on a specific machine it is your
responsibility to test updates on that machine.

Alternatively, one may ceate a tests folder for a specific machine in
`/enchanted-surrogates/tests/MACHINE_NAME_tests`, which should be executable via

    python -m pytest tests/MACHINE_NAME_tests

---

## Documentation
#### Main documentation
This project supports automatic documentation generation from docstrings of classes and modules, while still allowing manual documentation via Markdown (.md) files in the `docs/` directory.

There are three main approaches to documenting modules or classes:

1. Docstring-only documentation
    - You can rely entirely on docstrings to generate documentation. There is no need to create a .md file if all documentation is provided via docstrings.
    - Write a docstring in the module or class.
    - Important: the docstring must be placed at the very top of the file, before any imports, see other samplers and executors as example.
    - The documentation will be automatically extracted and included in the generated site.

2. Docstring + Markdown file
    - You can combine docstrings with a manually written Markdown file.
    - Important: include a docstring at the top of the file (before imports).
    - Docstrings will be automatically imported and rendered within the corresponding documentation page.
    - Ensure that your .md file and module/class file have same or almost same name.

3. Markdown-only documentation
    - You can rely exclusively on manually written Markdown documentation.
    - If you do not want docstrings to be imported:
        Either do not write a docstring at all, or
        Do not include a docstring at the top of the file (before imports). In this case, the documentation generator will skip importing the module/class.

##### Mathematical/physics formulas in documentation
Mathematical formulas are supported in the documentation and are rendered using LaTeX syntax. You can include both inline and block equations depending on the context.

Inline formulas: use `\(...\)`

Block formulas: use `\[...\]`

This allows you to properly represent complex mathematical expressions within your documentation pages.

As example of syntax:
```
Inline variant

\( \int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi} \)

More than one expession

\[
\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}, \quad
\lim_{x \to 0} \frac{\sin x}{x} = 1
\]

```

#### Plugin documentation

Documentation for plugins can also be imported from docstrings contained within plugin modules and classes. The same three documentation approaches described above can be applied to plugins as well: docstring-only documentation, docstring + Markdown file or Markdown only.

In case you want to add only Markdown or Markdown + docstring of your plugin to the official documentation, create a pull request to the `docs/plugins/` in the `enchanted-surrogates` repository, following the format used for existing plugins. Name file match your plugin name.

You can include docstring-based documentation from your plugin into the main documentation. Before make sure your plugin contain an __init__.py file (it can be empty, but it must exist so Python treats the directory as a package).
Fill configuration file plugins.yml to enable docstring import and create a Pull Request with your changes. You can add docstring of all module or just chosen one, see examples bellow.


🧩 Configuration Fields:

```yml
- name: <plugin_name>
  repo: <github_repo>
  module: <module_path>
  src_path: <source_directory>
```
There:
  name — plugin name (used for docs file naming)
  repo — GitHub repository path
  module — Python module path to import docstrings from
  src_path — path to source code inside the repository (e.g. src)

Example 1 (All module docstrings):
```yml
- name: template_plugin
  repo: DIGIfusion/enchanted-plugin-template
  module: enchanted_plugin_template
  src_path: src
```
This will scan the entire module and import docstrings from all .py files inside the package.

Example 2 (Only one file docstring):
```yml
- name: template_plugin
  repo: DIGIfusion/enchanted-plugin-template
  module: enchanted_plugin_template.template_runner
  src_path: src
```
This will import docstrings only from template_runner.py and ignore all other files.

Example 3 (Two or more file docstring):
```yml
- name: template_plugin
  repo: DIGIfusion/enchanted-plugin-template
  module:
    - enchanted_plugin_template.template_runner
    - enchanted_plugin_template.template_parser
  src_path: src
```
This will import docstrings from chosen two modules and ignore all other modules.


##### Personal access token 
To allow the main repository to access and build documentation from plugin repositories (including private ones), a Personal Access Token (PAT) is used.
This token provides secure, automated access to plugin source code during the documentation build process.

The token currently stored in the repository's secrets has no expiration date. If for some reason a new token needs to be generated, this can be done in the settings. Navigate to Developer settings, open Personal access tokens, click Tokens (classic), click Generate new token and choose repo scope. Copy new token and add it to GitHub secrets on main repository.

---


## Creating a release

!!! note
    Only for developers with write or push access.

Releases are accessed in [GitHub](https://github.com/DIGIfusion/enchanted-surrogates). Only users with write or push access to project can create release. Releases are generally created from `main` branch.

## Semantic versioning

Workflow uses GitHub actions to create Releases on GitHub. It runs automatically when you push a Git tag that matches the pattern `vX.Y.Z` where `X.Y.Z` is using [Semantic Versioning](https://semver.org/). It matches `major.minor.patch` type of semantic versioning:

1. Major `X` is increased when something backwards incompatible is created
2. Minor `Y` is increased when functionality is added in a backward compatible manner 
3. Patch `Z` is incremented when backward compatible bug fixes are made.

Increasing `X` resets `Y` and `Z`  to zero, and increasing `Y` resets `Z` to zero. Each increment is always one.

**Version should be incremented for each release**

## Triggering release

Currently workflow is triggered in two steps: first `git tag` and `git push` commands in development environment, then the release is created from [Releases subpage](https://github.com/DIGIfusion/enchanted-surrogates/releases/)






### Step 1: Local triggering

This requires user to create tag name `vX.Y.Z` according to semantic versioning described above. Workflow activates `.github/workflows/release.yaml`.


In `release.yaml` file linting and testing is run, then if those succeed package is built, and if build succeeds content of  `dist/*` folder is included in release. 

Triggering is done in the branch, that is to be released.
Use below commands to trigger first part of release workflow:
```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

### Step: 2 GitHub release
Check on [Actions](https://github.com/DIGIfusion/enchanted-surrogates/actions) page, that tests, build and artifact creation succeeded.

Go to [Releases](https://github.com/DIGIfusion/enchanted-surrogates/releases) page to create a new release:

1. Click Draft a new release. 
2. Select the tag `vX.Y.Z` which was used for local triggering
3. Click Generate release notes and use it as a base
4. Modify release notes if necessary and publish



