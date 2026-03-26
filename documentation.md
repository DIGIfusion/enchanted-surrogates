### Documentation
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
    - Create a .md file in the docs directory with a name similar to the module or class and write additional documentation in the .md file.
    - You can still use docstrings for part of the documentation.
    - Important: keep the same rule — include a docstring at the top of the file (before imports).
    - Docstrings will be automatically imported and rendered within the corresponding documentation page.
    - Ensure that your .md file and module/class file have same or almost same name.

3. Markdown-only documentation
    - You can rely exclusively on manually written Markdown documentation.
    - Create a .md file in the docs directory and write all documentation there.
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

First:
\( \int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi} \)

Second:

\[
\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}, \quad
\lim_{x \to 0} \frac{\sin x}{x} = 1
\]

#### Plugin documentation
Documentation for plugins can also be imported from docstrings contained within plugin modules and classes. The same three documentation approaches described above can be applied to plugins as well.
For more detailed information on importing documentation from plugins, please refer to the [Plugins](https://digifusion.github.io/enchanted-surrogates/plugins/#development-steps) section.

