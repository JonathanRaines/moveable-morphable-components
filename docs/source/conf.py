# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path

project = "MMC"
copyright = "2024, Jonathan Raines"
author = "Jonathan Raines"
release = "0.1.0"


sys.path.insert(
    0, Path("../../moveable_morphable_components").resolve().as_posix())

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "use_edit_page_button": True,
}
html_sidebars = {
    "**": [],
}
html_static_path = ["_static"]
html_context = {
    "github_url": "ttps://github.com",
    "github_user": "JonathanRaines",
    "github_repo": "moveable-morphable-components",
    "github_version": "main",
    "doc_path": "docs/source/",
}
