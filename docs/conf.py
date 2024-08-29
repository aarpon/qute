# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import datetime
import os
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

sys.path.insert(0, os.path.abspath(".."))
from qute import __version__

project = "qute"
year = str(datetime.datetime.now().year)
copyright = f"2022 - {year}, Aaron Ponti"
author = "Aaron Ponti"
release = f"{__version__}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
]

# Mock some imports that are not needed to build the documentation
autodoc_mock_imports = [
    "numpy",
    "userpaths",
    "torch",
    "pytorch_lightning",
    "monai",
    "scipy",
    "natsort",
    "imio",
    "typing_extensions",
    "torchmetrics",
    "tifffile",
    "nd2reader",
    "skimage",
    "sklearn",
    "tqdm",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
