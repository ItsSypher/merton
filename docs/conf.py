"""Sphinx configuration for the merton documentation."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
# --- end shim ----------------------------------------------------------------

project = "merton"
author = "The Merton Authors"
copyright = f"{datetime.now().year}, {author}"

try:
    from merton import __version__ as release
except ImportError:
    release = "0.0.0"
version = release

extensions = [
    # `myst_nb` loads `myst_parser` internally; listing both triggers
    # https://github.com/executablebooks/MyST-Parser/issues/962 (a
    # ValueError on Sphinx 8). `myst_nb` alone covers both `.md` and
    # `.ipynb` sources.
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "autoapi.extension",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.bibtex",
]

autoapi_type = "python"
autoapi_dirs = ["../src/merton"]
autoapi_root = "api"
autoapi_keep_files = False
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    # `imported-members` would re-document things like `merton.MertonResult`
    # under both `merton` and `merton.core.result`, causing autoapi to emit
    # hundreds of "duplicate object description" warnings — which `-W` in CI
    # turns into hard failures. Each symbol lives in exactly one place now.
]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "fieldlist",
    "colon_fence",
    "smartquotes",
    "tasklist",
]

nb_execution_mode = "auto"
nb_execution_timeout = 120
nb_execution_allow_errors = False

# `myst_nb` registers `.md` and `.ipynb` automatically; we just add
# `.rst` here so docutils handles it.
source_suffix = {
    ".rst": "restructuredtext",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "jupyter_execute"]

html_theme = "furo"
html_title = "merton"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/ItsSypher/merton/",
    "source_branch": "main",
    "source_directory": "docs/",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

bibtex_bibfiles = ["references.bib"]
bibtex_default_style = "alpha"

# Napoleon (NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True
