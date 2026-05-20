"""Sphinx configuration for the merton documentation."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
    "pyarrow": ("https://arrow.apache.org/docs/", None),
}

# --- Nitpicky-mode tuning ----------------------------------------------------
# `-n` (nitpicky) flags every cross-reference sphinx can't resolve, which is
# great for catching genuine typos but noisy for our type-alias / TypeVar /
# extras-only-module references. Two layers of suppression:
#
#  1. `nitpick_ignore_regex` — exact patterns of refs to silence. Each entry
#     is `(reftype, regex_against_target)`. The regex must match the FULL
#     target string.
#
#  2. `suppress_warnings` — broad-stroke categories. `autoapi.python_import_resolution`
#     and `ref.python` cover the dataclass-attribute duplicates from
#     sphinx-autoapi documenting `MertonResult.firm` twice.
nitpick_ignore_regex = [
    # Type aliases in the private `_typing` module. Aliases aren't classes,
    # so sphinx can't hyperlink them.
    ("py:class", r"merton\._typing\..*"),
    ("py:class", r"(ArrayLike|FloatArray|IntArray|BoolArray|Scalar|NumberLike|SupportsArray)"),
    # TypeVars (F, T, R, …) aren't classes either.
    ("py:class", r"[A-Z]"),
    ("py:class", r"Ellipsis"),
    # Bare names that *do* resolve via FQN; the bare-name ref is a doctring
    # convention we don't want to fail the build on.
    (
        "py:class",
        r"(Firm|FirmPanel|MertonResult|StructuralResult|MertonError|BacktestResult"
        r"|LossDistribution|BlackCoxModel|GeskeModel|LongstaffSchwartzModel"
        r"|LelandToftModel|JumpDiffusionModel|CreditGradesModel|ClimateScenario"
        r"|DefaultPointLike|DefaultPointCallable|EDFMap)",
    ),
    # Internal helpers referenced from docstrings but not exported.
    ("py:func", r"merton\.core\.compute_default_point|jmr_iterative"),
    ("py:class", r"merton\.core\.MertonResult|merton\.portfolio\.Portfolio"),
    ("py:func", r"merton\.greeks\.equity_theta"),
    ("py:mod", r"merton\.calibration\.implied_vol"),
    # External libs without published intersphinx inventories.
    ("py:mod", r"typer|structlog|openpyxl|rich\.progress|joblib|dask|ray|emcee"),
    ("py:class", r"structlog\..*|pd|stderr|matplotlib\..*|joblib\..*"),
    ("py:class", r"Path|emcee\..*|SpreadInput"),
    # Functions / modules that exist internally but aren't documented
    # under their bare-name reference path.
    ("py:func", r"merton\.calibration\.available_methods"),
    ("py:func", r"jax\.(grad|vmap|jit)"),
    # Bare names cross-referenced from autoapi-generated module summaries
    # that don't have a fully-qualified path.
    ("py:class", r"MertonModel|Portfolio|VasicekFactor"),
    ("py:class", r"merton\.MertonResult|merton\.MertonModel|merton\.Firm"),
    # Internal modules referenced in autoapi-generated summaries.
    ("py:mod", r"merton\._backend\._survival|merton\._config"),
    ("py:class", r"merton\.core\.default_point\.DefaultPointLike"),
    ("py:class", r"merton\.calibration\.BootstrapResult|merton\.calibration\.CalibrationResult"),
    ("py:meth", r"MertonResult\.pd_term_structure"),
    ("py:func", r"__getattr__|_warm_numba_cache"),
]
# autoapi documents dataclass attributes both at the class scope (via
# the dataclass annotation) and as their own attribute nodes — sphinx
# then warns "duplicate object description". This is a known autoapi
# behavior; suppress it rather than re-architect every dataclass.
suppress_warnings = [
    "ref.python",  # autoapi duplicates for dataclass attrs
    "autoapi.python_import_resolution",
    "autoapi.toc_reference",
]

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


def setup(app):  # type: ignore[no-untyped-def]
    """Sphinx setup hook — install a logging filter for the noisy autoapi
    `duplicate object description` warnings that we can't reach through
    `suppress_warnings` (the python domain emits them without a subtype).

    Dataclasses with `slots=True, frozen=True` end up documented twice by
    sphinx-autoapi: once at the class signature, once as standalone
    attribute nodes. The pages render correctly either way; we just want
    CI to stop yelling about it.
    """
    import logging

    class _DropAutoapiDuplicateAttrs(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            msg = record.getMessage()
            if "duplicate object description" in msg:
                return False
            return True

    sphinx_logger = logging.getLogger("sphinx")
    sphinx_logger.addFilter(_DropAutoapiDuplicateAttrs())
    # The "document isn't included in any toctree" check fires for the
    # autoapi-generated `api/index.rst`. It IS reachable from the main
    # toctree (`api/merton/index`), just not at the top level.
    return {"parallel_read_safe": True}
