# Configuration file for the Sphinx documentation builder.

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/page/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import sys
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))


# -- Project information -----------------------------------------------------

# NOTE: If you installed your project in editable mode, this might be stale.
#       If this is the case, reinstall it to refresh the metadata
info = metadata("pydeseq2")
project = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}"
version = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
repository_url = urls["Source"]

# The full version, including alpha/beta/rc tags
release = info["Version"]

bibtex_bibfiles = ["references.bib"]
templates_path = ["_templates"]
nitpicky = True  # Warn about broken links
needs_sphinx = "4.0"

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "scverse",
    "github_repo": project,
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
# They can be extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.googleanalytics",
    "sphinx_autodoc_typehints",
    "sphinx_design",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxext.opengraph",
    "scanpydoc",  # theme + elegant type hints
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]

autosummary_generate = True
autodoc_member_order = "groupwise"
autodoc_default_options = {
    "show-inheritance": False,
    "inherited-members": False,
    "members": True,
}
add_module_names = False
default_role = "literal"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
myst_heading_anchors = 6  # create anchors for h1-h6
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto")
typehints_defaults = "braces"
always_use_bars_union = (
    True  # use `|` instead of `Union` in types even when building with Python ≤3.14
)

ogp_social_cards = {"image": "_static/pydeseq2_logo.png"}
googleanalytics_id = "UA-83738774-2"

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

intersphinx_mapping = {
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "python": ("https://docs.python.org/3", None),
    "anndata": ("https://anndata.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "README.md",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "scanpydoc"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_logo = "_static/pydeseq2_logo.svg"
html_favicon = "_static/favicon.ico"
html_show_sourcelink = False

html_title = project

html_theme_options = {
    "repository_url": repository_url,
    "repository_branch": "main",
    "use_repository_button": True,
    "use_issues_button": True,
    "path_to_docs": "docs/",
    "navigation_with_keys": False,
    # The dark green of the logo. scanpydoc exposes it as the `--accent-color`
    # CSS variable, which colors the mobile header and the project name.
    "accent_color": "#078013",
    "show_toc_level": 2,
    # Without this the author list is repeated above the copyright line.
    "footer_content_items": ["copyright.html"],
}

pygments_style = "default"

nitpick_ignore = [
    # anndata re-exports AnnData from a private submodule; autodoc resolves the
    # runtime path for base-class references which intersphinx cannot remap.
    ("py:class", "anndata._core.anndata.AnnData"),
    # pandas re-exports DataFrame from pandas.core.frame, but intersphinx only
    # knows the public path (pandas.DataFrame). autodoc resolves type annotations
    # like `pd.DataFrame` to the internal module path at runtime.
    ("py:class", "pandas.core.frame.DataFrame"),
    ("py:class", "numpy._typing._generic_alias.ScalarType"),
    # Bare types written in docstrings rather than resolved from annotations.
    ("py:class", "ndarray"),
    ("py:class", "optional"),
]
