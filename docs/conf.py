from datetime import datetime
from importlib.metadata import metadata

from scverse_doc.config import DEFAULTS
from scverse_doc.registry import intersphinx

info = metadata("pydeseq2")
project = info["Name"]
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}"
version = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
source_repository = urls["Source"]
release = info["Version"]

bibtex_bibfiles = ["references.bib"]
templates_path = ["_templates"]

extensions = [
    "scverse_doc",
    "sphinx_gallery.gen_gallery",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.googleanalytics",
    "sphinx_autodoc_typehints",
    "IPython.sphinxext.ipython_console_highlighting",
]

add_module_names = False
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

intersphinx_mapping = intersphinx()

exclude_patterns = [
    *DEFAULTS.exclude_patterns,
    "README.md",
    # sphinx-gallery writes a notebook next to each example, which myst-nb would also read
    "auto_examples/*.ipynb",
]

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_logo = "_static/pydeseq2_logo.svg"
html_favicon = "_static/favicon.ico"
html_show_sourcelink = False
# The logo greens, as in scverse/scverse.github.io#380; switch to `{"secondary_accent": True}` once scverse-doc's registry has them.
html_theme_options = {
    "accent": "oklch(0.676 0.189 145.9)",
    "secondary_accent": "oklch(0.521 0.17 143.3)",
}

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
