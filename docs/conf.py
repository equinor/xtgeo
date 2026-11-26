from datetime import date

import autoclasstoc
from autoclasstoc import AutoClassToc, PublicMethods, load_static_assets
from docutils import nodes

import xtgeo

version = xtgeo.__version__
release = xtgeo.__version__
project = "xtgeo"
current_year = date.today().year
copyright = "Equinor 2019 - " + str(current_year) + f" (XTGeo release {release})"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_autodoc_typehints",
]


class RemainingPublicMethods(PublicMethods):
    # skip dunder methods
    def predicate(self, name, attr, meta):
        return super().predicate(name, attr, meta) and not name.startswith("__")


autoclass_content = "both"
autoclasstoc_sections = [
    "public-attrs",
    "public-methods",
]
autosummary_generate = True
autodoc_default_options = {
    "inherited-members": True,
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
}
autodoc_typehints = "none"
napoleon_include_special_with_doc = False

# The suffix of source filenames.
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

# The master toctree document.
master_doc = "index"

templates_path = ["_templates"]
exclude_patterns = ["_build"]
pygments_style = "sphinx"
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "style_nav_header_background": "#C0C0C0",
}
html_logo = "images/xtgeo-logo.png"


class ClassToc(AutoClassToc):
    """Render a class table without a transition inside the class directive."""

    def run(self):
        """Return the table nodes allowed inside the class directive."""
        return [
            node for node in super().run() if not isinstance(node, nodes.transition)
        ]


def setup(app):
    """Register the class table directive without overriding Sphinx autodoc."""
    # See https://github.com/kalekundert/autoclasstoc/issues/41.
    autoclasstoc.nodes.setup(app)
    app.add_config_value("autoclasstoc_sections", autoclasstoc_sections, "env")
    app.add_directive("autoclasstoc", ClassToc)
    app.connect("config-inited", load_static_assets)
