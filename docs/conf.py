# noqa # pylint: skip-file
from datetime import date

import xtgeo
from autoclasstoc import PublicMethods

version = xtgeo.__version__
release = xtgeo.__version__
project = "xtgeo"
current_year = date.today().year
copyright = "Equinor 2019 - " + str(current_year) + f" (XTGeo release {release})"

extensions = [
    "myst_parser",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx_toolbox.more_autodoc.typehints",
    "sphinx_toolbox.more_autosummary",
    "autoclasstoc",
]

autosummary_generate = True

autoclass_content = "both"


class RemainingPublicMethods(PublicMethods):
    # skip dunder methods
    def predicate(self, name, attr, meta):
        return super().predicate(name, attr, meta) and not name.startswith("__")


autoclasstoc_sections = [
    "public-attrs",
    "public-methods",
]

autodoc_typehints = "description"

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
