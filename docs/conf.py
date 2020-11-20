#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import sys
import os

import xtgeo

print("VERSION is {}".format(xtgeo.__version__))
print("SYS PATH is {}".format(sys.path))
print("XTGEO file is {}".format(xtgeo.__file__))
# -- General configuration ---------------------------------------------

# The short X.Y version.
version = xtgeo.__version__
# The full version, including alpha/beta/rc tags.
release = xtgeo.__version__

extensions = [
    "sphinxcontrib.apidoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "recommonmark",
]

apidoc_module_dir = "../src/xtgeo"
apidoc_output_dir = "apiref"
apidoc_excluded_paths = ["tests"]
apidoc_separate_modules = True
apidoc_module_first = True
apidoc_extra_args = ["-H", "XTGeo API description"]

# The suffix of source filenames.
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "xtgeo"
copyright = "Equinor"


# Sort members by input order in classes
autodoc_member_order = "bysource"
autodoc_default_flags = ["members", "show_inheritance"]

exclude_patterns = ["_build"]

pygments_style = "sphinx"

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "style_nav_header_background": "#C0C0C0",
}


html_logo = "images/xtgeo-logo.png"

# The name of an image file (within the static path) to use as favicon
# of the docs.  This file should be a Windows icon file (.ico) being
# 16x16 or 32x32 pixels large.
# html_favicon = None


# If not '', a 'Last updated on:' timestamp is inserted at every page
# bottom, using the given strftime format.
# html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
# html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names
# to template names.
# html_additional_pages = {}

# If false, no module index is generated.
# html_domain_indices = True

# If false, no index is generated.
# html_use_index = True

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, links to the reST sources are added to the pages.
# html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer.
# Default is True.
# html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer.
# Default is True.
# html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages
# will contain a <link> tag referring to it.  The value of this option
# must be the base URL from which the finished HTML is served.
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = "xtgeodoc"


# -- Options for LaTeX output ------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #'preamble': '',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass
# [howto/manual]).
latex_documents = [
    ("index", "xtgeo.tex", "xtgeo Documentation", "Jan C. Rivenaes", "manual")
]

# The name of an image file (relative to this directory) to place at
# the top of the title page.
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings
# are parts, not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_domain_indices = True


# -- Options for manual page output ------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [("index", "xtgeo", "XTGeo Documentation", ["Equinor"], 1)]

# If true, show URL addresses after external links.
# man_show_urls = False


# -- Options for Texinfo output ----------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        "index",
        "xtgeo",
        "XTGeo Documentation",
        "Equinor",
        "xtgeo",
        "Library for subsurface reservoir modelling",
        "Miscellaneous",
    )
]

# Documents to append as an appendix to all manuals.
# texinfo_appendices = []

# If false, no module index is generated.
# texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
# texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
# texinfo_no_detailmenu = False
