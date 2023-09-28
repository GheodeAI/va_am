# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../src/va_am'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = 'va_am'
copyright = '2023, cosminmarina'
author = 'cosminmarina'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
	"sphinx.ext.viewcode",
	"sphinx.ext.autodoc",
	"sphinx.ext.autosummary",
	"sphinx.ext.napoleon"
]

add_module_names = False

html_theme_options = {"show_toc_level" : 2,
		      "navbar_start": ["navbar-logo"],
		      #"logo": {
		      #        "link": "https://climateintelligence.eu",
		      #},
		      "navbar_center": ["navbar-nav"],
		      "navbar_end": ["navbar-icon-links"],
		      "navbar_persistent": ["search-button"],
		      "primary_sidebar_end": ["sidebar-ethical-ads"],
		      "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink"],
		      "navbar_align": "left"}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_sidebars = {
    "**": ["sidebar-nav-bs", "sidebar-ethical-ads"]
}

#html_logo = "_static/CLINT.png"

#autodoc_member_order = 'bysource'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
