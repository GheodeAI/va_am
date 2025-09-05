# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
#sys.path.insert(0, os.path.abspath('../src/va_am'))
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../src'))


# -- Project information -----------------------------------------------------

project = 'va_am'
copyright = '2023, cosminmarina'
author = 'cosminmarina'
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
	"sphinx.ext.viewcode",
	"sphinx.ext.autodoc",
	"sphinx.ext.autosummary",
	"sphinx.ext.napoleon",
	'sphinx_licenseinfo',
    'sphinx_copybutton'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

add_module_names = False

html_theme_options = {"show_toc_level" : 4,
		      "navbar_start": ["navbar-logo"],
		      "logo": {
      				"image_light": "_static/va-am.png",
      				"image_dark": "_static/va-am2.png",
					},
		      "navbar_center": ["navbar-nav"],
		      "navbar_end": ["navbar-icon-links"],
		      "navbar_persistent": ["search-button"],
		      "primary_sidebar_end": ["sidebar-ethical-ads"],
		      "secondary_sidebar_items": ["page-toc", "edit-this-page", "sourcelink", "custom-citation"],
		      "navbar_align": "left"}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

autodoc_member_order = 'bysource'

html_sidebars = {
    "**": ["sidebar-nav-bs", "sidebar-ethical-ads"]
}

html_theme = 'pydata_sphinx_theme'
#html_logo = '_static/va-am.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
templates_path = ['_templates']

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"
copybutton_here_doc_delimiter = "EOT"
copybutton_selector = "div.highlight pre"

def setup(app):
    app.add_css_file('custom.css')