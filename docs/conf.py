# conf.py - Configuration file for Sphinx documentation

import os
import sys
import datetime

# -- Path setup & datasafari import ------------------------------------------

# Add project directory to sys.path to find datasafari package
sys.path.insert(0, os.path.abspath('../'))

import datasafari

# -- Project information -----------------------------------------------------

project = 'DataSafari'
copyright = f"{datetime.datetime.now().year}, George Dreemer"
author = "George Dreemer"

# The short X.Y version
version = datasafari.__version__
# The full version, including alpha/beta/rc tags
release = datasafari.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.viewcode', 'sphinx.ext.napoleon',
    'sphinx.ext.autosummary', 'sphinx_favicon', 'sphinxext.opengraph',
    'sphinx_prompt', 'sphinx_copybutton', 'sphinx.ext.mathjax',
    'sphinxemoji.sphinxemoji', 'sphinx.ext.githubpages'
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
source_suffix = '.rst'
master_doc = 'index'
language = 'en'
pygments_style = 'sphinx'

# Metadata of project (opengraph)
ogp_site_url = 'https://www.datasafari.dev/docs'
ogp_site_name = 'DataSafari Documentation'
ogp_image = '_static/thumbs/ds-branding-thumb-main-docs.png'
ogp_description_length = 300
ogp_type = 'website'
ogp_description = 'DataSafari simplifies complex data science tasks into straightforward, powerful one-liners.'

ogp_social_cards = {
    "enable": False
}
# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'
html_theme_options = {
    "light_logo": "logos/ds-branding-logo-big-lightmode.png",
    "dark_logo": "logos/ds-branding-logo-big-darkmode.png",
}
html_static_path = ['_static']
html_favicon = '_static/favicons/favicon.ico'
html_title = 'DataSafari Documentation'

# -- Options for LaTeX output ------------------------------------------------

latex_documents = [
    (master_doc, 'datasafari.tex', 'Data Safari Documentation',
     'George Dreemer', 'manual'),
]

# -- Options for manual page output ------------------------------------------

man_pages = [
    (master_doc, 'datasafari', 'Data Safari Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (master_doc, 'datasafari', 'Data Safari Documentation',
     author, 'datasafari', 'One line description of project.', 'Miscellaneous'),
]

# -- Sphinx extension settings -----------------------------------------------

# Ensure function signature parameters are displayed in the documentation
python_maximum_signature_line_length = 1

# -- End of conf.py ----------------------------------------------------------
