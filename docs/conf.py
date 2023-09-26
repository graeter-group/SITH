# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SITH'
copyright = '2022, Mikaela Farrugia, Daniel Sucerquia'
author = 'Mikaela Farrugia, Daniel Sucerquia'
release = '0.1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.coverage', 'sphinx.ext.napoleon']#, 'sphinx_rtd_theme']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
import os
import sys
sys.path.insert(0, os.path.abspath('../src/'))



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#remote_theme: rundocs/jekyll-rtd-theme

# html_theme = 'sphinx_rtd_theme'
# html_theme_options = {
#     'display_version' : True,
#     'style_external_links' : True
# }
html_static_path = ['_static']
