# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NeuroP'
copyright = '2023, Andreas Andreou, Gert Cauwenberghs, Shantanu Chakrabartty, Zihao Chen, Nik Dennler, Michael Furlong, Chad Harper, Johannes Leugering, Adil Malik, Omowuyi Olajide, Riccardo Pignari, Zhili Xiao'
author = 'Andreas Andreou, Gert Cauwenberghs, Shantanu Chakrabartty, Zihao Chen, Nik Dennler, Michael Furlong, Chad Harper, Johannes Leugering, Adil Malik, Omowuyi Olajide, Riccardo Pignari, Zhili Xiao'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
exclude_patterns = []

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    "sphinx.ext.napoleon",
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
]

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
html_title = 'NeuroP library for neuromorphic optimization'
