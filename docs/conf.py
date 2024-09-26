import os
import sys
sys.path.insert(0, os.path.abspath('../ETIA'))

# Project information
project = 'ETIA'
author = 'Antonios Ntroumpogiannis'
release = '0.2'

# Sphinx configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
]

autosummary_generate = True
add_module_names = False  # Avoid showing the module names in the class/method signatures
autodoc_member_order = 'bysource'  # Keep the order as in the source code

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
