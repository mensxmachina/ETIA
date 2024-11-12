import os
import sys
sys.path.insert(0, os.path.abspath('../../.'))

# Project information
project = 'ETIA'
author = 'Antonios Ntroumpogiannis'
release = '1.0'

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
autodoc_default_flags = ['members', 'undoc-members', 'show-inheritance']
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
