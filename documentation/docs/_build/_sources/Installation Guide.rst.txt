Installation Guide
==================

This page provides detailed installation instructions for AutoCD.

Requirements
------------
- Python 3.6 or newer
- pip (Python package installer)
- causalnex==0.12.1
- cdt==0.6.0
- joblib==1.2.0
- JPype1==1.5.0
- networkx==3.2.1
- numpy==1.22.4
- pandas==1.4.2
- pgmpy==0.1.25
- pywhy_graphs==0.1.0
- scikit_learn==1.4.1.post1
- setuptools==68.1.2
- tigramite==5.2.3.1

Installing AutoCD
-----------------
You can install AutoCD using pip:

.. code-block:: bash

   pip install AutoCD

Installing from Source
----------------------
If you have access to the source code, you can install AutoCD directly from the source using the setup.py script:

1. First, clone the repository or download the source code:

   .. code-block:: bash

      git clone https://github.com/droubo/AutoCD.git
      cd AutoCD

2. Once you are in the project directory, run the following command:

   .. code-block:: bash

      python setup.py install

   This command will build and install the package locally.

Verify Installation
-------------------
To verify that AutoCD has been installed correctly, run the following command:

.. code-block:: python

   import AutoCD
   print(AutoCD.version())
