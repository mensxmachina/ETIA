=========================
Installation Guide
=========================

Welcome to the installation guide for ETIA, a comprehensive automated causal discovery library. Follow the steps below to install the library and its dependencies.

Prerequisites
-------------
Before installing ETIA, ensure that you have the following dependencies:

- **Python 3.8++**
- **Java 17** (required for Tetrad algorithms in the Causal Learning module)
- **R 4.4+++** (required for some feature selection algorithms in the AFS module)

You can download and install these dependencies from their official websites:

- [Python](https://www.python.org/downloads/)
- [Java](https://www.oracle.com/java/technologies/javase-jdk17-downloads.html)
- [R](https://www.r-project.org/)

ETIA Installation
-----------------

### Installing via PyPi (Upcoming)

Once ETIA is available on PyPi, you will be able to install it directly using `pip`:

.. code-block:: bash

    pip install etia

### Installing from Source

To install ETIA from the source code, follow the steps below:

1. Clone the repository:

.. code-block:: bash

    git clone <repository-url>
    cd etia

2. Install the required dependencies:

.. code-block:: bash

    pip install -r requirements.txt

3. Compile the necessary components:

.. code-block:: bash

    make all

Java and R Configuration
------------------------
For using Tetrad algorithms and certain feature selection algorithms, ensure that Java and R are correctly installed.

### Setting up Java:

1. Install Java 17 from the [official website](https://www.oracle.com/java/technologies/javase-jdk17-downloads.html).
2. Ensure that the `JAVA_HOME` environment variable is set:

.. code-block:: bash

    export JAVA_HOME=/path/to/java

### Setting up R:

1. Install R 4.4+ from the [official website](https://www.r-project.org/).
2. Make sure that R is in your system’s PATH:

.. code-block:: bash

    R --version

3. Install MxM package (this part may take a while)
Note: Depending on the OS you may need to install CMake and GSL (GNU Scientific Library)

.. code-block:: bash

    Rscipt --vanilla "install.packages("MXM", repos = "http://cran.us.r-project.org")"

Verify Installation
-------------------
After installing the library, you can verify the installation by importing the ETIA modules:

.. code-block:: python

    import ETIA.AFS as AFS

    afs = AFS()


If no errors occur, the installation was successful.

Next Steps
----------
Once you have installed ETIA, you can proceed to explore its functionalities. Check out the **Example Usage** section to learn how to use the library effectively.
