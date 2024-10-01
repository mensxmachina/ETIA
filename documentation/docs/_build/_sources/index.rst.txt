ETIA: A Comprehensive Automated Causal Discovery Library
=========================================================

ETIA (Αιτία (pronounced etía): "cause" in Greek) is a cutting-edge automated causal discovery library that extends beyond traditional methods to tackle complex, real-world problems. It automates the entire causal discovery process, from feature selection and causal structure learning to causal reasoning validation, making it a comprehensive solution for both researchers and practitioners.

Library Overview
----------------

ETIA provides:

- **Dimensionality Reduction** through the **Automated Feature Selection (AFS)** module, ensuring that only the most relevant variables are used, even in high-dimensional datasets.
- **Causal Structure Learning** via the **Causal Learning (CL)** module, automating the discovery of causal graphs that best fit the data.
- **Causal Reasoning Validation (CRV)**, which offers tools to compute confidence in discovered relationships and visualize causal paths for easy interpretation.

Unlike other causal discovery tools, **ETIA** offers a **fully automated pipeline** that optimizes each step of the process. This ensures that results are robust, interpretable, and reliable, whether you are a researcher exploring new hypotheses or an industrial practitioner making data-driven decisions.

Why ETIA is Unique
------------------

ETIA stands apart from other causal discovery libraries with the following key features:

- **End-to-End Automation**: ETIA automates the entire causal discovery process, combining various algorithms to find the best configuration for your dataset, unlike other libraries that rely on manual selection and tuning of algorithms.
- **Out-of-Sample Causal Tuning**: This method selects the best causal graph without the need for manually tuned parameters, making it highly suitable for unsupervised environments.
- **Confidence Estimation and Visualization**: ETIA goes beyond discovering causal graphs by estimating the confidence of each causal relationship through bootstrapping, and providing visualization tools for interpreting results.
- **Dimensionality Reduction with Causal Insight**: The AFS module applies advanced techniques to reduce the dimensionality of datasets without compromising on causal accuracy, even in high-dimensional data.
- **Handling Latent Variables**: ETIA can identify causal relationships even in datasets with hidden confounders, using algorithms like FCI and GFCI that most other libraries cannot handle.

Core Features
-------------

### 1. Automated Feature Selection (AFS)

The AFS module identifies the **Markov Boundary** of the outcome of interest, ensuring that only the most causally relevant features are selected. This reduces noise and redundancy, resulting in more accurate causal discovery.

**Why It Matters**: In datasets with hundreds of variables, AFS ensures that only the most important features are considered for causal analysis, making subsequent steps more efficient and accurate.

### 2. Causal Learning (CL)

ETIA’s CL module automates the discovery of causal structures using a variety of algorithms, which are automatically optimized to fit the dataset. Its **causal tuning** mechanism ensures the selection of the best causal structure without user intervention.

**Why It Matters**: ETIA’s automated pipeline selects the most appropriate algorithm for your dataset, saving time and ensuring the discovery of accurate causal structures, even in the presence of latent variables.

### 3. Causal Reasoning Validation (CRV)

CRV offers tools for evaluating the discovered causal graph, providing confidence estimates and comprehensive visualizations. It can answer specific causal queries, making it an invaluable tool for researchers and decision-makers.

**Why It Matters**: ETIA’s ability to compute confidence in causal relationships ensures that users can trust the discovered causal graphs, backed by statistical validation and clear visualizations.

Modules Overview
----------------

Explore the core modules of ETIA:

- `Automated Feature Selection (AFS) <AFS>`_
- `Causal Learning (CL) <CausalLearning>`_
- `Causal Reasoning Validation (CRV) <CRV>`_

.. toctree::
   :maxdepth: 2
   :caption: Menu

   installation_guide
   example_usage
   experiments
   afs_index
   cl_index
   crv_index
