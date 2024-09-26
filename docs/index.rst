Welcome to ETIA's documentation!
=================================

We introduce ETIA, a system designed for Automated Causal Discovery (AutoCD). AutoCD aims to fully automate the application of causal discovery and causal reasoning methods, delivering comprehensive causal information and answering user queries. ETIA performs dimensionality reduction, causal structure learning, and causal reasoning, making sophisticated causal analysis accessible to all, including non-experts.


Introduction to ETIA
---------------------
ETIA stands for an automated system that simplifies causal discovery and reasoning, processes traditionally requiring extensive expertise. ETIA makes these sophisticated methods accessible to non-experts by automating the analysis pipeline and delivering insights comparable to those of a human analyst.

**What is Automated Causal Discovery (AutoCD)?**
AutoCD refers to systems that automate the application of causal discovery and reasoning methods. The primary goal is to provide all causal information and answers to user queries, mimicking the output of an expert analyst. ETIA embodies this concept by optimizing data representation and causal discovery algorithms to derive the best-fitting causal models.

**Core Features of ETIA**:
- **Dimensionality Reduction**: Uses automated feature selection to handle high-dimensional data, reducing the problem space to essential variables.

- **Causal Structure Learning**: Employs a variety of state-of-the-art causal discovery algorithms and optimizes their hyper-parameters to learn the best causal model.

- **Causal Reasoning and Visualization**: Offers tools to answer causal queries, calculate confidence measures, and visualize causal relationships.

**Modules in ETIA**:
1. **Automated Feature Selection (AFS)**: Identifies the minimal set of variables necessary for accurate causal modeling.

2. **Causal Learning (CL)**: Optimizes the causal discovery process, selecting the best algorithm and parameters.

3. **Causal Reasoning and Visualization (CRV)**: Provides insights and visualizations of the causal model, including confidence measures and adjustment sets.

**Use Case Example**:
ETIA has been benchmarked on synthetic datasets, showcasing its ability to handle various causal discovery problems. A detailed use case example illustrates ETIA's functionalities and performance.

Getting Started
---------------
To start using ETIA, follow the installation instructions provided in the next section. ETIA's user-friendly interface and automated processes make it accessible for users with varying levels of expertise in causal discovery.

Contents:

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   Installation Guide
   Use Cases
   Experiments
   AFS
   CausalLearning
   CRV
   simulation

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
