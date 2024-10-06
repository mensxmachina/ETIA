Automated Feature Selection (AFS)
=================================

The Automated Feature Selection (AFS) module plays a critical role in automating the selection of relevant features from large, high-dimensional datasets. Its primary purpose is to identify the **Markov Boundary (Mb)** of a target variable. This significantly reduces the complexity of subsequent causal modeling or prediction tasks by focusing only on the variables most relevant to the target. By automatically selecting features and configuring prediction models, AFS streamlines the data analysis process and supports researchers in efficiently building robust, interpretable models.

AFS operates within the broader **ETIA** framework, designed for automated causal discovery and reasoning. It serves as the first step in a pipeline where dimensionality reduction is essential to improve the efficiency of downstream causal learning and predictive modeling tasks. The ability to handle various data types—continuous, categorical, and mixed—allows AFS to be adaptable to numerous problem domains. Its flexible architecture and seamless integration with different algorithms enable it to cater to both non-experts and experienced researchers.

Core Objectives
---------------

The core objectives of AFS include:

- Identifying the Markov boundary of the target variable(s).
- Selecting and configuring predictive models to assess feature relevance.
- Optimizing predictive performance while ensuring minimal feature selection.
- Handling large datasets efficiently, leveraging parallel processing.

How AFS Works
--------------

AFS employs a two-stage process:

1. **Predictive Configuration Generator (CG)**:
   This module generates multiple configurations of feature selection and predictive algorithms. It uses a predefined search space of hyperparameters tailored to each dataset and target. Feature selection algorithms like **FBED** and **SES** are configured and applied to identify features that are statistically equivalent or most relevant.

2. **Predictive Configuration Evaluator (CE)**:
   The CE assesses the performance of the generated configurations using cross-validation (5-fold by default). It measures the predictive performance based on metrics like the **Area Under the Receiver Operating Characteristic (AUROC)** for classification tasks or the **coefficient of determination (R²)** for regression tasks. The best-performing configuration is selected and applied to all data, returning the final set of selected features along with the optimal predictive model.

AFS Output
----------

The output of AFS includes:

- A set of selected features, which are the Markov boundaries of the target(s).
- The best-performing predictive model.
- An evaluation of the model’s predictive performance.
- The reduced dataset
AFS ensures that the selected features are not only statistically relevant but also optimized for prediction, improving both the efficiency and accuracy of subsequent analysis.


Available Algorithms
--------------------

The AFS module includes several feature selection and prediction algorithms. Below is a table summarizing the available algorithms and their hyperparameters:

**Feature Selection Algorithms**

+------------+-------------------+---------------------------------------+
| Algorithm  | Hyperparameters   | Default Values                        |
+============+===================+=======================================+
| FBED       | alpha             | [0.05, 0.01]                          |
|            | k                 | [3, 5]                                |
|            | ind_test_name     | ['testIndFisher']                     |
+------------+-------------------+---------------------------------------+
| SES        | alpha             | [0.05, 0.01]                          |
|            | k                 | [3, 5]                                |
|            | ind_test_name     | ['testIndFisher']                     |
+------------+-------------------+---------------------------------------+


**Predictive Algorithms**

+------------------+-------------------+-----------------------------------+
| Algorithm        | Hyperparameters   | Default Values                    |
+==================+===================+===================================+
| Random Forest    | n_estimators      | [50, 100]                         |
|                  | min_samples_leaf  | [0.1]                             |
|                  | max_features      | ['sqrt']                          |
+------------------+-------------------+-----------------------------------+
| Linear Regression| None              |                                   |
+------------------+-------------------+-----------------------------------+

Main Class
----------

The main entry point for using the AFS module is through the `AFS` class. This class provides methods to configure, execute feature selection, and manage results. It integrates preprocessing steps, feature selection, and predictive modeling in a seamless workflow.

.. autoclass:: ETIA.AFS.AFS
    :members:
    :undoc-members:
    :show-inheritance:
    :noindex:


Helper Classes
---------------

Below is a list of available classes in the AFS module:

### Helper Classes

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   AFS.feature_selector.rst
   AFS.oos.rst
   AFS.predictive_configurator.rst
   AFS.predictive_model.rst
   AFS.preprocessor.rst
   AFS.utils.rst


Each class is responsible for different aspects of the feature selection and prediction pipeline, ensuring flexibility and modularity in the system.

