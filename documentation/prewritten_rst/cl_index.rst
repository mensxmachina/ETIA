Causal Learning (CL)
=====================

The Causal Learning (CL) module is a core component of the **ETIA** framework, designed to automate the discovery of causal relationships in complex, high-dimensional datasets. It is responsible for learning a causal graph from the features selected by the Automated Feature Selection (AFS) module. This causal graph captures the directed dependencies between variables, facilitating further tasks such as causal reasoning and prediction.

The CL module optimizes the entire causal discovery pipeline by exploring a configuration space of algorithms and hyperparameters. It searches for the best-fitting causal model based on the available data, ensuring that the discovered relationships are accurate and interpretable. By supporting a wide variety of causal discovery algorithms, independence tests, and scoring functions, CL can be adapted to different data types (continuous, mixed, categorical) and assumptions about the underlying system (e.g., causal sufficiency, latent confounders).

Core Objectives
-----------------

The main goals of the CL module include:

- Learning an accurate causal graph from the selected features.
- Optimizing the causal discovery process by searching over various algorithms and configurations.
- Supporting different data types, including continuous, categorical, and mixed variables.
- Handling both causally sufficient and insufficient systems (i.e., with or without latent confounders).
- Allowing flexible integration with downstream reasoning and visualization tasks.

How CL Works
---------------

The CL module operates in three stages:

1. **Causal Configuration Generator (CG)**:
   The generator explores the configuration space of causal discovery algorithms, independence tests, and scoring functions. It selects appropriate configurations based on the characteristics of the input data, including the type (continuous, mixed, or categorical) and any assumptions regarding causal sufficiency.

2. **Causal Discovery**:
   Once the best configuration is selected, the CL module applies the causal discovery algorithm to the data. The output is a causal graph that captures the directed dependencies between variables. This graph can be further analyzed to identify key causal relationships, intervention points, or adjustment sets.

3. **Causal Evaluation**:
   The discovered causal graphs are evaluated using scoring functions to assess their fit to the data. The evaluation considers the accuracy of the learned structure in representing the true causal relationships.

Available Algorithms
---------------------

The CL module offers a variety of causal discovery algorithms, each suited for different data types and assumptions. These algorithms are listed below:

+--------------------+-------------------+--------------------------------------------------+
| Algorithm          | Data Type         | Description                                      |
+====================+===================+==================================================+
| **PC**             | Continuous, Mixed,| A constraint-based algorithm that uses           |
|                    | Categorical       | conditional independence tests to learn          |
|                    |                   | the causal structure. Assumes causal sufficiency.|
+--------------------+-------------------+--------------------------------------------------+
| **CPC**            | Continuous, Mixed,| A variant of PC that improves stability by       |
|                    | Categorical       | handling non-faithful distributions.             |
+--------------------+-------------------+--------------------------------------------------+
| **FGES**           | Continuous, Mixed,| A score-based algorithm that does not assume     |
|                    | Categorical       | causal sufficiency. Suitable for high-           |
|                    |                   | dimensional data.                                |
+--------------------+-------------------+--------------------------------------------------+
| **GFCI**           | Continuous, Mixed,| A hybrid algorithm combining constraint-based    |
|                    | Categorical       | and score-based methods. Allows for latent       |
|                    |                   | confounders.                                     |
+--------------------+-------------------+--------------------------------------------------+
| **LiNGAM**         | Continuous        | A linear non-Gaussian causal discovery method.   |
|                    |                   | Suitable for discovering linear causal           |
|                    |                   | relationships in continuous data.                |
+--------------------+-------------------+--------------------------------------------------+
| **DirectLiNGAM**   | Continuous        | A fast variant of the LiNGAM algorithm that      |
|                    |                   | performs well on high-dimensional datasets.      |
+--------------------+-------------------+--------------------------------------------------+
| **NOTEARS**        | Continuous        | An optimization-based algorithm that learns      |
|                    |                   | causal structure using least squares and         |
|                    |                   | L1-regularization.                               |
+--------------------+-------------------+--------------------------------------------------+

Available Independence Tests
------------------------------

The CL module supports a range of conditional independence tests, enabling flexibility in testing relationships between variables across different data types:

+--------------------+-------------------+--------------------------------------------------+
| Test Name          | Data Type         | Description                                      |
+====================+===================+==================================================+
| **FisherZ**        | Continuous        | A widely used test for continuous data.          |
+--------------------+-------------------+--------------------------------------------------+
| **CG-LRT**         | Mixed             | Conditional Gaussian Likelihood Ratio Test for   |
|                    |                   | mixed data (continuous and categorical).         |
+--------------------+-------------------+--------------------------------------------------+
| **DG-LRT**         | Mixed             | Discrete Gaussian Likelihood Ratio Test for      |
|                    |                   | mixed data (discrete and Gaussian).              |
+--------------------+-------------------+--------------------------------------------------+
| **Chi-Square**     | Categorical       | Test for independence in categorical data.       |
+--------------------+-------------------+--------------------------------------------------+
| **G-Square**       | Categorical       | Another test for independence in categorical     |
|                    |                   | data, based on the G-statistic.                  |
+--------------------+-------------------+--------------------------------------------------+
| **ParCor**         | Continuous        | Test based on partial correlation.               |
+--------------------+-------------------+--------------------------------------------------+
| **CMIknn**         | Continuous        | Conditional Mutual Information test using        |
|                    |                   | nearest neighbors.                               |
+--------------------+-------------------+--------------------------------------------------+

Available Scoring Functions
----------------------------

To evaluate the causal graphs, the CL module includes several scoring functions, allowing flexibility in selecting the most appropriate metric for the data:

+--------------------+-------------------+------------------------------------------------+
| Score Name         | Data Type         | Description                                    |
+====================+===================+================================================+
| **SEM BIC Score**  | Continuous        | Bayesian Information Criterion for Structural  |
|                    |                   | Equation Models. Suitable for continuous data. |
+--------------------+-------------------+------------------------------------------------+
| **BDeu**           | Categorical       | Bayesian Dirichlet equivalent uniform score    |
|                    |                   | for categorical data.                          |
+--------------------+-------------------+------------------------------------------------+
| **CG-BIC**         | Mixed             | BIC score for mixed data models (continuous    |
|                    |                   | and categorical).                              |
+--------------------+-------------------+------------------------------------------------+
| **DG-BIC**         | Mixed             | BIC score for discrete Gaussian models.        |
+--------------------+-------------------+------------------------------------------------+

### CL Output
The output of the CL module includes:

- A causal graph representing the learned structure between variables.
- The best-performing causal discovery configuration, including the selected algorithm, independence test, and scoring function.

By providing an optimized causal discovery pipeline, the CL module ensures that the causal relationships discovered are both accurate and interpretable, facilitating further analysis and reasoning.

Main Class
---------------
The main entry point for using the CL module is the `CausalLearner` class. This class allows users to configure and run the causal discovery process, selecting from a variety of algorithms, tests, and scoring functions. The causal graphs generated can then be passed on for downstream reasoning or visualization tasks.

.. autoclass:: ETIA.CausalLearning.CausalLearner
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:



Helper Classes
---------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   CausalLearning.algorithms.causalnex_algorithm.rst
   CausalLearning.algorithms.cdt_algorithms.rst
   CausalLearning.algorithms.jar_files.rst
   CausalLearning.algorithms.rst
   CausalLearning.algorithms.tetrad_algorithm.rst
   CausalLearning.algorithms.tigramite_algorithm.rst
   CausalLearning.CausalModel.rst
   CausalLearning.CDHPO.OCT.causal_graph_utils.rst
   CausalLearning.CDHPO.OCT.rst
   CausalLearning.CDHPO.rst
   CausalLearning.configurations.rst
   CausalLearning.data.rst
   CausalLearning.model_validation_protocols.kfold.rst
   CausalLearning.model_validation_protocols.rst
   CausalLearning.regressors.rst
   CausalLearning.utils.oct_functions.rst
   CausalLearning.utils.rst
