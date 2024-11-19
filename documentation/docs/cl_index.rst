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
-----------------------

The CL module offers a variety of causal discovery algorithms, each suited for different data types and assumptions. These algorithms are listed below:

+--------------------+----------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **Algorithm**      | **Data Type**              | **Description**                                                                                                                                            |
+====================+============================+============================================================================================================================================================+
| **PC**             | Continuous, Mixed,         | A constraint-based algorithm that uses conditional independence tests to learn the causal structure. Assumes causal sufficiency. Supports continuous,      |
|                    | Categorical                | mixed, and categorical data.                                                                                                                               |
+--------------------+----------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **CPC**            | Continuous, Mixed,         | A variant of the PC algorithm that improves stability by handling non-faithful distributions. Assumes causal sufficiency. Supports continuous, mixed,      |
|                    | Categorical                | and categorical data.                                                                                                                                      |
+--------------------+----------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **FGES**           | Continuous, Mixed,         | A score-based algorithm that does not assume causal sufficiency. Suitable for high-dimensional data. Utilizes various scoring functions like SEM BIC       |
|                    | Categorical                | Score, BDeu, Discrete BIC, CG BIC, and DG BIC.                                                                                                             |
+--------------------+----------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **FCI**            | Continuous, Mixed,         | A constraint-based algorithm that accounts for latent confounders. Uses conditional independence tests to infer the causal structure. Supports continuous, |
|                    | Categorical                | mixed, and categorical data.                                                                                                                               |
+--------------------+----------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **FCI-Max**        | Continuous, Mixed,         | An extension of the FCI algorithm that maximizes certain criteria for improved causal discovery. Assumes the presence of latent variables. Supports        |
|                    | Categorical                | continuous, mixed, and categorical data.                                                                                                                   |
+--------------------+----------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **RFCI**           | Continuous, Mixed,         | A relaxed version of the FCI algorithm that offers faster performance with slightly relaxed constraints. Assumes the presence of latent variables. Supports|
|                    | Categorical                | continuous, mixed, and categorical data.                                                                                                                   |
+--------------------+----------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **GFCI**           | Continuous, Mixed,         | A hybrid algorithm combining constraint-based and score-based methods. Allows for latent confounders and utilizes various conditional independence tests.  |
|                    | Categorical                | Supports continuous, mixed, and categorical data.                                                                                                          |
+--------------------+----------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **CFCI**           | Continuous, Mixed,         | Combines features of the FCI and RFCI algorithms to enhance causal discovery in the presence of latent variables. Supports continuous, mixed, and          |
|                    | Categorical                | categorical data.                                                                                                                                          |
+--------------------+----------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **sVAR-FCI**       | Continuous, Mixed,         | A time-series variant of the FCI algorithm that accounts for temporal dependencies. Supports time series data along with continuous, mixed, and            |
|                    | Categorical (Time Series)  | categorical data.                                                                                                                                          |
+--------------------+----------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **svargFCI**       | Continuous, Mixed,         | An extension of sVAR-FCI that incorporates additional scoring functions like SEM BIC Score, BDeu, Discrete BIC, CG BIC, and DG BIC for enhanced causal     |
|                    | Categorical (Time Series)  | discovery in time-series data.                                                                                                                             |
+--------------------+----------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **PCMCI**          | Continuous, Mixed,         | A time-series causal discovery algorithm that does not assume causal sufficiency. Utilizes conditional mutual information tests and various                |
|                    | Categorical (Time Series)  |  correlation-based methods like ParCor, RobustParCor, GPDC, CMIknn, ParCorrWLS, Gsquared, CMIsymb, and RegressionCI.                                       |
+--------------------+----------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **PCMCI+**         | Continuous, Mixed,         | An enhanced version of PCMCI with improved handling of time lags and dependencies. Utilizes the same set of conditional mutual information tests and       |
|                    | Categorical (Time Series)  | correlation-based methods as PCMCI.                                                                                                                        |
+--------------------+----------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **LPCMCI**         | Continuous, Mixed,         | A latent-variable variant of PCMCI that accounts for unobserved confounders. Utilizes conditional mutual information tests and correlation-based           |
|                    | Categorical (Time Series)  | methods similar to PCMCI.                                                                                                                                  |
+--------------------+----------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **SAM**            | Continuous, Mixed          | A neural network-based causal discovery algorithm that does not assume causal sufficiency. Includes parameters like learning rate, regularization,         |
|                    |                            | hidden neurons, training/testing epochs, batch size, and loss type.                                                                                        |
+--------------------+----------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+
| **NOTEARS**        | Continuous, Mixed,         | An optimization-based algorithm that learns causal structures using least squares and L1-regularization. Assumes causal sufficiency. Supports continuous,  |
|                    | Categorical                | mixed, and categorical data.                                                                                                                               |
+--------------------+----------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------+

Available Independence Tests
--------------------------------

The CL module supports a range of conditional independence tests, enabling flexibility in testing relationships between variables across different data types:

+--------------------+-------------------+--------------------------------------------------+
| **Test Name**      | **Data Type**     | **Description**                                  |
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
| **RobustParCor**   | Continuous        | A robust version of the partial correlation test,|
|                    |                   | less sensitive to outliers.                      |
+--------------------+-------------------+--------------------------------------------------+
| **GPDC**           | Continuous        | Gaussian Process-based Dependency Criterion.     |
+--------------------+-------------------+--------------------------------------------------+
| **CMIknn**         | Continuous        | Conditional Mutual Information test using        |
|                    |                   | nearest neighbors.                               |
+--------------------+-------------------+--------------------------------------------------+
| **ParCorrWLS**     | Continuous        | Partial Correlation with Weighted Least Squares. |
+--------------------+-------------------+--------------------------------------------------+
| **Gsquared**       | Mixed             | G-squared test adapted for mixed data types.     |
+--------------------+-------------------+--------------------------------------------------+
| **CMIsymb**        | Mixed             | Symmetric Conditional Mutual Information test.   |
+--------------------+-------------------+--------------------------------------------------+
| **RegressionCI**   | Mixed             | Regression-based Conditional Independence test.  |
+--------------------+-------------------+--------------------------------------------------+

Available Scoring Functions
--------------------------------

To evaluate the causal graphs, the CL module includes several scoring functions, allowing flexibility in selecting the most appropriate metric for the data:

+--------------------+-------------------+------------------------------------------------+
| **Score Name**     | **Data Type**     | **Description**                                |
+====================+===================+================================================+
| **SEM BIC Score**  | Continuous        | Bayesian Information Criterion for Structural  |
|                    |                   | Equation Models. Suitable for continuous data. |
+--------------------+-------------------+------------------------------------------------+
| **BDeu**           | Categorical       | Bayesian Dirichlet equivalent uniform score    |
|                    |                   | for categorical data.                          |
+--------------------+-------------------+------------------------------------------------+
| **Discrete BIC**   | Categorical       | Bayesian Information Criterion for discrete    |
|                    |                   | data models.                                   |
+--------------------+-------------------+------------------------------------------------+
| **CG-BIC**         | Mixed             | BIC score for mixed data models (continuous    |
|                    |                   | and categorical).                              |
+--------------------+-------------------+------------------------------------------------+
| **DG-BIC**         | Mixed             | BIC score for discrete Gaussian models.        |
+--------------------+-------------------+------------------------------------------------+
| **GFCI Score**     | Mixed             | Scoring function used by the GFCI algorithm to |
|                    |                   | evaluate causal structures.                    |
+--------------------+-------------------+------------------------------------------------+
| **svargFCI Score** | Mixed             | Enhanced scoring function for svargFCI with    |
|                    |                   | additional metrics like SEM BIC Score, BDeu,   |
|                    |                   | Discrete BIC, CG BIC, and DG BIC.              |
+--------------------+-------------------+------------------------------------------------+

Algorithm Parameters
--------------------------------


Each algorithm may have specific parameters that can be tuned to optimize performance based on the dataset and requirements. Below are the parameters for each available algorithm:

**PC Algorithm Parameters**


.. list-table:: **PC Algorithm Parameters**
   :widths: 20 20 60
   :header-rows: 1

   * - **Parameter**
     - **Type**
     - **Description**
   * - `ci_test`
     - string
     - Type of conditional independence test to use. Options: FisherZ, cg_lrt, dg_lrt, chisquare, gsquare.
   * - `stable`
     - boolean
     - Whether to use the stable version of the PC algorithm.

**CPC Parameters**


.. list-table:: **CPC Parameters**
   :widths: 20 20 60
   :header-rows: 1

   * - **Parameter**
     - **Type**
     - **Description**
   * - `ci_test`
     - string
     - Type of conditional independence test to use. Options: FisherZ, cg_lrt, dg_lrt, chisquare, gsquare.
   * - `stable`
     - boolean
     - Whether to use the stable version of the CPC algorithm.

**FGES Parameters**


.. list-table:: **FGES Parameters**
   :widths: 20 20 60
   :header-rows: 1

   * - **Parameter**
     - **Type**
     - **Description**
   * - `score`
     - string
     - Scoring function to use. Options: sem_bic_score, bdeu, discrete_bic, cg_bic, dg_bic.

**FCI Parameters**


.. list-table:: **FCI Parameters**
   :widths: 20 20 60
   :header-rows: 1

   * - **Parameter**
     - **Type**
     - **Description**
   * - `ci_test`
     - string
     - Type of conditional independence test to use. Options: FisherZ, cg_lrt, dg_lrt, chisquare, gsquare.
   * - `stable`
     - boolean
     - Whether to use a stable version of the FCI algorithm.

**FCI-Max Parameters**


.. list-table:: **FCI-Max Parameters**
   :widths: 20 20 60
   :header-rows: 1

   * - **Parameter**
     - **Type**
     - **Description**
   * - `ci_test`
     - string
     - Type of conditional independence test to use. Options: FisherZ, cg_lrt, dg_lrt, chisquare, gsquare.
   * - `stable`
     - boolean
     - Whether to use a stable version of the FCI-Max algorithm.

**RFCI Parameters**


.. list-table:: **RFCI Parameters**
   :widths: 20 20 60
   :header-rows: 1

   * - **Parameter**
     - **Type**
     - **Description**
   * - `ci_test`
     - string
     - Type of conditional independence test to use. Options: FisherZ, cg_lrt, dg_lrt, chisquare, gsquare.
   * - `stable`
     - boolean
     - Whether to use a stable version of the RFCI algorithm.

**GFCI Parameters**


.. list-table:: **GFCI Parameters**
   :widths: 20 20 60
   :header-rows: 1

   * - **Parameter**
     - **Type**
     - **Description**
   * - `ci_test`
     - string
     - Type of conditional independence test to use. Options: FisherZ, cg_lrt, dg_lrt, chisquare, gsquare.
   * - `stable`
     - boolean
     - Whether to use a stable version of the GFCI algorithm.
   * - `score`
     - string
     - Additional scoring functions (optional): sem_bic_score, bdeu, discrete_bic, cg_bic, dg_bic.

**CFCI Parameters**


.. list-table:: **CFCI Parameters**
   :widths: 20 20 60
   :header-rows: 1

   * - **Parameter**
     - **Type**
     - **Description**
   * - `ci_test`
     - string
     - Type of conditional independence test to use. Options: FisherZ, cg_lrt, dg_lrt, chisquare, gsquare.
   * - `stable`
     - boolean
     - Whether to use a stable version of the CFCI algorithm.

**sVAR-FCI Parameters**


.. list-table:: **sVAR-FCI Parameters**
   :widths: 20 20 60
   :header-rows: 1

   * - **Parameter**
     - **Type**
     - **Description**
   * - `ci_test`
     - string
     - Type of conditional independence test to use. Options: FisherZ, cg_lrt, dg_lrt, chisquare, gsquare.
   * - `stable`
     - boolean
     - Whether to use a stable version of the sVAR-FCI algorithm.
   * - `time_series`
     - boolean
     - Indicates if the data is a time series.

**svargFCI Parameters**


.. list-table:: **svargFCI Parameters**
   :widths: 20 20 60
   :header-rows: 1

   * - **Parameter**
     - **Type**
     - **Description**
   * - `ci_test`
     - string
     - Type of conditional independence test to use. Options: FisherZ, cg_lrt, dg_lrt, chisquare, gsquare.
   * - `stable`
     - boolean
     - Whether to use a stable version of the svargFCI algorithm.
   * - `score`
     - string
     - Additional scoring functions: sem_bic_score, bdeu, discrete_bic, cg_bic, dg_bic.
   * - `time_series`
     - boolean
     - Indicates if the data is a time series.

**PCMCI Parameters**


.. list-table:: **PCMCI Parameters**
   :widths: 20 20 60
   :header-rows: 1

   * - **Parameter**
     - **Type**
     - **Description**
   * - `ci_test`
     - string
     - Type of conditional independence test to use. Options: ParCor, RobustParCor, GPDC, CMIknn, ParCorrWLS, Gsquared, CMIsymb, RegressionCI.

**PCMCI+ Parameters**


.. list-table:: **PCMCI+ Parameters**
   :widths: 20 20 60
   :header-rows: 1

   * - **Parameter**
     - **Type**
     - **Description**
   * - `ci_test`
     - string
     - Type of conditional independence test to use. Options: ParCor, RobustParCor, GPDC, CMIknn, ParCorrWLS, Gsquared, CMIsymb, RegressionCI.

**LPCMCI Parameters**


.. list-table:: **LPCMCI Parameters**
   :widths: 20 20 60
   :header-rows: 1

   * - **Parameter**
     - **Type**
     - **Description**
   * - `ci_test`
     - string
     - Type of conditional independence test to use. Options: ParCor, RobustParCor, GPDC, CMIknn, ParCorrWLS, Gsquared, CMIsymb, RegressionCI.

**SAM Parameters**


.. list-table:: **SAM Parameters**
   :widths: 20 20 60
   :header-rows: 1

   * - **Parameter**
     - **Type**
     - **Description**
   * - `lr`
     - float
     - Learning rate. Options: 0.001, 0.01, 0.1.
   * - `dlr`
     - float
     - Decay learning rate. Options: 0.0001, 0.001, 0.01.
   * - `lambda1`
     - float
     - Regularization parameter 1. Options: 1, 10, 100.
   * - `lambda2`
     - float
     - Regularization parameter 2. Options: 0.0001, 0.001, 0.01.
   * - `nh`
     - int
     - Number of hidden neurons. Options: 10, 20, 50.
   * - `dnh`
     - int
     - Decay hidden neurons. Options: 100, 200, 300.
   * - `train_epochs`
     - int
     - Number of training epochs. Options: 1000, 3000, 5000.
   * - `test_epochs`
     - int
     - Number of testing epochs. Options: 500, 1000, 1500.
   * - `batch_size`
     - int
     - Batch size. Options: 50, 100, 200.
   * - `losstype`
     - string
     - Type of loss function. Options: fgan, gan, mse.

**NOTEARS Parameters**


.. list-table:: **NOTEARS Parameters**
   :widths: 20 20 60
   :header-rows: 1

   * - **Parameter**
     - **Type**
     - **Description**
   * - `max_iter`
     - int
     - Maximum number of iterations. Options: 100, 500, 1000.
   * - `h_tol`
     - float
     - Tolerance for convergence. Options: 1e-7, 1e-5, 1e-3.
   * - `threshold`
     - float
     - Threshold for edge inclusion. Options: 0.0, 0.5, 0.8.

Key Details
--------------------------------


- **Latent Variables Supported:**
  - **✓**: Supports latent (unobserved) variables.
  - **✕**: Does **not** support latent variables (causal sufficiency assumed).

- **Tests/Scores Used:**
  - **Conditional Independence Tests (`ci_test`):** Methods like FisherZ, CG-LRT, DG-LRT, Chi-Square, G-Square, ParCor, RobustParCor, GPDC, CMIknn, ParCorrWLS, Gsquared, CMIsymb, RegressionCI.
  - **Scores (`score`):** Metrics like SEM BIC Score, BDeu, Discrete BIC, CG-BIC, DG-BIC, GFCI Score, svargFCI Score.
  - **Additional Parameters:** Algorithms like SAM and NOTEARS have specific parameters relevant to their optimization and learning processes.

- **Data Type:**
  - **Continuous:** Numeric data without discrete categories.
  - **Mixed:** Combination of continuous and categorical data.
  - **Categorical:** Data with discrete categories.
  - **Time Series:** Data that includes temporal dependencies.

Notes
--------------------------------


- **Assumptions:**
  - **Causal Sufficiency:** If set to `False`, the algorithm accounts for potential latent variables.
  - **Assume Faithfulness:** Indicates whether the algorithm assumes the faithfulness condition holds, impacting its ability to recover the true causal graph.

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
   CausalLearning.algorithms.rst
   CausalLearning.algorithms.tetrad_algorithm.rst
   CausalLearning.algorithms.tigramite_algorithm.rst
   CausalLearning.CDHPO.OCT.causal_graph_utils.rst
   CausalLearning.CDHPO.OCT.rst
   CausalLearning.CDHPO.rst
   CausalLearning.configurations.rst
   CausalLearning.model_validation_protocols.kfold.rst
   CausalLearning.model_validation_protocols.rst
   CausalLearning.regressors.rst
   CausalLearning.utils.oct_functions.rst
   CausalLearning.utils.rst
