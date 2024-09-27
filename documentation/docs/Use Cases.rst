Use Cases
=========

1. Extract the causal relationships for a single temporal dataset
-----------------------------------------------------------------

In this use case, we demonstrate how to extract causal relationships from a single temporal dataset without focusing on a particular target variable. As such, the Automated Feature Selection (AFS) step is not required.

**Objective:** Identify causal relationships in a temporal dataset.

**Steps:**

1. **Load and preprocess data:**

    .. code-block:: python

        from data.Dataset import Dataset

        data = Dataset('temporal_dataset.csv', data_time_info={'n_lags': 0, 'time_lagged': False},
                       remove_constants=True, remove_missing=True)
        data.get_dataset().head()

    **Explanation:**
    Here, we load the dataset `temporal_dataset.csv` and preprocess it by removing constant columns and handling missing values. Since this is a temporal dataset, no time lags are added.

2. **Causal model learning:**

    .. code-block:: python

        from CausalLearning.CausalLearner import CausalLearner

        CL = CausalLearner(dataset=data, n_jobs=8)
        opt_conf, matrix_mec_graph, run_time, _ = CL.learn_model()
        CL.print_results()

    **Explanation:**
    The `CausalLearner` class is used to discover causal structures in the dataset. By leveraging parallel processing (`n_jobs=8`), the computation is expedited, and the model outputs the optimal configuration and the learned causal graph.

3. **Confidence estimation and visualization:**

    .. code-block:: python

        from CRV.confidence import calculate_confidence
        from CRV.visualization import Visualization

        edge_consistency, edge_similarity = calculate_confidence(data, opt_conf, 50)
        visualizer = Visualization(matrix_mec_graph, 'Temporal Data', 'Causal Relationships')
        visualizer.plot_cytoscape()
        visualizer.plot_edge_confidence(edge_similarity)

    **Explanation:**
    We calculate the confidence of each causal edge using bootstrap methods and visualize the results using the `Visualization` class, providing a clear depiction of the causal relationships and their confidence levels.


2. Extract the causal relationships for a single time-series dataset
---------------------------------------------------------------------

In this use case, we handle a time-series dataset, where time lags are created to account for temporal dependencies.

**Objective:** Identify causal relationships in a time-series dataset with time lags.

**Steps:**

1. **Load and preprocess data:**

    .. code-block:: python

        from data.Dataset import Dataset

        data = Dataset('time_series_dataset.csv', data_time_info={'n_lags': 5, 'time_lagged': True},
                       remove_constants=True, remove_missing=True)
        data.get_dataset().head()

    **Explanation:**
    We load the `time_series_dataset.csv` and preprocess it by adding time lags (`n_lags=5`) to capture temporal dependencies.

2. **Causal model learning:**

    .. code-block:: python

        from CausalLearning.CausalLearner import CausalLearner

        CL = CausalLearner(dataset=data, n_jobs=8)
        opt_conf, matrix_mec_graph, run_time, _ = CL.learn_model()
        CL.print_results()

    **Explanation:**
    Similar to the previous use case, we use the `CausalLearner` class to discover causal structures in the time-lagged dataset.

3. **Confidence estimation and visualization:**

    .. code-block:: python

        from CRV.confidence import calculate_confidence
        from CRV.visualization import Visualization

        edge_consistency, edge_similarity = calculate_confidence(data, opt_conf, 50)
        visualizer = Visualization(matrix_mec_graph, 'Time-Series Data', 'Causal Relationships')
        visualizer.plot_cytoscape()
        visualizer.plot_edge_confidence(edge_similarity)

    **Explanation:**
    We calculate the confidence of the causal edges and visualize the results, showing the causal relationships and their confidence scores.


3. Extract the causal relationships of a variable of interest for a single temporal dataset
-------------------------------------------------------------------------------------------

In this use case, we focus on extracting the causal relationships of a specific variable of interest in a temporal dataset, using the Automated Feature Selection (AFS) step.

**Objective:** Identify causal relationships related to a specific variable in a temporal dataset.

**Steps:**

1. **Load and preprocess data:**

    .. code-block:: python

        from data.Dataset import Dataset

        data = Dataset('temporal_dataset.csv', data_time_info={'n_lags': 0, 'time_lagged': False},
                       remove_constants=True, remove_missing=True)
        data.get_dataset().head()

    **Explanation:**
    We load the dataset `temporal_dataset.csv` and preprocess it by removing constant columns and handling missing values.

2. **Feature selection:**

    .. code-block:: python

        from AFS import AFS

        reduced_data = AFS(data)

    **Explanation:**
    The `AFS` class is used to select the most significant features that impact the variable of interest, reducing the dataset's dimensionality and enhancing computational efficiency.

3. **Causal model learning:**

    .. code-block:: python

        from CausalLearning.CausalLearner import CausalLearner

        CL = CausalLearner(dataset=reduced_data, n_jobs=8)
        opt_conf, matrix_mec_graph, run_time, _ = CL.learn_model()
        CL.print_results()

    **Explanation:**
    We use the `CausalLearner` class to discover causal structures in the reduced dataset.

4. **Confidence estimation and visualization:**

    .. code-block:: python

        from CRV.confidence import calculate_confidence
        from CRV.visualization import Visualization

        edge_consistency, edge_similarity = calculate_confidence(reduced_data, opt_conf, 50)
        visualizer = Visualization(matrix_mec_graph, 'Temporal Data', 'Variable of Interest')
        visualizer.plot_cytoscape()
        visualizer.plot_edge_confidence(edge_similarity)

    **Explanation:**
    We calculate the confidence of the causal edges and visualize the results, showing the causal relationships and their confidence scores.

5. **Causal querying:**

    .. code-block:: python

        from CRV.queries import one_path_anytype

        path = one_path_anytype(matrix_mec_graph, start='Variable_of_Interest', end='Target_Variable')
        print('The path from Variable_of_Interest to Target_Variable is: ', path)

    **Explanation:**
    We query the causal graph to identify specific paths related to the variable of interest, providing detailed causal insights.


4. Extract the causal relationships of a variable of interest for a single time-series dataset
------------------------------------------------------------------------------------------------

This use case focuses on identifying the causal relationships of a specific variable in a time-series dataset, with the addition of time lags and using AFS.

**Objective:** Identify causal relationships related to a specific variable in a time-series dataset with time lags.

**Steps:**

1. **Load and preprocess data:**

    .. code-block:: python

        from data.Dataset import Dataset

        data = Dataset('time_series_dataset.csv', data_time_info={'n_lags': 5, 'time_lagged': True},
                       remove_constants=True, remove_missing=True)
        data.get_dataset().head()

    **Explanation:**
    We load the `time_series_dataset.csv` and preprocess it by adding time lags (`n_lags=5`) to capture temporal dependencies.

2. **Feature selection:**

    .. code-block:: python

        from AFS import AFS

        reduced_data = AFS(data)

    **Explanation:**
    The `AFS` class is used to select the most significant features that impact the variable of interest, reducing the dataset's dimensionality and enhancing computational efficiency.

3. **Causal model learning:**

    .. code-block:: python

        from CausalLearning.CausalLearner import CausalLearner

        CL = CausalLearner(dataset=reduced_data, n_jobs=8)
        opt_conf, matrix_mec_graph, run_time, _ = CL.learn_model()
        CL.print_results()

    **Explanation:**
    We use the `CausalLearner` class to discover causal structures in the reduced and time-lagged dataset.

4. **Confidence estimation and visualization:**

    .. code-block:: python

        from CRV.confidence import calculate_confidence
        from CRV.visualization import Visualization

        edge_consistency, edge_similarity = calculate_confidence(reduced_data, opt_conf, 50)
        visualizer = Visualization(matrix_mec_graph, 'Time-Series Data', 'Variable of Interest')
        visualizer.plot_cytoscape()
        visualizer.plot_edge_confidence(edge_similarity)

    **Explanation:**
    We calculate the confidence of the causal edges and visualize the results, showing the causal relationships and their confidence scores.

5. **Causal querying:**

    .. code-block:: python

        from CRV.queries import one_path_anytype

        path = one_path_anytype(matrix_mec_graph, start='Variable_of_Interest', end='Target_Variable')
        print('The path from Variable_of_Interest to Target_Variable is: ', path)

    **Explanation:**
    We query the causal graph to identify specific paths related to the variable of interest, providing detailed causal insights.


5. Configure own hyperparameters for dataset analysis
-----------------------------------------------------

In this use case, we demonstrate how to configure specific hyperparameters for analyzing a dataset, such as choosing which algorithms to run and setting their hyperparameters.

**Objective:** Customize the analysis by setting specific hyperparameters for the causal discovery algorithms.

**Steps:**

1. **Load and preprocess data:**

    .. code-block:: python

        from data.Dataset import Dataset

        data = Dataset('custom_dataset.csv', data_time_info={'n_lags': 0, 'time_lagged': False},
                       remove_constants=True, remove_missing=True)
        data.get_dataset().head()

    **Explanation:**
    We load the dataset `custom_dataset.csv` and preprocess it as needed.

2. **Set custom hyperparameters:**

    .. code-block:: python

        custom_params = {
            'algorithm': 'pc',  # Use the PC algorithm for causal discovery
            'alpha': 0.05,      # Significance level for statistical tests
            'ind_test': 'FisherZ'    # Independence test
        }

    **Explanation:**
    We define a dictionary `custom_params` containing the desired hyperparameters for the causal discovery algorithm.

3. **Causal model learning with custom hyperparameters:**

    .. code-block:: python

        from CausalLearning.CausalLearner import CausalLearner
        from CausalLearner.configurations import Configurations

        conf =  Configurations(custom_params)
        CL = CausalLearner(dataset=data, configurations = conf, n_jobs=8)
        opt_conf, matrix_mec_graph, run_time, _ = CL.learn_model()
        CL.print_results()

    **Explanation:**
    We initialize the `CausalLearner` with the custom hyperparameters and run the causal discovery process.

4. **Confidence estimation and visualization:**

    .. code-block:: python

        from CRV.confidence import calculate_confidence
        from CRV.visualization import Visualization

        edge_consistency, edge_similarity = calculate_confidence(data, opt_conf, 50)
        visualizer = Visualization(matrix_mec_graph, 'Custom Data', 'Custom Hyperparameters')
        visualizer.plot_cytoscape()
        visualizer.plot_edge_confidence(edge_similarity)

    **Explanation:**
    We calculate the confidence of the causal edges and visualize the results, showing the causal relationships and their confidence scores.


6. Restart analysis with new hyperparameters or algorithms
----------------------------------------------------------

This use case demonstrates how to restart an analysis with new hyperparameters or causal discovery algorithms after completing an initial analysis.

**Objective:** Conduct a new analysis with updated parameters or algorithms.

**Steps:**

1. **Load and preprocess data:**

    .. code-block:: python

        from data.Dataset import Dataset

        data = Dataset('custom_dataset.csv', data_time_info={'n_lags': 0, 'time_lagged': False},
                       remove_constants=True, remove_missing=True)
        data.get_dataset().head()

    **Explanation:**
    We load the dataset `custom_dataset.csv` and preprocess it as needed.

2. **Set new hyperparameters:**

    .. code-block:: python

        new_params = {
            'algorithm': 'fci',  # Switch to the FCI algorithm for causal discovery
            'alpha': 0.01,       # Use a more stringent significance level
        }

    **Explanation:**
    We define a new dictionary `new_params` containing updated hyperparameters for the causal discovery algorithm.

3. **Causal model learning with new hyperparameters:**

    .. code-block:: python

        from CausalLearning.CausalLearner import CausalLearner

        CL = load('causal_learner') #Load previous analysis
        CL.configurations.add_parameters(new_params)
        opt_conf, matrix_mec_graph, run_time, _ = CL.learn_model_new()
        CL.print_results()

    **Explanation:**
    We initialize the `CausalLearner` with the new hyperparameters and run the causal discovery process again.

4. **Confidence estimation and visualization:**

    .. code-block:: python

        from CRV.confidence import calculate_confidence
        from CRV.visualization import Visualization

        edge_consistency, edge_similarity = calculate_confidence(data, opt_conf, 50)
        visualizer = Visualization(matrix_mec_graph, 'Custom Data', 'New Hyperparameters')
        visualizer.plot_cytoscape()
        visualizer.plot_edge_confidence(edge_similarity)

    **Explanation:**
    We calculate the confidence of the causal edges and visualize the results, showing the causal relationships and their confidence scores.


7. Add and test a custom causal discovery algorithm
---------------------------------------------------

This use case describes how to add a custom causal discovery algorithm to the library and test its performance against existing algorithms on multiple datasets.

**Objective:** Integrate a custom algorithm and evaluate its performance.

**Steps:**

1. **Add custom algorithm:**

    .. code-block:: python

        class CustomAlgorithm:
            def __init__(self, params):
                self.params = params

            def fit(self, data):
                # Implement the algorithm's fitting procedure
                pass

            def predict(self, data):
                # Implement the prediction procedure
                return causal_graph

    **Explanation:**
    We define a `CustomAlgorithm` class implementing the necessary methods (`fit` and `predict`) for the causal discovery process.

2. **Integrate into the library:**

    .. code-block:: python

        from CausalLearning.algorithms import register_algorithm

        register_algorithm('custom_algorithm', CustomAlgorithm)

    **Explanation:**
    We use the `register_algorithm` function to integrate the `CustomAlgorithm` into the AutoCD library.

3. **Evaluate performance on multiple datasets:**

    .. code-block:: python

        datasets = ['dataset1.csv', 'dataset2.csv', 'dataset3.csv']
        results = {}

        for dataset in datasets:
            data = Dataset(dataset, data_time_info={'n_lags': 0, 'time_lagged': False},
                           remove_constants=True, remove_missing=True)
            CL = CausalLearner(dataset=data, algorithm='custom_algorithm', params={'param1': value1})
            opt_conf, matrix_mec_graph, run_time, _ = CL.learn_model()
            results[dataset] = CL.results

    **Explanation:**
    We iterate over a list of datasets, applying the `CustomAlgorithm` and recording its performance metrics (e.g., runtime).

4. **Compare performance:**

    .. code-block:: python

        for dataset, runtime in results.items():
            print(f'Dataset: {dataset}, Runtime: {runtime} seconds')

    **Explanation:**
    We print the performance metrics for each dataset, allowing for comparison between the `CustomAlgorithm` and existing algorithms.


8. Get the best model between multiple algorithms
-------------------------------------------------

This use case demonstrates how to identify the best causal discovery model by comparing different algorithms using the AutoCD library.

**Objective:** Find the best causal model by comparing multiple causal discovery algorithms.

**Steps:**

1. **Load and preprocess data:**

    .. code-block:: python

        from data.Dataset import Dataset

        data = Dataset('comparison_dataset.csv', data_time_info={'n_lags': 0, 'time_lagged': False},
                       remove_constants=True, remove_missing=True)
        data.get_dataset().head()

    **Explanation:**
    We load and preprocess the dataset `comparison_dataset.csv`.

2. **Compare multiple algorithms:**

    .. code-block:: python

        from CausalLearning.CausalLearner import CausalLearner

        CL = CausalLearner(dataset=data, n_jobs=8)
        CL.learn_model()
        algorithms = ['pc', 'fci']
        best_config = CL.get_best_model_between_algorithms(algorithms)

        print('Best Configuration:', best_config)

    **Explanation:**
    We use the `get_best_model_between_algorithms` method to compare the performance of the 'pc', 'fci', and 'gcastle' algorithms, and find the best configuration.


9. Get the best model within a family of algorithms
---------------------------------------------------

This use case shows how to select the best causal discovery model within a family of algorithms by setting specific constraints.

**Objective:** Find the best causal model within a family of algorithms with specific constraints.

**Steps:**

1. **Load and preprocess data:**

    .. code-block:: python

        from data.Dataset import Dataset

        data = Dataset('family_comparison_dataset.csv', data_time_info={'n_lags': 0, 'time_lagged': False},
                       remove_constants=True, remove_missing=True)
        data.get_dataset().head()

    **Explanation:**
    We load and preprocess the dataset `family_comparison_dataset.csv`.

2. **Set constraints and find the best algorithm:**

    .. code-block:: python

        from CausalLearning.CausalLearner import CausalLearner

        CL = CausalLearner(dataset=data, n_jobs=8)
        CL.learn_model()
        best_family_config = CL.get_best_model_between_family(
            admit_latent_variables=True,
            assume_faithfulness=True,
            is_output_mec=False,
            accepts_missing_values=True
        )

        print('Best Family Configuration:', best_family_config)

    **Explanation:**
    We use the `get_best_model_between_family` method with specific constraints to find the best configuration within a family of algorithms.


10. Simulate a dataset for future experiments
--------------------------------------------

In this use case, we simulate a dataset to serve as a gold standard for future experiments.

**Objective:** Generate a synthetic dataset with known causal relationships for benchmarking.

**Steps:**

1. **Simulate dataset:**

    .. code-block:: python

        from Simulation import simulate_data

        tDag, tCpdag, tData, dag_pd, cpdag_pd, data_pd = simulate_data(n_nodes=5, n_samples=10000, avg_degree=3, max_degree=5, simulation_type='LeeHastie',
                          minCategories = 2, maxCategories = 4, percentDiscrete = 50, n_latents=0, n_lags=0,  seed=None)

    **Explanation:**
    We use the `simulate_data` function to generate a dataset based on a predefined causal model (`true_model`), specifying the number of samples.

2. **Save simulated dataset:**

    .. code-block:: python

        data_pd.to_csv('simulated_dataset.csv', index=False)

    **Explanation:**
    We save the simulated dataset to a CSV file for future use.

3. **Visualize true causal graph:**

    .. code-block:: python

        from CRV.visualization import Visualization

        visualizer = Visualization(tDag, 'Simulated Data', 'True Causal Model')
        visualizer.plot_cytoscape()

    **Explanation:**
    We visualize the true causal graph of the simulated dataset to serve as a reference for future experiments.

---

These use cases cover a wide range of scenarios demonstrating the flexibility and power of the AutoCD library. Each example is detailed with clear explanations and code snippets to guide users through the processes of causal discovery and analysis.
