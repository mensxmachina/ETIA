Beginner Example
==================

This demo walks through the full pipeline of feature selection, causal discovery, and causal reasoning.
We will use the AFS module to perform feature selection, then pass the reduced dataset to CL for causal discovery,
and finally use CRV for causal reasoning and validation on the learned causal model.

**Note:** Ensure that Cytoscape is open before running the visualization steps in Step 6.

Step 1: Import Required Modules
-------------------------------

.. code-block:: python

    import pandas as pd
    from ETIA.AFS import AFS
    from ETIA.CausalLearning import CausalLearner

    # Additional imports for visualization and path finding
    from ETIA.CRV.visualization import Visualization  # Visualization class provided
    from ETIA.CRV.queries import one_potentially_directed_path  # Function provided
    from ETIA.CRV import find_adjset  # Function provided

Step 2: Load Example Dataset
----------------------------

We start by loading the example dataset ``example_dataset.csv`` which contains several features and two target variables.

.. code-block:: python

    data = pd.read_csv('example_dataset.csv')

    # Display the first few rows of the dataset
    print("Original Dataset:")
    print(data.head())

Step 3: Define Target Features
------------------------------

We define two target variables (``'t1'`` and ``'t2'``) for which we want to perform feature selection and causal discovery.

.. code-block:: python

    target_features = {'t1': 'categorical', 't2': 'categorical'}

Step 4: Run Automated Feature Selection (AFS)
---------------------------------------------

Now, we initialize the AFS module and run it on the dataset to select the most relevant features.

.. code-block:: python

    # Initialize the AFS module with depth 1
    afs_instance = AFS(depth=1)

    # Run AFS to select features for the target variables
    afs_result = afs_instance.run_AFS(data=data, target_features=target_features)

    # Display the selected features and the best configuration found
    print("Selected Features by AFS:")
    print(afs_result['selected_features'])

    print("Best AFS Configuration:")
    print(afs_result['best_config'])

    # Extract the reduced dataset containing only the selected features and the target variables
    reduced_data = afs_result['reduced_data']

Step 5: Run Causal Learner (CL)
-------------------------------

Next, we use the CausalLearner (CL) to discover causal relationships between the selected features and the target variables.
The reduced dataset from AFS is passed as input to CL.

.. code-block:: python

    # Initialize the CausalLearner with the reduced dataset
    learner = CausalLearner(dataset_input=reduced_data)

    # Run the causal discovery process
    opt_conf, matrix_mec_graph, run_time, library_results = learner.learn_model()

    # Display the results of causal discovery
    print("Optimal Causal Discovery Configuration from CL:")
    print(opt_conf)

    print("MEC Matrix Graph (Markov Equivalence Class):")
    print(matrix_mec_graph)

Step 6: Run Causal Reasoning Validator (CRV)
--------------------------------------------

Finally, we use the Causal Reasoning Validator (CRV) to perform causal reasoning and validation on the learned causal model from CL.

**Note:** Ensure that Cytoscape is open before running this step, as the visualization requires Cytoscape to be running.

### Visualize the Causal Graph using Cytoscape

We use the ``Visualization`` class to send the graph to Cytoscape for visualization.

.. code-block:: python

    # Initialize the Visualization class with the adjacency matrix
    visualization = Visualization(matrix_pd=matrix_mec_graph, net_name='CausalGraph', collection_name='CausalAnalysis')

    # Plot the graph in Cytoscape
    visualization.plot_cytoscape()

    # Optionally, set a specific layout and export the visualization
    visualization.set_layout(layout_name='force-directed')
    visualization.export_to_png(file_path='causal_graph.png')

### Find a Path from a Variable to a Target Variable

We can find a potentially directed path from a variable to a target using the ``one_potentially_directed_path`` function.

.. code-block:: python

    # Define the variable names (ensure they exist in your dataset and graph)
    source_variable = 'X1'  # Replace with an actual variable name from your dataset
    target_variable = 't1'  # Target variable

    # Get the adjacency matrix as a NumPy array
    adjacency_matrix = matrix_mec_graph.values
    node_names = list(matrix_mec_graph.columns)
    node_indices = {name: idx for idx, name in enumerate(node_names)}

    # Find one potentially directed path from source to target
    path = one_potentially_directed_path(
        matrix=adjacency_matrix,
        start=node_indices[source_variable],
        end=node_indices[target_variable]
    )

    if path:
        path_variables = [node_names[idx] for idx in path]
        print(f"\nA potentially directed path from {source_variable} to {target_variable}:")
        print(" -> ".join(path_variables))
    else:
        print(f"\nNo potentially directed path found from {source_variable} to {target_variable}.")

### Compute the Adjustment Set

We compute the adjustment set for estimating the causal effect of the source variable on the target variable.

.. code-block:: python

    # Define the graph type (e.g., 'pag' for Partial Ancestral Graph)
    graph_type = 'pag'  # Adjust based on your graph's type

    # Find the adjustment set using the provided function
    adj_set_can, adj_set_min = find_adjset(
        graph_pd=matrix_mec_graph,
        graph_type=graph_type,
        target_name=[target_variable],
        exposure_names=[source_variable],
        r_path='/path/to/Rscript'  # Replace with the correct path
    )

    print(f"\nCanonical Adjustment Set for {source_variable} and {target_variable}:")
    print(adj_set_can if adj_set_can else "No canonical adjustment set found.")

    print(f"\nMinimal Adjustment Set for {source_variable} and {target_variable}:")
    print(adj_set_min if adj_set_min else "No minimal adjustment set found.")

### Calculate Edge Confidence (Optional)

We can estimate the confidence of the edges in the causal graph by performing bootstrapping.

.. code-block:: python

    # Calculate edge consistency and similarity confidence
    edge_consistency, edge_similarity = calculate_confidence(
        dataset=learner.dataset,
        opt_conf=opt_conf,
        n_bootstraps=50  # Adjust the number of bootstraps as needed
    )

    print("\nEdge Consistency:")
    print(edge_consistency)

    print("\nEdge Similarity:")
    print(edge_similarity)

Step 7: (Optional) Save Progress
--------------------------------

You can save the progress of the experiment if needed.

.. code-block:: python

    learner.save_progress(path="causal_pipeline_progress.pkl")

    # To load the saved progress later:
    # learner = learner.load_progress(path="causal_pipeline_progress.pkl")

---

Explanation
-----------

### Overview

This example demonstrates the complete pipeline of using the AFS, CL, and CRV modules for causal analysis:

1. **Feature Selection (AFS)**: Identifies the most relevant features for the target variables.
2. **Causal Discovery (CL)**: Discovers causal relationships among the selected features.
3. **Causal Reasoning and Validation (CRV)**: Validates the causal model, visualizes it, finds causal paths, and computes adjustment sets.

### Visualization with Cytoscape

- **Visualization Class**: We use the ``Visualization`` class to handle graph visualization in Cytoscape.
- **Plotting**: The ``plot_cytoscape`` method sends the graph to Cytoscape for visualization.
- **Layout and Export**: Use ``set_layout`` and ``export_to_png`` to adjust the layout and save the visualization.

### Finding Paths

- **``one_potentially_directed_path`` Function**: Searches for a potentially directed path from a start node to an end node in the causal graph.
- **Node Mapping**: Maps node names to indices for processing and back to interpret the results.

### Computing Adjustment Sets

- **``find_adjset`` Function**: Uses the ``dagitty`` R package to compute adjustment sets for causal effect estimation.
- **Parameters**:
  - ``graph_pd``: The adjacency matrix as a pandas DataFrame.
  - ``graph_type``: Type of the graph (e.g., ``'dag'``, ``'cpdag'``, ``'mag'``, ``'pag'``).
  - ``target_name``: The target variable.
  - ``exposure_names``: The exposure variable(s).
  - ``r_path``: Path to the Rscript executable.

### Calculating Edge Confidence

- **Bootstrap Methods**: Functions like ``bootstrapping_causal_graph`` and ``edge_metrics_on_bootstraps`` estimate the confidence of edges via bootstrapping.
- **Edge Consistency and Similarity**: Metrics to assess the stability of the discovered causal relationships.

### Dependencies and Setup

- **Cytoscape**: Ensure Cytoscape is installed and running.
- **R and dagitty**: The ``find_adjset`` function requires R and the ``dagitty`` package.
- **Python Packages**: Install required Python packages (e.g., ``py4cytoscape``, ``numpy``, ``pandas``).

### Variable Names

- **Source and Target Variables**: Replace ``'X1'`` and ``'t1'`` with actual variable names from your dataset.
- **Node Names**: Ensure node names in the adjacency matrix match those used in your dataset.

### Error Handling

- **Module Imports**: Confirm all modules and functions are correctly imported.
- **Path Corrections**: Update paths like ``/path/to/Rscript`` to correct locations on your system.
- **Function Compatibility**: Verify method compatibility with your module versions.

---

By following these steps, you can utilize the full pipeline provided by the AFS, CL, and CRV modules to perform comprehensive causal analysis on your dataset. This includes selecting relevant features, discovering causal structures, visualizing the causal graph, finding causal paths, computing adjustment sets, and assessing the confidence of causal relationships.
