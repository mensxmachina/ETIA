Advanced Example
=================

This advanced example demonstrates a comprehensive pipeline for **Automated Feature Selection (AFS)**, **Causal Learning (CL)**, and **Causal Reasoning Validation (CRV)**. It showcases advanced configurations, parallel processing, and in-depth causal analysis, providing a robust framework for complex datasets.

**Prerequisites:**

- Ensure all prerequisites from the **Prerequisites** section are met.
- Familiarity with Python programming and causal analysis concepts.
- Cytoscape must be installed and running for visualization steps.

Step 1: Import Required Modules
--------------------------------

Begin by importing all necessary modules, including those for feature selection, causal learning, visualization, and path finding.

.. code-block:: python
    :caption: Importing Required Modules

    import pandas as pd
    from ETIA.AFS import AFS
    from ETIA.CausalLearning import CausalLearner, Configurations
    from ETIA.CRV.visualization import Visualization  # Visualization class for graph plotting
    from ETIA.CRV.queries import one_potentially_directed_path  # Function to find directed paths

Step 2: Load and Inspect the Dataset
------------------------------------

Load your dataset and perform an initial inspection to understand its structure.

.. code-block:: python
    :caption: Loading and Displaying the Dataset

    # Load the dataset from a CSV file
    data = pd.read_csv('example_dataset.csv')

    # Display the first few rows of the dataset
    print("Original Dataset:")
    print(data.head())

Step 3: Define Target and Exposure Features
-------------------------------------------

Specify the target variables and exposure features for feature selection and causal analysis.

.. code-block:: python
    :caption: Defining Target and Exposure Features

    # Define the target features with their data types
    target_features = {'target': 'continuous'}

    # Specify the names of exposure variables
    exposure_names = ['feature4', 'feature5']

Step 4: Initialize Automated Feature Selection (AFS)
-----------------------------------------------------

Set up the AFS module with a specified search depth to control the complexity of feature selection.

.. code-block:: python
    :caption: Initializing Automated Feature Selection (AFS)

    # Initialize the AFS module with depth 2
    afs_instance = AFS(depth=2)

Step 5: Define Prediction Configurations for AFS
-------------------------------------------------

Configure the parameters for the feature selection model. Here, two configurations using Random Forest are defined with different alpha values.

.. code-block:: python
    :caption: Defining Prediction Configurations for AFS

    pred_configs = [
        {
            'model': 'random_forest',
            'n_estimators': 100,
            'min_samples_leaf': 0.1,
            'max_features': 'sqrt',
            'fs_name': 'fbed',
            'alpha': 0.05,
            'k': 3,
            'ind_test_name': 'testIndFisher'
        },
        {
            'model': 'random_forest',
            'n_estimators': 100,
            'min_samples_leaf': 0.1,
            'max_features': 'sqrt',
            'fs_name': 'fbed',
            'alpha': 0.1,
            'k': 3,
            'ind_test_name': 'testIndFisher'
        }
    ]

Step 6: Run AFS for Target Features
-----------------------------------

Execute the AFS process to select features relevant to the target variable.

.. code-block:: python
    :caption: Running AFS for Target Features

    # Run AFS to select features relevant to the target variable
    afs_result = afs_instance.run_AFS(
        data=data,
        target_features=target_features,
        pred_configs=pred_configs
    )

    # Retrieve the selected features for the target
    selected_features_target = afs_result['selected_features']

    # Initialize a set with the target's selected features
    selected_feature_set = selected_features_target

Step 7: Run AFS for Exposure Features with Parallel Processing
--------------------------------------------------------------

Perform AFS for each exposure variable using parallel processing to enhance performance.

.. code-block:: python
    :caption: Running AFS for Exposure Features with Parallel Processing

    # AFS on each exposure
    for e_name in exposure_names:
        # Initialize AFS with a search depth of 1 and utilize 12 processors for parallel processing
        afs = AFS(depth=1, num_processors=12)
        # Run AFS to select features relevant to the current exposure
        results = afs.run_AFS(
            data=data,
            target_features={e_name: 'continuous'},
            pred_configs=pred_configs
        )
        # Retrieve the selected features for the current exposure
        selected_features_exposure = results['selected_features']
        # Update the overall set of selected features
        selected_feature_set.update(selected_features_exposure)

Step 8: Aggregate and Display Selected Features
-----------------------------------------------

Combine all selected features into a unique set to avoid duplicates and display them.

.. code-block:: python
    :caption: Aggregating and Displaying Selected Features

    # Collect all unique selected feature names
    unique_selected_features = set()

    # Iterate over the selected feature lists and add them to the unique set
    for feature_list in selected_feature_set.values():
        unique_selected_features.update(feature_list)

    # Convert the set of unique selected features to a list
    unique_selected_features = list(unique_selected_features)

    # Display the selected features from AFS
    print("Selected Features by AFS:")
    print(unique_selected_features)

    # Display the best configuration found by AFS
    print("Best AFS Configuration:")
    print(afs_result['best_config'])

Step 9: Prepare the Reduced Dataset
-----------------------------------

Create a new dataset containing only the selected features to reduce dimensionality.

.. code-block:: python
    :caption: Preparing the Reduced Dataset

    # Extract the reduced dataset containing only the selected features
    reduced_data = afs_result['original_data'][unique_selected_features]

Step 10: Initialize Causal Learner (CL)
---------------------------------------

Load configurations and initialize the CausalLearner with the reduced dataset.

.. code-block:: python
    :caption: Initializing Causal Learner (CL)

    # Load configurations from a JSON file for causal learning
    conf = Configurations(conf_file='conf.json')

    # Initialize the CausalLearner with the loaded configurations
    learner = CausalLearner(configurations=conf)

.. code-block:: python
    :caption: conf.json

    {
       "Dataset":
            {
                    "dataset_name": "example_dataset.csv",
                    "time_lagged": false,
                    "n_lags": 0
            },
        "Results_folder_path": "./",
        "causal_sufficiency": false,
        "assume_faithfulness": true,
        "OCT":
            {
                    "alpha": 0.01,
                    "n_permutations": 100,
                    "variables_type": "mixed",
                    "out_of_sample_protocol":
                        {
                            "name": "KFoldCV",
                            "parameters":
                            {
                                "folds": 10,
                                "folds_to_run": 5
                            }
                        },
                    "Regressor_parameters":
                        {
                            "name": "RandomForestRegressor",
                            "parameters":
                                {
                                    "n_trees": 100,
                                    "min_samples_leaf": 0.01,
                                    "max_depth": 10
                                }
                        },
                    "CausalDiscoveryAlgorithms": {
                        "exclude_algs": ["fcimax", "gfci", "rfci", "cfci"]
                    }

            }

    }

Step 11: Run Causal Discovery Process
-------------------------------------

Execute the causal discovery process to identify causal relationships among the selected features.

.. code-block:: python
    :caption: Running the Causal Discovery Process

    # Run the causal discovery process
    cl_results = learner.learn_model()

    # Display the results of causal discovery
    print("Optimal Causal Discovery Configuration from CL:")
    print(cl_results['optimal_conf'])

    print("MEC Matrix Graph (Markov Equivalence Class):")
    print(cl_results['matrix_mec_graph'])

Step 12: Visualize the Causal Graph with Cytoscape
---------------------------------------------------

Use the Visualization class to send the causal graph to Cytoscape for interactive visualization.

**Note:** Ensure that Cytoscape is open before running this step.

.. code-block:: python
    :caption: Visualizing the Causal Graph with Cytoscape

    # Initialize the Visualization object with the MEC graph
    viz = Visualization(cl_results['matrix_mec_graph'], 'Collection', 'Graph')
    # Plot the MEC graph using Cytoscape
    viz.plot_cytoscape()

Step 13: Identify Directed Paths in the Causal Graph
-----------------------------------------------------

Find a potentially directed path from a specified source variable to the target variable within the causal graph.

.. code-block:: python
    :caption: Identifying Directed Paths in the Causal Graph

    # Find a potentially directed path from "feature1" to "target"
    path = one_potentially_directed_path(cl_results['matrix_mec_graph'], "feature1", "target")

    # Display the identified path
    print('The path from feature1 to target is:', path)

Step 14: Save and Load Progress (Optional)
------------------------------------------

Optionally, save the progress of the causal learning process for future use.

.. code-block:: python
    :caption: Saving and Loading Progress

    # Save the progress of the causal learning process
    learner.save_progress(path="causal_pipeline_progress.pkl")

    # To load the saved progress later:
    # learner = learner.load_progress(path="causal_pipeline_progress.pkl")