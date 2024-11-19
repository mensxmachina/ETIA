import pandas as pd
from ETIA.AFS import AFS
from ETIA.CausalLearning import CausalLearner, Configurations
from ETIA.CRV.visualization import Visualization  # Import Visualization class for graph plotting
from ETIA.CRV.queries import one_potentially_directed_path  # Import function to find directed paths

# Load the dataset from a CSV file
data = pd.read_csv('example_dataset.csv')

# Display the first few rows of the dataset
print("Original Dataset:")
print(data.head())

# Define the target features with their data types
target_features = {'target': 'continuous'}
# Specify the names of exposure variables
exposure_names = ['feature4', 'feature5']

# Initialize the AFS (Automated Feature Selection) module with a search depth of 2
afs_instance = AFS(depth=2)

# Define prediction configurations for feature selection
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

# Perform AFS for each exposure variable
for exposure_name in exposure_names:
    # Initialize AFS with a search depth of 1 and utilize 12 processors for parallel processing
    afs = AFS(depth=1, num_processors=12)
    # Run AFS to select features relevant to the current exposure
    results = afs.run_AFS(
        data=data,
        target_features={exposure_name: 'continuous'},
        pred_configs=pred_configs
    )
    # Retrieve the selected features for the current exposure
    selected_features_exposure = results['selected_features']
    # Update the overall set of selected features
    selected_feature_set.update(selected_features_exposure)

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

# Extract the reduced dataset containing only the selected features
reduced_data = data[unique_selected_features]

# Load configurations from a JSON file for causal learning
conf = Configurations(conf_file='conf.json')

# Initialize the CausalLearner with the loaded configurations
learner = CausalLearner(configurations=conf)

# Assign the reduced dataset to the learner (assuming CausalLearner accepts a dataset)
learner.data = reduced_data

# Execute the causal discovery process
cl_results = learner.learn_model()

# Display the optimal configuration identified by the causal learner
print("Optimal Causal Discovery Configuration from CL:")
print(cl_results['optimal_conf'])

# Display the Markov Equivalence Class (MEC) matrix graph
print("MEC Matrix Graph (Markov Equivalence Class):")
print(cl_results['matrix_mec_graph'])

# Initialize the Visualization object with the MEC graph
viz = Visualization(cl_results['matrix_mec_graph'], 'Collection', 'Graph')
# Plot the MEC graph using Cytoscape
viz.plot_cytoscape()

# Find a potentially directed path from "feature1" to "target" within the MEC graph
path = one_potentially_directed_path(cl_results['matrix_mec_graph'], "feature1", "target")

# Display the identified path
print('The path from feature1 to target is:', path)
