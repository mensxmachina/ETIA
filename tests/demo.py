import pandas as pd
from ETIA.AFS import AFS
from ETIA.CausalLearning import CausalLearner
from ETIA.CRV.visualization import Visualization  # Visualization class provided
from ETIA.CRV.queries import one_potentially_directed_path  # Function provided
from ETIA.CRV import find_adjset  # Function provided

data = pd.read_csv('example_dataset.csv')

# Display the first few rows of the dataset
print("Original Dataset:")
print(data.head())

target_features = {'target': 'continuous'}

# Initialize the AFS module with depth 1
afs_instance = AFS(depth=1)

# Run AFS to select features for the target variables
afs_result = afs_instance.run_AFS(data=data, target_features=target_features, pred_configs=0.1)

# Display the selected features and the best configuration found
print("Selected Features by AFS:")
print(afs_result['selected_features'])

print("Best AFS Configuration:")
print(afs_result['best_config'])

# Extract the reduced dataset containing only the selected features and the target variables
reduced_data = afs_result['reduced_data']

# Initialize the CausalLearner with the reduced dataset
learner = CausalLearner(dataset_input=reduced_data)

# Run the causal discovery process
cl_results = learner.learn_model()

# Display the results of causal discovery
print("Optimal Causal Discovery Configuration from CL:")
print(cl_results['optimal_conf'])

print("MEC Matrix Graph (Markov Equivalence Class):")
print(cl_results['matrix_mec_graph'])

viz = Visualization(cl_results['matrix_mec_graph'], 'Collection', 'Graph')
viz.plot_cytoscape()