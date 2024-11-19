import pandas as pd
import numpy as np
from ETIA.AFS import AFS

def main():
    # Generate synthetic data for testing
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42
    )

    # Convert to pandas DataFrame
    feature_names = [f'feature_{i}' for i in range(20)]
    data = pd.DataFrame(X, columns=feature_names)
    data['target'] = y

    # Define target features and their types
    target_features = {
        'Category': 'categorical'
    }

    # Define custom configurations
    custom_configs = [
        {
            'fs_name': 'ses',
            'ind_test_name': 'testIndReg',
            'alpha': 0.05,
            'k': 0,
            'preprocess_method': None,
            'r_path': 'R'
        },
        {
            'fs_name': 'ses',
            'ind_test_name': 'testIndReg',
            'alpha': 0.05,
            'k': 0,
            'preprocess_method': None,
            'r_path': 'R'
        },
    ]

    # Initialize the AFS class with depth
    afs = AFS(depth=1, verbose=True, random_seed=42)

    # Run the AFS process using custom configurations
    results = afs.run_AFS(
        data='data_sample1.csv',
        target_features=target_features,
        pred_configs=0.1,  # Use custom configurations
        dataset_name='synthetic_dataset'
    )

    # Access and print the results
    selected_features = results['selected_features']
    best_config = results['best_config']
    reduced_data = results['reduced_data']

    print("\nSelected Features for each target:")
    for target, features in selected_features.items():
        print(f"{target}: {features}")

    print("\nBest Configuration:")
    print(best_config)

    print("\nReduced Data (first 5 rows):")
    print(reduced_data.head())

if __name__ == '__main__':
    main()
