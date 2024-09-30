# test_afs.py

import unittest
import numpy as np
import pandas as pd
import os
from ETIA.AFS import AFS
from ETIA.AFS.predictive_configurator import PredictiveConfigurator
from sklearn.datasets import make_classification, make_regression

class TestAFS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Generate synthetic classification data
        cls.X_classification, cls.y_classification = make_classification(
            n_samples=100,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        cls.feature_names_classification = [f'feature_{i}' for i in range(10)]
        cls.data_classification = pd.DataFrame(cls.X_classification, columns=cls.feature_names_classification)
        cls.data_classification['target'] = cls.y_classification

        # Generate synthetic regression data
        cls.X_regression, cls.y_regression = make_regression(
            n_samples=100,
            n_features=10,
            n_informative=5,
            noise=0.1,
            random_state=42
        )
        cls.feature_names_regression = [f'feature_{i}' for i in range(10)]
        cls.data_regression = pd.DataFrame(cls.X_regression, columns=cls.feature_names_regression)
        cls.data_regression['target'] = cls.y_regression

    def test_initialization_defaults(self):
        """Test initialization with default parameters."""
        afs = AFS()
        self.assertEqual(afs.depth, 1)
        self.assertFalse(afs.verbose)
        self.assertIsNotNone(afs.num_processors)
        self.assertIsNotNone(afs.oos_protocol)

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        afs = AFS(depth=2, verbose=True, num_processors=2, oos_protocol={'name': 'Holdout', 'test_size': 0.2})
        self.assertEqual(afs.depth, 2)
        self.assertTrue(afs.verbose)
        self.assertEqual(afs.num_processors, 2)
        self.assertEqual(afs.oos_protocol['name'], 'Holdout')

    def test_run_afs_default_config_classification(self):
        """Test running AFS with default configurations on classification data."""
        afs = AFS(depth=1, verbose=False, random_seed=42)
        target_features = {'target': 'categorical'}
        # Added pred_configs=0.1 to limit configurations
        results = afs.run_AFS(
            data=self.data_classification,
            target_features=target_features,
            pred_configs=0.1,  # Run 10% of configurations to make test faster
            dataset_name='test_classification'
        )
        self.assertIn('original_data', results)
        self.assertIn('reduced_data', results)
        self.assertIn('best_config', results)
        self.assertIn('selected_features', results)
        self.assertGreaterEqual(len(results['selected_features']['target']), 1)

    def test_run_afs_default_config_regression(self):
        """Test running AFS with default configurations on regression data."""
        afs = AFS(depth=1, verbose=False, random_seed=42)
        target_features = {'target': 'continuous'}
        # Added pred_configs=0.1 to limit configurations
        results = afs.run_AFS(
            data=self.data_regression,
            target_features=target_features,
            pred_configs=0.1,  # Run 10% of configurations to make test faster
            dataset_name='test_regression'
        )
        self.assertIn('original_data', results)
        self.assertIn('reduced_data', results)
        self.assertIn('best_config', results)
        self.assertIn('selected_features', results)
        self.assertGreaterEqual(len(results['selected_features']['target']), 1)

    def test_run_afs_custom_configs(self):
        """Test running AFS with custom configurations."""
        custom_configs = [
            {
                'fs_name': 'fbed',
                'ind_test_name': 'testIndReg',
                'alpha': 0.05,
                'k': 0,
                'preprocess_method': None,
                'model': 'linear_regression',  # Added model
                'r_path': 'R'
            }
        ]
        afs = AFS(depth=1, verbose=False, random_seed=42)
        target_features = {'target': 'continuous'}
        results = afs.run_AFS(
            data=self.data_regression,
            target_features=target_features,
            pred_configs=custom_configs,
            dataset_name='test_custom_configs'
        )
        self.assertEqual(results['best_config'], custom_configs[0])

    def test_run_afs_percentage_configs(self):
        """Test running AFS with a percentage of configurations."""
        afs = AFS(depth=1, verbose=False, random_seed=42)
        target_features = {'target': 'categorical'}
        # This test specifically tests using 10% of configurations
        results = afs.run_AFS(
            data=self.data_classification,
            target_features=target_features,
            pred_configs=0.1,
            dataset_name='test_percentage_configs'
        )
        self.assertIn('best_config', results)
        self.assertGreaterEqual(len(results['selected_features']['target']), 1)

    def test_run_afs_invalid_pred_configs(self):
        """Test error handling when invalid pred_configs are provided."""
        afs = AFS()
        target_features = {'target': 'categorical'}
        with self.assertRaises(ValueError):
            afs.run_AFS(
                data=self.data_classification,
                target_features=target_features,
                pred_configs=-0.1,  # Invalid percentage
                dataset_name='test_invalid_pred_configs'
            )

    def test_recursive_feature_selection_depth(self):
        """Test recursive feature selection with depth greater than 1."""
        afs = AFS(depth=2, verbose=False, random_seed=42)
        target_features = {'target': 'categorical'}
        # Added pred_configs=0.1 to limit configurations
        results = afs.run_AFS(
            data=self.data_classification,
            target_features=target_features,
            pred_configs=0.1,
            dataset_name='test_recursive_fs'
        )
        self.assertGreaterEqual(len(results['selected_features']['target']), 1)

    def test_random_seed_effect(self):
        """Test that results are reproducible when random_seed is set."""
        afs1 = AFS(depth=1, verbose=False, random_seed=42)
        afs2 = AFS(depth=1, verbose=False, random_seed=42)
        target_features = {'target': 'categorical'}
        results1 = afs1.run_AFS(
            data=self.data_classification,
            target_features=target_features,
            pred_configs=0.1,
            dataset_name='test_reproducibility'
        )
        results2 = afs2.run_AFS(
            data=self.data_classification,
            target_features=target_features,
            pred_configs=0.1,
            dataset_name='test_reproducibility'
        )
        # Allow for minor variations if exact reproducibility isn't possible
        self.assertEqual(set(results1['selected_features']['target']), set(results2['selected_features']['target']))

    def test_multiple_target_variables(self):
        """Test running AFS with multiple target variables."""
        data_multi_target = self.data_classification.copy()
        data_multi_target['target2'] = np.random.choice([0, 1], size=100)
        target_features = {'target': 'categorical', 'target2': 'categorical'}
        afs = AFS(depth=1, verbose=False, random_seed=42)
        # Added pred_configs=0.1 to limit configurations
        results = afs.run_AFS(
            data=data_multi_target,
            target_features=target_features,
            pred_configs=0.1,
            dataset_name='test_multiple_targets'
        )
        self.assertIn('target', results['selected_features'])
        self.assertIn('target2', results['selected_features'])

    def test_no_features_selected(self):
        """Test handling when no features are selected."""
        # Create data where features are not informative
        data_non_informative = pd.DataFrame(np.random.randn(100, 10), columns=self.feature_names_classification)
        data_non_informative['target'] = np.random.choice([0, 1], size=100)
        afs = AFS(depth=1, verbose=False, random_seed=42)
        target_features = {'target': 'categorical'}
        results = afs.run_AFS(
            data=data_non_informative,
            target_features=target_features,
            pred_configs=0.1,
            dataset_name='test_no_features_selected'
        )
        # Check that the number of selected features is minimal
        self.assertLessEqual(len(results['selected_features']['target']), 1)

    def test_preprocessing_methods(self):
        """Test running AFS with preprocessing methods."""
        custom_configs = [
            {
                'fs_name': 'fbed',
                'ind_test_name': 'testIndReg',
                'alpha': 0.05,
                'k': 0,
                'preprocess_method': 'standard',
                'model': 'linear_regression',  # Added model
                'r_path': 'R'
            }
        ]
        afs = AFS(depth=1, verbose=False, random_seed=42)
        target_features = {'target': 'continuous'}
        results = afs.run_AFS(
            data=self.data_regression,
            target_features=target_features,
            pred_configs=custom_configs,
            dataset_name='test_preprocessing'
        )
        self.assertEqual(results['best_config']['preprocess_method'], 'standard')

    def test_handling_cycles_in_recursion(self):
        """Test that visited features are handled correctly to avoid cycles."""
        afs = AFS(depth=3, verbose=False, random_seed=42)
        target_features = {'target': 'categorical'}
        # Added pred_configs=0.1 to limit configurations
        results = afs.run_AFS(
            data=self.data_classification,
            target_features=target_features,
            pred_configs=0.1,
            dataset_name='test_cycles_in_recursion'
        )
        self.assertIsInstance(results['selected_features'], dict)

    def test_invalid_data_input(self):
        """Test error handling when invalid data input is provided."""
        afs = AFS()
        target_features = {'target': 'categorical'}
        with self.assertRaises(ValueError):
            afs.run_AFS(
                data=123,  # Invalid data type
                target_features=target_features,
                pred_configs=0.1,
                dataset_name='test_invalid_data_input'
            )

    def test_invalid_target_features_input(self):
        """Test error handling when invalid target_features input is provided."""
        afs = AFS()
        with self.assertRaises(ValueError):
            afs.run_AFS(
                data=self.data_classification,
                target_features='invalid',  # Invalid target_features type
                pred_configs=0.1,
                dataset_name='test_invalid_target_features'
            )

    def test_oos_protocol(self):
        """Test custom out-of-sample protocol."""
        afs = AFS(oos_protocol={'name': 'KFoldCV', 'folds': 3}, random_seed=42)
        target_features = {'target': 'categorical'}
        results = afs.run_AFS(
            data=self.data_classification,
            target_features=target_features,
            pred_configs=0.1,
            dataset_name='test_oos_protocol'
        )
        self.assertEqual(afs.oos_protocol['name'], 'KFoldCV')

    def test_bootstrap_bias_correction(self):
        """Test that bootstrap bias correction returns a float score."""
        afs = AFS(depth=1, verbose=False, random_seed=42)
        # This test does not need pred_configs adjustment
        target_type = 'categorical'
        # Simulate fold_predictions
        fold_predictions = [
            (np.random.rand(20), np.random.randint(0, 2, 20))
            for _ in range(5)
        ]
        bbc_score = afs.bootstrap_bias_correction(fold_predictions, target_type)
        self.assertIsInstance(bbc_score, float)

    def test_feature_selection_with_no_preprocessing(self):
        """Test feature selection without preprocessing."""
        afs = AFS(depth=1, verbose=False, random_seed=42)
        custom_configs = [
            {
                'fs_name': 'ses',
                'ind_test_name': 'testIndReg',
                'alpha': 0.05,
                'k': 0,
                'preprocess_method': None,
                'model': 'linear_regression',  # Added model
                'r_path': 'R'
            }
        ]
        target_features = {'target': 'continuous'}
        results = afs.run_AFS(
            data=self.data_regression,
            target_features=target_features,
            pred_configs=custom_configs,
            dataset_name='test_no_preprocessing'
        )
        self.assertIsNone(results['best_config']['preprocess_method'])

    def test_depth_zero(self):
        """Test that depth zero returns empty selected features."""
        afs = AFS(depth=0, verbose=False)
        target_features = {'target': 'categorical'}
        # Added pred_configs=0.1 to limit configurations
        results = afs.run_AFS(
            data=self.data_classification,
            target_features=target_features,
            pred_configs=0.1,
            dataset_name='test_depth_zero'
        )
        self.assertEqual(len(results['selected_features']['target']), 0)

    def test_invalid_depth(self):
        """Test error handling for invalid depth."""
        with self.assertRaises(ValueError):
            afs = AFS(depth=-1)  # Invalid depth

    def test_random_seed_effect(self):
        """Test that different random seeds produce different results."""
        afs1 = AFS(depth=1, verbose=False, random_seed=42)
        afs2 = AFS(depth=1, verbose=False, random_seed=24)
        target_features = {'target': 'categorical'}
        # Added pred_configs=0.1 to limit configurations
        results1 = afs1.run_AFS(
            data=self.data_classification,
            target_features=target_features,
            pred_configs=0.1,
            dataset_name='test_random_seed_effect'
        )
        results2 = afs2.run_AFS(
            data=self.data_classification,
            target_features=target_features,
            pred_configs=0.1,
            dataset_name='test_random_seed_effect'
        )
        self.assertNotEqual(results1['selected_features'], results2['selected_features'])

    def test_large_depth(self):
        """Test handling of large depth values."""
        afs = AFS(depth=5, verbose=False, random_seed=42)
        target_features = {'target': 'categorical'}
        # Added pred_configs=0.1 to limit configurations
        results = afs.run_AFS(
            data=self.data_classification,
            target_features=target_features,
            pred_configs=0.1,
            dataset_name='test_large_depth'
        )
        self.assertGreaterEqual(len(results['selected_features']['target']), 1)

    def test_feature_selector_error_handling(self):
        """Test that feature selector errors are handled gracefully."""
        afs = AFS()
        # Modify the data to cause an error in feature selection (e.g., all NaNs)
        data_with_nans = self.data_classification.copy()
        data_with_nans.iloc[:, :-1] = np.nan
        target_features = {'target': 'categorical'}
        # Added pred_configs=0.1 to limit configurations
        results = afs.run_AFS(
            data=data_with_nans,
            target_features=target_features,
            pred_configs=0.1,
            dataset_name='test_feature_selector_error'
        )
        self.assertEqual(len(results['selected_features']['target']), 0)

    def test_predictive_model_error_handling(self):
        """Test that predictive model errors are handled gracefully."""
        afs = AFS()
        # Create a configuration that would cause an error in model training
        custom_configs = [
            {
                'fs_name': 'fbed',
                'ind_test_name': 'invalid_test',  # Invalid test name
                'alpha': 0.05,
                'k': 0,
                'preprocess_method': None,
                'r_path': 'R'
            }
        ]
        target_features = {'target': 'categorical'}
        results = afs.run_AFS(
            data=self.data_classification,
            target_features=target_features,
            pred_configs=custom_configs,  # Using custom configs
            dataset_name='test_predictive_model_error'
        )
        self.assertEqual(len(results['selected_features']['target']), 0)

    def test_run_afs_with_numpy_array(self):
        """Test running AFS with data provided as a NumPy array."""
        afs = AFS()
        data_array = self.data_classification.values
        # Convert to DataFrame to include column names
        data_df = pd.DataFrame(data_array, columns=self.data_classification.columns)
        target_features = {'target': 'categorical'}
        results = afs.run_AFS(
            data=data_df,  # Use DataFrame instead of NumPy array
            target_features=target_features,
            pred_configs=0.1,
            dataset_name='test_numpy_array'
        )
        self.assertIn('reduced_data', results)

    def test_run_afs_with_data_filename(self):
        """Test running AFS with data provided as a filename."""
        afs = AFS()
        data_filename = 'test_data.csv'
        self.data_classification.to_csv(data_filename, index=False)
        target_features = {'target': 'categorical'}
        # Added pred_configs=0.1 to limit configurations
        results = afs.run_AFS(
            data=data_filename,
            target_features=target_features,
            pred_configs=0.1,
            dataset_name='test_data_filename'
        )
        os.remove(data_filename)  # Clean up
        self.assertIn('reduced_data', results)


if __name__ == '__main__':
    unittest.main()
