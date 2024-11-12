# test_afs.py

import unittest
import pandas as pd
import numpy as np
import logging
import warnings

# Import the AFS class from your module
from ETIA.AFS import AFS  # Replace 'afs_module_name' with the actual module name

# Suppress warnings during testing
warnings.filterwarnings("ignore")

# Configure logging to show only warnings and errors during testing
logging.basicConfig(level=logging.WARNING)

class TestAFSInitialization(unittest.TestCase):
    def test_default_initialization(self):
        afs = AFS()
        self.assertEqual(afs.depth, 1)
        self.assertFalse(afs.verbose)
        self.assertIsNotNone(afs.num_processors)
        self.assertEqual(afs.oos_protocol, {"name": "KFoldCV", "folds": 5, "folds_to_run": 2})
        self.assertIsNone(afs.random_seed)

    def test_custom_initialization(self):
        afs = AFS(depth=3, verbose=True, num_processors=4, oos_protocol={"name": "Holdout", "test_size": 0.3}, random_seed=42)
        self.assertEqual(afs.depth, 3)
        self.assertTrue(afs.verbose)
        self.assertEqual(afs.num_processors, 4)
        self.assertEqual(afs.oos_protocol, {"name": "Holdout", "test_size": 0.3})
        self.assertEqual(afs.random_seed, 42)

    def test_invalid_depth(self):
        with self.assertRaises(ValueError):
            AFS(depth=-1)

    def test_random_seed_reproducibility(self):
        afs1 = AFS(random_seed=42)
        afs2 = AFS(random_seed=42)
        self.assertEqual(afs1.random_seed, afs2.random_seed)

class TestAFSRunAFS(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Generate features
        feature1 = np.random.rand(100)
        feature2 = np.random.rand(100)
        feature3 = np.random.rand(100)
        feature4 = np.random.rand(100)
        # Create a target that is correlated with feature1 and feature2
        target_prob = 0.3 * feature1 + 0.7 * feature2
        target = np.random.binomial(1, target_prob / target_prob.max(), size=100)
        # Feature5 is a noisy version of the target
        feature5 = target + np.random.normal(0, 0.1, size=100)
        # Construct the DataFrame
        self.df = pd.DataFrame({
            'feature1': feature1,
            'feature2': feature2,
            'feature3': feature3,
            'feature4': feature4,
            'feature5': feature5,
            'target': target
        })
        self.target_features = {'target': 'categorical'}
        self.afs = AFS(random_seed=42)

    def test_run_afs_with_dataframe(self):
        result = self.afs.run_AFS(data=self.df, target_features=self.target_features)
        self.assertIn('original_data', result)
        self.assertIn('reduced_data', result)
        self.assertIn('best_config', result)
        self.assertIn('selected_features', result)
        self.assertIsInstance(result['original_data'], pd.DataFrame)
        self.assertIsInstance(result['reduced_data'], pd.DataFrame)
        self.assertIsInstance(result['selected_features'], dict)
        # Check that some features have been selected
        self.assertTrue(len(result['selected_features']['target']) > 0)
        print("Selected features:", result['selected_features']['target'])

    def test_run_afs_with_invalid_data_type(self):
        with self.assertRaises(ValueError):
            self.afs.run_AFS(data=123, target_features=self.target_features)

    def test_run_afs_with_invalid_target_features(self):
        with self.assertRaises(ValueError):
            self.afs.run_AFS(data=self.df, target_features='invalid_type')

    def test_run_afs_with_numpy_array(self):
        data_np = self.df.to_records(index=False)
        result = self.afs.run_AFS(data=data_np, target_features=self.target_features)
        self.assertIn('original_data', result)

    def test_run_afs_with_invalid_numpy_array(self):
        data_np = self.df.values  # No column names
        with self.assertRaises(ValueError):
            self.afs.run_AFS(data=data_np, target_features=self.target_features)

    def test_run_afs_with_custom_pred_configs(self):
        pred_configs = [{
            'model': 'random_forest',
            'n_estimators': 50,
            'max_features': 'sqrt',
            'min_samples_leaf': 0.01,
            'fs_name': 'ses',
            'alpha': 0.05,
            'k': 3,
            'ind_test_name': 'testIndFisher',
            'r_path': 'Rscript'  # Adjust if Rscript is in a different location
        }]
        result = self.afs.run_AFS(
            data=self.df,
            target_features=self.target_features,
            pred_configs=pred_configs
        )
        self.assertIn('best_config', result)
        self.assertEqual(result['best_config'], pred_configs[0])
        # Check that some features have been selected
        self.assertTrue(len(result['selected_features']['target']) > 0)
        print("Selected features with custom configs:", result['selected_features']['target'])

    def test_run_afs_with_pred_configs_sampling(self):
        result = self.afs.run_AFS(data=self.df, target_features=self.target_features, pred_configs=0.1)
        self.assertIn('best_config', result)

    def test_run_afs_with_invalid_pred_configs(self):
        with self.assertRaises(ValueError):
            self.afs.run_AFS(data=self.df, target_features=self.target_features, pred_configs=-0.1)

        with self.assertRaises(ValueError):
            self.afs.run_AFS(data=self.df, target_features=self.target_features, pred_configs=1.5)

class TestAFSErrorHandling(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Generate features
        feature1 = np.random.rand(100)
        feature2 = np.random.rand(100)
        feature3 = np.random.rand(100)
        feature4 = np.random.rand(100)
        # Create a target that is correlated with feature1 and feature2
        target_prob = 0.3 * feature1 + 0.7 * feature2
        target = np.random.binomial(1, target_prob / target_prob.max(), size=100)
        # Feature5 is a noisy version of the target
        feature5 = target + np.random.normal(0, 0.1, size=100)
        # Construct the DataFrame
        self.df = pd.DataFrame({
            'feature1': feature1,
            'feature2': feature2,
            'feature3': feature3,
            'feature4': feature4,
            'feature5': feature5,
            'target': target
        })
        self.target_features = {'target': 'categorical'}
        self.afs = AFS(random_seed=42)

    def test_invalid_oos_protocol(self):
        afs = AFS(oos_protocol={'name': 'UnsupportedProtocol'})
        with self.assertRaises(ValueError):
            afs.run_AFS(data=self.df, target_features=self.target_features)

    def test_empty_data(self):
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            self.afs.run_AFS(data=empty_df, target_features=self.target_features)

    def test_no_features_left(self):
        afs = AFS(depth=1)
        result = afs.run_AFS(data=self.df[['target']], target_features=self.target_features)
        self.assertEqual(result['selected_features'], [])

class TestAFSPrivateMethods(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Generate features
        feature1 = np.random.rand(100)
        feature2 = np.random.rand(100)
        feature3 = np.random.rand(100)
        feature4 = np.random.rand(100)
        # Create a target that is correlated with feature1 and feature2
        target_prob = 0.3 * feature1 + 0.7 * feature2
        target = np.random.binomial(1, target_prob / target_prob.max(), size=100)
        # Feature5 is a noisy version of the target
        feature5 = target + np.random.normal(0, 0.1, size=100)
        # Construct the DataFrame
        self.df = pd.DataFrame({
            'feature1': feature1,
            'feature2': feature2,
            'feature3': feature3,
            'feature4': feature4,
            'feature5': feature5,
            'target': target
        })
        self.target_features = {'target': 'categorical'}
        self.afs = AFS(random_seed=42)

    def test_process_target(self):
        result = self.afs._process_target(
            data=self.df,
            target_feature='target',
            target_type='categorical',
            pred_configs=[{
                'model': 'random_forest',
                'n_estimators': 10,
                'fs_name': 'fbed',
                'alpha': 0.05,
                'k': 3,
                'ind_test_name': 'testIndFisher',
                'r_path': 'Rscript'
            }],
            dataset_name='test_dataset',
            depth=1
        )
        self.assertIn('selected_features', result)

    def test_recursive_fs_for_target(self):
        result = self.afs.recursive_fs_for_target(
            data=self.df,
            target_feature='target',
            target_type='categorical',
            pred_configs=[{
                'model': 'random_forest',
                'n_estimators': 10,
                'fs_name': 'fbed',
                'alpha': 0.05,
                'k': 3,
                'ind_test_name': 'testIndFisher',
                'r_path': 'Rscript'
            }],
            dataset_name='test_dataset',
            depth=1,
            visited_features=set()
        )
        self.assertIn('selected_features', result)

    def test_bootstrap_bias_correction(self):
        predictions = np.random.rand(100)
        true_values = np.random.randint(0, 2, 100)
        bbc_score, ci = self.afs.bootstrap_bias_correction(
            fold_predictions=[(predictions, true_values)],
            target_type='categorical'
        )
        self.assertIsInstance(bbc_score, float)
        self.assertIsInstance(ci, np.ndarray)

class TestAFSAdditionalScenarios(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Generate features
        feature1 = np.random.rand(100)
        feature2 = np.random.rand(100)
        feature3 = np.random.rand(100)
        feature4 = np.random.rand(100)
        feature5 = np.random.rand(100)
        # Create a continuous target
        target = 0.5 * feature1 + 0.3 * feature2 + np.random.normal(0, 0.1, 100)
        # Construct the DataFrame
        self.df = pd.DataFrame({
            'feature1': feature1,
            'feature2': feature2,
            'feature3': feature3,
            'feature4': feature4,
            'feature5': feature5,
            'target': target
        })
        self.target_features = {'target': 'continuous'}
        self.afs = AFS(random_seed=42)

    def test_continuous_target(self):
        result = self.afs.run_AFS(data=self.df, target_features=self.target_features)
        self.assertIn('selected_features', result)
        # Check that some features have been selected
        self.assertTrue(len(result['selected_features']['target']) > 0)
        print("Selected features for continuous target:", result['selected_features']['target'])

    def test_exception_in_feature_selection(self):
        # Mock the FeatureSelector to raise an exception
        original_feature_selection = self.afs.recursive_fs_for_target

        def mock_feature_selection(*args, **kwargs):
            raise RuntimeError("Mocked exception")

        self.afs.recursive_fs_for_target = mock_feature_selection

        with self.assertRaises(RuntimeError):
            self.afs.run_AFS(data=self.df, target_features=self.target_features)

        # Restore the original method
        self.afs.recursive_fs_for_target = original_feature_selection

if __name__ == '__main__':
    unittest.main()
