# ETIA/CausalLearning/test/CL_Tests.py

import unittest
import pandas as pd
import numpy as np
import os
import tempfile
import logging

from ETIA.CausalLearning.CausalLearner import CausalLearner
from ETIA.CausalLearning.configurations.configurations import Configurations
from ETIA.CausalLearning.data.Dataset import Dataset

class TestCausalLearner(unittest.TestCase):
    """Unit tests for the CausalLearner module without using mocks."""

    def setUp(self):
        """Set up test fixtures."""

        # Create a simple DataFrame for testing
        df = pd.DataFrame({
            'A': np.random.randint(0, 2, size=100),
            'B': np.random.randint(0, 2, size=100),
            'C': np.random.randn(100),
            'D': np.random.randn(100)
        })

        # Initialize Dataset object
        self.dataset = Dataset(
            data=df,
            data_time_info={'n_lags': 0, 'time_lagged': False},
            time_series=False,
            dataset_name='Test Dataset'
        )

        # Create a temporary directory for results
        self.temp_dir = tempfile.TemporaryDirectory()

        # Define default configurations
        self.default_configurations = Configurations(
            dataset=self.dataset,
            n_lags=0,
            time_lagged=False,
            time_series=False,
            n_jobs=1  # Ensure n_jobs is set
        )

    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()

    def test_default_configurations(self):
        """Test that CausalLearner uses default configurations when none are provided."""
        learner = CausalLearner(
            dataset_input=self.dataset,
            configurations=None,  # Use default
            verbose=False,
            n_jobs=1,
            random_seed=42
        )
        self.assertIsNotNone(learner.configurations)
        self.assertEqual(learner.configurations.n_lags, 0)
        self.assertFalse(learner.configurations.time_lagged)

    def test_initialization_with_dataframe(self):
        """Test initializing CausalLearner with a Dataset."""
        try:
            learner = CausalLearner(
                dataset_input=self.dataset,
                configurations=self.default_configurations,
                verbose=False,
                n_jobs=1,
                random_seed=42
            )
            self.assertIsInstance(learner, CausalLearner)
        except Exception as e:
            self.fail(f"Initialization with Dataset failed with exception: {e}")

    def test_initialization_with_invalid_dataset(self):
        """Test initializing CausalLearner with an invalid dataset input."""
        with self.assertRaises(ValueError):
            CausalLearner(
                dataset_input=123,  # Invalid type
                configurations=self.default_configurations,
                verbose=False
            )

    def test_run_causal_discovery(self):
        """Test running causal discovery without errors."""
        learner = CausalLearner(
            dataset_input=self.dataset,
            configurations=self.default_configurations,
            verbose=False,
            n_jobs=1,
            random_seed=42
        )
        try:
            opt_conf, mec_graph, run_time, library_results = learner.learn_model()
            self.assertIsNotNone(mec_graph)
            self.assertIsNotNone(library_results)
            self.assertIsInstance(mec_graph, pd.DataFrame)  # Assuming output is a DataFrame
        except Exception as e:
            self.fail(f"Causal discovery run failed with exception: {e}")

    def test_output_graph_structure(self):
        """Test the structure of the output causal graph."""
        learner = CausalLearner(
            dataset_input=self.dataset,
            configurations=self.default_configurations,
            verbose=False,
            n_jobs=1,
            random_seed=42
        )
        try:
            opt_conf, mec_graph, run_time, library_results = learner.learn_model()
            # Check that the graph has expected columns, e.g., 'source', 'target', 'relation'
            expected_columns = {'source', 'target', 'relation'}
            self.assertTrue(expected_columns.issubset(mec_graph.columns), "Output graph missing expected columns.")
        except Exception as e:
            self.fail(f"Causal discovery run failed with exception: {e}")

    def test_save_and_load_progress(self):
        """Test saving progress to a file and loading it back."""
        learner = CausalLearner(
            dataset_input=self.dataset,
            configurations=self.default_configurations,
            verbose=False,
            n_jobs=1,
            random_seed=42
        )
        # Run the model to generate progress
        try:
            learner.learn_model()
        except Exception as e:
            self.fail(f"Causal discovery run failed with exception: {e}")

        # Save progress
        progress_path = os.path.join(self.temp_dir.name, 'progress.pkl')
        try:
            learner.save_progress(progress_path)
            self.assertTrue(os.path.isfile(progress_path), "Progress file was not created.")
        except Exception as e:
            self.fail(f"Saving progress failed with exception: {e}")

        # Load progress
        try:
            loaded_learner = CausalLearner.load_progress(progress_path)
            self.assertIsInstance(loaded_learner, CausalLearner, "Loaded object is not an instance of CausalLearner.")
            # Verify specific attributes
            self.assertEqual(loaded_learner.configurations.dataset.dataset_name, 'Test Dataset', "Loaded dataset name mismatch.")
            self.assertIsNotNone(loaded_learner.opt_conf, "Loaded learner's optimal configuration is None.")
        except Exception as e:
            self.fail(f"Loading progress failed with exception: {e}")

    def test_handle_missing_values(self):
        """Test that CausalLearner handles missing values appropriately."""
        # Introduce missing values
        df_with_nan = self.dataset.data.copy()
        df_with_nan.loc[0, 'A'] = np.nan
        self.dataset.data = df_with_nan

        learner = CausalLearner(
            dataset_input=self.dataset,
            configurations=self.default_configurations,
            verbose=False,
            n_jobs=1,
            random_seed=42
        )
        try:
            opt_conf, mec_graph, run_time, library_results = learner.learn_model()
            # Depending on implementation, check how missing values are handled
            # For example, ensure that missing values were imputed or rows were dropped
            self.assertFalse(mec_graph.isnull().values.any(), "Output graph contains null values.")
        except Exception as e:
            self.fail(f"Causal discovery with missing values failed with exception: {e}")

    def test_run_with_categorical_data(self):
        """Test running causal discovery on categorical data."""
        categorical_df = pd.DataFrame({
            'A': np.random.choice(['yes', 'no'], size=100),
            'B': np.random.choice(['high', 'low'], size=100),
            'C': np.random.choice(['type1', 'type2', 'type3'], size=100)
        })
        categorical_dataset = Dataset(
            data=categorical_df,
            data_time_info={'n_lags': 0, 'time_lagged': False},
            time_series=False,
            dataset_name='Categorical Dataset'
        )

        configurations = Configurations(
            dataset=categorical_dataset,
            n_lags=0,
            time_lagged=False,
            time_series=False,
            n_jobs=1
        )

        learner = CausalLearner(
            dataset_input=categorical_dataset,
            configurations=configurations,
            verbose=False,
            n_jobs=1,
            random_seed=42
        )
        try:
            opt_conf, mec_graph, run_time, library_results = learner.learn_model()
            self.assertIsInstance(mec_graph, pd.DataFrame)
            self.assertFalse(mec_graph.empty, "Output graph is empty for categorical data.")
        except Exception as e:
            self.fail(f"Causal discovery on categorical data failed with exception: {e}")

    def test_run_with_continuous_data(self):
        """Test running causal discovery on continuous data."""
        continuous_df = pd.DataFrame({
            'A': np.random.randn(100),
            'B': np.random.randn(100),
            'C': np.random.randn(100)
        })
        continuous_dataset = Dataset(
            data=continuous_df,
            data_time_info={'n_lags': 0, 'time_lagged': False},
            time_series=False,
            dataset_name='Continuous Dataset'
        )

        configurations = Configurations(
            dataset=continuous_dataset,
            n_lags=0,
            time_lagged=False,
            time_series=False,
            n_jobs=1
        )

        learner = CausalLearner(
            dataset_input=continuous_dataset,
            configurations=configurations,
            verbose=False,
            n_jobs=1,
            random_seed=42
        )
        try:
            opt_conf, mec_graph, run_time, library_results = learner.learn_model()
            self.assertIsInstance(mec_graph, pd.DataFrame)
            self.assertFalse(mec_graph.empty, "Output graph is empty for continuous data.")
        except Exception as e:
            self.fail(f"Causal discovery on continuous data failed with exception: {e}")

    def test_run_with_time_lagged_data(self):
        """Test running causal discovery with time-lagged data."""
        time_series_df = pd.DataFrame({
            'A_t0': np.random.randn(100),
            'A_t1': np.random.randn(100),
            'B_t0': np.random.randn(100),
            'B_t1': np.random.randn(100)
        })
        time_series_dataset = Dataset(
            data=time_series_df,
            data_time_info={'n_lags': 1, 'time_lagged': True},
            time_series=True,
            dataset_name='Time Series Dataset'
        )

        configurations = Configurations(
            dataset=time_series_dataset,
            n_lags=1,
            time_lagged=True,
            time_series=True,
            n_jobs=1
        )

        learner = CausalLearner(
            dataset_input=time_series_dataset,
            configurations=configurations,
            verbose=False,
            n_jobs=1,
            random_seed=42
        )
        try:
            opt_conf, mec_graph, run_time, library_results = learner.learn_model()
            self.assertIsInstance(mec_graph, pd.DataFrame)
            self.assertFalse(mec_graph.empty, "Output graph is empty for time-lagged data.")
        except Exception as e:
            self.fail(f"Causal discovery with time-lagged data failed with exception: {e}")

    def test_run_with_large_dataset(self):
        """Test running causal discovery on a larger dataset."""
        large_df = pd.DataFrame({
            'A': np.random.randint(0, 2, size=1000),
            'B': np.random.randint(0, 2, size=1000),
            'C': np.random.randn(1000),
            'D': np.random.randn(1000),
            'E': np.random.randn(1000)
        })
        large_dataset = Dataset(
            data=large_df,
            data_time_info={'n_lags': 0, 'time_lagged': False},
            time_series=False,
            dataset_name='Large Dataset'
        )

        configurations = Configurations(
            dataset=large_dataset,
            n_lags=0,
            time_lagged=False,
            time_series=False,
            n_jobs=2  # Utilize multiple cores if supported
        )

        learner = CausalLearner(
            dataset_input=large_dataset,
            configurations=configurations,
            verbose=False,
            n_jobs=2,
            random_seed=42
        )
        try:
            opt_conf, mec_graph, run_time, library_results = learner.learn_model()
            self.assertIsInstance(mec_graph, pd.DataFrame)
            self.assertGreater(len(mec_graph), 0, "Output graph is empty for large dataset.")
        except Exception as e:
            self.fail(f"Causal discovery on large dataset failed with exception: {e}")

    def test_save_progress_creates_file(self):
        """Test that saving progress creates the expected file."""
        learner = CausalLearner(
            dataset_input=self.dataset,
            configurations=self.default_configurations,
            verbose=False,
            n_jobs=1,
            random_seed=42
        )
        # Run the model to generate progress
        try:
            learner.learn_model()
        except Exception as e:
            self.fail(f"Causal discovery run failed with exception: {e}")

        # Save progress
        progress_path = os.path.join(self.temp_dir.name, 'progress.pkl')
        try:
            learner.save_progress(progress_path)
            self.assertTrue(os.path.isfile(progress_path), "Progress file was not created.")
        except Exception as e:
            self.fail(f"Saving progress failed with exception: {e}")

    def test_load_progress_retrieves_data(self):
        """Test that loading progress retrieves the expected state."""
        learner = CausalLearner(
            dataset_input=self.dataset,
            configurations=self.default_configurations,
            verbose=False,
            n_jobs=1,
            random_seed=42
        )
        # Run the model to generate progress
        try:
            learner.learn_model()
        except Exception as e:
            self.fail(f"Causal discovery run failed with exception: {e}")

        # Save progress
        progress_path = os.path.join(self.temp_dir.name, 'progress.pkl')
        try:
            learner.save_progress(progress_path)
            self.assertTrue(os.path.isfile(progress_path), "Progress file was not created.")
        except Exception as e:
            self.fail(f"Saving progress failed with exception: {e}")

        # Load progress
        try:
            loaded_learner = CausalLearner.load_progress(progress_path)
            self.assertIsInstance(loaded_learner, CausalLearner, "Loaded object is not an instance of CausalLearner.")
            # Verify specific attributes
            self.assertEqual(loaded_learner.configurations.dataset.dataset_name, 'Test Dataset', "Loaded dataset name mismatch.")
            self.assertIsNotNone(loaded_learner.opt_conf, "Loaded learner's optimal configuration is None.")
        except Exception as e:
            self.fail(f"Loading progress failed with exception: {e}")



#tst = TestCausalLearner()
#tst.setUp()
#tst.test_default_configurations()