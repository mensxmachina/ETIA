# test_causal_learner.py

import unittest
import pandas as pd
import numpy as np
import os
import shutil
import tempfile
import warnings
import json

# Suppress warnings during testing
warnings.filterwarnings("ignore")

# Import the CausalLearner class
from ETIA.CausalLearning.CausalLearner import CausalLearner
from ETIA.data.Dataset import Dataset
from ETIA.CausalLearning.configurations import Configurations

# Import other necessary modules (mocking where appropriate)
from unittest.mock import MagicMock, patch

class TestCausalLearnerInitialization(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for saving progress
        self.test_dir = tempfile.mkdtemp()

        # Sample data
        np.random.seed(42)
        self.df = pd.DataFrame({
            'A': np.random.rand(100),
            'B': np.random.rand(100),
            'C': np.random.rand(100),
            'D': np.random.rand(100)
        })

        # Save DataFrame to a CSV file to match the configuration
        self.dataset_file = os.path.join(self.test_dir, 'test.csv')
        self.df.to_csv(self.dataset_file, index=False)

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_init_with_configuration_file(self):
        # Create the configuration as specified
        default_conf = {
            "Dataset": {
                "dataset_name": self.dataset_file,
                "time_lagged": False,
                "n_lags": 0  # Adjusted to 0 since we don't have time-lagged data
            },
            "Results_folder_path": self.test_dir,
            "causal_sufficiency": True,
            "assume_faithfulness": False,
            "OCT": {
                "alpha": 0.01,
                "n_permutations": 10,
                "variables_type": "continuous",
                "out_of_sample_protocol": {
                    "name": "KFoldCV",
                    "parameters": {
                        "folds": 5,
                        "folds_to_run": 2
                    }
                },
                "Regressor_parameters": {
                    "name": "RandomForestRegressor",
                    "parameters": {
                        "n_trees": 100,
                        "min_samples_leaf": 0.01,
                        "max_depth": 10
                    }
                },
                "CausalDiscoveryAlgorithms": {
                    "include_algs": ['pc']
                }
            }
        }

        # Save the configuration to a JSON file
        config_path = os.path.join(self.test_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(default_conf, f)

        # Initialize the CausalLearner with the configuration file
        configurations = Configurations(
            dataset=None,
            conf_file=config_path,
            verbose=False
        )
        learner = CausalLearner(
            dataset_input=configurations.dataset,
            configurations=configurations,
            verbose=False
        )
        self.assertIsInstance(learner.configurations, Configurations)
        self.assertEqual(learner.configurations.results_folder, self.test_dir)
        print(configurations.cdhpo_params.configs)
        self.assertTrue('pc' in learner.configurations.cdhpo_params.configs)

    def test_init_with_invalid_configuration_file(self):
        # Provide an invalid configuration file path
        invalid_config_path = os.path.join(self.test_dir, 'invalid_config.json')
        with self.assertRaises(FileNotFoundError):
            configurations = Configurations(
                dataset=None,
                conf_file=invalid_config_path,
                verbose=False
            )

    def test_init_with_incomplete_configuration(self):
        # Create an incomplete configuration
        incomplete_conf = {
            "Dataset": {
                # Missing 'dataset_name'
                "time_lagged": False,
                "n_lags": 0
            },
            "OCT": {
                "alpha": 0.01,
                "n_permutations": 10,
                "variables_type": "continuous",
            }
        }

        # Save the incomplete configuration to a JSON file
        incomplete_config_path = os.path.join(self.test_dir, 'incomplete_config.json')
        with open(incomplete_config_path, 'w') as f:
            json.dump(incomplete_conf, f)

        with self.assertRaises(ValueError):
            configurations = Configurations(
                dataset=None,
                conf_file=incomplete_config_path,
                verbose=False
            )

class TestCausalLearnerMethods(unittest.TestCase):
    def setUp(self):
        # Create sample data
        np.random.seed(42)
        self.df = pd.DataFrame({
            'A': np.random.rand(100),
            'B': np.random.rand(100),
            'C': np.random.rand(100),
            'D': np.random.rand(100)
        })

        # Create a temporary directory for saving progress
        self.test_dir = tempfile.mkdtemp()

        # Save DataFrame to a CSV file to match the configuration
        self.dataset_file = os.path.join(self.test_dir, 'test.csv')
        self.df.to_csv(self.dataset_file, index=False)

        # Create the configuration as specified
        self.default_conf = {
            "Dataset": {
                "dataset_name": self.dataset_file,
                "time_lagged": False,
                "n_lags": 0
            },
            "Results_folder_path": self.test_dir,
            "causal_sufficiency": True,
            "assume_faithfulness": True,
            "OCT": {
                "alpha": 0.01,
                "n_permutations": 10,
                "variables_type": "continuous",
                "out_of_sample_protocol": {
                    "name": "KFoldCV",
                    "parameters": {
                        "folds": 5,
                        "folds_to_run": 2
                    }
                },
                "Regressor_parameters": {
                    "name": "RandomForestRegressor",
                    "parameters": {
                        "n_trees": 100,
                        "min_samples_leaf": 0.01,
                        "max_depth": 10
                    }
                },
                "CausalDiscoveryAlgorithms": {
                    "include_algs": ['pc']
                }
            }
        }

        # Save the configuration to a JSON file
        self.config_path = os.path.join(self.test_dir, 'config.json')
        with open(self.config_path, 'w') as f:
            json.dump(self.default_conf, f)

        # Initialize the CausalLearner with the configuration file
        self.configurations = Configurations(
            dataset=None,
            conf_file=self.config_path,
            verbose=False
        )
        self.learner = CausalLearner(
            dataset_input=self.configurations.dataset,
            configurations=self.configurations,
            verbose=False
        )

        # Mocking the CDHPO object to avoid actual computation
        self.learner.cdhpo = MagicMock()
        self.learner.cdhpo.run.return_value = (
            {'name': 'pc', 'alpha': 0.01},  # opt_conf
            np.zeros((4, 4)),               # matrix_mec_graph
            np.zeros((4, 4)),               # matrix_graph
            {}                              # library_results
        )

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_learn_model(self):
        result = self.learner.learn_model()
        self.assertIn('optimal_conf', result)
        self.assertIn('matrix_mec_graph', result)
        self.assertIn('matrix_graph', result)
        self.assertIn('run_time', result)
        self.assertIn('library_results', result)
        self.assertEqual(result['optimal_conf']['name'], 'pc')
        self.assertEqual(result['optimal_conf']['alpha'], 0.01)

    def test_print_results(self):
        # Mock the print function
        with patch('builtins.print') as mocked_print:
            self.learner.opt_conf = {'name': 'pc', 'alpha': 0.01}
            self.learner.matrix_mec_graph = np.zeros((4, 4))
            self.learner.print_results()
            mocked_print.assert_any_call('Best Causal Discovery configuration was:', 'pc')
            mocked_print.assert_any_call('With parameters:')
            mocked_print.assert_any_call('alpha: 0.01')

    def test_save_and_load_progress(self):
        # Save progress
        save_path = os.path.join(self.test_dir, 'progress.pkl')
        learner = CausalLearner(
            dataset_input=self.configurations.dataset,
            configurations=self.configurations,
            verbose=False
        )

        # Set random values to avoid computations and
        # using Mock CDHPO instance (mock instances cause problems in pickle)
        learner.opt_conf = {}
        learner.opt_conf['name'] = 'pc'
        learner.opt_conf['alpha'] = 0.01

        learner.save_progress(path=save_path)
        # Load the progress
        loaded_learner = CausalLearner.load_progress(path=save_path)
        self.assertIsInstance(loaded_learner, CausalLearner)
        self.assertEqual(loaded_learner.opt_conf['name'], 'pc')
        self.assertEqual(loaded_learner.opt_conf['alpha'], 0.01)

    def test_add_configurations_from_file(self):
        # Create an additional configuration file to add
        additional_conf = {
            "OCT": {
                "CausalDiscoveryAlgorithms": {
                    "fci": {
                        "alpha": [0.05],
                        "ci_test": ["FisherZ"]
                    }
                }
            }
        }
        additional_config_path = os.path.join(self.test_dir, 'additional_config.json')
        with open(additional_config_path, 'w') as f:
            json.dump(additional_conf, f)

        # Mock the add_configurations_from_file method
        self.learner.configurations.add_configurations_from_file(additional_config_path)
        # Check if the new algorithm is added
        self.assertTrue('fci' in self.learner.configurations.cdhpo_params.configs)

    def test_update_learnt_model(self):
        # Mock the cdhpo.run_new method
        self.learner.cdhpo.run_new.return_value = (
            {'name': 'fci', 'alpha': 0.05},
            np.zeros((4, 4)),
            np.zeros((4, 4))
        )
        self.learner.update_learnt_model()
        self.assertEqual(self.learner.opt_conf['name'], 'fci')
        self.assertEqual(self.learner.opt_conf['alpha'], 0.05)

    def test_get_best_model_between_algorithms(self):
        # Mock the cdhpo.find_best_config method
        self.learner.cdhpo.find_best_config.return_value = {'name': 'pc', 'alpha': 0.01}
        best_config = self.learner.get_best_model_between_algorithms(['pc', 'fci'])
        self.assertEqual(best_config, {'name': 'pc', 'alpha': 0.01})

    def test_get_best_model_between_family(self):
        # Mock the cdhpo.find_best_config method
        self.learner.cdhpo.find_best_config.return_value = {'name': 'pc', 'alpha': 0.01}
        best_config = self.learner.get_best_model_between_family(causal_sufficiency=False)
        self.assertEqual(best_config, {'name': 'pc', 'alpha': 0.01})

class TestCausalLearnerErrorHandling(unittest.TestCase):
    def setUp(self):
        # Create sample data
        np.random.seed(42)
        self.df = pd.DataFrame({
            'A': np.random.rand(100),
            'B': np.random.rand(100),
            'C': np.random.rand(100),
            'D': np.random.rand(100)
        })

        # Create a temporary directory for saving progress
        self.test_dir = tempfile.mkdtemp()

        # Save DataFrame to a CSV file to match the configuration
        self.dataset_file = os.path.join(self.test_dir, 'test.csv')
        self.df.to_csv(self.dataset_file, index=False)

        # Create the configuration as specified
        self.default_conf = {
            "Dataset": {
                "dataset_name": self.dataset_file,
                "time_lagged": False,
                "n_lags": 0
            },
            "Results_folder_path": self.test_dir,
            "causal_sufficiency": True,
            "assume_faithfulness": True,
            "OCT": {
                "alpha": 0.01,
                "n_permutations": 10,
                "variables_type": "continuous",
                "out_of_sample_protocol": {
                    "name": "KFoldCV",
                    "parameters": {
                        "folds": 5,
                        "folds_to_run": 2
                    }
                },
                "Regressor_parameters": {
                    "name": "RandomForestRegressor",
                    "parameters": {
                        "n_trees": 100,
                        "min_samples_leaf": 0.01,
                        "max_depth": 10
                    }
                },
                "CausalDiscoveryAlgorithms": {
                    "include_algs": ['pc']
                }
            }
        }

        # Save the configuration to a JSON file
        self.config_path = os.path.join(self.test_dir, 'config.json')
        with open(self.config_path, 'w') as f:
            json.dump(self.default_conf, f)

        # Initialize the CausalLearner with the configuration file
        self.configurations = Configurations(
            dataset=None,
            conf_file=self.config_path,
            verbose=False
        )
        self.learner = CausalLearner(
            dataset_input=self.configurations.dataset,
            configurations=self.configurations,
            verbose=False
        )

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_learn_model_error(self):
        # Use patch.object to mock the 'run' method of cdhpo
        with patch.object(
                self.learner.cdhpo,
                'run',
                side_effect=Exception('CDHPO run failed')
        ):
            with self.assertRaises(Exception) as context:
                self.learner.learn_model()

            # Optional: Verify the exception message
            self.assertIn('CDHPO run failed', str(context.exception))

    def test_invalid_algorithm_in_get_best_model(self):
        # Use patch.object to mock find_best_config method of cdhpo
        with patch.object(
                self.learner.cdhpo,
                'find_best_config',
                side_effect=RuntimeError('Algorithm not found')
        ):
            with self.assertRaises(RuntimeError) as context:
                self.learner.get_best_model_between_algorithms(['invalid_algo'])

            # Optional: Verify the exception message
            self.assertEqual(str(context.exception), 'Algorithm not found')

    def test_invalid_configuration_file(self):
        # Provide a non-existent configuration file
        with self.assertRaises(FileNotFoundError):
            configurations = Configurations(
                dataset=None,
                conf_file='non_existent_file.json',
                verbose=False
            )

    def test_invalid_dataset_input(self):
        # Try initializing with invalid dataset input
        with self.assertRaises(ValueError):
            CausalLearner(dataset_input=123)

class TestCausalLearnerIntegration(unittest.TestCase):
    def setUp(self):
        # Create sample data with a known causal structure
        np.random.seed(42)
        A = np.random.rand(100)
        B = 2 * A + np.random.rand(100) * 0.1
        C = 3 * B + np.random.rand(100) * 0.1
        D = np.random.rand(100)
        self.df = pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D})

        # Create a temporary directory for saving progress
        self.test_dir = tempfile.mkdtemp()

        # Save DataFrame to a CSV file
        self.dataset_file = os.path.join(self.test_dir, 'test.csv')
        self.df.to_csv(self.dataset_file, index=False)

        # Create the configuration as specified
        self.default_conf = {
            "Dataset": {
                "dataset_name": self.dataset_file,
                "time_lagged": False,
                "n_lags": 0
            },
            "Results_folder_path": self.test_dir,
            "causal_sufficiency": True,
            "assume_faithfulness": True,
            "OCT": {
                "alpha": 0.01,
                "n_permutations": 10,
                "variables_type": "continuous",
                "out_of_sample_protocol": {
                    "name": "KFoldCV",
                    "parameters": {
                        "folds": 5,
                        "folds_to_run": 2
                    }
                },
                "Regressor_parameters": {
                    "name": "RandomForestRegressor",
                    "parameters": {
                        "n_trees": 100,
                        "min_samples_leaf": 0.01,
                        "max_depth": 10
                    }
                },
                "CausalDiscoveryAlgorithms": {
                    "include_algs": ['pc']
                }
            }
        }

        # Save the configuration to a JSON file
        self.config_path = os.path.join(self.test_dir, 'config.json')
        with open(self.config_path, 'w') as f:
            json.dump(self.default_conf, f)

        # Initialize the CausalLearner with the configuration file
        self.configurations = Configurations(
            dataset=None,
            conf_file=self.config_path,
            verbose=False
        )
        self.learner = CausalLearner(
            dataset_input=self.configurations.dataset,
            configurations=self.configurations,
            random_seed=42,
            verbose=False
        )

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_full_causal_learning(self):
        # Run the full causal learning process
        learner = CausalLearner(dataset_input='example_dataset.csv')
        result = learner.learn_model()
        self.assertIn('optimal_conf', result)
        self.assertIn('matrix_mec_graph', result)
        self.assertIn('matrix_graph', result)
        self.assertIn('run_time', result)
        self.assertIn('library_results', result)
        # Check that the learned graph has the expected shape
        self.assertEqual(result['matrix_mec_graph'].shape, (5, 5))
        # Since we are using the PC algorithm, we might expect certain edges
        # Note: In a real test, we would compare the learned graph with the true causal graph

if __name__ == '__main__':
    unittest.main()
