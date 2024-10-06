import unittest
import os
import pandas as pd
from ETIA.CausalLearning.CausalLearner import CausalLearner
from ETIA.data import Dataset
from ETIA.CausalLearning.configurations import Configurations

class TestCausalLearner(unittest.TestCase):

    def create_sample_learner(self):
        """Helper function to create a sample CausalLearner instance with a real dataset."""
        data = {
            'A': [1, 2, 3, 4, 5, 1, 5, 6, 7, 5, 2],
            'B': [5, 4, 3, 2, 1, 4, 2, 4, 5, 6, 7],
            'C': [2, 3, 4, 5, 6, 5, 4, 5, 5, 6, 6]
        }
        df = pd.DataFrame(data)
        dataset = Dataset(
            data=df,
            data_time_info={'n_lags': 0, 'time_lagged': False},
            time_series=False,
            dataset_name='Test Dataset'
        )
        configurations = Configurations(dataset=dataset, verbose=True)
        learner = CausalLearner(dataset, configurations, verbose=True, random_seed=42)
        return learner

    def test_init_with_dataset_instance(self):
        learner = self.create_sample_learner()
        self.assertIsInstance(learner.dataset, Dataset)

    def test_init_with_pandas_dataframe(self):
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        learner = CausalLearner(df)
        self.assertEqual(learner.dataset.dataset_name, 'Preloaded Dataset')

    def test_set_configurations_valid(self):
        learner = self.create_sample_learner()
        new_config = Configurations(dataset=learner.dataset)
        learner.set_configurations(new_config)
        self.assertEqual(learner.configurations, new_config)

    def test_set_configurations_invalid(self):
        learner = self.create_sample_learner()
        with self.assertRaises(TypeError):
            learner.set_configurations('invalid_configuration')

    def test_set_dataset_valid(self):
        learner = self.create_sample_learner()
        new_data = {
            'A': [5, 6, 7, 8],
            'B': [8, 7, 6, 5]
        }
        new_df = pd.DataFrame(new_data)
        new_dataset = Dataset(
            data=new_df,
            data_time_info={'n_lags': 0, 'time_lagged': False},
            time_series=False,
            dataset_name='New Test Dataset'
        )
        learner.set_dataset(new_dataset)
        self.assertEqual(learner.dataset, new_dataset)

    def test_set_dataset_invalid(self):
        learner = self.create_sample_learner()
        with self.assertRaises(TypeError):
            learner.set_dataset('invalid_dataset')

    def test_learn_model_success(self):
        learner = self.create_sample_learner()
        result = learner.learn_model()
        self.assertIn('optimal_conf', result)
        self.assertIn('matrix_mec_graph', result)
        self.assertIn('run_time', result)

    def test_save_progress_default_path(self):
        learner = self.create_sample_learner()
        with unittest.mock.patch('builtins.open', unittest.mock.mock_open()) as mocked_file:
            learner.save_progress()
            mocked_file.assert_called_once_with(os.path.join(learner.results_folder, 'Experiment.pkl'), 'wb')

    def test_save_progress_custom_path(self):
        learner = self.create_sample_learner()
        with unittest.mock.patch('builtins.open', unittest.mock.mock_open()) as mocked_file:
            learner.save_progress('custom_path.pkl')
            mocked_file.assert_called_once_with('custom_path.pkl', 'wb')

    def test_load_progress(self):
        learner = self.create_sample_learner()
        with unittest.mock.patch('builtins.open', unittest.mock.mock_open()):
            with unittest.mock.patch('pickle.load', return_value=learner):
                loaded_learner = CausalLearner.load_progress('path.pkl')
                self.assertEqual(loaded_learner, learner)

    def test_add_configurations_from_file(self):
        learner = self.create_sample_learner()
        with unittest.mock.patch('ETIA.CausalLearning.configurations.Configurations.add_configurations_from_file') as mock_add_configs:
            learner.add_configurations_from_file('config_file.json')
            mock_add_configs.assert_called_once_with('config_file.json')

    def test_update_learnt_model(self):
        learner = self.create_sample_learner()
        with unittest.mock.patch('ETIA.CausalLearning.CDHPO.OCT.run_new', return_value=('new_conf', 'new_mec_graph', None)):
            learner.update_learnt_model()
            self.assertEqual(learner.opt_conf, 'new_conf')
            self.assertEqual(learner.matrix_mec_graph, 'new_mec_graph')

    def test_get_best_model_between_algorithms(self):
        learner = self.create_sample_learner()
        with unittest.mock.patch('ETIA.CausalLearning.CDHPO.OCT.find_best_config', return_value='best_config'):
            result = learner.get_best_model_between_algorithms(['algo1', 'algo2'])
            self.assertEqual(result, 'best_config')

    def test_get_best_model_between_family(self):
        learner = self.create_sample_learner()
        with unittest.mock.patch('ETIA.CausalLearning.CDHPO.OCT.find_best_config', return_value='family_best'):
            result = learner.get_best_model_between_family()
            self.assertEqual(result, 'family_best')


if __name__ == '__main__':
    unittest.main()
