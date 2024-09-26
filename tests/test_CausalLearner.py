import unittest
from unittest.mock import MagicMock, patch
from ETIA.CausalLearning.CausalLearner import CausalLearner
from ETIA.CausalLearning.configurations.configurations import Configurations
from ETIA.data.Dataset import Dataset


class TestCausalLearner(unittest.TestCase):

    def setUp(self):
        self.dataset_mock = MagicMock(spec=Dataset)
        self.configurations_mock = MagicMock(spec=Configurations)
        self.configurations_mock.dataset = self.dataset_mock
        self.configurations_mock.results_folder = "/path/to/results"
        self.configurations_mock.cdhpo_params = {}

    @patch('ETIA.CausalLearning.util.logger.get_logger')
    @patch('ETIA.CausalLearning.configurations.configurations.Configurations')
    def test_initialization_with_default_configurations(self, mock_configurations, mock_get_logger):
        mock_logger = mock_get_logger.return_value
        mock_configurations.return_value = self.configurations_mock
        learner = CausalLearner(dataset=self.dataset_mock)
        self.assertIsInstance(learner.configurations, Configurations)
        self.assertEqual(learner.dataset, self.dataset_mock)
        mock_logger.info.assert_called_with('Experiment object initialized')

    @patch('ETIA.CausalLearning.util.logger.get_logger')
    def test_initialization_with_custom_configurations(self, mock_get_logger):
        mock_logger = mock_get_logger.return_value
        learner = CausalLearner(configurations=self.configurations_mock)
        self.assertEqual(learner.configurations, self.configurations_mock)
        self.assertEqual(learner.dataset, self.dataset_mock)
        mock_logger.info.assert_called_with('Experiment object initialized')

    def test_initialization_with_invalid_configurations(self):
        with self.assertRaises(TypeError):
            CausalLearner(configurations="invalid_configurations")

    @patch('ETIA.CausalLearning.util.logger.get_logger')
    @patch('ETIA.CausalLearning.CDHPO.OCT.OCT')
    def test_learn_model(self, mock_OCT, mock_get_logger):
        mock_logger = mock_get_logger.return_value
        mock_OCT_instance = mock_OCT.return_value
        mock_OCT_instance.run.return_value = ('opt_conf', 'matrix_mec_graph', 'library_results')

        learner = CausalLearner(configurations=self.configurations_mock)
        results = learner.learn_model()

        self.assertEqual(results, ('opt_conf', 'matrix_mec_graph', 'library_results'))
        mock_logger.info.assert_any_call('Starting OCT Run')
        mock_logger.info.assert_any_call('CDHPO Runtime: ' + learner.run_time)

    def test_set_dataset(self):
        learner = CausalLearner(configurations=self.configurations_mock)
        new_dataset_mock = MagicMock(spec=Dataset)
        learner.set_dataset(new_dataset_mock)
        self.assertEqual(learner.dataset, new_dataset_mock)

        with self.assertRaises(TypeError):
            learner.set_dataset("invalid_dataset")

    def test_set_configurations(self):
        learner = CausalLearner(configurations=self.configurations_mock)
        new_configurations_mock = MagicMock(spec=Configurations)
        new_configurations_mock.dataset = self.dataset_mock

        learner.set_configurations(new_configurations_mock)
        self.assertEqual(learner.configurations, new_configurations_mock)
        self.assertEqual(learner.dataset, self.dataset_mock)

        with self.assertRaises(TypeError):
            learner.set_configurations("invalid_configurations")

    @patch('pickle.dump')
    def test_save_progress(self, mock_pickle_dump):
        learner = CausalLearner(configurations=self.configurations_mock)
        learner.save_progress('/path/to/save')
        mock_pickle_dump.assert_called_once()

    @patch('pickle.load')
    def test_load_progress(self, mock_pickle_load):
        learner = CausalLearner(configurations=self.configurations_mock)
        mock_pickle_load.return_value = learner
        result = learner.load_progress('/path/to/load')
        self.assertEqual(result, learner)
        mock_pickle_load.assert_called_once()

    @patch('ETIA.CausalLearning.configurations.configurations.Configurations.add_conf_from_file')
    def test_add_configurations_from_file(self, mock_add_conf_from_file):
        learner = CausalLearner(configurations=self.configurations_mock)
        learner.add_configurations_from_file('config.json')
        mock_add_conf_from_file.assert_called_once_with('config.json')

    @patch('ETIA.CausalLearning.CDHPO.OCT.OCT')
    def test_update_learnt_model(self, mock_OCT):
        mock_OCT_instance = mock_OCT.return_value
        mock_OCT_instance.run_new.return_value = ('opt_conf', 'matrix_mec_graph', 'library_results')

        learner = CausalLearner(configurations=self.configurations_mock)
        learner.update_learnt_model()
        self.assertEqual(learner.opt_conf, 'opt_conf')
        self.assertEqual(learner.matrix_mec_graph, 'matrix_mec_graph')

    @patch('ETIA.CausalLearning.CDHPO.OCT.OCT')
    def test_get_best_model_between_algorithms(self, mock_OCT):
        mock_OCT_instance = mock_OCT.return_value
        mock_OCT_instance.find_best_config.return_value = 'best_config'

        learner = CausalLearner(configurations=self.configurations_mock)
        best_config = learner.get_best_model_between_algorithms(['algo1', 'algo2'])
        self.assertEqual(best_config, 'best_config')

    @patch('ETIA.CausalLearning.CDHPO.OCT.OCT')
    @patch('ETIA.CausalLearning.algorithms.causaldiscoveryalgorithms')
    def test_get_best_model_between_family(self, mock_causaldiscoveryalgorithms, mock_OCT):
        mock_OCT_instance = mock_OCT.return_value
        mock_OCT_instance.find_best_config.return_value = 'best_config'

        mock_causaldiscoveryalgorithms.cd_algorithms = {
            'algo1': MagicMock(admit_latent_variables=True, assume_faithfulness=True, is_output_mec=True,
                               accepts_missing_values=True),
            'algo2': MagicMock(admit_latent_variables=False, assume_faithfulness=False, is_output_mec=False,
                               accepts_missing_values=False),
        }

        learner = CausalLearner(configurations=self.configurations_mock)
        best_config = learner.get_best_model_between_family(admit_latent_variables=True, assume_faithfulness=True,
                                                            is_output_mec=True, accepts_missing_values=True)
        self.assertEqual(best_config, 'best_config')
        mock_OCT_instance.find_best_config.assert_called_once_with(['algo1'])


if __name__ == '__main__':
    unittest.main()