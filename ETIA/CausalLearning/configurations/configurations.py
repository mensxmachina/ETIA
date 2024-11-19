import json
import os
from typing import Optional

from ...data.Dataset import Dataset  # Corrected import
from .parameters import CDHPO_Parameters
from .default_parameters import default_conf
from .class_causal_configurator import CausalDiscoveryConfigurator
from ..utils import get_logger


class Configurations:
    """
    Configurations class for setting up the causal discovery experiment.

    Parameters
    ----------
    dataset : Dataset
        The dataset object.
    n_lags : int, optional
        Number of lags (for time series).
    time_lagged : bool, optional
        Indicates if the data is time-lagged.
    time_series : bool, optional
        Indicates if the dataset is time series data.
    conf_file : str, optional
        JSON configuration file containing parameters for the causal discovery experiment.
    n_jobs : int, optional
        Number of jobs to use for parallel processing.
    verbose : bool, optional
        Whether to print debug information.

    Attributes
    ----------
    cdhpo_params : CDHPOParameters
        Parameters for the CDHPO algorithm.
    results_folder : str
        Folder path for storing results.
    """

    def __init__(
        self,
        dataset: Dataset = None,
        n_lags: int = 0,
        time_lagged: bool = False,
        time_series: bool = False,
        conf_file: Optional[str] = None,
        n_jobs: Optional[int] = -1,
        verbose=False
    ):
        self.n_jobs = n_jobs
        self.dataset = dataset
        self.n_lags = n_lags
        self.time_lagged = time_lagged
        self.time_series = time_series
        self.verbose = verbose
        self.logger = get_logger(name=__name__, verbose=self.verbose)
        self.cdhpo_params = CDHPO_Parameters()
        self.results_folder = './'

        # If conf_file is provided, process it. Otherwise, set default configurations.
        if conf_file is not None and conf_file.endswith('.json'):
            self.conf_file = conf_file
            self.process_conf_file()  # Process the JSON file
        else:
            self.set_default_configuration()  # Set default configuration based on dataset

        self.logger.info('Configurations object has been initialized')

    def set_default_configuration(self):
        """
        Set default configurations based on the dataset when no JSON configuration file is provided.
        """
        default_conf['Dataset'] = {
            'dataset_name': self.dataset.dataset_name or 'Preloaded Dataset',
            'time_lagged': self.time_lagged,
            'n_lags': self.n_lags,
            'time_series': self.time_series,
        }

        self.dataset_info = self.dataset.get_info()

        # Configure CDHPO parameters using default configurations
        oct_json_params = default_conf['OCT']
        self.cdhpo_params.init_main_params(
            alpha=oct_json_params['alpha'],
            n_permutations=oct_json_params['n_permutations'],
            causal_sufficiency=default_conf['causal_sufficiency'],
            n_jobs=self.n_jobs,
            variables_type=oct_json_params['variables_type'],
        )
        self.cdhpo_params.set_regressor(
            name=oct_json_params['Regressor_parameters']['name'],
            parameters=oct_json_params['Regressor_parameters']['parameters'],
        )
        self.cdhpo_params.set_oos_protocol(
            name=oct_json_params['out_of_sample_protocol']['name'],
            parameters=oct_json_params['out_of_sample_protocol']['parameters'],
        )

        # Configure causal discovery algorithms
        if 'CausalDiscoveryAlgorithms' not in oct_json_params:
            configurator = CausalDiscoveryConfigurator()
            oct_json_params['CausalDiscoveryAlgorithms'] = configurator.create_causal_configs(
                data_type=self.dataset_info['data_type'],
                causal_sufficiency=default_conf['causal_sufficiency'],
                assume_faithfulness=default_conf['assume_faithfulness'],
                time_lagged=self.time_lagged,
                time_series=self.time_series,
            )
        else:
            if ('include_algs' in oct_json_params['CausalDiscoveryAlgorithms'] or 'exclude_algs' in oct_json_params[
                'CausalDiscoveryAlgorithms']):
                configurator = CausalDiscoveryConfigurator()
                if ('exclude_algs' in oct_json_params['CausalDiscoveryAlgorithms'] and 'include_algs' in
                        oct_json_params['CausalDiscoveryAlgorithms']):
                    exlcude_algs = oct_json_params['CausalDiscoveryAlgorithms']['exclude_algs']
                    include_algs = oct_json_params['CausalDiscoveryAlgorithms']['include_algs']
                elif ('exclude_algs' in oct_json_params['CausalDiscoveryAlgorithms']):
                    exlcude_algs = oct_json_params['CausalDiscoveryAlgorithms']['exclude_algs']
                    include_algs = []
                else:
                    exlcude_algs = []
                    include_algs = oct_json_params['CausalDiscoveryAlgorithms']['include_algs']

                oct_json_params['CausalDiscoveryAlgorithms'] = configurator.create_causal_configs(
                    data_type=self.dataset_info['data_type'],
                    causal_sufficiency=default_conf['causal_sufficiency'],
                    assume_faithfulness=default_conf['assume_faithfulness'],
                    time_lagged=self.time_lagged,
                    time_series=self.time_series,
                    include_algs=include_algs,
                    exclude_algs=exlcude_algs
                )

        self.cdhpo_params.set_cd_algorithms(oct_json_params['CausalDiscoveryAlgorithms'], self.dataset_info)

    def process_conf_file(self):
        """
        Process the JSON file containing all vital information, such as algorithms, algorithm parameters, run mode, etc.
        """
        with open(self.conf_file) as f:
            conf = json.load(f)

        dataset_conf = conf.get('Dataset', {})
        self.time_lagged = dataset_conf.get('time_lagged', self.time_lagged)
        self.n_lags = dataset_conf.get('n_lags', self.n_lags)
        self.time_series = dataset_conf.get('time_series', self.time_series)

        # Initialize the dataset from the configuration file
        if 'dataset_name' in dataset_conf:
            self.dataset = Dataset(
                filename=dataset_conf['dataset_name'],
                data_time_info={'n_lags': self.n_lags, 'time_lagged': self.time_lagged},
                time_series=self.time_series,
            )
        else:
            raise ValueError("Configuration file must include 'dataset_name' in 'Dataset' section.")

        self.results_folder = conf.get('Results_folder_path', self.results_folder)

        # Process CDHPO parameters
        oct_json_params = conf['OCT']
        if('n_jobs' in oct_json_params):
            self.n_jobs = oct_json_params['n_jobs']
        else:
            self.n_jobs = os.cpu_count()
        self.cdhpo_params.init_main_params(
            alpha=oct_json_params['alpha'],
            n_permutations=oct_json_params['n_permutations'],
            causal_sufficiency=conf['causal_sufficiency'],
            variables_type=oct_json_params['variables_type'],
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        )
        self.cdhpo_params.set_regressor(
            name=oct_json_params['Regressor_parameters']['name'],
            parameters=oct_json_params['Regressor_parameters']['parameters'],
        )
        self.cdhpo_params.set_oos_protocol(
            name=oct_json_params['out_of_sample_protocol']['name'],
            parameters=oct_json_params['out_of_sample_protocol']['parameters'],
        )

        self.dataset_info = self.dataset.get_info()

        # Configure causal discovery algorithms
        if 'CausalDiscoveryAlgorithms' not in oct_json_params:
            configurator = CausalDiscoveryConfigurator()
            oct_json_params['CausalDiscoveryAlgorithms'] = configurator.create_causal_configs(
                data_type=self.dataset_info['data_type'],
                causal_sufficiency=conf['causal_sufficiency'],
                assume_faithfulness=conf['assume_faithfulness'],
                time_lagged=self.time_lagged,
                time_series=self.time_series,
            )
        else:
            if('include_algs' in oct_json_params['CausalDiscoveryAlgorithms'] or 'exclude_algs' in oct_json_params['CausalDiscoveryAlgorithms']):
                configurator = CausalDiscoveryConfigurator()
                if('exclude_algs' in oct_json_params['CausalDiscoveryAlgorithms'] and 'include_algs' in oct_json_params['CausalDiscoveryAlgorithms']):
                    exlcude_algs = oct_json_params['CausalDiscoveryAlgorithms']['exclude_algs']
                    include_algs = oct_json_params['CausalDiscoveryAlgorithms']['include_algs']
                elif('exclude_algs' in oct_json_params['CausalDiscoveryAlgorithms']):
                    exlcude_algs = oct_json_params['CausalDiscoveryAlgorithms']['exclude_algs']
                    include_algs = []
                else:
                    exlcude_algs = []
                    include_algs = oct_json_params['CausalDiscoveryAlgorithms']['include_algs']

                oct_json_params['CausalDiscoveryAlgorithms'] = configurator.create_causal_configs(
                    data_type=self.dataset_info['data_type'],
                    causal_sufficiency=conf['causal_sufficiency'],
                    assume_faithfulness=conf['assume_faithfulness'],
                    time_lagged=self.time_lagged,
                    time_series=self.time_series,
                    include_algs=include_algs,
                    exclude_algs=exlcude_algs
                )

        self.cdhpo_params.set_cd_algorithms(oct_json_params['CausalDiscoveryAlgorithms'], self.dataset_info)

    def add_configurations_from_file(self, filename: str) -> None:
        """
        Add additional configurations to the experiment from a JSON file.

        Parameters
        ----------
        filename : str
            The filename of the JSON file containing configurations.
        """
        with open(filename) as f:
            conf = json.load(f)
        oct_json_params = conf['OCT']
        causal_algorithms = oct_json_params['CausalDiscoveryAlgorithms']
        for algo, params in causal_algorithms.items():
            if algo not in self.cdhpo_params.configs:
                self.cdhpo_params.add_cd_algorithm(algo, params, self.dataset_info)
            else:
                self.cdhpo_params.add_cd_algorithm_parameters(algo, params)
        self.logger.info(f'Additional configurations added from {filename}')
