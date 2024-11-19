# ETIA/CausalLearning/CausalLearner.py

import pickle
import time
from typing import Optional, Union
import pandas as pd

from .algorithms import causaldiscoveryalgorithms
from ..data.Dataset import Dataset
from .configurations import Configurations, configurations
from .configurations.causal_configs import causal_configs
from .CDHPO import OCT
from ..utils import get_logger
from .CausalModel.utils import pywhy_graph_to_matrix
from ..utils.jvm_manager import *  # Import the JVMManager

class CausalLearner:
    """
    CausalLearner class for automated causal discovery.

    Parameters
    ----------
    dataset_input : str or Dataset
        Either a file path to the dataset or a Dataset instance containing the data.
    configurations : Configurations, optional
        A Configurations object containing experiment configurations. If None, default configurations are used.
    verbose : bool, optional
        If True, prints detailed logs. Default is False.
    n_jobs : int, optional
        Number of jobs for parallel processing. Default is the number of CPU cores.
    random_seed : int, optional
        Seed for random number generator to ensure reproducibility. Default is None.

    Methods
    -------
    learn_model()
        Runs the causal discovery process.
    print_results(opt_conf=None)
        Prints the results of the causal discovery process.
    set_dataset(dataset)
        Sets the dataset for the causal learner.
    set_configurations(configurations)
        Sets the configurations for the causal learner.
    save_progress(path=None)
        Saves the progress of the experiment to a file.
    load_progress(path)
        Loads the progress of the experiment from a file.
    add_configurations_from_file(filename)
        Adds additional configurations to the experiment from a JSON file.
    update_learnt_model()
        Updates the learnt model with new configurations.
    get_best_model_between_algorithms(algorithms)
        Gets the best model between specified algorithms.
    get_best_model_between_family(**kwargs)
        Gets the best model within a family of algorithms based on specified criteria.
    """

    def __init__(
        self,
        dataset_input: Optional[Union[str, Dataset]] = None,
        configurations: Optional[Configurations] = None,
        verbose: bool = False,
        n_jobs: Optional[int] = None,
        random_seed: Optional[int] = None,
    ):
        if n_jobs is None:
            n_jobs = os.cpu_count()
        start_jvm()
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_seed = random_seed

        # Setup logging
        self.logger = get_logger(name=__name__, verbose=self.verbose)

        self.logger.debug('Initializing CausalLearner')

        # Initialize configurations
        if configurations is None:
            # Initialize dataset
            if isinstance(dataset_input, Dataset):
                self.dataset = dataset_input
            elif isinstance(dataset_input, pd.DataFrame):
                # If a plain DataFrame is provided, initialize Dataset with a default name
                self.dataset = Dataset(
                    data=dataset_input,
                    data_time_info={'n_lags': 0, 'time_lagged': False},
                    time_series=False,
                    dataset_name='Preloaded Dataset'
                )
            elif isinstance(dataset_input, str):
                self.dataset = Dataset(filename=dataset_input)
            else:
                raise ValueError("dataset_input must be either a file path, a Dataset instance, or a pandas DataFrame")

            self.configurations = Configurations(dataset=self.dataset, verbose=self.verbose, n_jobs=n_jobs)
        elif not isinstance(configurations, Configurations):
            self.logger.error('Configurations must be of type Configurations.')
            raise TypeError('Configurations must be of type Configurations.')
        else:
            self.configurations = configurations
            self.dataset = configurations.dataset
        self.results_folder = self.configurations.results_folder

        # Initialize CDHPO (Combined Discovery and Hyperparameter Optimization)
        self.cdhpo = OCT(
            oct_params=self.configurations.cdhpo_params,
            data=self.dataset,
            results_folder=self.results_folder, verbose=self.verbose
        )
        self.opt_conf = None
        self.matrix_mec_graph = None
        self.run_time = None

        # Set random seed for reproducibility
        if self.random_seed is not None:
            import numpy as np
            import random
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
    def learn_model(self):
        """
        Runs the causal discovery process using the OCT algorithm.

        Returns
        -------
        Tuple containing:
            - opt_conf: The optimal configuration found.
            - matrix_mec_graph: The MEC graph matrix.
            - matrix_graph: The graph matrix
            - run_time: The runtime of the CDHPO process.
            - library_results: Results from the causal discovery libraries.
        """
        self.logger.debug(
'Starting OCT Run')
        start_time = time.time()
        try:
            self.opt_conf, self.matrix_mec_graph, self.matrix_graph, library_results = self.cdhpo.run()
        except AttributeError as e:
            self.logger.error(f"Attribute error during CDHPO run: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error during CDHPO run: {e}")
            raise
        end_time = time.time()
        self.run_time = str(end_time - start_time)
        self.logger.debug(f'CDHPO Runtime: {self.run_time}')
        self.print_results()
        return {
            'optimal_conf': self.opt_conf,
            'matrix_mec_graph': self.matrix_mec_graph,
            'matrix_graph' : self.matrix_graph,
            'run_time': self.run_time,
            'library_results': library_results
        }

    def print_results(self, opt_conf=None):
        """
        Prints the results of the causal discovery process.

        Parameters
        ----------
        opt_conf : dict, optional
            The optimal configuration to print. If None, uses self.opt_conf.
        """
        if opt_conf is None:
            opt_conf = self.opt_conf

        print('Best Causal Discovery configuration was:', opt_conf.get('name'))
        print('With parameters:')
        for par, val in opt_conf.items():
            if par not in ['is_cat_var', 'model', 'var_type', 'indexes']:
                print(f'{par}: {val}')

        print('The MEC matrix graph is:')
        print(self.matrix_mec_graph)

    def set_dataset(self, dataset):
        """
        Sets the dataset for the causal learner.

        Parameters
        ----------
        dataset : Dataset
            The Dataset object to set.

        Raises
        ------
        TypeError
            If dataset is not of type Dataset.
        """
        if not isinstance(dataset, Dataset):
            raise TypeError('Dataset must be of type Dataset.')
        self.dataset = dataset

    def set_configurations(self, configurations):
        """
        Sets the configurations for the causal learner.

        Parameters
        ----------
        configurations : Configurations
            The Configurations object to set.

        Raises
        ------
        TypeError
            If configurations is not of type Configurations.
        """
        if not isinstance(configurations, Configurations):
            raise TypeError('Configurations must be of type Configurations.')
        self.configurations = configurations
        self.set_dataset(self.configurations.dataset)

    def save_progress(self, path=None):
        """
        Saves the progress of the experiment to a file.

        Parameters
        ----------
        path : str, optional
            The file path to save the progress to. If None, saves to 'Experiment.pkl' in results_folder.
        """
        if path is None:
            path = os.path.join(self.results_folder, 'Experiment.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        self.logger.debug(
f'Progress saved to {path}')

    @staticmethod
    def load_progress(path):
        """
        Loads the progress of the experiment from a file.

        Parameters
        ----------
        path : str
            The file path to load the progress from.

        Returns
        -------
        CausalLearner
            The loaded CausalLearner object.
        """
        with open(path, 'rb') as f:
            learner = pickle.load(f)
        learner.logger.debug(
f'Progress loaded from {path}')
        return learner

    def add_configurations_from_file(self, filename):
        """
        Adds additional configurations to the experiment from a JSON file.

        Parameters
        ----------
        filename : str
            The filename of the JSON file containing configurations.
        """
        self.configurations.add_configurations_from_file(filename)
        self.logger.debug(f'Configurations added from {filename}')

    def update_learnt_model(self):
        """
        Updates the learnt model with the new configurations.
        """
        self.logger.debug('Updating learnt model with new configurations')
        self.opt_conf, self.matrix_mec_graph, _ = self.cdhpo.run_new()

    def get_best_model_between_algorithms(self, algorithms):
        """
        Gets the best model between specified algorithms.

        Parameters
        ----------
        algorithms : list
            A list of algorithm names to consider.

        Returns
        -------
        dict
            The best configuration among the specified algorithms.
        """
        best_config = self.cdhpo.find_best_config(algorithms)
        self.logger.debug(f'Best configuration among specified algorithms: {best_config}')
        return best_config

    def get_best_model_between_family(
        self,
        causal_sufficiency=None,
        assume_faithfulness=None,
        is_output_mec=None,
        accepts_missing_values=None
    ):
        """
        Gets the best model within a family of algorithms based on specified criteria.

        Parameters
        ----------
        causal_sufficiency : bool, optional
            Filter algorithms that admit latent variables.
        assume_faithfulness : bool, optional
            Filter algorithms based on faithfulness assumption.
        is_output_mec : bool, optional
            Filter algorithms that output MEC graphs.
        accepts_missing_values : bool, optional
            Filter algorithms that accept missing values.

        Returns
        -------
        dict
            The best configuration among the filtered algorithms.
        """
        algorithms = []
        for algo in causal_configs:
            if causal_sufficiency is not None and causal_configs[algo]['causal_sufficiency'] != causal_sufficiency:
                continue
            if assume_faithfulness is not None and causal_configs[algo]['assume_faithfulness'] != assume_faithfulness:
                continue
            if accepts_missing_values is not None and causal_configs[algo]['missing_values'] != accepts_missing_values:
                continue
            algorithms.append(algo)

        best_config = self.cdhpo.find_best_config(algorithms)
        self.logger.debug(f'Best configuration among filtered algorithms: {best_config}')
        return best_config
