from typing import Optional

from ..model_validation_protocols.available_mvp_protocols import available_mvp_protocols
from ..regressors import regressors
import copy
from ..algorithms.causaldiscoveryalgorithms import *
from ..algorithms.tetrad_algorithm import TetradAlgorithm
from ..algorithms.tigramite_algorithm import TigramiteAlgorithm
from ..algorithms.causalnex_algorithm import NoTears
from ..algorithms.cdt_algorithms import SAMAlgorithm
from ..utils.logger import get_logger

class MVP_Parameters:
    """
    Class to manage the out-of-sample (OOS) protocol for model validation.

    Parameters
    ----------
    protocol_name : str, optional
        The name of the OOS protocol. Default is 'KFoldCV'.
    parameters : dict, optional
        Parameters for the chosen OOS protocol. Default is {'folds': 10, 'folds_to_run': 1}.
    verbose : bool, optional
        If True, enables detailed logging. Default is False.

    Attributes
    ----------
    protocol_name : str
        Name of the chosen OOS protocol.
    protocol : object
        Instance of the chosen OOS protocol.
    parameters : dict
        Parameters for the OOS protocol.

    Methods
    -------
    __init__(protocol_name='KFoldCV', parameters={'folds': 10, 'folds_to_run': 1}, verbose=False)
        Initializes the OOS protocol with the given or default parameters.
    """

    def __init__(self, protocol_name='KFoldCV', parameters={'folds': 10, 'folds_to_run': 1}, verbose=False):
        self.protocol_name = protocol_name
        self.parameters = parameters
        self.verbose = verbose
        self.logger = get_logger(name=__name__, verbose=self.verbose)

        # Check if the specified protocol name exists in available protocols
        if protocol_name in available_mvp_protocols:
            self.protocol = available_mvp_protocols[protocol_name]
            available_mvp_protocols[protocol_name].set_params(parameters, self.verbose)
        else:
            self.logger.error('The protocol you chose is not available')
            raise Exception('The protocol you chose is not available')


class Regressor_parameters:
    """
    Class to manage the configuration of regressors for the CDHPO process.

    Parameters
    ----------
    regressor_name : str, optional
        The name of the regressor. Default is 'RandomForestRegressor'.
    parameters : dict, optional
        Parameters for the regressor. Default is {'n_trees': 100, 'min_samples_leaf': 0.1, 'max_depth': 10}.
    verbose : bool, optional
        If True, enables detailed logging. Default is False.

    Attributes
    ----------
    regressor_name : str
        Name of the chosen regressor.
    regressor : object
        Instance of the chosen regressor.
    parameters : dict
        Parameters for the regressor.

    Methods
    -------
    __init__(regressor_name='RandomForestRegressor', parameters={'n_trees': 100, 'min_samples_leaf': 0.1, 'max_depth': 10}, verbose=False)
        Initializes the regressor with the given or default parameters.
    """

    def __init__(self, regressor_name='RandomForestRegressor', parameters={'n_trees': 100, 'min_samples_leaf': 0.1, 'max_depth': 10}, verbose=False):
        self.regressor_name = regressor_name
        self.parameters = parameters
        self.verbose = verbose
        self.logger = get_logger(name=__name__, verbose=self.verbose)

        # Check if the specified regressor name exists in available regressors
        if regressor_name in regressors.available_regressors:
            self.regressor = regressors.available_regressors[regressor_name].set_regressor_params(parameters)
        else:
            self.logger.error('The regressor you chose is not available')
            raise Exception('The regressor you chose is not available')


class CDHPO_Parameters:
    """
    Class to manage the configuration of the CDHPO (Causal Discovery with Hyperparameter Optimization) process.

    Methods
    -------
    init_main_params(alpha=0.01, n_permutations=200, causal_sufficiency=True, variables_type='mixed', n_jobs: Optional[int] = 1, verbose=False)
        Initializes the main parameters for CDHPO.
    set_regressor(name, parameters)
        Sets the regressor used in the CDHPO process.
    set_oos_protocol(name, parameters)
        Sets the out-of-sample (OOS) protocol for validation.
    set_cd_algorithms(algorithms, data_info)
        Sets the causal discovery algorithms for CDHPO.
    check_configs(data_info)
        Verifies the configurations for the causal discovery algorithms.
    add_cd_algorithm(algorithm, parameters, data_info)
        Adds a new causal discovery algorithm with parameters.
    add_cd_algorithm_parameters(algorithm, parameters)
        Adds parameters to an existing causal discovery algorithm.
    """

    def init_main_params(self, alpha=0.01, n_permutations=200, causal_sufficiency=True, variables_type='mixed', n_jobs: Optional[int] = 1, verbose=False):
        """
        Initializes the main parameters for the CDHPO process.

        Parameters
        ----------
        alpha : float, optional
            The significance level. Default is 0.01.
        n_permutations : int, optional
            The number of permutations for statistical tests. Default is 200.
        causal_sufficiency : bool, optional
            Whether to assume causal sufficiency. Default is True.
        variables_type : str, optional
            Type of variables in the data (e.g., 'mixed', 'discrete', 'continuous'). Default is 'mixed'.
        n_jobs : int, optional
            Number of parallel jobs. Default is 1.
        verbose : bool, optional
            If True, enables detailed logging. Default is False.

        Returns
        -------
        None
        """
        self.n_jobs = n_jobs
        self.alpha = alpha
        self.n_permutations = n_permutations
        self.causal_sufficiency = causal_sufficiency
        self.variables_type = variables_type
        self.oos_protocol = MVP_Parameters()
        self.regressor = Regressor_parameters()
        self.verbose = verbose
        self.logger = get_logger(name=__name__, verbose=self.verbose)
        self.logger.info('CDHPO Parameters have been initialized')
        self.configs = {}
    def set_regressor(self, name, parameters):
        """
        Sets the regressor for the CDHPO process.

        Parameters
        ----------
        name : str
            The name of the regressor.
        parameters : dict
            The parameters for the regressor.

        Returns
        -------
        None
        """
        self.regressor = Regressor_parameters(name, parameters, self.verbose)
        self.logger.info('Regressor_parameters has been set')

    def set_oos_protocol(self, name, parameters):
        """
        Sets the out-of-sample (OOS) protocol for validation.

        Parameters
        ----------
        name : str
            The name of the OOS protocol.
        parameters : dict
            The parameters for the OOS protocol.

        Returns
        -------
        None
        """
        self.oos_protocol = MVP_Parameters(name, parameters, self.verbose)
        self.logger.info('OOS protocol has been set')

    def set_cd_algorithms(self, algorithms, data_info):
        """
        Sets the causal discovery algorithms for CDHPO.

        Parameters
        ----------
        algorithms : dict
            Dictionary of algorithms and their configurations.
        data_info : dict
            Information about the dataset.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If an algorithm is not found in the database.
        """
        for algo in algorithms:
            if algo in cd_algorithms['tetrad']['algorithms']:
                algorithms[algo]['model'] = [TetradAlgorithm(algo, self.verbose)]
                algorithms[algo]['model'][0].init_algo(data_info)
                algorithms[algo]['library'] = ['tetrad']
                self.configs[algo] = algorithms[algo]
            elif algo in cd_algorithms['tigramite']['algorithms']:
                algorithms[algo]['model'] = [TigramiteAlgorithm(algo, self.verbose)]
                algorithms[algo]['model'][0].init_algo(data_info)
                algorithms[algo]['library'] = ['tigramite']
                self.configs[algo] = algorithms[algo]
            elif algo in cd_algorithms['causalnex']['algorithms']:
                algorithms[algo]['model'] = [NoTears.NoTearsAlgorithm(algo, self.verbose)]
                algorithms[algo]['library'] = ['causalnex']
                self.configs[algo] = algorithms[algo]
            elif algo in cd_algorithms['cdt']['algorithms']:
                algorithms[algo]['model'] = [SAMAlgorithm.SAMAlgorithm(algo, self.verbose)]
                algorithms[algo]['library'] = ['cdt']
                self.configs[algo] = algorithms[algo]
            else:
                raise RuntimeError("This algorithm (" + algo + ") does not exist in the database")
        self.configs = algorithms

    def check_configs(self, data_info):
        """
        Verifies the configurations for the causal discovery algorithms.

        Parameters
        ----------
        data_info : dict
            Information about the dataset.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If any algorithm has incorrect parameters.
        """
        for algo in self.configs:
            if not self.configs[algo]['model'][0].check_parameters(self.configs[algo], data_info):
                raise RuntimeError(f'Algorithm {algo} has wrong parameters or parameter values')

    def add_cd_algorithm(self, algorithm, parameters, data_info):
        """
        Adds a new causal discovery algorithm to the CDHPO configuration.

        Parameters
        ----------
        algorithm : str
            Name of the algorithm to add.
        parameters : dict
            Configuration parameters for the algorithm.
        data_info : dict
            Information about the dataset.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the algorithm is not found in the database.
        """
        if algorithm not in self.configs:
            if algorithm in cd_algorithms['tetrad']['algorithms']:
                parameters['model'] = [TetradAlgorithm(algorithm)]
                parameters['model'][0].init_algo(data_info)
                parameters['library'] = ['tetrad']
            elif algorithm in cd_algorithms['tigramite']['algorithms']:
                parameters['model'] = [TigramiteAlgorithm(algorithm)]
                parameters['model'][0].init_algo(data_info)
                parameters['library'] = ['tigramite']
            elif algorithm in cd_algorithms['causalnex']['algorithms']:
                parameters['model'] = [NoTears.NoTearsAlgorithm(algorithm)]
                parameters['library'] = ['tigramite']
            elif algorithm in cd_algorithms['cdt']['algorithms']:
                parameters['model'] = [SAMAlgorithm.SAMAlgorithm(algorithm)]
                parameters['library'] = ['tigramite']
            else:
                raise RuntimeError("This algorithm does not exist in the database")
            self.configs[algorithm] = parameters
        else:
            raise Warning('Causal discovery algorithm already added')

    def add_cd_algorithm_parameters(self, algorithm, parameters):
        """
        Adds parameters to an existing causal discovery algorithm.

        Parameters
        ----------
        algorithm : str
            Name of the algorithm.
        parameters : dict
            Additional parameters for the algorithm.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If the algorithm does not exist or the parameters are invalid.
        """
        if algorithm not in self.configs:
            self.logger.error('Tried to add parameters for a non-existent algorithm')
            raise RuntimeError('Tried to add parameters for a non-existent algorithm')
        for param in parameters:
            if param not in self.configs[algorithm]:
                self.logger.error('Tried to add an invalid parameter')
                raise RuntimeError('Tried to add an invalid parameter')
            for val in parameters[param]:
                if val not in self.configs[algorithm][param]:
                    self.configs[algorithm][param].append(val)
        self.logger.info(f'Parameters added to algorithm {algorithm}')
