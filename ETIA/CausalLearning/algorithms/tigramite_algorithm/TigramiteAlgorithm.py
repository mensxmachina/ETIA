import numpy as np
import pandas as pd
import tigramite
from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.lpcmci import LPCMCI

from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.independence_tests.parcorr_wls import ParCorrWLS
from tigramite.independence_tests.gpdc import GPDC
from tigramite.independence_tests.cmiknn import CMIknn
from tigramite.independence_tests.cmisymb import CMIsymb
from tigramite.independence_tests.gsquared import Gsquared
from tigramite.independence_tests.regressionCI import RegressionCI
from ..utils import prepare_data_tigramite
from ...utils.logger import get_logger
from ...CausalModel.utils import matrix_to_pywhy_graph


class TigramiteAlgorithm:
    """
    A class that implements causal discovery using the Tigramite library.

    Methods
    -------
    init_algo(data_info)
        Initializes the algorithm with the data type and time information.
    prepare_data(Data, parameters)
        Prepares the dataset for Tigramite algorithms.
    _ci_test(parameters)
        Configures the conditional independence test to be used in the algorithm.
    _algo(dataframe_, parameters, ci_test)
        Configures and runs the selected causal discovery algorithm.
    output_to_array(output)
        Converts the Tigramite graph output into a numpy array representation.
    run(data, parameters, prepare_data=True)
        Runs the specified Tigramite algorithm on the provided data.
    """

    def __init__(self, algorithm, verbose=False):
        """
        Initializes the TigramiteAlgorithm class.

        Parameters
        ----------
        algorithm : str
            The name of the algorithm to use (e.g., 'PCMCI', 'LPCMCI', 'PCMCI+').
        verbose : bool, optional
            If True, enables verbose logging. Default is False.
        """
        self.algorithm = algorithm
        self.data = None
        self.verbose = verbose
        self.logger = get_logger(name=__name__, verbose=self.verbose)

    def init_algo(self, data_info):
        """
        Initializes the algorithm with data type and time lag information.

        Parameters
        ----------
        data_info : dict
            Dictionary containing the data type information and time lag details.
        """
        self.data_type_info = data_info['data_type_info']
        self.data_time_info = data_info['data_time_info']
        self.n_lags = self.data_time_info['n_lags']
        self.var_names = data_info['var_names_lagged']  # names in lags

    def prepare_data(self, Data, parameters):
        """
        Prepares the dataset for use in the Tigramite algorithm.

        Parameters
        ----------
        Data : object
            The dataset to be prepared.
        parameters : dict
            Additional parameters for data preparation.

        Returns
        -------
        pd.DataFrame
            Prepared dataset in Tigramite format.
        """
        dataframe_ = prepare_data_tigramite(Data, parameters)
        self.data = dataframe_
        return dataframe_

    def _ci_test(self, parameters):
        """
        Configures the conditional independence test to be used in the algorithm.

        Parameters
        ----------
        parameters : dict
            A dictionary of parameters specifying the conditional independence test (e.g., 'ParCor').

        Returns
        -------
        ci_test : object
            The configured conditional independence test.
        """
        if parameters['ci_test'] == 'ParCor':
            ci_test = ParCorr()
        elif parameters['ci_test'] == 'RobustParCor':
            ci_test = RobustParCorr()
        elif parameters['ci_test'] == 'GPDC':
            ci_test = GPDC(significance='analytic', gp_params=None)
        elif parameters['ci_test'] == 'CMIknn':
            ci_test = CMIknn(significance='fixed_thres', model_selection_folds=3)
        elif parameters['ci_test'] == 'ParCorrWLS':
            ci_test = ParCorrWLS(significance='analytic')
        elif parameters['ci_test'] == 'Gsquared':  # for discrete variables
            ci_test = Gsquared(significance='analytic')
        elif parameters['ci_test'] == 'CMIsymb':
            ci_test = CMIsymb(significance='shuffle_test')
        elif parameters['ci_test'] == 'RegressionCI':
            ci_test = RegressionCI(significance='analytic')
        else:
            raise ValueError(f"{parameters['ci_test']} ci test not included")

        return ci_test

    def _algo(self, dataframe_, parameters, ci_test):
        """
        Configures and runs the specified causal discovery algorithm.

        Parameters
        ----------
        dataframe_ : pd.DataFrame
            The dataset to be used for causal discovery.
        parameters : dict
            The parameters for the algorithm (e.g., significance level).
        ci_test : object
            The configured conditional independence test.

        Returns
        -------
        dict
            The output of the Tigramite algorithm.
        """
        if self.algorithm == 'PCMCI':
            alg = PCMCI(dataframe=dataframe_, cond_ind_test=ci_test, verbosity=0)
            output = alg.run_pcmci(tau_max=self.n_lags, pc_alpha=parameters['significance_level'],
                                   alpha_level=parameters['significance_level'])
        elif self.algorithm == 'PCMCI+':
            alg = PCMCI(dataframe=dataframe_, cond_ind_test=ci_test, verbosity=0)
            output = alg.run_pcmciplus(tau_max=self.n_lags, pc_alpha=parameters['significance_level'])
        elif self.algorithm == 'LPCMCI':
            alg = LPCMCI(dataframe=dataframe_, cond_ind_test=ci_test, verbosity=0)
            output = alg.run_lpcmci(tau_max=self.n_lags, pc_alpha=parameters['significance_level'])
        else:
            raise ValueError(f"{self.algorithm} cd alg not included")

        return output

    def output_to_array(self, output):
        """
        Converts the Tigramite graph output to a numpy array representation.

        Parameters
        ----------
        output : dict
            The output of the Tigramite algorithm containing the graph.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame representing the adjacency matrix of the learned graph.
        """
        t_graph = output['graph']
        n_nodes = t_graph.shape[0]
        T = t_graph.shape[2]

        matrix = np.zeros((n_nodes * T, n_nodes * T), dtype=int)

        for step in range(T):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if t_graph[i, j, step] != '':
                        for t in range(step, T):
                            i_ = n_nodes * t + i
                            j_ = n_nodes * (t - step) + j
                            edge = t_graph[i, j, step]

                            if edge == 'o-o':
                                matrix[i_, j_] = 1
                                matrix[j_, i_] = 1
                            elif edge == '-->':
                                matrix[i_, j_] = 2
                                matrix[j_, i_] = 3
                            elif edge == '<--':
                                matrix[j_, i_] = 2
                                matrix[i_, j_] = 3
                            elif edge == '<->':
                                matrix[i_, j_] = 2
                                matrix[j_, i_] = 2
                            elif edge == 'o->':
                                matrix[i_, j_] = 2
                                matrix[j_, i_] = 1
                            elif edge == '<-o':
                                matrix[j_, i_] = 2
                                matrix[i_, j_] = 1
                            elif edge == 'x-x':
                                matrix[i_, j_] = 1
                                matrix[j_, i_] = 1
                            elif edge == 'x->':
                                matrix[i_, j_] = 2
                                matrix[j_, i_] = 1
                            elif edge == '<-x':
                                matrix[j_, i_] = 2
                                matrix[i_, j_] = 1
                            else:
                                raise ValueError(f"{edge} edge not included")

        matrix_pd = pd.DataFrame(matrix, columns=self.var_names)

        return matrix_pd

    def run(self, data, parameters, prepare_data=True):
        """
        Runs the Tigramite algorithm on the provided data.

        Parameters
        ----------
        data : object
            The dataset to be used in the algorithm.
        parameters : dict
            The parameters for the algorithm (e.g., significance level, ci_test).
        prepare_data : bool, optional
            If True, prepares the data before running the algorithm. Default is True.

        Returns
        -------
        tuple
            A tuple containing the learned graph and a dictionary of results.
        """
        if prepare_data:
            dataframe_ = self.prepare_data(data, parameters)
        else:
            dataframe_ = data

        ci_test = self._ci_test(parameters)
        output = self._algo(dataframe_, parameters, ci_test)
        mec_graph_pd = self.output_to_array(output)

        library_results = {'mec': output}
        mec_graph = matrix_to_pywhy_graph(mec_graph_pd)

        return mec_graph, library_results
