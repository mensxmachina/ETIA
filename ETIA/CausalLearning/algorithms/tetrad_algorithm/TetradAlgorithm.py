import jpype
import numpy as np
import pandas as pd
from ....utils.jvm_manager import start_jvm
from jpype import JPackage, JProxy
from ...utils.logger import get_logger
from ..utils import prepare_data_tetrad
from ...CausalModel.utils import matrix_to_pywhy_graph


class TetradAlgorithm:
    """
    A class that implements various causal discovery algorithms using the Tetrad library.

    Methods
    -------
    mute_java_output()
        Mutes Java's standard output and error streams.
    configure_java_logging()
        Configures Java's Log4j logging for the Tetrad package.
    init_algo(data_info)
        Initializes algorithm-specific data, such as lags and time information.
    prepare_data(Data, parameters=None)
        Prepares the data in a format suitable for the Tetrad algorithms.
    time_knowledge(ds)
        Defines the time-knowledge for time-lagged datasets.
    _ci_test(ds, parameters)
        Configures the appropriate conditional independence test for the algorithm.
    _score(ds, parameters)
        Configures the appropriate score-based test for the algorithm.
    _algo(parameters, ind_test, score)
        Configures and returns the specified causal discovery algorithm.
    output_to_array(tetrad_graph_, var_map)
        Converts the Tetrad graph to a numpy array representing the causal structure.
    check_parameters(parameters, data_info)
        Validates the parameters required to run the Tetrad algorithm.
    run(data, parameters, prepare_data=True)
        Runs the specified Tetrad algorithm on the provided data and returns the results.
    """

    def __init__(self, algorithm, verbose=False):
        """
        Initializes the TetradAlgorithm class.

        Parameters
        ----------
        algorithm : str
            The name of the algorithm to be used (e.g., 'pc', 'fci', 'fges').
        verbose : bool, optional
            If True, enables verbose logging. Default is False.
        """
        self.algorithm = algorithm
        self.verbose = verbose
        self.logger = get_logger(name=__name__, verbose=self.verbose)

        if not self.verbose:
            self.mute_java_output()

    def mute_java_output(self):
        """
        Mutes Java's standard output and error streams to suppress logs and output.
        """
        try:
            class NullOutputStream:
                def write(self, b):
                    pass

                def flush(self):
                    pass

            OutputStream = JPackage('java.io').OutputStream
            null_out = JProxy(OutputStream, inst=NullOutputStream())
            null_err = JProxy(OutputStream, inst=NullOutputStream())

            java_lang_System = JPackage('java.lang').System
            java_lang_System.setOut(null_out)
            java_lang_System.setErr(null_err)

            self.logger.info("Java System.out and System.err have been muted.")
        except Exception as e:
            self.logger.error(f"Failed to mute Java output streams: {e}")

    def configure_java_logging(self):
        """
        Configures Java's Log4j logging level for Tetrad algorithms.
        """
        try:
            log4j = JPackage('org').apache.log4j
            tetrad_logger = log4j.Logger.getLogger("edu.cmu.tetrad")

            if self.verbose:
                tetrad_logger.setLevel(log4j.Level.INFO)
                self.logger.info("Log4j logging set to INFO level for Tetrad.")
            else:
                tetrad_logger.setLevel(log4j.Level.ERROR)
                self.logger.info("Log4j logging set to ERROR level for Tetrad.")

            appenders = tetrad_logger.getAllAppenders()
            while appenders.hasMoreElements():
                appender = appenders.nextElement()
                tetrad_logger.removeAppender(appender)

            if self.verbose:
                console_appender = log4j.ConsoleAppender()
                layout = log4j.PatternLayout("%d{ISO8601} %-5p [%c{1}] %m%n")
                console_appender.setLayout(layout)
                console_appender.setThreshold(log4j.Level.INFO)
                console_appender.activateOptions()
                tetrad_logger.addAppender(console_appender)
        except Exception as e:
            self.logger.error(f"Failed to configure Log4j logging: {e}")

    def init_algo(self, data_info):
        """
        Initializes the algorithm with data type and time information.

        Parameters
        ----------
        data_info : dict
            Dictionary containing information about data types and time lags.
        """
        self.data_type_info = data_info['data_type_info']
        self.data_time_info = data_info['data_time_info']
        self.n_lags = self.data_time_info['n_lags']

    def prepare_data(self, Data, parameters=None):
        """
        Prepares the dataset for use in the Tetrad algorithms.

        Parameters
        ----------
        Data : object
            The dataset to be used in the algorithm.
        parameters : dict, optional
            Additional parameters for data preparation. Default is None.

        Returns
        -------
        tuple
            A tuple containing the prepared dataset and a mapping of variable names.
        """
        ds, name_map_pd = prepare_data_tetrad(Data, parameters)
        return ds, name_map_pd

    def time_knowledge(self, ds):
        """
        Generates temporal knowledge for time-lagged data.

        Parameters
        ----------
        ds : object
            The dataset in Tetrad format.

        Returns
        -------
        knowledge : object
            A Tetrad Knowledge object that encodes the temporal relationships in the data.
        """
        data = jpype.JPackage("edu.cmu.tetrad.data")
        knowledge = data.Knowledge()
        var_names = list(ds.getVariableNames())

        for t, tier in zip(range(self.n_lags + 1), reversed(range(self.n_lags + 1))):
            for i, var in enumerate(var_names):
                if (t == 0) and (':' not in var):
                    knowledge.addToTier(tier, var_names[i])
                elif (t > 0) and (':' + str(t) in var):
                    knowledge.addToTier(tier, var_names[i])

        return knowledge

    def add_knowledge(self, ds, var_map, prior_knowledge):
        data = jpype.JPackage("edu.cmu.tetrad.data")
        knowledge = data.Knowledge()
        var_names = list(ds.getVariableNames())
        i = 0
        var_to_tetrad = var_map.set_index("var_name")["tetrad_name"].to_dict()
        for tier in prior_knowledge:
            for var in prior_knowledge[tier]:
                tetrad_name = var_to_tetrad.get(var, None)
                idx = var_names.index(tetrad_name)
                knowledge.addToTier(i, var_names[idx])
            i += 1
        return knowledge
    def _ci_test(self, ds, parameters):
        """
        Configures the conditional independence test for the algorithm.

        Parameters
        ----------
        ds : object
            The dataset in Tetrad format.
        parameters : dict
            Algorithm parameters, including the choice of CI test and significance level.

        Returns
        -------
        ind_test : object
            The configured CI test object.
        """
        test = jpype.JPackage("edu.cmu.tetrad.search.test")

        if 'stable' not in parameters:
            parameters['stable'] = True

        if parameters['ci_test'] == 'FisherZ':
            ind_test = test.IndTestFisherZ(ds, parameters['significance_level'])
        elif parameters['ci_test'] == 'cci':
            ind_test = test.IndTestConditionalCorrelation(ds, parameters['significance_level'])
        elif parameters['ci_test'] == 'cg_lrt':
            discretize = True
            ind_test = test.IndTestConditionalGaussianLrt(ds, parameters['significance_level'], discretize)
        elif parameters['ci_test'] == 'dg_lrt':
            ind_test = test.IndTestDegenerateGaussianLrt(ds)
            ind_test.setAlpha(parameters['significance_level'])
        elif parameters['ci_test'] == 'chisquare':
            ind_test = test.IndTestChiSquare(ds, parameters['significance_level'])
        elif parameters['ci_test'] == 'gsquare':
            ind_test = test.IndTestGSquare(ds, parameters['significance_level'])
        else:
            raise ValueError(f"{parameters['ci_test']} CI test not included")

        return ind_test

    def _score(self, ds, parameters):
        """
        Configures the score-based test for the algorithm.

        Parameters
        ----------
        ds : object
            The dataset in Tetrad format.
        parameters : dict
            Algorithm parameters, including the choice of score and penalty.

        Returns
        -------
        score_ : object
            The configured score object.
        """
        score = jpype.JPackage("edu.cmu.tetrad.search.score")

        if parameters['score'] == 'sem_bic_score':
            score_ = score.SemBicScore(ds, True)
            score_.setPenaltyDiscount(parameters['penalty_discount'])
        elif parameters['score'] == 'bdeu':
            score_ = score.BdeuScore(ds)
            score_.setStructurePrior(parameters['structure_prior'])
        elif parameters['score'] == 'discrete_bic':
            score_ = score.DiscreteBicScore(ds)
            score_.setPenaltyDiscount(parameters['penalty_discount'])
            score_.setStructurePrior(parameters['structure_prior'])
        elif parameters['score'] == 'cg_bic':
            discretize = True
            score_ = score.ConditionalGaussianScore(ds, parameters['penalty_discount'], discretize)
        elif parameters['score'] == 'dg_bic':
            score_ = score.DegenerateGaussianScore(ds, True)
            score_.setPenaltyDiscount(parameters['penalty_discount'])
        else:
            raise ValueError(f"{parameters['score']} score not included")

        return score_

    def _algo(self, parameters, ind_test, score):
        """
        Configures and returns the selected causal discovery algorithm.

        Parameters
        ----------
        parameters : dict
            The algorithm parameters.
        ind_test : object
            The conditional independence test to be used.
        score : object
            The score-based test to be used.

        Returns
        -------
        alg : object
            The configured Tetrad algorithm.
        """
        search = jpype.JPackage("edu.cmu.tetrad.search")

        if self.algorithm == 'pc':
            alg = search.Pc(ind_test)
            alg.setGuaranteeCpdag(True)
            alg.setStable(parameters['stable'])
        elif self.algorithm == 'cpc':
            alg = search.Cpc(ind_test)
            alg.setStable(parameters['stable'])
            alg.setGuaranteeCpdag(True)
        elif self.algorithm == 'fges':
            alg = search.Fges(score)
        elif self.algorithm == 'fci':
            alg = search.Fci(ind_test)
        elif self.algorithm == 'fcimax':
            alg = search.FciMax(ind_test)
        elif self.algorithm == 'rfci':
            alg = search.Rfci(ind_test)
        elif self.algorithm == 'gfci':
            alg = search.GFci(ind_test, score)
        elif self.algorithm == 'cfci':
            alg = search.Cfci(ind_test)
        elif self.algorithm == 'svarfci':
            alg = search.SvarFci(ind_test)
        elif self.algorithm == 'svargfci':
            alg = search.SvarGfci(ind_test, score)
        else:
            raise ValueError(f"{self.algorithm} algorithm not included")

        alg.setVerbose(False)
        return alg

    def output_to_array(self, tetrad_graph_, var_map):
        """
        Converts the Tetrad graph to a numpy array representation.

        Parameters
        ----------
        tetrad_graph_ : object
            The Tetrad graph object to be converted.
        var_map : pd.DataFrame
            A DataFrame mapping Tetrad variable names to original variable names.

        Returns
        -------
        matrix_pd : pd.DataFrame
            A pandas DataFrame representing the adjacency matrix of the learned graph.
        """
        n_nodes_ = tetrad_graph_.getNumNodes()
        edges = tetrad_graph_.getEdges()
        edgesIterator = edges.iterator()

        matrix = np.zeros(shape=(n_nodes_, n_nodes_), dtype=int)

        while edgesIterator.hasNext():
            curEdge = edgesIterator.next()

            Nodei = str(curEdge.getNode1().toString())
            Nodej = str(curEdge.getNode2().toString())

            iToj = str(curEdge.getEndpoint2().toString())
            jToi = str(curEdge.getEndpoint1().toString())

            i = np.where(var_map['tetrad_name'] == Nodei)
            j = np.where(var_map['tetrad_name'] == Nodej)

            if iToj in ['Circle', 'CIRCLE']:
                matrix[i, j] = 1
            elif iToj in ['Arrow', 'ARROW']:
                matrix[i, j] = 2
            elif iToj in ['Tail', 'TAIL']:
                matrix[i, j] = 3

            if jToi in ['Circle', 'CIRCLE']:
                matrix[j, i] = 1
            elif jToi in ['Arrow', 'ARROW']:
                matrix[j, i] = 2
            elif jToi in ['Tail', 'TAIL']:
                matrix[j, i] = 3

        matrix_pd = pd.DataFrame(matrix, columns=var_map['var_name'], index=var_map['var_name'])
        return matrix_pd

    def check_parameters(self, parameters, data_info):
        """
        Checks the validity of the parameters for running the Tetrad algorithm.

        Parameters
        ----------
        parameters : dict
            The algorithm parameters.
        data_info : dict
            Information about the dataset, such as variable types.

        Returns
        -------
        bool
            True if all parameters are valid, raises RuntimeError otherwise.
        """
        ind_tests = ['FisherZ', 'chisquare', 'gsquare', 'cg_lrt', 'dg_lrt']
        score_tests = ['sem_bic_score', 'bdeu', 'discrete_bic', 'cg_bic', 'dg_bic']

        if data_info['contains_constant_vars']:
            self.logger.error(f"{self.algorithm} cannot run on datasets containing constant variables")
            raise RuntimeError(f"{self.algorithm} cannot run on datasets containing constant variables")

        if 'significance_level' in parameters.keys():
            for alpha in parameters['significance_level']:
                if not (0 < alpha < 1):
                    self.logger.error(f"Invalid alpha value in {self.algorithm}")
                    raise RuntimeError(f"Invalid alpha value in {self.algorithm}")

        if 'ci_test' in parameters.keys():
            for ind_test in parameters['ci_test']:
                if ind_test not in ind_tests:
                    self.logger.error(f"Invalid independence test in {self.algorithm}: {ind_test}")
                    raise RuntimeError(f"Invalid independence test in {self.algorithm}: {ind_test}")
                if data_info['dataset_type'] in ['continuous', 'mixed'] and ind_test in ['chisquare', 'gsquare']:
                    self.logger.error(f"{ind_test} cannot be used with continuous variables in {self.algorithm}")
                    raise RuntimeError(f"{ind_test} cannot be used with continuous variables in {self.algorithm}")

        return True

    def run(self, data, parameters, prepare_data=True):
        """
        Runs the Tetrad algorithm on the provided data.

        Parameters
        ----------
        data : object
            The dataset to be used in the algorithm.
        parameters : dict
            The parameters for running the algorithm.
        prepare_data : bool, optional
            If True, prepares the data before running the algorithm. Default is True.

        Returns
        -------
        tuple
            A tuple containing the learned MEC graph and a dictionary of results.
        """
        start_jvm()

        graph = jpype.JPackage("edu.cmu.tetrad.graph")
        if prepare_data:
            ds, var_map = self.prepare_data(data, parameters)
        else:
            ds, var_map = data, self.var_map

        if 'ci_test' in parameters.keys():
            ind_test = self._ci_test(ds, parameters)
        else:
            ind_test = None

        if 'score' in parameters.keys():
            score_ = self._score(ds, parameters)
        else:
            score_ = None

        alg = self._algo(parameters, ind_test, score_)

        if self.data_time_info['time_lagged']:
            tetrad_knowledge = self.time_knowledge(ds)
            alg.setKnowledge(tetrad_knowledge)

        tetrad_mec_graph = alg.search()

        if parameters['causal_sufficiency']:
            tetrad_graph = graph.GraphTransforms.dagFromCpdag(tetrad_mec_graph)
        else:
            tetrad_graph = graph.GraphTransforms.magFromPag(tetrad_mec_graph)

        mec_graph_pd = self.output_to_array(tetrad_mec_graph, var_map)
        graph_pd = self.output_to_array(tetrad_graph, var_map)

        library_results = {
            'mec': tetrad_mec_graph,
            'graph': tetrad_graph,
            'map': var_map,
            'matrix_graph': graph_pd
        }

        if parameters['causal_sufficiency']:
            mec_graph = matrix_to_pywhy_graph(mec_graph_pd, 'CPDAG')
            graph = matrix_to_pywhy_graph(graph_pd, 'DAG')
        else:
            mec_graph = matrix_to_pywhy_graph(mec_graph_pd, 'PAG')
            graph = matrix_to_pywhy_graph(graph_pd, 'MAG')

        return mec_graph_pd, graph_pd, library_results
