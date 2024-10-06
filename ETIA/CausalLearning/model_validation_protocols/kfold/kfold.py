import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from ..MVP_ProtocolBase import MVP_ProtocolBase, get_logger


class KFoldCV(MVP_ProtocolBase):
    """
    Class implementing a K-Fold Cross-Validation protocol for running a causal discovery algorithm.

    Attributes
    ----------
    folds : int
        Number of folds to be used in the cross-validation. Default is 10.
    folds_to_run : int
        Number of folds to run the cross-validation for. Default is 1.
    train_indexes : list of int
        A list of indexes for the training samples.
    test_indexes : list of int
        A list of indexes for the test samples.
    data_train : list of pd.DataFrame
        A list of training data samples for each fold.
    data_test : list of pd.DataFrame
        A list of test data samples for each fold.

    Methods
    -------
    set_params(parameters, verbose=False)
        Set the number of folds and the number of folds to run the protocol for.
    run_cd_algorithm(data, algorithm, parameters, fold)
        Run the causal discovery algorithm on the specified fold.
    init_protocol(data)
        Initialize the K-Fold protocol.
    run_protocol(data, algorithm, parameters, n_jobs=1)
        Run the K-Fold cross-validation protocol.
    """

    def __init__(self):
        """Initializes the KFoldCV class with default values for folds and folds_to_run."""
        self.folds = 10
        self.folds_to_run = 1
        self.train_indexes = []
        self.test_indexes = []
        self.data_train = []
        self.data_test = []

    def set_params(self, parameters, verbose=False):
        """
        Set the number of folds and the number of folds to run the protocol for.

        Parameters
        ----------
        parameters : dict
            A dictionary of parameters, including the number of folds and the number of folds to run.
        verbose : bool, optional
            If True, enables detailed logging. Default is False.
        """
        self.folds = parameters['folds']
        self.folds_to_run = parameters['folds_to_run']
        self.verbose = verbose
        self.logger = get_logger(name=__name__, verbose=self.verbose)

    def run_cd_algorithm(self, data, algorithm, parameters, fold):
        """
        Run the causal discovery algorithm on the specified fold.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset on which to run the causal discovery algorithm.
        algorithm : object
            The causal discovery algorithm to be used.
        parameters : dict
            A dictionary of parameters to pass to the algorithm.
        fold : int
            The current fold number for which to run the algorithm.

        Returns
        -------
        list of np.ndarray
            A list containing the MEC graph and library results produced by the causal discovery algorithm.
        """
        # Causal discovery
        parameters['indexes'] = self.train_indexes[fold]
        mec_graph, graph, library_results = algorithm.run(data, parameters, prepare_data=True)
        self.logger.debug('Causal discovery algorithm has been run for fold ' + str(fold))

        return [mec_graph, graph, library_results]

    def init_protocol(self, data):
        """
        Initialize the K-Fold protocol by splitting the data into training and test sets for each fold.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset to be used for the cross-validation.
        """
        data = data.get_dataset()
        kf = KFold(n_splits=self.folds)
        self.train_indexes = []
        self.test_indexes = []
        for train_index, test_index in kf.split(data):
            self.train_indexes.append(train_index)
            self.test_indexes.append(test_index)

    def run_protocol(self, data, algorithm, parameters, n_jobs=1):
        """
        Run the K-Fold cross-validation protocol with the specified causal discovery algorithm.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset on which to run the algorithm.
        algorithm : object
            The causal discovery algorithm to use.
        parameters : dict
            A dictionary of parameters to be passed to the algorithm.
        n_jobs : int, optional
            The number of CPU cores to use for parallel computation. Default is 1.

        Returns
        -------
        list of np.ndarray
            A list containing the results of the protocol, with the MEC graphs and other results.
        """
        results = Parallel(n_jobs=n_jobs)(
            delayed(self.run_cd_algorithm)(data, algorithm, parameters, fold)
            for fold in range(self.folds_to_run)
        )
        results = np.array(results)

        return [results[:, 0], results[:, 1], results[:, 2]]
