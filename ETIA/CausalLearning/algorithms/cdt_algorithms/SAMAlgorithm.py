import pandas as pd
from cdt.causality.graph import SAM
from ...utils.logger import get_logger
from ...CausalModel import DAGWrapper


class SAMAlgorithm:
    """
    A class that implements the SAM (Structural Agnostic Model) algorithm for causal discovery.

    Methods
    -------
    check_parameters()
        Validates the parameters for running the SAM algorithm.
    prepare_data(data)
        Prepares the data for the SAM algorithm by converting it to a pandas DataFrame.
    set_parameters(parameters)
        Sets the algorithm's parameters from a provided dictionary, using defaults where necessary.
    run(data, parameters, prepare_data=True)
        Runs the SAM algorithm on the provided data and parameters, and returns the learned causal structure.
    """

    def __init__(self, algorithm='sam', verbose=False):
        """
        Initializes the SAMAlgorithm class.

        Parameters
        ----------
        algorithm : str, optional
            The name of the algorithm. Default is 'sam'.
        verbose : bool, optional
            If True, enables verbose logging. Default is False.
        """
        self.algorithm = algorithm
        self.verbose = verbose
        self.logger = get_logger(name=__name__, verbose=self.verbose)

    def check_parameters(self):
        """
        Validates the parameters required for running the SAM algorithm.

        Returns
        -------
        bool
            True if all parameters are valid, raises ValueError otherwise.

        Raises
        ------
        ValueError
            If any parameter is out of the valid range or not of the expected type.
        """
        if not (0 < self.lr < 1) or not (0 < self.dlr < 1):
            self.logger.error('Learning rates (lr and dlr) must be between 0 and 1')
            return False

        if not isinstance(self.mixed_data, bool):
            self.logger.error('mixed_data must be a boolean value')
            return False

        if self.lambda1 < 0 or self.lambda2 < 0:
            self.logger.error('lambda1 and lambda2 must be non-negative')
            return False

        if not isinstance(self.nh, int) or self.nh <= 0 or not isinstance(self.dnh, int) or self.dnh <= 0:
            self.logger.error('nh and dnh must be positive integers')
            return False

        if not isinstance(self.train_epochs, int) or self.train_epochs <= 0 or not isinstance(self.test_epochs, int) or self.test_epochs <= 0:
            self.logger.error('train_epochs and test_epochs must be positive integers')
            return False

        if not isinstance(self.batch_size, int) or self.batch_size < 0:
            self.logger.error('batch_size must be a non-negative integer')
            return False

        if self.losstype not in ['fgan', 'gan', 'mse']:
            self.logger.error("losstype must be one of 'fgan', 'gan', or 'mse'")
            return False

        return True

    def prepare_data(self, data):
        """
        Prepares the data for the SAM algorithm by converting it to a pandas DataFrame if necessary.

        Parameters
        ----------
        data : any
            The dataset to be used. Can be a pandas DataFrame or an object that implements the `get_dataset` method.

        Returns
        -------
        data : pandas.DataFrame
            The prepared dataset as a pandas DataFrame.
        """
        if not isinstance(data, pd.DataFrame):
            data = data.get_dataset()
        return data

    def set_parameters(self, parameters):
        """
        Sets the parameters for the SAM algorithm, using default values where necessary.

        Parameters
        ----------
        parameters : dict
            A dictionary containing the parameters to set, such as learning rates, lambda values, number of hidden units, etc.

        Raises
        ------
        ValueError
            If any of the parameters are invalid.
        """
        self.lr = parameters.get('lr', 0.01)
        self.dlr = parameters.get('dlr', 0.001)
        self.mixed_data = parameters.get('mixed_data', False)
        self.lambda1 = parameters.get('lambda1', 10)
        self.lambda2 = parameters.get('lambda2', 0.001)
        self.nh = parameters.get('nh', 20)
        self.dnh = parameters.get('dnh', 200)
        self.train_epochs = parameters.get('train_epochs', 3000)
        self.test_epochs = parameters.get('test_epochs', 1000)
        self.batch_size = parameters.get('batch_size', 100)
        self.losstype = parameters.get('losstype', 'fgan')

        if not self.check_parameters():
            raise ValueError("Invalid parameters for SAM algorithm")

    def run(self, data, parameters, prepare_data=True):
        """
        Runs the SAM algorithm on the provided data and parameters.

        Parameters
        ----------
        data : any
            The dataset to be used, either as a pandas DataFrame or an object implementing `get_dataset`.
        parameters : dict
            The parameters to configure the SAM algorithm.
        prepare_data : bool, optional
            If True, the data will be prepared before running the algorithm. Default is True.

        Returns
        -------
        tuple
            A tuple containing:
            - mec_graph : DAGWrapper
                The learned causal structure represented as a graph.
            - library_results : dict
                A dictionary containing the resulting graph and additional results.

        Raises
        ------
        ValueError
            If any of the parameters are invalid.
        """
        if prepare_data:
            data = self.prepare_data(data)
        self.set_parameters(parameters)

        # Initialize and run the SAM algorithm with the class parameters
        sam = SAM(
            lr=self.lr, dlr=self.dlr, mixed_data=self.mixed_data,
            lambda1=self.lambda1, lambda2=self.lambda2, nh=self.nh,
            dnh=self.dnh, train_epochs=self.train_epochs,
            test_epochs=self.test_epochs, batch_size=self.batch_size,
            losstype=self.losstype
        )
        output = sam.predict(data)

        # Wrap the learned structure in a DAGWrapper
        mec_graph = DAGWrapper()
        mec_graph.dag = output

        library_results = {'mec': mec_graph}

        return mec_graph, library_results
