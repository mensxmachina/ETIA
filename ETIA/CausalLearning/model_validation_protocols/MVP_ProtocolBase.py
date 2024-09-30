from ..utils.logger import get_logger


class MVP_ProtocolBase:
    """
    A base class for running protocols for causal discovery algorithms.

    This class provides the foundation for implementing various protocols to evaluate causal discovery algorithms.
    Derived classes should implement specific protocols (e.g., KFoldCV, Holdout). This class should not be
    instantiated directly.

    Methods
    -------
    set_params(parameters)
        Sets the parameters for the protocol.
    run_protocol(data, algorithm, parameters, n_jobs=1)
        Runs the protocol and returns the results in array format.
    """

    def set_params(self, parameters):
        """
        Sets the parameters of the protocol.

        Parameters
        ----------
        parameters : dict
            A dictionary containing the protocol-specific parameters to set. Each key corresponds to a parameter
            name and its value defines the parameter's value.

        Returns
        -------
        None
        """
        pass

    def run_protocol(self, data, algorithm, parameters, n_jobs=1):
        """
        Runs the protocol using the specified causal discovery algorithm and dataset.

        Parameters
        ----------
        data : Any
            The dataset on which to run the causal discovery algorithm. Can be in various formats (e.g., pandas DataFrame).
        algorithm : Any
            The causal discovery algorithm to evaluate within the protocol.
        parameters : dict
            A dictionary of parameters for both the protocol and the algorithm.
        n_jobs : int, optional
            The number of parallel jobs to run during the evaluation. Default is 1.

        Returns
        -------
        Any
            The results of the protocol in array format, which may vary based on the specific implementation.
        """
        pass
