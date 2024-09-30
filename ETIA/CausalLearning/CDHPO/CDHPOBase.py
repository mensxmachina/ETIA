from ..utils.logger import get_logger

class CDHPOBase:
    """
    A base class for Causal Discovery Hyperparameter Optimization (CDHPO) algorithms.
    This class defines the basic structure for implementing hyperparameter optimization algorithms.

    Methods
    -------
    run()
        Runs the hyperparameter optimization process. This method should be overridden by subclasses.
    run_new()
        Re-runs the optimization process with new or updated configurations. This method should be overridden by subclasses.
    find_best_config(algorithms)
        Finds the best configuration from a list of algorithms. This method should be overridden by subclasses.
    save_progress(path)
        Saves the current progress of the optimization process.
    load_progress(path)
        Loads the progress of a previously run optimization from the specified path.
    """

    def __init__(self, configurations, dataset):
        """
        Initializes the CDHPO algorithm with the provided configurations and dataset.

        Parameters
        ----------
        configurations : object
            An instance of the configurations object containing hyperparameters and settings.
        dataset : object
            The dataset to be used for hyperparameter optimization.
        """
        self.configurations = configurations
        self.dataset = dataset
        self.results = None

    def run(self):
        """
        Runs the hyperparameter optimization process.
        This method should be overridden by subclasses to provide the specific optimization algorithm.

        Returns
        -------
        object
            The optimal configuration after the hyperparameter optimization process.
        """
        raise NotImplementedError("The run method should be implemented in subclasses.")

    def run_new(self):
        """
        Re-runs the optimization process, typically with new or updated configurations.
        This method should be overridden by subclasses to define the behavior for restarting or continuing optimization.
        """
        raise NotImplementedError("The run_new method should be implemented in subclasses.")

    def find_best_config(self, algorithms):
        """
        Finds the best configuration among the given algorithms.

        Parameters
        ----------
        algorithms : list
            A list of algorithms to evaluate and choose the best configuration from.

        Returns
        -------
        object
            The best configuration determined from the list of algorithms.
        """
        raise NotImplementedError("The find_best_config method should be implemented in subclasses.")

    def save_progress(self, path):
        """
        Saves the progress of the optimization process to a file.

        Parameters
        ----------
        path : str
            The file path where the progress should be saved.
        """
        raise NotImplementedError("The save_progress method should be implemented in subclasses.")

    def load_progress(self, path):
        """
        Loads the progress of a previously run optimization from a file.

        Parameters
        ----------
        path : str
            The file path from where the progress should be loaded.
        """
        raise NotImplementedError("The load_progress method should be implemented in subclasses.")
