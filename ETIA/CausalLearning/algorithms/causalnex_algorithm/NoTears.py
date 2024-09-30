import pandas as pd
import numpy as np
from causalnex.structure.notears import from_pandas
from sklearn.preprocessing import LabelEncoder
from ...utils.logger import get_logger
from ...CausalModel.utils import matrix_to_pywhy_graph


class NoTearsAlgorithm:
    """
    Implements the NOTEARS algorithm for learning causal structures from data.

    Methods
    -------
    prepare_data(Data, parameters=None)
        Prepares data for the NOTEARS algorithm.
    check_parameters(parameters, data_info)
        Checks if the provided parameters are valid for the NOTEARS algorithm.
    structure_model_to_matrix(StructureModel)
        Converts a causal structure model to an adjacency matrix.
    run(data, parameters, prepare_data=True)
        Runs the NOTEARS algorithm on the provided data and parameters.
    """

    def __init__(self, algorithm='notears', verbose=False):
        """
        Initializes the NoTearsAlgorithm class.

        Parameters
        ----------
        algorithm : str, optional
            The name of the algorithm. Default is 'notears'.
        verbose : bool, optional
            If True, enables verbose logging. Default is False.
        """
        self.algorithm = algorithm
        self.verbose = verbose
        self.logger = get_logger(name=__name__, verbose=self.verbose)
        self.data = None

    def prepare_data(self, Data, parameters=None):
        """
        Prepares data for the NOTEARS algorithm. This function can be adapted to include specific data preparation steps for NOTEARS.

        Parameters
        ----------
        Data : object
            The dataset to be used in the algorithm.
        parameters : dict, optional
            Additional parameters for data preparation, if any. Default is None.

        Returns
        -------
        tuple
            Prepared dataset and additional preparation info, if any.
        """
        return Data.get_dataset(), None

    def check_parameters(self, parameters, data_info):
        """
        Checks if the provided parameters are valid for the NOTEARS algorithm.

        Parameters
        ----------
        parameters : dict
            Parameters to be used in the algorithm.
        data_info : dict
            Information about the dataset, such as data type and time info.

        Returns
        -------
        bool
            True if parameters are valid, raises ValueError otherwise.

        Raises
        ------
        ValueError
            If an invalid parameter value is provided, such as a threshold outside the range [0, 1].
        """
        if parameters.get('threshold') and not (0 <= parameters['threshold'] <= 1):
            self.logger.error('Invalid threshold value')
            raise ValueError('Invalid threshold value')
        return True

    def _label_encode_data(self):
        """
        Converts non-numeric columns in the dataset to numeric values using label encoding.
        """
        non_numeric_columns = list(self.data.select_dtypes(exclude=[np.number]).columns)
        le = LabelEncoder()
        for col in non_numeric_columns:
            self.data[col] = le.fit_transform(self.data[col])

    def structure_model_to_matrix(self, StructureModel):
        """
        Converts a StructureModel to an adjacency matrix representation.

        Parameters
        ----------
        StructureModel : causalnex.StructureModel
            A StructureModel object representing the learned structure.

        Returns
        -------
        matrix : numpy.ndarray
            A numpy array representing the adjacency matrix of the DAG, where
            2 represents an edge from source to target, and 3 represents a reverse edge.
        """
        nodes = list(StructureModel.nodes())
        node_indices = {node: i for i, node in enumerate(nodes)}

        matrix = np.zeros((len(nodes), len(nodes)), dtype=int)

        for source, target in StructureModel.edges():
            matrix[node_indices[source], node_indices[target]] = 2
            matrix[node_indices[target], node_indices[source]] = 3

        return matrix

    def _run_notears(self, parameters):
        """
        Applies the NOTEARS algorithm to learn the causal structure from the dataset.

        Parameters
        ----------
        parameters : dict
            Parameters for running the NOTEARS algorithm, such as the threshold for edge inclusion.

        Returns
        -------
        causalnex.StructureModel
            The learned structure model from NOTEARS.
        """
        self._label_encode_data()
        sm = from_pandas(self.data, w_threshold=parameters.get('threshold', 0.8))
        return sm

    def run(self, data, parameters, prepare_data=True):
        """
        Runs the NoTears algorithm to learn a causal structure from the data.

        Parameters
        ----------
        data : object
            The dataset to be used in the algorithm.
        parameters : dict
            The parameters for the algorithm.
        prepare_data : bool, optional
            If True, prepares the data before running the algorithm. Default is True.

        Returns
        -------
        tuple
            A tuple containing the learned structure as a MEC graph (pywhy graph) and library results.

        Raises
        ------
        ValueError
            If parameters are invalid or other issues arise during the algorithm run.
        """
        if prepare_data:
            data_prepared, _ = self.prepare_data(data, parameters)
        else:
            data_prepared = data
        self.data = data_prepared
        self.check_parameters(parameters, {'data_type_info': None, 'data_time_info': None})
        learned_structure = self._run_notears(parameters)
        matrix_graph = self.structure_model_to_matrix(learned_structure)
        mec_graph_pywhy = matrix_to_pywhy_graph(matrix_graph)

        library_results = {'mec': mec_graph_pywhy, 'matrix_graph': matrix_graph}
        return mec_graph_pywhy, library_results
