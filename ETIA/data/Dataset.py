from .utils import *

class Dataset:
    """
    A class for representing datasets and providing functionalities for loading, manipulating, and processing datasets.

    Parameters
    ----------
    filename : str, optional
        The name of the CSV file containing the dataset. Default is None.
    data_time_info : dict, optional
        Dictionary containing time-related information (lags, etc.). Default is None.
    time_series : bool, optional
        Boolean indicating if the dataset is time series data. Default is False.
    data : pd.DataFrame, optional
        A pandas DataFrame containing preloaded data (e.g., from AFS). Default is None.
    dataset_name : str, optional
        The name of the dataset. If not provided, it defaults to 'Preloaded Dataset' or the filename.

    Attributes
    ----------
    dataset_name : str
        The name of the dataset.
    data_time_info : dict
        Information related to time and lags in the dataset.
    time_series : bool
        Boolean flag indicating if the data is a time series dataset.
    n_lags : int
        The number of time lags in the dataset.
    data : pd.DataFrame
        The loaded dataset.
    data_type_info : dict
        Information on the types of variables in the dataset.
    data_type : str
        General type of data (e.g., continuous, categorical).
    data_general_info : dict
        General information about the dataset.
    processed_data : dict
        Data after processing (currently empty).
    annotations : dict
        Annotations on the dataset (optional).

    Methods
    -------
    load_file(filename)
        Loads a new dataset from a CSV file.
    load_np_dataset(dataset, column_names)
        Loads a new dataset from a NumPy array.
    load_pd_dataset(dataset)
        Loads a new dataset from a pandas DataFrame.
    convert_to_time_lag(n_lags)
        Converts the dataset into time-lagged data.
    get_dataset()
        Returns the dataset stored in the Dataset instance.
    get_data_type_info()
        Returns the data type information of the dataset.
    get_data_time_info()
        Returns the time-related information of the dataset.
    get_info()
        Returns all the general information of the dataset including type and time-related info.
    annotate_dataset(annotations)
        Stores annotations for the dataset.
    """

    def __init__(self, filename=None, data_time_info=None, time_series=False, data=None, dataset_name=None):
        """
        Initializes the Dataset object, either from a file or from a preloaded pandas DataFrame.

        Parameters
        ----------
        filename : str, optional
            Name of the CSV file containing the dataset. Default is None.
        data_time_info : dict, optional
            Dictionary containing time-related information (lags, etc.). Default is None.
        time_series : bool, optional
            Boolean indicating if the dataset is time series data. Default is False.
        data : pd.DataFrame, optional
            A pandas DataFrame containing preloaded data (e.g., from AFS). Default is None.
        dataset_name : str, optional
            The name of the dataset. If not provided, defaults to 'Preloaded Dataset' or the filename.
        """
        if data_time_info is None:
            data_time_info = {'n_lags': 0, 'time_lagged': False}

        self.dataset_name = filename if filename else 'Preloaded Dataset'
        self.data_time_info = data_time_info
        self.time_series = time_series
        self.n_lags = data_time_info['n_lags']

        if data is not None:
            # Use the provided DataFrame
            self.data = data
        elif filename is not None:
            # Load from a CSV file
            self.data = pd.read_csv(filename, header=0)
        else:
            raise ValueError("Either a filename or a pandas DataFrame must be provided")

        # Process the data types
        _, self.data_type_info, self.data_type = var_types_and_categorical_encoding(self.data)
        self.data_general_info = get_data_info(self.data)

        if not self.data_time_info.get('time_lagged', False) and self.n_lags != 0:
            self.convert_to_time_lag(self.n_lags)

        self.processed_data = {}

    def load_file(self, filename):
        """
        Loads a new dataset from a CSV file.

        Parameters
        ----------
        filename : str
            Name of the CSV file to load the dataset from.
        """
        self.data = pd.read_csv(filename)
        _, self.data_type_info, self.data_type = var_types_and_categorical_encoding(self.data)
        self.data_general_info = get_data_info(self.data)

    def load_np_dataset(self, dataset, column_names):
        """
        Loads a new dataset from a NumPy array.

        Parameters
        ----------
        dataset : np.ndarray
            The dataset as a NumPy array.
        column_names : list
            List of column names for the dataset.

        Raises
        ------
        TypeError
            If the input is not a NumPy array.
        """
        if not isinstance(dataset, np.ndarray):
            raise TypeError('load_np_dataset requires numpy array as input')
        self.data = pd.DataFrame(dataset, columns=column_names)
        _, self.data_type_info, self.data_type = var_types_and_categorical_encoding(self.data)
        self.data_general_info = get_data_info(self.data)

    def load_pd_dataset(self, dataset):
        """
        Loads a new dataset from a pandas DataFrame.

        Parameters
        ----------
        dataset : pd.DataFrame
            The dataset as a pandas DataFrame.

        Raises
        ------
        TypeError
            If the input is not a pandas DataFrame.
        """
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError('load_pd_dataset requires pd.DataFrame as input')
        self.data = dataset
        _, self.data_type_info, self.data_type = var_types_and_categorical_encoding(self.data)
        self.data_general_info = get_data_info(self.data)

    def convert_to_time_lag(self, n_lags):
        """
        Converts the dataset into time-lagged data.

        Parameters
        ----------
        n_lags : int
            Number of time lags to add to the dataset.

        Returns
        -------
        pd.DataFrame
            The dataset with added time lags (if applicable).
        """
        # Placeholder for converting data
        # Implement the actual time-lagging logic here
        return self.data

    def get_dataset(self):
        """
        Returns the dataset stored in the Dataset instance.

        Returns
        -------
        pd.DataFrame
            The loaded dataset.
        """
        return self.data

    def get_data_type_info(self):
        """
        Returns the data type information of the dataset.

        Returns
        -------
        dict
            A dictionary containing information about the variable types in the dataset.
        """
        return self.data_type_info

    def get_data_time_info(self):
        """
        Returns the time-related information of the dataset.

        Returns
        -------
        dict
            A dictionary containing time-related information such as lags and whether the dataset is time-lagged.
        """
        return self.data_time_info

    def get_info(self):
        """
        Returns general information about the dataset, including data types and time-related information.

        Returns
        -------
        dict
            A dictionary containing:
            - data_type_info: Information about variable types in the dataset.
            - data_time_info: Time-related information about the dataset.
            - data_type: General type of data (e.g., continuous, categorical).
            - data_general_info: General information about the dataset.
            - dataset_name: The name of the dataset.
        """
        return {
            'data_type_info': self.data_type_info,
            'data_time_info': self.data_time_info,
            'data_type': self.data_type,
            'data_general_info': self.data_general_info,
            'dataset_name': self.dataset_name
        }

    def annotate_dataset(self, annotations):
        """
        Stores annotations for the dataset.

        Parameters
        ----------
        annotations : dict
            Dictionary of annotations related to the dataset.
        """
        self.annotations = annotations
