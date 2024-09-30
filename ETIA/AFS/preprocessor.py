from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Any
import logging


class Preprocessor:
    """
    Preprocessor class for data preprocessing.

    Methods
    -------
    fit_transform(data)
        Fits the preprocessor to the data and transforms it.
    transform(data)
        Transforms the data using the fitted preprocessor.
    """

    def __init__(self, method: str = 'standard'):
        """
        Initializes the Preprocessor.

        Parameters
        ----------
        method : str, optional
            The preprocessing method to use ('standard' or 'minmax'). Default is 'standard'.
        """
        self.method = method
        self.scaler = None
        self.logger = logging.getLogger(__name__)

    def fit_transform(self, data: Any) -> Any:
        """
        Fits the preprocessor to the data and transforms it.
        """
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported preprocessing method: {self.method}")
        return self.scaler.fit_transform(data)

    def transform(self, data: Any) -> Any:
        """
        Transforms the data using the fitted preprocessor.
        """
        if self.scaler is None:
            raise ValueError("Preprocessor has not been fitted yet.")
        return self.scaler.transform(data)
