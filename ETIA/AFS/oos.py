import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit, ShuffleSplit
from typing import Tuple, List, Dict, Any, Optional
import logging


class OOS:
    """
    Out-of-sample protocols for data splitting.

    Methods
    -------
    data_split(oos_protocol, X, y=None, target_type='continuous')
        Splits the data according to the specified out-of-sample protocol.
    """

    def __init__(self):
        """
        Initializes the OOS (Out-of-Sample) class.
        """
        self.logger = logging.getLogger(__name__)

    def data_split(
            self,
            oos_protocol: Dict[str, Any],
            X: Any,
            y: Optional[Any] = None,
            target_type: str = 'continuous'
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Splits the data according to the specified out-of-sample protocol.

        Parameters
        ----------
        oos_protocol : dict
            A dictionary that specifies the out-of-sample protocol.
            The 'name' key should specify the type of protocol (e.g., 'KFoldCV', 'Holdout').
            The 'folds' or 'test_size' key specifies the number of folds or test size (for holdout).
        X : array-like
            The feature data (input variables).
        y : array-like, optional
            The target vector (output variables). Required for stratified splits.
        target_type : str, optional
            Indicates whether the target is 'continuous' or 'categorical'. Default is 'continuous'.

        Returns
        -------
        train_inds : list of np.ndarray
            A list containing the training indices for each fold or holdout split.
        test_inds : list of np.ndarray
            A list containing the testing indices for each fold or holdout split.

        Raises
        ------
        ValueError
            If an unsupported protocol name is provided.
        """
        train_inds = []
        test_inds = []

        if oos_protocol['name'] == 'KFoldCV':
            n_splits = oos_protocol.get('folds', 5)
            if target_type == 'continuous':
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                kf_split = kf.split(X)
            else:
                kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                kf_split = kf.split(X, y)

            for train_index, test_index in kf_split:
                train_inds.append(train_index)
                test_inds.append(test_index)

        elif oos_protocol['name'] == 'Holdout':
            test_size = oos_protocol.get('test_size', 0.2)
            if target_type == 'continuous':
                sss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
            else:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
            for train_index, test_index in sss.split(X, y):
                train_inds.append(train_index)
                test_inds.append(test_index)

        else:
            raise ValueError(f"Protocol {oos_protocol['name']} not supported.")

        return train_inds, test_inds
