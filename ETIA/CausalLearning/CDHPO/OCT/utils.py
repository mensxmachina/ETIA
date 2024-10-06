from statistics import stdev
from math import log, pi, exp
import numpy as np

def is_dict_in_array(dictionary, array):
    """
    Check if a dictionary is already in an array of dictionaries.

    :param dictionary: the dictionary to check
    :type dictionary: dict
    :param array: the array of dictionaries to check
    :type array: list
    :return: True if the dictionary is in the array, False otherwise
    :rtype: bool
    """
    for d in array:
        if sorted(d.items()) == sorted(dictionary.items()):
            return True
    return False

def mutual_info_continuous(y, y_hat):

        """
        Computes the mutual information between two continuous variables, assuming Gaussian distribution
        Args:
            y (numpy array): vector of true values
            y_hat (numpy array): vector of predicted values

        Returns:
            mutual_info (float) : mutual information of y and y_hat
        """
        if stdev(y) == 0 or stdev(y_hat) == 0:
            raise ValueError("MutualInfo: zero st_dev")

        std_y = stdev(y)
        std_y_hat = stdev(y_hat)

        h_y = (1/2) * log(2 * pi * exp(1) * (std_y ** 2))
        h_y_hat = (1/2) * log(2 * pi * exp(1) * (std_y_hat ** 2))

        if np.array_equal(y, y_hat):
            mutual_info = h_y
        else:
            corr = np.corrcoef(y, y_hat)[0,1]
            mutual_info = -(1/2)*log(1-corr**2)

        return mutual_info