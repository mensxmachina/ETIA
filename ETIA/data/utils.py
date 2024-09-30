from enum import Enum
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder



class DataTypes(Enum):
    CONTINUOUS = 1
    DISCRETE = 2
    MIXED = 3
    GRAPH = 4
    COVARIANCE = 5
    ALL = 6
def var_types_and_categorical_encoding(data, unique_val_thr=5):
    '''
    Returns information about the data type (continuous or categorical) of each column in data.
    Args:
        data: pandas array with possible nan values, str, int, floats and objects
        unique_val_thr: int value to

    Returns:
        data_type_info : numpy array with two columns :
            1st column has the names of the variables and the
            2nd column has the information 'continuous' or 'catagorical'
    '''


    d = {'var_type': ['continuous' for i in data.columns],
         'n_domain': [0 for i in data.columns]}
    data_type_info = pd.DataFrame(data=d, index=data.columns)

    for var in data.columns:
        # check 1: check if the column has only str
        cur_col = pd.to_numeric(data[var], errors='coerce')
        if pd.isna(cur_col).all():  # input is str
            data_type_info.loc[var, 'var_type'] = 'categorical'

        # check 2: check if there are less than thr number of unique values
        else:
            # continuous
            if len(data[var].unique()) < unique_val_thr:
                data_type_info.loc[var, 'var_type'] = 'categorical'
            # else:
            #     data[var] = data[var].astype('float')

    #   apply ordinal encoding to categorical variables
    # categorical_var_names = data_type_info.index[data_type_info['var_type'] == 'categorical'].tolist()
    categorical_var_names = data_type_info.index[data_type_info['var_type'] == 'categorical'] #.tolist()
    ord_encoder = OrdinalEncoder()
    ord_encoder.fit(data[categorical_var_names])
    data[categorical_var_names] = ord_encoder.transform(data[categorical_var_names])

    #   how many classes they have
    for var in categorical_var_names:
        unique_classes = data[var].unique()
        data_type_info.loc[var, 'n_domain'] = np.nanmax(unique_classes) + 1  # [0,1,...,maxC]


    # data type
    if data_type_info['var_type'].eq('continuous').all():
        data_type = 'continuous'
    elif data_type_info['var_type'].eq('categorical').all():
        data_type = 'categorical'
    else:
        data_type = 'mixed'
    # Check if the DataFrame contains any missing values
    data_type_info['contains_missing_values'] = data.isnull().values.any()

    # Check if the DataFrame contains any constant variables
    data_type_info['contains_constant_vars'] = (data.apply(pd.Series.nunique) == 1).any()

    return data, data_type_info, data_type

def get_data_info(data):
    data_info = {}

    # Get the number of features (columns)
    data_info['num_features'] = len(data.columns)

    # Get the number of samples (rows)
    data_info['num_samples'] = len(data.index)

    # Check if the DataFrame contains any missing values
    data_info['contains_missing_values'] = data.isnull().values.any()

    # Check if the DataFrame contains any constant variables
    data_info['contains_constant_vars'] = (data.apply(pd.Series.nunique) == 1).any()