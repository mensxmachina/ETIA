# ETIA/algorithms/utils.py

import numpy as np
import pandas as pd
from tigramite import data_processing as pp
from jpype import JArray, JDouble, JInt, JPackage
from ...data.Dataset import Dataset
from ...utils.jvm_manager import *
from ...utils.logger import get_logger

logger = get_logger(__name__, False)

def prepare_data_tetrad(Data, parameters=None):
    if not isinstance(Data, Dataset):
        raise TypeError('The input parameter is incorrect type. Must be type Dataset')

    try:
        start_jvm()
    except:
        logger.debug('Error starting the JVM')

    try:
        jav = JPackage("java")
        data = JPackage("edu.cmu.tetrad.data")
        util = JPackage("java.util")

        data_type_info = Data.get_data_type_info()
        data_pd = Data.get_dataset()
        if 'indexes' in parameters:
            try:
                data_pd = data_pd.iloc[parameters['indexes']]
            except Exception as e:
                logger.error(f"Error indexing data: {e}")
                print(parameters['indexes'])
                print(data_pd)
                raise

        n_lags = Data.get_data_time_info()['n_lags']

        is_cat_var = Data.data_type_info['var_type'] == 'categorical'
        is_cat_var = is_cat_var.to_numpy()
        n_domain = data_type_info['n_domain'].to_numpy()

        data_np = data_pd.to_numpy()
        var_names = data_pd.columns.to_list()
        is_con_var = ~is_cat_var

        my_list = util.LinkedList()
        n_samples, n_cols = data_np.shape
        dataC = data_np[:, is_con_var]
        dataD = data_np[:, is_cat_var].astype(int)

        # Create variable names
        tetrad_names = []
        for lag in range(n_lags + 1):
            for i in range(int(n_cols / (n_lags + 1))):
                if lag == 0:
                    tetrad_names.append(f'X{i + 1}')
                else:
                    tetrad_names.append(f'X{i + 1}:{lag}')

        # Initialize variables
        name_map = []
        for i in range(n_cols):
            tetrad_name = tetrad_names[i]
            if is_cat_var[i]:
                var = data.DiscreteVariable(tetrad_name, n_domain[i])
            else:
                var = data.ContinuousVariable(tetrad_name)
            my_list.add(var)
            if var_names:
                name_map.append([i + 1, tetrad_name, var_names[i]])
            else:
                name_map.append([i + 1, tetrad_name])

        dsM = data.MixedDataBox(my_list, n_samples)

        # Prepare data for continuous and discrete variables
        if np.any(is_con_var):
            tdataC = data_np[:, is_con_var].T.astype(np.float64).tolist()
            java_rows = [JArray(JDouble)(row) for row in tdataC]
            dsC = data.VerticalDoubleDataBox(JArray(JArray(JDouble))(java_rows))
        if np.any(is_cat_var):
            tdataD = data_np[:, is_cat_var].T.astype(np.int32).tolist()
            java_rows = [JArray(JInt)(row) for row in tdataD]
            dsD = data.VerticalIntDataBox(JArray(JArray(JInt))(java_rows))

        # Populate dsM with data
        for i in range(n_samples):
            c = 0
            d = 0
            for node in range(n_cols):
                if is_con_var[node]:
                    dsM.set(i, node, dsC.get(i, c))
                    c += 1
                else:
                    dsM.set(i, node, dsD.get(i, d))
                    d += 1

        ds = data.BoxDataSet(dsM, my_list)
        name_map_pd = pd.DataFrame(name_map, columns=['index', 'tetrad_name', 'var_name'] if var_names else ['index', 'tetrad_name'])
        return ds, name_map_pd
    except Exception as e:
        logger.error(f"An unexpected error occurred in prepare_data_tetrad: {e}", exc_info=True)
        raise Exception("Unable to prepare data for Tetrad due to an unexpected error.") from e

def prepare_data_tigramite(Data, parameters):
    if not isinstance(Data, Dataset):
        raise TypeError('The input parameter is incorrect type. Must be type Dataset')

    data_type_info = Data.get_data_type_info()
    data_pd = Data.get_dataset()
    if 'indexes' in parameters:
        data_pd = data_pd.iloc[parameters['indexes']]
    if parameters.get('ci_test') != 'RegressionCI':
        dataframe_ = pp.DataFrame(data_pd.to_numpy(), var_names=data_pd.columns)
    else:
        data_type = np.zeros(data_pd.shape, dtype='int')
        data_type_ = data_type_info['var_type'].to_numpy()
        data_type_ = data_type_[0:data_pd.shape[1]]  # take only first lag
        data_type_[data_type_ == 'continuous'] = 0
        data_type_[data_type_ == 'categorical'] = 1
        data_type[:] = data_type_

        dataframe_ = pp.DataFrame(data_pd.to_numpy(), data_type=data_type, var_names=data_pd.columns)

    return dataframe_
