import re
import pandas as pd
import numpy as np
import jpype
from jpype import *
from jpype.types import *
import jpype.imports
import pickle
import multiprocessing as mp

from sklearn.utils import resample

from ETIA.data.Dataset import Dataset


# ----------------------------------------------------------------------------------------------
# Experiment: Apply the CRV module on the estimated causal graph to create bootstrapped graphs
# Author: kbiza@csd.uoc.gr
# ----------------------------------------------------------------------------------------------


def process_bootstrap(rep, params, files, B, output_queue):
    print('\n Rep %d' % rep)

    all_vars = files[rep]['true_dag'].columns
    union_study_names = files[rep]['opt_pag_union'].columns
    est_MB_names = files[rep]['est_MB_names']
    est_MB_names = sorted(est_MB_names, key=lambda x: int(x[1:]))
    data = files[rep]['data'][est_MB_names]
    opt_causal_config = files[rep]['opt_causal_config']

    # Create bootstrapped samples and apply the selected causal configuration
    samples_for_boostrap = data.values.copy()
    bootstrapped_samples_all = []
    bootstrapped_matrix_mec = []

    b = 0
    while b < B:
        if(b == 50):
            print('Bootstrap 50 for rep ', str(rep))
        bootstrapped_samples = resample(samples_for_boostrap,
                                         n_samples=samples_for_boostrap.shape[0], replace=True)
        dataset_obj = Dataset(
            data=pd.DataFrame(bootstrapped_samples,columns=data.columns),
            data_time_info={'n_lags': 0, 'time_lagged': False},
            time_series=False,
            dataset_name='Preloaded Dataset'
        )
        boost_mec_graph, _, _ = \
            opt_causal_config['model'].run(dataset_obj, opt_causal_config)
        if isinstance(boost_mec_graph, pd.DataFrame):
            bootstrapped_samples_all.append(bootstrapped_samples)

            boost_pag_all = pd.DataFrame(np.zeros((len(all_vars), len(all_vars)), dtype=int),
                                          columns=all_vars, index=all_vars)
            boost_pag_all.loc[boost_mec_graph.index, boost_mec_graph.columns] = (
                boost_mec_graph.loc)[boost_mec_graph.index, boost_mec_graph.columns]

            boost_pag_union = boost_pag_all.loc[union_study_names, union_study_names]

            bootstrapped_matrix_mec.append(boost_pag_union)

            b += 1
    print('Ended rep ', str(rep))
    output_queue.put((rep, bootstrapped_samples_all, bootstrapped_matrix_mec))


def main():
    # file names
    path = './files_results/'
    id = '500n_2500s_3ad_6md_1exp_10rep_'
    params_name = path + id + 'params.pkl'
    input_name = path + id + 'files_mb_cd.pkl'
    output_name = path + id + 'files_mb_cd_boot.pkl'

    # load data and simulation parameters
    with open(input_name, 'rb') as inp:
        files = pickle.load(inp)

    with open(params_name, 'rb') as inp:
        params = pickle.load(inp)

    B = 100
    output_queue = mp.Manager().Queue()
    processes = []

    for rep in range(params['n_rep']):
        p = mp.Process(target=process_bootstrap, args=(rep, params, files, B, output_queue))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # Collect results from the output queue
    results_collected = 0
    expected_results = params['n_rep']
    while results_collected < expected_results:
        rep, bootstrapped_samples_all, bootstrapped_matrix_mec = output_queue.get()
        files[rep]['bootstrapped_samples'] = bootstrapped_samples_all
        files[rep]['bootstrapped_mec'] = bootstrapped_matrix_mec
        results_collected += 1

    # Save
    with open(output_name, 'wb') as outp:
        pickle.dump(files, outp, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()