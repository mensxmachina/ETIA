
import json
import pickle
import time
import pandas as pd
import numpy as np
import jpype
from ETIA.data.Dataset import Dataset
from jpype import *
from jpype.types import *
import jpype.imports
from ETIA.CausalLearning.CausalModel import pywhy_graph_to_matrix
from metrics_evaluation.adjacency_precision_recall import *
from metrics_evaluation.shd_mag_pag import *
from ETIA.CausalLearning import CausalLearner

# ---------------------------------------------------------------------------------------------------
# Experiment: Apply the CL module on the reduced dataset to estimate the causal graph
# Author: kbiza@csd.uoc.gr
# ---------------------------------------------------------------------------------------------------


def main():

    # file names
    path = './files_results/'
    id = '500n_2500s_3ad_6md_1exp_10rep_'                 # change the file name if needed
    params_name = path + id + 'params.pkl'
    input_name = path + id + 'files_mb.pkl'
    output_name = path + id + 'files_mb_cd.pkl'

    # load data and simulation parameters
    with open(input_name, 'rb') as inp:
        files = pickle.load(inp)

    with open(params_name, 'rb') as inp:
        params = pickle.load(inp)

    for rep in range(params['n_rep']):

        print('\n Rep %d' %rep)
        t0 = time.time()

        true_dag = files[rep]['true_dag']
        est_MB_names = files[rep]['est_MB_names']
        est_MB_names = sorted(est_MB_names, key=lambda x: int(x[1:]))
        data = files[rep]['data'][est_MB_names]
        true_pag_study = files[rep]['true_pag_study']
        true_study_idx = files[rep]['true_study_idx']
        est_study_idx = files[rep]['est_study_idx']
        cl = CausalLearner(data)
        results = cl.learn_model()
        est_pag_study = results['matrix_mec_graph']
        # Graph matrices over all vars
        all_vars = true_dag.columns
        opt_pag_all = pd.DataFrame(np.zeros((len(all_vars), len(all_vars)), dtype=int),columns=all_vars, index=all_vars)
        opt_pag_all.loc[est_pag_study.index, est_pag_study.columns] = \
            (est_pag_study.loc)[est_pag_study.index, est_pag_study.columns]

        true_pag_all = pd.DataFrame(np.zeros((len(all_vars), len(all_vars)), dtype=int),columns=all_vars, index=all_vars)
        true_pag_all.loc[true_pag_study.index, true_pag_study.columns] = \
            (true_pag_study.loc)[true_pag_study.index, true_pag_study.columns]

        # Graph matrices over the union of true and est variables
        union_study_idx = list(set(true_study_idx + est_study_idx))
        union_study_idx.sort()
        union_study_names = true_dag.columns[union_study_idx]
        true_pag_union = true_pag_all.loc[union_study_names, union_study_names]
        opt_pag_union = opt_pag_all.loc[union_study_names, union_study_names]

        # Compute SHD
        shd_opt = shd_mag_pag(true_pag_union.to_numpy(), opt_pag_union.to_numpy())

        # Compute adjacency precision and recall
        adj_prec, adj_rec = adjacency_precision_recall(true_pag_union, opt_pag_union)
        print('Adj precision: ', adj_prec)
        print('Adj recall: ', adj_rec)
        print('SHD: ', shd_opt)
        causal_configs = cl.cdhpo.configs_ran
        # Evaluate OCT method
        shds_all = np.zeros((len(causal_configs), 1))
        est_pag_union_all = []
        dataset_obj = Dataset(
            data=data,
            data_time_info={'n_lags': 0, 'time_lagged': False},
            time_series=False,
            dataset_name='Preloaded Dataset'
        )
        mec_graphs = []
        for i in range(len(causal_configs)):
            mec_graph_i, _, _ = \
                causal_configs[i]['model'].run(dataset_obj, causal_configs[i])
            est_pag_i_all = pd.DataFrame(np.zeros((len(all_vars), len(all_vars)), dtype=int),
                                           columns=all_vars, index=all_vars)

            est_pag_i_all.loc[mec_graph_i.index, mec_graph_i.columns] = (
                mec_graph_i.loc)[mec_graph_i.index, mec_graph_i.columns]

            est_pag_i_union = est_pag_i_all.loc[union_study_names, union_study_names]

            shds_all[i, 0] = shd_mag_pag(true_pag_union.to_numpy(), est_pag_i_union.to_numpy())
            est_pag_union_all.append(est_pag_i_union)
            mec_graphs.append(mec_graph_i)

        print('Delta SHD: ', shd_opt - np.min(shds_all))
        # Save
        files[rep]['est_pag_study'] = est_pag_study
        files[rep]['opt_causal_config'] = results['optimal_conf']

        files[rep]['union_study_names'] = union_study_names
        files[rep]['opt_pag_union'] = opt_pag_union
        files[rep]['true_pag_union'] = true_pag_union
        files[rep]['est_pag_union_all'] = est_pag_union_all

        files[rep]['shd'] = shd_opt
        files[rep]['adj_prec'] = adj_prec
        files[rep]['adj_rec'] = adj_rec
        files[rep]['delta_shd'] = shd_opt - np.min(shds_all)
        files[rep]['shds_all'] = shds_all

        files[rep]['cd_exec_time'] = time.time() - t0
        files[rep]['causal_configs'] = causal_configs # similar for all repetitions

    # Print results
    adj_prec_ = [d.get('adj_prec') for d in files]
    print('mean Adj.Precision: %0.2f' %(np.mean(adj_prec_)), 'SE:%0.2f' %(np.std(adj_prec_) / np.sqrt(len(adj_prec_))))

    adj_rec_ = [d.get('adj_rec') for d in files]
    print('mean Adj. Recall: %0.2f' %(np.mean(adj_rec_)), 'SE:%0.2f' %(np.std(adj_rec_) / np.sqrt(len(adj_rec_))))

    delta_shd_ = [d.get('delta_shd') for d in files]
    print('mean DSHD:%0.2f' %(np.mean(delta_shd_)), 'SE:%0.2f' %(np.std(delta_shd_) / np.sqrt(len(delta_shd_))))

    with open(output_name, 'wb') as outp:
        pickle.dump(files, outp, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()