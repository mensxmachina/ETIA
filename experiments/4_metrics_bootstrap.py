import re
import pandas as pd
import numpy as np
import pickle

from sklearn.metrics import average_precision_score

from ETIA.CRV.causal_graph_utils.confidence_causal_findings import *

# --------------------------------------------------------------------------------------------------
# Experiment: Apply the CRV module on the bootstrapped causal graphs to compute the edge confidences
# Author: kbiza@csd.uoc.gr
# --------------------------------------------------------------------------------------------------


def main():

    # file names
    path = './files_results/'
    id = '1000n_2500s_3ad_6md_1exp_10rep_'                 # change the file name if needed
    params_name = path + id + 'params.pkl'
    input_name = path + id + 'files_mb_cd_boot.pkl'
    output_name = path + id + 'files_mb_cd_boot_evalConf.pkl'

    # load data and simulation parameters
    with open(input_name, 'rb') as inp:
        files = pickle.load(inp)

    with open(params_name, 'rb') as inp:
        params = pickle.load(inp)

    for rep in range(params['n_rep']):

        print('\n Rep %d' %rep)

        true_pag_union = files[rep]['true_pag_union']
        opt_pag_union = files[rep]['opt_pag_union']
        bootstrapped_matrix_mec = files[rep]['bootstrapped_mec']

        weight_data_pd_all, label_data_pd_all = compute_edge_weights(opt_pag_union, bootstrapped_matrix_mec,
                                                             true_graph=true_pag_union, all_edges=True)
        label_d = label_data_pd_all['edge_consistency'].to_numpy(dtype=float)
        fpr, tpr, thresholds = metrics.roc_curve(label_data_pd_all['edge_consistency'].to_numpy(dtype=float),
                                                 weight_data_pd_all['edge_consistency'].to_numpy(dtype=float), pos_label=1)
        auc_all = metrics.auc(fpr, tpr)
        avg_precision = average_precision_score(label_data_pd_all['edge_consistency'].to_numpy(dtype=float), weight_data_pd_all['edge_consistency'].to_numpy(dtype=float))
        # Save
        zrs = 0
        for i in range(len(label_d)):
            if(label_d[i] == 0):
                zrs += 1
        percentage = zrs/len(label_d)
        files[rep]['weight_data_pd_all'] = weight_data_pd_all
        files[rep]['label_data_pd_all'] = label_data_pd_all
        files[rep]['auc_all'] = auc_all
        files[rep]['avg_precision'] = avg_precision
        files[rep]['percentage'] = percentage

    with open(output_name, 'wb') as outp:
        pickle.dump(files, outp, pickle.HIGHEST_PROTOCOL)

    auc_all_ = [d.get('auc_all') for d in files if not np.isnan(d.get('auc_all'))]
    print('mean AUC: %0.2f' % (np.mean(auc_all_)), 'SE:%0.2f' % (np.std(auc_all_) / np.sqrt(len(auc_all_))))

    ap_all_ = [d.get('avg_precision') for d in files if not np.isnan(d.get('avg_precision'))]
    print('mean AP: %0.2f' % (np.mean(ap_all_)), 'SE:%0.2f' % (np.std(ap_all_) / np.sqrt(len(ap_all_))))

    percentage_all = [d.get('percentage') for d in files if not np.isnan(d.get('percentage'))]
    print('mean Percentage of negative: %0.2f' % (np.mean(percentage_all)))


if __name__ == "__main__":
    main()