
import jpype
from sklearn.metrics import r2_score

from ETIA.AFS.predictive_model import PredictiveModel
from jpype import *
from jpype.types import *
import jpype.imports
import json
import pickle
import time

from metrics_evaluation.evaluate_prec_rec_sets import *
from ETIA.AFS import AFS
# ----------------------------------------------------------------------------------------------------------
# Experiment: Apply the AFS module on the simulated data to reduce the dimensionality of the problem
# Author: kbiza@csd.uoc.gr
# ----------------------------------------------------------------------------------------------------------


def main():

    # file names
    path = './files_results/'
    id = '200n_2500s_3ad_6md_1exp_10rep_'           # change the file name if needed
    input_name = path + id + 'files.pkl'
    params_name = path + id + 'params.pkl'
    output_name = path + id + 'files_mb.pkl'

    # load data and simulation parameters
    with open(input_name, 'rb') as inp:
        files = pickle.load(inp)

    with open(params_name, 'rb') as inp:
        params = pickle.load(inp)

    for rep in range(params['n_rep']):

        print('\n Rep %d' %rep)
        t0 = time.time()

        target_name = files[rep]['target_name']
        data_train = files[rep]['data_train']
        data_test = files[rep]['data_test']
        exposure_names = files[rep]['exposure_names']
        true_MB_names = files[rep]['true_MB_names']
        target_idx = files[rep]['target_idx']
        exposure_idx = files[rep]['exposure_idx']
        true_pred_config = files[rep]['true_pred_config']
        true_mb_target = files[rep]['true_mb_target']
        start = time.time()
        afs = AFS(depth=2, num_processors=12)
        results = afs.run_AFS(data_train, {data_train.columns[target_idx]:'continuous'}, pred_configs=0.3)
        est_mb_names_target = results['selected_features']

        print(results['bbc_score'])
        print(results['ci'])
        # AFS on each Mb member of the target
        set_idx_ = est_mb_names_target
        print(set_idx_)

        # AFS on each exposure
        for e_name in exposure_names:
            afs = AFS(depth=1, num_processors=12)
            results = afs.run_AFS(data_train, {e_name:'continuous'}, pred_configs=0.3)
            mb_idx_ = results['selected_features']
            set_idx_.update(mb_idx_)

        # Get all unique elements from the arrays (values)
        unique_elements = set()

        # Iterate over all values (arrays) in the dictionary
        for value in set_idx_.values():
            unique_elements.update(value)

        # Convert the set of unique elements back to a list (optional)
        unique_sel_vars = list(unique_elements)


        unique_sel_vars.sort()
        est_MB_names = unique_sel_vars
        if(target_name in est_MB_names):
            est_MB_names.remove(target_name)
        prec_mb, rec_mb = evaluate_prec_rec_sets(true_MB_names, est_MB_names)
        print('True MB', true_MB_names)
        print('Est MB', est_MB_names)
        print(prec_mb)
        print(rec_mb)
        end = time.time()
        # Study vars : target + exposure + mb

        pm = PredictiveModel()
        true_model_target_given_mb = pm.fit(results['best_config'],
                                                              data_train[true_mb_target].to_numpy(),
                                                              data_train[target_name].to_numpy(),
                                                              None, None,
                                                                'continuous')
        y_test_truemb = pm.predict(data_test[true_mb_target].to_numpy())
        opt_pred_model_target_given_mb = PredictiveModel()
        opt_pred_model_target_given_mb.fit(results['best_config'],
                                                              data_train[est_MB_names].to_numpy(),
                                                              data_train[target_name].to_numpy(),
                                                              None, None,
                                                                'continuous')

        y_test_estmb = opt_pred_model_target_given_mb.predict(data_test[est_MB_names].to_numpy())
        r2_truemb = r2_score(data_test[target_name], y_test_truemb)
        r2_estmb = r2_score(data_test[target_name], y_test_estmb)

        print('True R2: ', r2_truemb)
        print('Est R2: ', r2_estmb)
        print('DeltaR: ', r2_truemb - r2_estmb)
        files[rep]['est_study_names'] = est_MB_names
        files[rep]['est_study_idx'] = list(data_train.columns.get_loc(col) for col in est_MB_names)

        files[rep]['est_MB_names'] = est_MB_names
        files[rep]['est_MB_idx'] = list(data_train.columns.get_loc(col) for col in est_MB_names)
        files[rep]['est_mb_names_target'] = est_mb_names_target

        files[rep]['prec_mb'] = prec_mb
        files[rep]['rec_mb'] = rec_mb
        files[rep]['r2_estmb'] = r2_estmb
        files[rep]['r2_truemb'] = r2_truemb
        files[rep]['deltar2'] = r2_truemb - r2_estmb
        files[rep]['opt_pred_config_target_given_mb'] = results['best_config']
        files[rep]['opt_pred_model_target_given_mb'] = opt_pred_model_target_given_mb
        files[rep]['afs_exec_time'] = time.time() - start

        print('Time: ', end-start)

    precs_mb = [d.get('prec_mb') for d in files]
    recs_mb = [d.get('rec_mb') for d in files]
    delta_r2s = [d.get('deltar2') for d in files]
    print('mean Prec:%0.2f' %(np.mean(precs_mb)), 'SE:%0.2f' %(np.std(precs_mb) / np.sqrt(len(precs_mb))))
    print('mean Rec:%0.2f' %(np.mean(recs_mb)), 'SE:%0.2f' %(np.std(recs_mb) / np.sqrt(len(recs_mb))))
    print('mean DeltaR2:%0.2f' % (np.mean(delta_r2s)), 'SE:%0.2f' % (np.std(delta_r2s) / np.sqrt(len(delta_r2s))))

    with open(output_name, 'wb') as outp:
        pickle.dump(files, outp, pickle.HIGHEST_PROTOCOL)
if __name__ == "__main__":
    main()
