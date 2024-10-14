import os
import copy
import pickle
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from joblib import Parallel, delayed
from sklearn.metrics import mutual_info_score
from sklearn.model_selection import ParameterGrid

from ...utils import get_logger
from .utils import mutual_info_continuous, is_dict_in_array
from ...CDHPO.CDHPOBase import CDHPOBase
from ...CausalModel.utils import pywhy_graph_to_matrix
from ....CRV.causal_graph_utils.markov_boundary import markov_boundary

class OCT(CDHPOBase):
    """
    Class for performing Order-Based Causal Transfer (OCT) procedure.

    Parameters
    ----------
    oct_params : CDHPOParameters
        Object containing the parameters required for the OCT procedure.
    data : Dataset
        Data to be used for the OCT procedure.
    results_folder : str
        Path to the folder where results will be saved.
    verbose : bool, optional
        If True, enables verbose logging. Default is False.

    Methods
    -------
    run()
        Executes the OCT procedure.
    run_new()
        Continues the OCT procedure with new configurations.
    find_best_config(algorithms)
        Finds the best configuration among specified algorithms.
    save_progress()
        Saves the current state of the OCT object to a file.
    load_progress(path)
        Loads the OCT object state from a file.
    fold_fit(target, c, mec_graphs_configs, train_indexes, test_indexes, fold)
        Performs Markov boundary identification and predictive modeling for a specific fold.
    nodes_parallel(target, c, mec_graphs_configs, train_indexes, test_indexes)
        Calculates the mutual information between the true values and predicted values of a target node in parallel.
    config_parallel(c, mec_graphs_configs, train_indexes, test_indexes)
        Calculates the mutual information scores for all target nodes in parallel.
    permutations(node, poolYhat_best, poolYhat_cur, idxs, poolY)
        Calculates the mutual information scores after swapping predictions between best and current configurations.
    permutations_nodes(node, c)
        Performs permutations for a single node across all permutations.
    calculate_pvalues(c)
        Calculates p-values to compare the current configuration with the best one.
    """

    def __init__(self, oct_params: Any, data: Any, results_folder: str, verbose=False):
        """
        Initializes the OCT class.

        Parameters
        ----------
        oct_params : CDHPOParameters
            Object containing the parameters required for the OCT procedure.
        data : Dataset
            Data to be used for the OCT procedure.
        results_folder : str
            Path to the folder where results will be saved.
        verbose : bool, optional
            If True, enables verbose logging. Default is False.
        """
        super().__init__(oct_params, data)
        self.oct_params = copy.deepcopy(oct_params)
        self.data = copy.deepcopy(data)
        self.results_folder = results_folder
        self.verbose = verbose
        self.logger = get_logger(__name__, verbose=self.verbose)

        self.n_samples, self.n_nodes = self.data.get_dataset().shape
        self.data_type_info = self.data.get_data_type_info()
        self.logger.debug('OCT object has been initialized')

        # Instance-level attributes to ensure no shared references
        self.results = []
        self.mb_configs = []
        self.saved_mb_configs = {}
        self.pred_configs = []
        self.saved_pred_configs = {}
        self.mu_configs = []
        self.mb_size = np.array([])
        self.total_configs_run = 0
        self.configs_ran = []
        self.saved_y_test = {}
        self.matrix_mec_graph = None
        self.matrix_graph = None
        self.var_map = None
        self.mec_graphs_configs = []
        self.p_values = np.array([])
        self.is_equal = np.array([])
        self.mean_mb = np.array([])
        self.y_test_nodes = []
        self.idxs = []
    def save_progress(self):
        """
        Saves the current state of the OCT object to a file.
        """
        path = os.path.join(self.results_folder, 'OCT.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        self.logger.debug(f'OCT progress saved to {path}')

    @staticmethod
    def load_progress(path: str) -> 'OCT':
        """
        Loads the OCT object state from a file.

        Parameters
        ----------
        path : str
            The file path to load the progress from.

        Returns
        -------
        OCT
            The loaded OCT object.
        """
        with open(path, 'rb') as f:
            oct_instance = pickle.load(f)
        oct_instance.logger.debug(f'OCT progress loaded from {path}')
        return oct_instance

    def fold_fit(
        self,
        target: int,
        c: int,
        mec_graphs_configs: List[Any],
        train_indexes: List[np.ndarray],
        test_indexes: List[np.ndarray],
        fold: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs Markov boundary identification and predictive modeling for a specific fold of a target variable.

        Parameters
        ----------
        target : int
            Target variable index.
        c : int
            Configuration index.
        mec_graphs_configs : list
            MEC graphs configurations.
        train_indexes : list
            List of training indices for each fold.
        test_indexes : list
            List of testing indices for each fold.
        fold : int
            Fold index.

        Returns
        -------
        mb : np.ndarray
            Markov boundary indices.
        prediction : np.ndarray
            Predicted values.
        y_test : np.ndarray
            Actual target values for the test data.
        """
        self.saved_mb_configs.setdefault(c, {}).setdefault(target, {})
        self.saved_pred_configs.setdefault(c, {}).setdefault(target, {})
        self.saved_y_test.setdefault(c, {}).setdefault(target, {})

        if fold in self.saved_mb_configs[c][target]:
            return (
                self.saved_mb_configs[c][target][fold],
                self.saved_pred_configs[c][target][fold],
                self.saved_y_test[c][target][fold],
            )

        mb = markov_boundary(target, self.graphs_configs[c][fold])

        self.mb_size[c, fold, target] = len(mb)

        data_ = self.data.get_dataset()
        X_train = data_.iloc[train_indexes[fold]].to_numpy()[:, mb]
        y_train = data_.iloc[train_indexes[fold]].to_numpy()[:, target]
        X_test = data_.iloc[test_indexes[fold]].to_numpy()[:, mb]
        y_test = data_.iloc[test_indexes[fold]].to_numpy()[:, target]

        if self.data_type_info['var_type'][target] == 'categorical':
            if len(mb) > 0:
                clf = copy.deepcopy(self.oct_params.regressor.regressor)
                clf.fit(X_train, y_train)
                prediction = clf.predict(X_test)
            else:
                values, counts = np.unique(y_train, return_counts=True)
                prediction = np.full(y_test.shape, values[np.argmax(counts)])
        else:
            if len(mb) > 0:
                clf = copy.deepcopy(self.oct_params.regressor.regressor)
                clf.fit(X_train, y_train)
                prediction = clf.predict(X_test)
            else:
                prediction = np.random.uniform(np.min(y_train), np.max(y_train), y_test.shape)

        self.logger.debug(f'Fold {fold} for variable {target} is fitted')

        self.saved_mb_configs[c][target][fold] = mb
        self.saved_pred_configs[c][target][fold] = prediction
        self.saved_y_test[c][target][fold] = y_test

        return mb, prediction, y_test

    def nodes_parallel(
        self,
        target: int,
        c: int,
        mec_graphs_configs: List[Any],
        train_indexes: List[np.ndarray],
        test_indexes: List[np.ndarray],
    ) -> Tuple[float, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Calculates the mutual information between the true values and predicted values of a target node in parallel.

        Parameters
        ----------
        target : int
            Target variable index.
        c : int
            Configuration index.
        mec_graphs_configs : list
            MEC graphs configurations.
        train_indexes : list
            List of training indices for each fold.
        test_indexes : list
            List of testing indices for each fold.

        Returns
        -------
        mu : float
            Mutual information score between the true values and predicted values.
        mb_folds : list
            List of Markov boundaries for each fold.
        pred_folds : list
            List of predictions for each fold.
        y_test_folds : list
            List of true values for each fold.
        """
        results = Parallel(n_jobs=self.oct_params.n_jobs)(
            delayed(self.fold_fit)(
                target,
                c,
                mec_graphs_configs,
                train_indexes,
                test_indexes,
                fold,
            )
            for fold in range(self.oct_params.oos_protocol.protocol.folds_to_run)
        )

        mb_folds, pred_folds, y_test_folds = zip(*results)

        pred_folds_np = np.concatenate(pred_folds, axis=0)
        y_test_folds_np = np.concatenate(y_test_folds, axis=0)

        if self.data_type_info['var_type'][target] == 'categorical':
            mu = mutual_info_score(y_test_folds_np, pred_folds_np)
        else:
            mu = mutual_info_continuous(y_test_folds_np, pred_folds_np)

        self.logger.debug(f'All folds for target {target} have been fitted')
        return mu, list(mb_folds), list(pred_folds), list(y_test_folds)

    def config_parallel(
        self,
        c: int,
        mec_graphs_configs: List[Any],
        train_indexes: List[np.ndarray],
        test_indexes: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[List[np.ndarray]], List[List[np.ndarray]], List[List[np.ndarray]]]:
        """
        Calculates the mutual information scores for all target nodes in parallel.

        Parameters
        ----------
        c : int
            Configuration index.
        mec_graphs_configs : list
            MEC graphs configurations.
        train_indexes : list
            List of training indices for each fold.
        test_indexes : list
            List of testing indices for each fold.

        Returns
        -------
        mu_list : np.ndarray
            Array of mutual information scores for each target node.
        mb_list : list
            List of Markov boundaries for each target node.
        pred_list : list
            List of predictions for each target node.
        y_test_list : list
            List of true values for each target node.
        """
        results = Parallel(n_jobs=self.oct_params.n_jobs)(
            delayed(self.nodes_parallel)(
                target,
                c,
                mec_graphs_configs,
                train_indexes,
                test_indexes,
            )
            for target in range(self.n_nodes)
        )

        mu_list, mb_list, pred_list, y_test_list = zip(*results)

        self.logger.debug(f'Mutual information calculated for all nodes for config {c}')
        self.save_progress()
        return np.array(mu_list), list(mb_list), list(pred_list), list(y_test_list)

    def permutations(
        self,
        node: int,
        poolYhat_best: np.ndarray,
        poolYhat_cur: np.ndarray,
        idxs: np.ndarray,
        poolY: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Calculates the mutual information scores after swapping predictions between best and current configurations.

        Parameters
        ----------
        node : int
            Node index.
        poolYhat_best : np.ndarray
            Predictions from the best configuration.
        poolYhat_cur : np.ndarray
            Predictions from the current configuration.
        idxs : np.ndarray
            Indices for permutation.
        poolY : np.ndarray
            Actual target values.

        Returns
        -------
        x : float
            Mutual information score for the best configuration after swap.
        y : float
            Mutual information score for the current configuration after swap.
        """
        swap_best = poolYhat_best.copy()
        swap_cur = poolYhat_cur.copy()

        swap_best[idxs] = poolYhat_cur[idxs]
        swap_cur[idxs] = poolYhat_best[idxs]

        if self.data_type_info['var_type'][node] == 'categorical':
            x = mutual_info_score(poolY, swap_best)
            y = mutual_info_score(poolY, swap_cur)
        else:
            x = mutual_info_continuous(poolY, swap_best)
            y = mutual_info_continuous(poolY, swap_cur)

        return x, y

    def permutations_nodes(self, node: int, c: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs permutations for a single node across all permutations.

        Parameters
        ----------
        node : int
            Node index.
        c : int
            Configuration index.

        Returns
        -------
        swap_best_metric : np.ndarray
            Array of mutual information scores for the best configuration after swaps.
        swap_cur_metric : np.ndarray
            Array of mutual information scores for the current configuration after swaps.
        """
        poolYhat_best = np.concatenate(self.pred_configs[self.opt_config][node], axis=0)
        poolYhat_cur = np.concatenate(self.pred_configs[c][node], axis=0)
        poolY = np.concatenate(self.y_test_nodes[node], axis=0)

        results = Parallel(n_jobs=self.oct_params.n_jobs)(
            delayed(self.permutations)(
                node,
                poolYhat_best,
                poolYhat_cur,
                self.idxs[i],
                poolY,
            )
            for i in range(self.oct_params.n_permutations)
        )

        swap_best_metric, swap_cur_metric = zip(*results)
        self.logger.debug(f'All permutations for node {node} have been completed')
        return np.array(swap_best_metric), np.array(swap_cur_metric)

    def calculate_pvalues(self, c: int):
        """
        Calculates p-values to compare the current configuration with the best one.

        Parameters
        ----------
        c : int
            Configuration index.
        """
        if c == self.opt_config:
            return

        swap_best_metric = np.zeros((self.oct_params.n_permutations, self.n_nodes))
        swap_cur_metric = np.zeros((self.oct_params.n_permutations, self.n_nodes))

        results = Parallel(n_jobs=self.oct_params.n_jobs)(
            delayed(self.permutations_nodes)(node, c) for node in range(self.n_nodes)
        )

        for node, (best_metric, cur_metric) in enumerate(results):
            swap_best_metric[:, node] = best_metric
            swap_cur_metric[:, node] = cur_metric

        cur_metric = self.mean_mu_configs[c]
        best_metric = np.max(self.mean_mu_configs)
        obs_t_stat = best_metric - cur_metric
        t_stat = np.mean(swap_best_metric, axis=1) - np.mean(swap_cur_metric, axis=1)

        p_val = np.count_nonzero(t_stat >= obs_t_stat) / self.oct_params.n_permutations
        self.p_values[c] = p_val

        self.logger.debug(f'P-value for configuration {c} calculated: {p_val}')
        self.is_equal[c] = p_val > self.oct_params.alpha

    def run(self) -> Tuple[Dict[str, Any], np.ndarray, Any]:
        """
        Executes the OCT procedure.

        Returns
        -------
        opt_config : dict
            The optimal configuration found.
        matrix_mec_graph : np.ndarray
            The MEC graph matrix of the optimal configuration.
        matrix_graph : nd.nd.array
            The graph matrix of optimal configuration
        library_results : Any
            Results from the causal discovery library.
        """
        if self.configs_ran:
            self.logger.error('Previous configurations have already been run. Use run_new() to continue.')
            raise RuntimeError('Cannot run OCT when previous configurations have already been run.')

        self.oct_params.oos_protocol.protocol.init_protocol(self.data)
        mec_graphs_ = []
        graphs_ = []
        for algo, algo_configs in self.oct_params.configs.items():
            cd_configs = list(ParameterGrid(algo_configs))
            self.logger.info(f'Running algorithm: {algo}')
            for params in cd_configs:
                params['var_type'] = self.data_type_info['var_type']
                matrix_mec_graph, matrix_graph, _ = self.oct_params.oos_protocol.protocol.run_protocol(
                    self.data, params['model'], params
                )
                mec_graphs = [mec_graph.to_numpy()
                              for mec_graph in matrix_mec_graph]
                graphs = [graph.to_numpy() for graph in matrix_graph]
                mec_graphs_.append(mec_graphs)
                graphs_.append(graphs)
                self.save_progress()
                self.logger.debug(f'Protocol for params {params} has been run')
                params['name'] = algo
                self.configs_ran.append(params)

        self.mec_graphs_configs = mec_graphs_
        self.graphs_configs = graphs_
        num_configs = len(self.configs_ran)
        num_folds = self.oct_params.oos_protocol.protocol.folds_to_run
        self.mb_size = np.zeros((num_configs, num_folds, self.n_nodes))

        config_results = Parallel(n_jobs=self.oct_params.n_jobs)(
            delayed(self.config_parallel)(
                c,
                self.mec_graphs_configs,
                self.oct_params.oos_protocol.protocol.train_indexes,
                self.oct_params.oos_protocol.protocol.test_indexes,
            )
            for c in range(num_configs)
        )

        self.mu_configs, self.mb_configs, self.pred_configs, y_test_nodes_list = zip(*config_results)
        self.y_test_nodes = y_test_nodes_list[0]

        self.mu_configs_np = np.stack(self.mu_configs, axis=0)
        self.mean_mu_configs = np.nanmean(self.mu_configs_np, axis=1)
        self.opt_config = int(np.argmax(self.mean_mu_configs))

        pred_co = np.concatenate(self.pred_configs[0][0], axis=0)
        self.idxs = [np.random.choice(len(pred_co), len(pred_co) // 2, replace=False) for _ in range(self.oct_params.n_permutations)]

        self.p_values = np.ones(num_configs)
        self.is_equal = np.ones(num_configs, dtype=bool)
        self.mean_mb = np.mean(np.mean(self.mb_size, axis=2), axis=1)

        Parallel(n_jobs=self.oct_params.n_jobs)(
            delayed(self.calculate_pvalues)(c) for c in range(num_configs)
        )

        OCTs_c = self.opt_config
        for c in range(num_configs):
            if self.is_equal[c] and self.mean_mb[c] < self.mean_mb[OCTs_c]:
                OCTs_c = c

        self.logger.debug(f'Best causal configuration is {self.configs_ran[OCTs_c]}')

        self.matrix_mec_graph, self.graph, library_results = self.configs_ran[OCTs_c]['model'].run(
            self.data, self.configs_ran[OCTs_c]
        )
        self.save_progress()

        np.save(os.path.join(self.results_folder, 'matrix_mec_graph.npy'), self.matrix_mec_graph)
        return self.configs_ran[OCTs_c], self.matrix_mec_graph, self.graph, library_results

    def run_new(self) -> Tuple[Dict[str, Any], np.ndarray, Any]:
        """
        Continues the OCT procedure with new configurations.

        Returns
        -------
        opt_config : dict
            The optimal configuration found.
        matrix_mec_graph : np.ndarray
            The MEC graph matrix of the optimal configuration.
        library_results : Any
            Results from the causal discovery library.
        """
        if not self.configs_ran:
            self.logger.error('No previous configurations have been run. Use run() to start the OCT procedure.')
            raise RuntimeError('No previous run!')

        self.oct_params.oos_protocol.protocol.init_protocol(self.data)
        initial_config_count = len(self.configs_ran)
        mec_graphs_ = []
        graphs_ = []
        for algo, algo_configs in self.oct_params.configs.items():
            cd_configs = list(ParameterGrid(algo_configs))
            self.logger.debug(f'Running new configurations for algorithm: {algo}')
            for params in cd_configs:
                params['var_type'] = self.data_type_info['var_type']
                params['name'] = algo

                if is_dict_in_array(params, self.configs_ran):
                    continue

                matrix_mec_graph, matrix_graph, _ = self.oct_params.oos_protocol.protocol.run_protocol(
                    self.data, params['model'], params
                )
                mec_graphs = [mec_graph.to_numpy() for mec_graph in matrix_mec_graph]
                graphs = [graph.to_numpy() for graph in matrix_graph]
                mec_graphs_.append(mec_graphs)
                graphs_.append(graphs)
                self.save_progress()
                self.configs_ran.append(params)

        if not mec_graphs_:
            self.logger.warning('No new configurations to run.')
            return self.configs_ran[self.opt_config], self.matrix_mec_graph, None

        self.mec_graphs_configs.extend(mec_graphs_)
        self.graphs_configs.extend(graphs_)
        num_configs = len(self.configs_ran)
        num_folds = self.oct_params.oos_protocol.protocol.folds_to_run
        self.mb_size = np.zeros((num_configs, num_folds, self.n_nodes))

        config_results = Parallel(n_jobs=self.oct_params.n_jobs)(
            delayed(self.config_parallel)(
                c,
                self.mec_graphs_configs,
                self.oct_params.oos_protocol.protocol.train_indexes,
                self.oct_params.oos_protocol.protocol.test_indexes,
            )
            for c in range(num_configs)
        )

        self.mu_configs, self.mb_configs, self.pred_configs, y_test_nodes_list = zip(*config_results)
        self.y_test_nodes = y_test_nodes_list[0]

        self.mu_configs_np = np.stack(self.mu_configs, axis=0)
        self.mean_mu_configs = np.nanmean(self.mu_configs_np, axis=1)
        self.opt_config = int(np.argmax(self.mean_mu_configs))

        self.p_values = np.concatenate([self.p_values, np.ones(num_configs - initial_config_count)])
        self.is_equal = np.concatenate([self.is_equal, np.ones(num_configs - initial_config_count, dtype=bool)])
        self.mean_mb = np.concatenate([self.mean_mb, np.mean(np.mean(self.mb_size, axis=2), axis=1)])

        Parallel(n_jobs=self.oct_params.n_jobs)(
            delayed(self.calculate_pvalues)(c) for c in range(initial_config_count, num_configs)
        )

        OCTs_c = self.opt_config
        for c in range(num_configs):
            if self.is_equal[c] and self.mean_mb[c] < self.mean_mb[OCTs_c]:
                OCTs_c = c

        self.matrix_mec_graph, library_results = self.configs_ran[OCTs_c]['model'].run(
            self.data, self.configs_ran[OCTs_c]
        )

        self.logger.info(f'Best causal configuration is {self.configs_ran[OCTs_c]}')
        self.save_progress()

        np.save(os.path.join(self.results_folder, 'matrix_mec_graph.npy'), self.matrix_mec_graph)
        return self.configs_ran[OCTs_c], self.matrix_mec_graph, library_results

    def find_best_config(self, algorithms: List[str]) -> Tuple[Dict[str, Any], np.ndarray, Any]:
        """
        Finds the best configuration among specified algorithms.

        Parameters
        ----------
        algorithms : list
            List of algorithm names to consider.

        Returns
        -------
        best_config : dict
            The best configuration among the specified algorithms.
        matrix_mec_graph : np.ndarray
            The MEC graph matrix of the best configuration.
        library_results : Any
            Results from the causal discovery library.

        Raises
        ------
        RuntimeError
            If no configurations have been run for the specified algorithms.
        """
        indices = [i for i, config in enumerate(self.configs_ran) if config['name'] in algorithms]

        if not indices:
            self.logger.error('Cannot find best configuration among algorithms that have not been run yet.')
            raise RuntimeError('Cannot find best configuration among algorithms that have not been run yet.')

        best_index = indices[np.argmax(self.mean_mu_configs[indices])]

        OCTs_c = best_index
        for c in indices:
            if self.is_equal[c] and self.mean_mb[c] < self.mean_mb[OCTs_c]:
                OCTs_c = c

        matrix_mec_graph, library_results = self.configs_ran[OCTs_c]['model'].run(
            self.data, self.configs_ran[OCTs_c]
        )

        self.logger.info(f'Best configuration among specified algorithms is {self.configs_ran[OCTs_c]}')
        return self.configs_ran[OCTs_c], matrix_mec_graph, library_results
