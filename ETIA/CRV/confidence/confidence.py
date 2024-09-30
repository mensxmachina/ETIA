from joblib import Parallel, delayed
import numpy as np
from ...CausalLearning.CausalModel.utils import matrix_to_pywhy_graph, pywhy_graph_to_matrix
from sklearn.utils import resample

def is_consistent_edge_L(m1_ij, m1_ji, m2_ij, m2_ji):
    """
    Check if two edges are consistent based on their types.

    Parameters
    ----------
    m1_ij : int
        Type of edge from node i to node j in the first matrix.
    m1_ji : int
        Type of edge from node j to node i in the first matrix.
    m2_ij : int
        Type of edge from node i to node j in the second matrix.
    m2_ji : int
        Type of edge from node j to node i in the second matrix.

    Returns
    -------
    bool
        True if the edges are consistent, False otherwise.
    """
    if m1_ij == m2_ij and m1_ji == m2_ji:
        return True
    else:
        if m1_ij == 1 and m1_ji == 1 and m2_ij != 0 and m2_ji != 0:
            return True
        elif m1_ij == 2 and m1_ji == 1:
            return m2_ij == 2 and m2_ji == 2 or m2_ij == 2 and m2_ji == 3 or m2_ij == 1 and m2_ji == 1
        elif m1_ij == 1 and m1_ji == 2:
            return m2_ij == 2 and m2_ji == 2 or m2_ij == 3 and m2_ji == 2 or m2_ij == 1 and m2_ji == 1
        elif m1_ij == 2 and m1_ji == 3:
            return m2_ij == 2 and m2_ji == 1 or m2_ij == 1 and m2_ji == 1
        elif m1_ij == 3 and m1_ji == 2:
            return m2_ij == 1 and m2_ji == 2 or m2_ij == 1 and m2_ji == 1
        elif m1_ij == 2 and m1_ji == 2:
            return m2_ij == 1 and m2_ji == 1 or m2_ij == 2 and m2_ji == 1 or m2_ij == 1 and m2_ji == 2
        elif m1_ij == 0 and m2_ij != 0:
            return False
        elif m1_ij != 0 and m2_ij == 0:
            return False
        else:
            return False

def bootstrapping_causal_graph_parallel(input_data, config, tiers, is_cat_var):
    """
    Perform bootstrapping of causal graphs in parallel.

    Parameters
    ----------
    input_data : numpy.ndarray
        The input data.
    config : dict
        The configuration for the causal model.
    tiers : list
        Tiers for variable selection.
    is_cat_var : list of bool
        Boolean array indicating if the variable is categorical.

    Returns
    -------
    list
        Bootstrapped samples, matrix graphs, and matrix MEC graphs.
    """
    bootstrapped_ = resample(input_data, replace=True)
    matrix_mec_graph, matrix_graph, var_map = config['model'].run(input_data, config)
    matrix_mec_graph = pywhy_graph_to_matrix(matrix_mec_graph)
    matrix_graph = pywhy_graph_to_matrix(matrix_graph)
    return [bootstrapped_, matrix_graph, matrix_mec_graph]

def bootstrapping_causal_graph(n_bootstraps, input_data, tiers, best_config, is_cat_var):
    """
    Perform bootstrapping of causal graphs.

    Parameters
    ----------
    n_bootstraps : int
        Number of bootstrap repetitions.
    input_data : numpy.ndarray
        The input data.
    tiers : list
        Tiers for variable selection.
    best_config : dict
        The best causal configuration to estimate the bootstrapped graphs.
    is_cat_var : list of bool
        Boolean array indicating if the variable is categorical.

    Returns
    -------
    list
        Bootstrapped MEC matrix and bootstrapped graph matrix.
    """
    bootstrapped_samples = []
    bootstrapped_matrix = []

    results = Parallel(n_jobs=8)(
        delayed(bootstrapping_causal_graph_parallel)(input_data, best_config, tiers, is_cat_var) for nb in range(n_bootstraps))
    results = np.array(results)
    bootstrapped_samples = results[:, 0]
    bootstrapped_matrix = results[:, 1][0]
    bootstrapped_mec_matrix = results[:, 2]

    return [bootstrapped_mec_matrix, bootstrapped_matrix]

def edge_metrics_on_bootstraps(best_mec_matrix, isPAG, bootstrapped_mec_matrix):
    """
    Calculate edge consistency and similarity based on bootstrapped MEC matrices.

    Parameters
    ----------
    best_mec_matrix : numpy.ndarray
        The best MEC matrix.
    isPAG : bool
        True if the matrix is a PAG, False otherwise.
    bootstrapped_mec_matrix : list of numpy.ndarray
        Bootstrapped MEC matrices.

    Returns
    -------
    tuple
        Edge consistency and edge similarity.
    """
    best_mec_matrix = pywhy_graph_to_matrix(best_mec_matrix)
    n_bootstraps = len(bootstrapped_mec_matrix)
    n_nodes = best_mec_matrix.shape[0]
    n_edges = int(np.count_nonzero(best_mec_matrix) / 2)
    consistency_count = np.zeros((n_edges, 1), dtype=int)
    similarity_count = np.zeros((n_edges, 1), dtype=int)

    c = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if best_mec_matrix[i, j] != 0:
                for nb in range(n_bootstraps):
                    if is_consistent_edge_L(best_mec_matrix[i, j], best_mec_matrix[j, i],
                                             bootstrapped_mec_matrix[nb][i, j], bootstrapped_mec_matrix[nb][j, i]):
                        consistency_count[c] += 1
                    if bootstrapped_mec_matrix[nb][i, j] == best_mec_matrix[i, j] and \
                            bootstrapped_mec_matrix[nb][j, i] == best_mec_matrix[j, i]:
                        similarity_count[c] += 1

                c += 1

    edge_consistency = consistency_count / n_bootstraps
    edge_similarity = similarity_count / n_bootstraps

    return edge_consistency, edge_similarity

def calculate_confidence(dataset, opt_conf, n_bootstraps=50):
    """
    Calculate edge consistency and similarity confidence.

    Parameters
    ----------
    dataset : object
        The dataset.
    opt_conf : dict
        The optimal configuration.
    n_bootstraps : int, optional
        Number of bootstrap repetitions. Default is 50.

    Returns
    -------
    tuple
        Edge consistency and edge similarity.
    """
    if opt_conf is None:
        raise RuntimeError("You need to have an optimal configuration before you can calculate the edge confidences")

    bootstrapped_mec_matrix, bootstrapped_graph_matrix = bootstrapping_causal_graph(n_bootstraps=n_bootstraps,
                                                                                     input_data=dataset.get_dataset(), tiers=None,
                                                                                     best_config=opt_conf,
                                                                                     is_cat_var=opt_conf.data_type_info['var_type'])
    np.save('bootstraped_graphs', bootstrapped_mec_matrix)

    edge_consistency, edge_similarity = edge_metrics_on_bootstraps(
        best_mec_matrix=opt_conf.matrix_mec_graph, isPAG=True, bootstrapped_mec_matrix=bootstrapped_mec_matrix)

    return edge_consistency, edge_similarity
