import numpy as np
import pandas as pd
from sklearn import metrics

# Functions for computing edge/path consistency and similarity using bootstrapped graphs
# Author: kbiza@csd.uoc.gr



def is_consistent_edge(m1_ij, m1_ji, m2_ij, m2_ji):
    '''
    Checks if two edges are consistent
    Args:
        m1_ij(int):  notation of matrix1[i,j]
        m1_ji(int):  notation of matrix1[j,i]
        m2_ij(int):  notation of matrix2[i,j]
        m2_ji(int):  notation of matrix2[j,i]

    Returns:
        is consistent(bool) : True or False
    '''

    # identical edges (or identical absence of edge)
    if m1_ij == m2_ij and m1_ji == m2_ji:
        is_consistent = True

    # consistent edges
    else:
        # i o-o j  is consistent with  io->j, i<->j, i-->j,  i<--j, i<-oj
        if m1_ij == 1 and m1_ji == 1 and m2_ij != 0 and m2_ji != 0:
            is_consistent = True

        # i o-> j  is consistent with  i<->j, i-->j, i o-o j
        elif m1_ij == 2 and m1_ji == 1:
            if m2_ij == 2 and m2_ji == 2:
                is_consistent = True
            elif m2_ij == 2 and m2_ji == 3:
                is_consistent = True
            elif m2_ij == 1 and m2_ji == 1:
                is_consistent = True
            else:
                is_consistent = False

        # i <-o j  is consistent with  i<->j, i<--j, i o-o j
        elif m1_ij == 1 and m1_ji == 2:
            if m2_ij == 2 and m2_ji == 2:
                is_consistent = True
            elif m2_ij == 3 and m2_ji == 2:
                is_consistent = True
            elif m2_ij == 1 and m2_ji == 1:
                is_consistent = True
            else:
                is_consistent=False

        # i --> j is consistent with  io->j, i o-o j
        elif m1_ij == 2 and m1_ji == 3:
            if m2_ij == 2 and m2_ji == 1:
                is_consistent = True
            elif m2_ij == 1 and m2_ji == 1:
                is_consistent = True
            else:
                is_consistent = False

        # i <-- j  is consistent with  i<-oj, i o-o j
        elif m1_ij == 3 and m1_ji == 2:
            if m2_ij == 1 and m2_ji == 2:
                is_consistent = True
            elif m2_ij == 1 and m2_ji == 1:
                is_consistent = True
            else:
                is_consistent = False

        # i <-> j  is consistent with  io-oj  io->j, i<-oj
        elif m1_ij == 2 and m1_ji == 2:
            if m2_ij == 1 and m2_ji == 1:
                is_consistent = True
            elif m2_ij == 2 and m2_ji == 1:
                is_consistent = True
            elif m2_ij == 1 and m2_ji == 2:
                is_consistent = True
            else:
                is_consistent = False

        # no edge in m1, edge in m2
        elif m1_ij == 0 and m2_ij != 0:
            is_consistent = False

        # edge in m1, no edge in m2
        elif m1_ij != 0 and m2_ij == 0 :
            is_consistent = False

        # no edge in m1, no edge in m2
        elif m1_ij == 0 and m2_ij == 0:
            is_consistent = True

        else:
            print("problem with notation")
            is_consistent = False

    return is_consistent


def compute_edge_weights(best_mec_matrix, bootstrapped_mec_matrices, all_edges=True, true_graph=None):

    '''
    Compute edge consistency and edge frequency for each edge
    Parameters
    ----------
        best_mec_matrix
        bootstrapped_mec_matrices(list):
        all_edges(bool): if True it checks all possible edges n(n-1)/2  and evaluates missing edges
                         if False it evaluates only the edges that appear in best_mec_matrix
        true_graph

    Returns
    -------

    '''

    n_bootstraps = len(bootstrapped_mec_matrices)

    row_names = best_mec_matrix.columns.to_list()
    column_names = best_mec_matrix.columns.to_list()

    n_nodes = best_mec_matrix.shape[0]
    if all_edges:
        n_edges = int(n_nodes * (n_nodes-1) / 2)
    else:
        n_edges = int(np.count_nonzero(best_mec_matrix) / 2)

    weight_data = np.empty((n_edges, 4), dtype='object')
    label_data = np.empty((n_edges, 4), dtype='object')

    c = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):

            if not all_edges and best_mec_matrix.iloc[i, j] == 0:
                continue

            edge_consistency = 0
            edge_discovery = 0
            for nb in range(n_bootstraps):

                # consistent edges
                if is_consistent_edge(best_mec_matrix.iloc[i, j], best_mec_matrix.iloc[j, i],
                                      bootstrapped_mec_matrices[nb].iloc[i, j],
                                      bootstrapped_mec_matrices[nb].iloc[j, i]):
                    edge_consistency += 1

                # same edges
                if bootstrapped_mec_matrices[nb].iloc[i, j] == best_mec_matrix.iloc[i, j] and \
                        bootstrapped_mec_matrices[nb].iloc[j, i] == best_mec_matrix.iloc[j, i]:
                    edge_discovery += 1

            weight_data[c] = [row_names[i], column_names[j],
                              edge_consistency / n_bootstraps,
                             edge_discovery / n_bootstraps]


            # Compare estimated graph with true
            if isinstance(true_graph, pd.DataFrame):
                if is_consistent_edge(best_mec_matrix.iloc[i, j], best_mec_matrix.iloc[j, i],
                                      true_graph.iloc[i, j], true_graph.iloc[j, i]):
                    label_consistency = 1
                else:
                    label_consistency = 0

                if true_graph.iloc[i, j] == best_mec_matrix.iloc[i, j] and \
                        true_graph.iloc[j, i] == best_mec_matrix.iloc[j, i]:
                    label_discovery = 1
                else:
                    label_discovery = 0

                label_data[c] = [row_names[i], column_names[j], label_consistency, label_discovery]

            c += 1

    weight_data_pd = pd.DataFrame(data=weight_data,
                                  columns=['source', 'target', 'edge_consistency', 'edge_discovery'])

    label_data_pd = pd.DataFrame(data=label_data,
                                  columns=['source', 'target', 'edge_consistency', 'edge_discovery'])


    return weight_data_pd, label_data_pd



def compute_path_weight(bootstrapped_mec_matrices, best_mec_matrix, path):

    n_bootstraps = len(bootstrapped_mec_matrices)
    path_consistency = 0
    path_discovery = 0

    for nb in range(n_bootstraps):

        is_consistent_path = True
        is_similar_path = True

        for i in range(len(path) - 1):
            node_i = path[i]
            node_j = path[i + 1]

            # if at least one edge is not consistent -- > the path is not consistent nor similar
            if not is_consistent_edge(best_mec_matrix.loc[node_i, node_j], best_mec_matrix.loc[node_j, node_i],
                                  bootstrapped_mec_matrices[nb].loc[node_i, node_j],
                                  bootstrapped_mec_matrices[nb].loc[node_j, node_i]):

                is_consistent_path = False
                is_similar_path = False
                break

            else:
                # if at least one edge is consistent but not similar --> the path is not similar
                if (bootstrapped_mec_matrices[nb].loc[node_i, node_j] != best_mec_matrix.loc[node_i, node_j] or
                        bootstrapped_mec_matrices[nb].loc[node_j, node_i] != best_mec_matrix.loc[node_j, node_i]):
                    is_similar_path = False
                    break


        if is_consistent_path:
            path_consistency += 1

        if is_similar_path:
            path_discovery += 1

    return path_consistency / n_bootstraps, path_discovery / n_bootstraps


def paths_metrics(best_mec_matrix, bootstrapped_mec_matrices, paths):

    '''
    Compute path consistency and path discovery for each path
    Parameters
    ----------
        paths(dictionary): dictionary with lists of paths
        bootstrapped_graphs(list): bootstrapped graphs
        opt_graph(pandas Dataframe): adjacency matrix of graph

    Returns
    -------
        path_consistency(dictionary): consistency values based on the input paths dictionary
        path_discovery(dictionary) : discovery values based on the input paths dictionary
    '''


    # Compute path metrics
    path_consistency = {}
    path_discovery = {}
    for key_path in paths.keys():
        print(key_path)
        consistency_ = np.zeros(len(paths[key_path]))
        discovery_ = np.zeros(len(paths[key_path]))
        for i, path in enumerate(paths[key_path]):
            consistency_[i], discovery_[i] = (
                compute_path_weight(bootstrapped_mec_matrices, best_mec_matrix, path))

        path_consistency[key_path] = consistency_
        path_discovery[key_path] = discovery_

    return path_consistency, path_discovery