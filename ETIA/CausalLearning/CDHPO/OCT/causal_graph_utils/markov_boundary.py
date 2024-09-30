import numpy as np
import pandas as pd


def bidirected_path(i, matrix):
    '''
    Recursive function to find the nodes that are reachable in any bidirected path starting from the node i
    Author : kbiza@csd.uoc.gr
    Args:
        i (int): the starting node (not a list of integer!!)
        matrix (numpy array): matrix of size N*N where N is the number of nodes in tetrad_graph
                matrix(i, j) = 2 and matrix(j, i) = 3: i-->j
                matrix(i, j) = 1 and matrix(j, i) = 1: io-oj
                matrix(i, j) = 2 and matrix(j, i) = 2: i<->j
                matrix(i, j) = 3 and matrix(j, i) = 3: i---j
                matrix(i, j) = 2 and matrix(j, i) = 1: io->j

    Returns:
        list_nodes (list): the nodes that are reachable in any bidirected path starting from node i
    '''

    bidirected_neighbors = np.where(np.logical_and(matrix[i, :] == 2, matrix[:, i] == 2))[0]
    bidirected_neighbors = bidirected_neighbors.tolist()

    if len(bidirected_neighbors) == 0:
        return

    list_nodes = []
    list_nodes = list_nodes+bidirected_neighbors

    matrix[i, :] = 0
    matrix[:, i] = 0

    for j in bidirected_neighbors:
        next_neighbors = bidirected_path(j, matrix)
        if next_neighbors:
            list_nodes = list_nodes+next_neighbors

    return list_nodes



def markov_boundary(target, matrix_pd):

    '''
    Identify the markov boundary of the target node.
    Function for DAGs, PDAGs and MAGs  (not for PAGs)
    Author:kbiza@csd.uoc.gr
    Args:
        target (int): index of the target node in the matrix (not a list of int!!)
        matrix_pd (pandas Dataframe): an array of size N*N where N is the number of nodes in tetrad_graph
                matrix(i, j) = 2 and matrix(j, i) = 3: i-->j
                matrix(i, j) = 1 and matrix(j, i) = 1: io-oj    in PAGs
                matrix(i, j) = 2 and matrix(j, i) = 2: i<->j    in MAGs and PAGs
                matrix(i, j) = 3 and matrix(j, i) = 3: i---j    in PDAGs
                matrix(i, j) = 2 and matrix(j, i) = 1: io->j    in PAGs

    Returns:
        markov_boundary (list) : list of indexes for the markov boundary ot the target

    '''
    if matrix_pd is pd.DataFrame:
        matrix = matrix_pd.to_numpy()
    else:
        matrix = matrix_pd
    # check if the input matrix is PAG
    if np.where(matrix == 1)[0].size > 0:
        raise ValueError('cannot find MB due to an undirected edge (need DAG or MAG)')

    # Common for PDAGs and MAGs
    parents = np.where(np.logical_and(matrix[target, :] == 3, matrix[:, target] == 2))[0].tolist()
    children = np.where(np.logical_and(matrix[target, :] == 2, matrix[:, target] == 3))[0].tolist()
    parents_of_children = []
    for child in children:
        parents_of_children += np.where(np.logical_and(matrix[child, :] == 3, matrix[:, child] == 2))[0].tolist()

    # Undirected adjacencies (in PDAGs)
    other_adjacencies = np.where(np.logical_and(matrix[target, :] == 3, matrix[:, target] == 3))[0].tolist()

    # District sets in MAGs
    district_i = bidirected_path(target, np.copy(matrix))
    district_children = []
    for child in children:
        district_child = bidirected_path(child, np.copy(matrix))
        if district_child:
            district_children += district_child

    parents_of_district_i = []
    if district_i:
        for di in district_i:
            parents_of_district_i += np.where(np.logical_and(matrix[di, :] == 3, matrix[:, di] == 2))[0].tolist()
    else:
        district_i =[] # bidirected_path returns none (fix if needed)

    parents_of_district_children = []
    if district_children:
        for dchild in district_children:
            parents_of_district_children += np.where(np.logical_and(matrix[dchild, :] == 3, matrix[:, dchild] == 2))[0].tolist()


    markov_boundary = parents + children + parents_of_children + \
                      other_adjacencies +\
                      district_i + district_children + \
                      parents_of_district_i + parents_of_district_children


    markov_boundary = list(set(markov_boundary))
    if target in markov_boundary:
        markov_boundary.remove(target)

    return markov_boundary
