import numpy as np
import pandas as pd

from .DAG import DAGWrapper
from .CPDAG import CPDAGWrapper
from .PAG import PAGWrapper
from .MAG import MAGWrapper

def matrix_to_pywhy_graph(matrix, graph_type='DAG'):
    """
    Convert a matrix representation to a pywhy-graphs graph.

    Parameters
    ----------
    matrix : numpy.ndarray or pandas.DataFrame
        The matrix representation of the graph.
    graph_type : str, optional
        The type of graph to be created. Default is 'DAG'.
        Supported types are 'DAG', 'CPDAG', 'PAG', and 'MAG'.

    Returns
    -------
    GraphWrapperBase
        The pywhy-graphs graph object.
    """
    if graph_type == 'DAG':
        graph = DAGWrapper()
    elif graph_type == 'CPDAG':
        graph = CPDAGWrapper()
    elif graph_type == 'PAG':
        graph = PAGWrapper()
    elif graph_type == 'MAG':
        graph = MAGWrapper()
    else:
        raise ValueError("Unsupported graph type")

    if isinstance(matrix, pd.DataFrame):
        matrix = matrix.to_numpy()
    n = matrix.shape[0]
    for i in range(n):
        graph.add_node(str(i))
        for j in range(n):
            if matrix[i, j] == 2 and matrix[j, i] == 3:
                graph.add_directed_edge(str(i), str(j))
            elif matrix[i, j] == 1 and matrix[j, i] == 1:
                if graph_type == 'PAG':
                    graph.add_circle_edge(str(i), str(j))
                    graph.add_circle_edge(str(j), str(i))
                else:
                    raise TypeError(
                        f"Unsupported edge type detected in the provided matrix for the graph type '{graph_type}'. The matrix contains edges that are not compatible with the specified graph type.")
            elif matrix[i, j] == 2 and matrix[j, i] == 1:
                if graph_type == 'PAG':
                    graph.add_directed_edge(str(i), str(j))
                    graph.add_circle_edge(str(j), str(i))
                else:
                    raise TypeError(
                        f"Unsupported edge type detected in the provided matrix for the graph type '{graph_type}'. The matrix contains edges that are not compatible with the specified graph type.")
            elif matrix[i, j] == 2 and matrix[j, i] == 2:
                if graph_type == 'MAG' or graph_type == 'PAG':
                    graph.add_bidirected_edge(str(i), str(j))
                else:
                    raise TypeError(
                        f"Unsupported edge type detected in the provided matrix for the graph type '{graph_type}'. The matrix contains edges that are not compatible with the specified graph type.")
            elif matrix[i, j] == 3 and matrix[j, i] == 3 and graph_type == 'CPDAG':
                if graph_type == 'CPDAG':
                    graph.add_undirected_edge(str(i), str(j))
                else:
                    raise TypeError(
                        f"Unsupported edge type detected in the provided matrix for the graph type '{graph_type}'. The matrix contains edges that are not compatible with the specified graph type.")
            elif(matrix[i, j] == 0 and matrix[j, i] != 0) or (matrix[j, i] == 0 and matrix[i, j] != 0):
                raise TypeError(
                        f"Unsupported edge type detected in the provided matrix for the graph type '{graph_type}'. The matrix contains edges that are not compatible with the specified graph type.")
    return graph

def pywhy_graph_to_matrix(graph):
    """
    Convert a pywhy-graphs graph to a matrix representation.

    Parameters
    ----------
    graph : GraphWrapperBase
        The pywhy-graphs graph object.

    Returns
    -------
    numpy.ndarray
        The matrix representation of the graph.
    """
    nodes = list(graph.get_nodes())
    node_indices = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    matrix = np.zeros((n, n), dtype=int)
    edges = graph.get_edges()
    if isinstance(graph, DAGWrapper):
        for edge in edges:
            source, target = edge
            matrix[node_indices[source], node_indices[target]] = 2
            matrix[node_indices[target], node_indices[source]] = 3
    else:
        for edge_type in edges:
            for edge in edges[edge_type]:
                source, target = edge
                # Assign values based on edge types
                if edge_type == 'directed':
                    matrix[node_indices[source], node_indices[target]] = 2
                    matrix[node_indices[target], node_indices[source]] = 3
                elif edge_type == 'circle':
                    matrix[node_indices[source], node_indices[target]] = 1
                elif edge_type == 'bidirected':
                    matrix[node_indices[source], node_indices[target]] = 2
                    matrix[node_indices[target], node_indices[source]] = 2
                elif edge_type == 'undirected':
                    matrix[node_indices[source], node_indices[target]] = 3
                    matrix[node_indices[target], node_indices[source]] = 3
                else:
                    raise TypeError(
                        f"Unsupported edge type.")

    return matrix

'''
import numpy as np

# Define the matrix
matrix = np.array([[0, 2, 1],
                   [3, 0, 2],
                   [2, 3, 0]])

# Convert the matrix to a PAG
pag = matrix_to_pywhy_graph(matrix, graph_type='PAG')
print(pag.get_edges())
# Convert the PAG back to a matrix
converted_matrix = pywhy_graph_to_matrix(pag)

# Print the converted matrix
print(converted_matrix)

'''
