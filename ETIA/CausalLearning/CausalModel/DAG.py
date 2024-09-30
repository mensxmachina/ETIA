import networkx as nx
from .GraphWrapperBase import GraphWrapperBase

class DAGWrapper(GraphWrapperBase):
    def __init__(self):
        """
        Initialize a Directed Acyclic Graph (DAG) wrapper.
        """
        self.dag = nx.DiGraph()

    def add_node(self, node):
        """
        Add a node to the DAG.

        Parameters
        ----------
        node : hashable object
            The node to be added.
        """
        self.dag.add_node(node)

    def remove_node(self, node):
        """
        Remove a node from the DAG.

        Parameters
        ----------
        node : hashable object
            The node to be removed.
        """
        self.dag.remove_node(node)

    def add_directed_edge(self, source, target):
        """
        Add a directed edge to the DAG.
        Checks for cycles and raises an error if an edge creates a cycle.

        Parameters
        ----------
        source : hashable object
            The source node of the directed edge.
        target : hashable object
            The target node of the directed edge.
        """
        self.dag.add_edge(source, target, edge_type='directed')
        if not nx.is_directed_acyclic_graph(self.dag):
            self.dag.remove_edge(source, target)
            raise ValueError("Adding the edge ({}, {}) would create a cycle.".format(source, target))

    def remove_edge(self, source, target):
        """
        Remove an edge from the DAG.

        Parameters
        ----------
        source : hashable object
            The source node of the edge to be removed.
        target : hashable object
            The target node of the edge to be removed.
        """
        self.dag.remove_edge(source, target)

    def get_nodes(self):
        """
        Return the nodes of the DAG.

        Returns
        -------
        list
            List of nodes in the DAG.
        """
        return self.dag.nodes

    def get_edges(self):
        """
        Return the edges of the DAG.

        Returns
        -------
        list
            List of edges in the DAG.
        """
        return self.dag.edges


    # Additional DAG-specific methods can be added as needed
