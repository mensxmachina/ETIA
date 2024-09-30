from pywhy_graphs import CPDAG
from .GraphWrapperBase import GraphWrapperBase

class CPDAGWrapper(GraphWrapperBase):
    def __init__(self,
                 incoming_directed_edges=None,
                 incoming_undirected_edges=None):
        """
        Initialize a Completed Partially Directed Acyclic Graph (CPDAG) wrapper.

        Parameters
        ----------
        incoming_directed_edges : list, optional
            List of tuples representing directed edges.
        incoming_undirected_edges : list, optional
            List of tuples representing undirected edges.
        """
        self.cpdag = CPDAG(incoming_directed_edges=incoming_directed_edges,
                           incoming_undirected_edges=incoming_undirected_edges)

    def add_node(self, node):
        """
        Add a node to the CPDAG.

        Parameters
        ----------
        node : hashable object
            The node to be added.
        """
        self.cpdag.add_node(node)

    def remove_node(self, node):
        """
        Remove a node from the CPDAG.

        Parameters
        ----------
        node : hashable object
            The node to be removed.
        """
        self.cpdag.remove_node(node)

    def add_directed_edge(self, source, target):
        """
        Add a directed edge to the CPDAG.

        Parameters
        ----------
        source : hashable object
            The source node of the directed edge.
        target : hashable object
            The target node of the directed edge.
        """
        self.cpdag.add_edge(source, target, edge_type=self.cpdag.directed_edge_name)

    def add_undirected_edge(self, source, target):
        """
        Add an undirected edge to the CPDAG.

        Parameters
        ----------
        source : hashable object
            The source node of the undirected edge.
        target : hashable object
            The target node of the undirected edge.
        """
        self.cpdag.add_edge(source, target, edge_type=self.cpdag.undirected_edge_name)

    def remove_edge(self, source, target):
        """
        Remove an edge from the CPDAG.

        Parameters
        ----------
        source : hashable object
            The source node of the edge to be removed.
        target : hashable object
            The target node of the edge to be removed.
        """
        self.cpdag.remove_edge(source, target)

    def get_nodes(self):
        """
        Return the nodes of the CPDAG.

        Returns
        -------
        list
            List of nodes in the CPDAG.
        """
        return self.cpdag.nodes

    def get_edges(self):
        """
        Return the edges of the CPDAG.

        Returns
        -------
        dict
            Dictionary containing directed and undirected edges of the CPDAG.
        """
        return {
            "directed": self.cpdag.directed_edges,
            "undirected": self.cpdag.undirected_edges
        }

    def possible_children(self, node):
        """
        Return an iterator over possible children of a node.

        Parameters
        ----------
        node : hashable object
            The node whose possible children are to be retrieved.

        Returns
        -------
        iterator
            Iterator over possible children of the given node.
        """
        return self.cpdag.possible_children(node)

    def possible_parents(self, node):
        """
        Return an iterator over possible parents of a node.

        Parameters
        ----------
        node : hashable object
            The node whose possible parents are to be retrieved.

        Returns
        -------
        iterator
            Iterator over possible parents of the given node.
        """
        return self.cpdag.possible_parents(node)

    def orient_uncertain_edge(self, u, v):
        """
        Orient an undirected edge into a directed edge.

        Parameters
        ----------
        u : hashable object
            One endpoint of the undirected edge.
        v : hashable object
            The other endpoint of the undirected edge.
        """
        self.cpdag.orient_uncertain_edge(u, v)
