from .GraphWrapperBase import GraphWrapperBase
from pywhy_graphs import PAG

class PAGWrapper(GraphWrapperBase):
    def __init__(self,
                 incoming_directed_edges=None,
                 incoming_undirected_edges=None,
                 incoming_bidirected_edges=None,
                 incoming_circle_edges=None):
        """
        Initialize a Partial Ancestral Graph (PAG) wrapper.

        Parameters
        ----------
        incoming_directed_edges : list, optional
            List of tuples representing directed edges.
        incoming_undirected_edges : list, optional
            List of tuples representing undirected edges.
        incoming_bidirected_edges : list, optional
            List of tuples representing bidirected edges.
        incoming_circle_edges : list, optional
            List of tuples representing circle edges.
        """
        self.pag = PAG(incoming_directed_edges=incoming_directed_edges,
                       incoming_undirected_edges=incoming_undirected_edges,
                       incoming_bidirected_edges=incoming_bidirected_edges,
                       incoming_circle_edges=incoming_circle_edges)

    def add_node(self, node):
        """
        Add a node to the PAG.

        Parameters
        ----------
        node : hashable object
            The node to be added.
        """
        self.pag.add_node(node)

    def remove_node(self, node):
        """
        Remove a node from the PAG.

        Parameters
        ----------
        node : hashable object
            The node to be removed.
        """
        self.pag.remove_node(node)

    def add_directed_edge(self, source, target):
        """
        Add a directed edge to the PAG.

        Parameters
        ----------
        source : hashable object
            The source node of the directed edge.
        target : hashable object
            The target node of the directed edge.
        """
        self.pag.add_edge(source, target, edge_type=self.pag.directed_edge_name)

    def add_bidirected_edge(self, source, target):
        """
        Add a bidirected edge to the PAG.

        Parameters
        ----------
        source : hashable object
            The source node of the bidirected edge.
        target : hashable object
            The target node of the bidirected edge.
        """
        self.pag.add_edge(source, target, edge_type=self.pag.bidirected_edge_name)

    def add_undirected_edge(self, source, target):
        """
        Add an undirected edge to the PAG.

        Parameters
        ----------
        source : hashable object
            The source node of the undirected edge.
        target : hashable object
            The target node of the undirected edge.
        """
        self.pag.add_edge(source, target, edge_type=self.pag.undirected_edge_name)

    def add_circle_edge(self, source, target):
        """
        Add a circle edge (circle edge on one side) to the PAG.

        Parameters
        ----------
        source : hashable object
            One side of the circle edge.
        target : hashable object
            The other side of the circle edge.
        """
        self.pag.add_edge(source, target, edge_type=self.pag.circle_edge_name)

    def remove_edge(self, source, target):
        """
        Remove an edge from the PAG.

        Parameters
        ----------
        source : hashable object
            The source node of the edge to be removed.
        target : hashable object
            The target node of the edge to be removed.
        """
        self.pag.remove_edge(source, target)

    def get_nodes(self):
        """
        Return the nodes of the PAG.

        Returns
        -------
        list
            List of nodes in the PAG.
        """
        return self.pag.nodes

    def get_edges(self):
        """
        Return the edges of the PAG.

        Returns
        -------
        dict
            Dictionary containing directed, bidirected, undirected, and circle edges of the PAG.
        """
        return {
            "directed": self.pag.directed_edges,
            "bidirected": self.pag.bidirected_edges,
            "undirected": self.pag.undirected_edges,
            "circle": self.pag.circle_edges
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
        return self.pag.possible_children(node)

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
        return self.pag.possible_parents(node)

    def children(self, node):
        """
        Return the definite children of a node.

        Parameters
        ----------
        node : hashable object
            The node whose definite children are to be retrieved.

        Returns
        -------
        list
            List of definite children of the given node.
        """
        return self.pag.children(node)

    def parents(self, node):
        """
        Return the definite parents of a node.

        Parameters
        ----------
        node : hashable object
            The node whose definite parents are to be retrieved.

        Returns
        -------
        list
            List of definite parents of the given node.
        """
        return self.pag.parents(node)
