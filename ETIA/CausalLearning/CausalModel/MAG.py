from pywhy_graphs import ADMG  # Assuming ADMG is imported from pywhy_graphs
from .GraphWrapperBase import GraphWrapperBase

class MAGWrapper(GraphWrapperBase):
    def __init__(self, incoming_directed_edges=None, incoming_bidirected_edges=None):
        """
        Initialize a Maximal Ancestral Graph (MAG) wrapper using the ADMG class.

        Parameters
        ----------
        incoming_directed_edges : list, optional
            List of tuples representing directed edges.
        incoming_bidirected_edges : list, optional
            List of tuples representing bidirected edges.
        """
        self.mag = ADMG(incoming_directed_edges=incoming_directed_edges,
                        incoming_bidirected_edges=incoming_bidirected_edges)

    def add_node(self, node):
        """
        Add a node to the MAG.

        Parameters
        ----------
        node : hashable object
            The node to be added.
        """
        self.mag.add_node(node)

    def remove_node(self, node):
        """
        Remove a node from the MAG.

        Parameters
        ----------
        node : hashable object
            The node to be removed.
        """
        self.mag.remove_node(node)

    def add_directed_edge(self, source, target):
        """
        Add a directed edge to the MAG.

        Parameters
        ----------
        source : hashable object
            The source node of the directed edge.
        target : hashable object
            The target node of the directed edge.
        """
        self.mag.add_edge(source, target, edge_type=self.mag.directed_edge_name)

    def add_bidirected_edge(self, source, target):
        """
        Add a bidirected edge to the MAG.

        Parameters
        ----------
        source : hashable object
            The source node of the bidirected edge.
        target : hashable object
            The target node of the bidirected edge.
        """
        self.mag.add_edge(source, target, edge_type=self.mag.bidirected_edge_name)

    def remove_edge(self, source, target):
        """
        Remove an edge from the MAG.

        Parameters
        ----------
        source : hashable object
            The source node of the edge to be removed.
        target : hashable object
            The target node of the edge to be removed.
        """
        self.mag.remove_edge(source, target)

    def get_nodes(self):
        """
        Return the nodes of the MAG.

        Returns
        -------
        list
            List of nodes in the MAG.
        """
        return self.mag.nodes

    def get_edges(self):
        """
        Return the edges of the MAG.

        Returns
        -------
        dict
            Dictionary containing directed and bidirected edges of the MAG.
        """
        return {
            "directed": self.mag.directed_edges,
            "bidirected": self.mag.bidirected_edges
        }

    # Additional MAG-specific methods can be added as needed
