class GraphWrapperBase:
    def __init__(self):
        """
        Base class for graph wrappers. This class cannot be instantiated directly.
        """
        raise NotImplementedError("This is a base class and cannot be instantiated directly.")

    def add_node(self, node):
        """
        Add a node to the graph.

        Parameters
        ----------
        node : hashable object
            The node to be added.
        """
        raise NotImplementedError

    def remove_node(self, node):
        """
        Remove a node from the graph.

        Parameters
        ----------
        node : hashable object
            The node to be removed.
        """
        raise NotImplementedError

    def add_edge(self, source, target):
        """
        Add an edge to the graph.

        Parameters
        ----------
        source : hashable object
            The source node of the edge.
        target : hashable object
            The target node of the edge.
        """
        raise NotImplementedError

    def remove_edge(self, source, target):
        """
        Remove an edge from the graph.

        Parameters
        ----------
        source : hashable object
            The source node of the edge to be removed.
        target : hashable object
            The target node of the edge to be removed.
        """
        raise NotImplementedError

    def get_nodes(self):
        """
        Return the nodes of the graph.

        Returns
        -------
        list
            List of nodes in the graph.
        """
        raise NotImplementedError

    def get_edges(self):
        """
        Return the edges of the graph.

        Returns
        -------
        list
            List of edges in the graph.
        """
        raise NotImplementedError

    # You can also include other common methods that might be relevant across different graph types.
