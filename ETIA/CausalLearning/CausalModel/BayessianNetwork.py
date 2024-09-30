from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel

class BayesianNetwork():
    def __init__(self, edges):
        """
        Initialize a Bayesian Network.

        Parameters
        ----------
        edges : list
            List of tuples representing directed edges between nodes.
        """
        self.model = BayesianModel(edges)
        self.infer = VariableElimination(self.model)

    def add_node(self, node):
        """
        Add a node to the Bayesian Network.

        Parameters
        ----------
        node : str
            The name of the node to be added.
        """
        self.model.add_node(node)

    def remove_node(self, node):
        """
        Remove a node from the Bayesian Network.

        Parameters
        ----------
        node : str
            The name of the node to be removed.
        """
        self.model.remove_node(node)

    def add_edge(self, edge):
        """
        Add an edge between two nodes in the Bayesian Network.

        Parameters
        ----------
        edge : tuple
            A tuple representing the directed edge between two nodes.
        """
        self.model.add_edge(*edge)

    def remove_edge(self, edge):
        """
        Remove an edge between two nodes in the Bayesian Network.

        Parameters
        ----------
        edge : tuple
            A tuple representing the directed edge between two nodes.
        """
        self.model.remove_edge(*edge)

    def get_parents(self, node):
        """
        Get the parents of a given node in the Bayesian Network.

        Parameters
        ----------
        node : str
            The name of the node whose parents are to be retrieved.

        Returns
        -------
        list
            List of parent nodes.
        """
        return self.model.predecessors(node)

    def get_children(self, node):
        """
        Get the children of a given node in the Bayesian Network.

        Parameters
        ----------
        node : str
            The name of the node whose children are to be retrieved.

        Returns
        -------
        list
            List of child nodes.
        """
        return self.model.successors(node)

    def get_nodes(self):
        """
        Get all the nodes in the Bayesian Network.

        Returns
        -------
        list
            List of all nodes.
        """
        return self.model.nodes()

    def get_edges(self):
        """
        Get all the edges in the Bayesian Network.

        Returns
        -------
        list
            List of all edges.
        """
        return self.model.edges()

    def get_cpds(self):
        """
        Get all the Conditional Probability Distributions (CPDs) in the Bayesian Network.

        Returns
        -------
        list
            List of CPDs.
        """
        return self.model.get_cpds()

    def get_inference(self):
        """
        Get the VariableElimination object for inference.

        Returns
        -------
        VariableElimination
            Object for performing inference.
        """
        return self.infer

    def set_evidence(self, evidence):
        """
        Set the evidence for inference in the Bayesian Network.

        Parameters
        ----------
        evidence : dict
            Dictionary where keys are node names and values are observed states.
        """
        self.infer.evidence = evidence

    def query(self, nodes, show_progress=False):
        """
        Perform inference and return the marginal probabilities for the given nodes.

        Parameters
        ----------
        nodes : str or list
            Single node or list of nodes for which marginal probabilities are to be computed.
        show_progress : bool, optional
            If True, display a progress bar during inference. Default is False.

        Returns
        -------
        dict
            Dictionary containing marginal probabilities of the queried nodes.
        """
        return self.infer.query(nodes, show_progress=show_progress)

    def map_query(self, show_progress=False):
        """
        Perform inference and return the most probable states of the nodes.

        Parameters
        ----------
        show_progress : bool, optional
            If True, display a progress bar during inference. Default is False.

        Returns
        -------
        dict
            Dictionary containing the most probable states of the nodes.
        """
        return self.infer.map_query(show_progress=show_progress)

    def maximum_likelihood_estimation(self, data):
        """
        Estimate the parameters of the Bayesian Network using Maximum Likelihood Estimation.

        Parameters
        ----------
        data : pandas DataFrame
            The dataset for parameter estimation.
        """
        mle = MaximumLikelihoodEstimator(self.model, data)
        self.model = mle.estimate()

    def bayesian_parameter_estimation(self, data):
        """
        Estimate the parameters of the Bayesian Network using Bayesian Parameter Estimation.

        Parameters
        ----------
        data : pandas DataFrame
            The dataset for parameter estimation.
        """
        bpe = BayesianEstimator(self.model, data)
        self.model = bpe.estimate()
