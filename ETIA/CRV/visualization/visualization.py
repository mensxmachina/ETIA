import numpy as np
import py4cytoscape as p4c
from py4cytoscape import create_visual_style
from .cytoscape_utils import *


class Visualization:
    """
    A class to create and manage network visualizations in Cytoscape based on causal discovery results.

    Parameters
    ----------
    matrix_pd : pd.DataFrame
        A pandas DataFrame containing the adjacency matrix of the graph to visualize.
    net_name : str
        The name of the network to create in Cytoscape.
    collection_name : str
        The name of the collection to store the network in Cytoscape.

    Attributes
    ----------
    style_name : str
        The name of the visual style applied to the network in Cytoscape.
    network_suid : int
        The unique identifier for the network in Cytoscape.

    Methods
    -------
    plot_cytoscape()
        Plots the graph in Cytoscape based on the adjacency matrix.
    create_visual_style(node_size=35, node_shape='ellipse', node_color='#FDD49E')
        Creates a visual style for the Cytoscape network.
    set_node_color(node_names, color)
        Sets the color of specified nodes.
    hide_nodes(nodes)
        Hides specific nodes from the Cytoscape network.
    plot_edge_confidence(edge_confidence)
        Plots edge confidence by adjusting edge widths and opacities.
    hide_edges(threshold)
        Hides edges based on a specified confidence threshold.
    set_layout(layout_name='force-directed')
        Applies a layout to the network in Cytoscape.
    export_to_png(file_path='network.png')
        Exports the network visualization as a PNG file.
    """

    def __init__(self, matrix_pd, net_name, collection_name):
        self.matrix_pd = matrix_pd
        self.net_name = net_name
        self.collection_name = collection_name
        self.style_name = 'AutoCD_Visualization_Style'
        self.network_suid = None

    def plot_cytoscape(self):
        """
        Plots the graph in Cytoscape based on the adjacency matrix.

        This method converts the adjacency matrix into a Cytoscape-readable format and visualizes the graph
        in the Cytoscape application.

        Returns
        -------
        None
        """
        cytoscape_ping()
        cytoscape_version_info()
        cyto_edges = matrix_to_cyto(self.matrix_pd)
        self.network_suid = create_network_from_data_frames(cyto_edges, title=self.net_name,
                                                            collection=self.collection_name)

        self.create_visual_style()

    def create_visual_style(self, node_size=35, node_shape='ellipse', node_color='#FDD49E'):
        """
        Creates a visual style for the Cytoscape network.

        Parameters
        ----------
        node_size : int, optional
            The size of the nodes in the network. Default is 35.
        node_shape : str, optional
            The shape of the nodes in the network. Default is 'ellipse'.
        node_color : str, optional
            The fill color of the nodes in the network. Default is '#FDD49E'.

        Returns
        -------
        None
        """
        defaults = {
            'NODE_SHAPE': node_shape,
            'NODE_SIZE': node_size,
            'NODE_FILL_COLOR': node_color
        }
        p4c.create_visual_style(self.style_name, defaults=defaults)
        p4c.set_visual_style(self.style_name)
        p4c.set_edge_target_arrow_shape_mapping(
            'interaction_type',
            table_column_values=['Circle-Arrow', 'Arrow-Circle', 'Circle-Tail', 'Tail-Circle', 'Arrow-Tail',
                                 'Tail-Arrow', 'Arrow-Arrow', 'Circle-Circle', 'Tail-Tail'],
            shapes=['ARROW', 'CIRCLE', 'NONE', 'CIRCLE', 'NONE', 'ARROW', 'ARROW', 'CIRCLE', 'NONE'],
            style_name=self.style_name)

        p4c.set_edge_source_arrow_shape_mapping(
            'interaction_type',
            table_column_values=['Circle-Arrow', 'Arrow-Circle', 'Circle-Tail', 'Tail-Circle', 'Arrow-Tail',
                                 'Tail-Arrow', 'Arrow-Arrow', 'Circle-Circle', 'Tail-Tail'],
            shapes=['CIRCLE', 'ARROW', 'CIRCLE', 'NONE', 'ARROW', 'NONE', 'ARROW', 'CIRCLE', 'NONE'],
            style_name=self.style_name)

    def set_node_color(self, node_names, color):
        """
        Sets the color for a specified list of nodes by their names.

        Parameters
        ----------
        node_names : list of str
            List of node names to set the color for.
        color : str
            The color to apply to the nodes (e.g., '#ADD8E6').

        Returns
        -------
        None
        """
        p4c.set_node_color_bypass(node_names, color)

    def hide_nodes(self, nodes):
        """
        Hides a group of nodes identified by their names.

        Parameters
        ----------
        nodes : list of str
            List of node names to hide.

        Returns
        -------
        None
        """
        print(p4c.select_nodes(nodes, by_col='name', network=self.network_suid))
        print(p4c.get_selected_nodes())
        print(p4c.delete_selected_nodes())

    def plot_edge_confidence(self, edge_confidence):
        """
        Plots edge confidence by adjusting edge widths and opacities.

        Parameters
        ----------
        edge_confidence : pd.DataFrame
            A DataFrame containing edge confidence data.

        Returns
        -------
        None
        """
        edge_width_controler = 'edge_confidence'
        edge_color_controler = 'edge_consistency'
        edge_width_mapping = {'input_values': [0, 0.5, 0.7, 1.0], 'width_values': [0.4, 1, 2, 3]}
        edge_opacity_controler = edge_width_controler
        edge_opacity_mapping = {'input_values': ['0', '1'], 'opacity_values': [150, 250]}

        p4c.set_edge_line_width_mapping(edge_width_controler, edge_width_mapping['input_values'],
                                        edge_width_mapping['width_values'], 'c', style_name=self.style_name)

        p4c.set_edge_opacity_mapping(edge_opacity_controler, edge_opacity_mapping['input_values'],
                                     edge_opacity_mapping['opacity_values'], 'c', style_name=self.style_name)

    def hide_edges(self, threshold):
        """
        Hides a group of edges based on a threshold value.

        Parameters
        ----------
        threshold : float
            Threshold for edge confidence. Edges below this value will be hidden.

        Returns
        -------
        None
        """
        edge_list = p4c.get_edge_list(edge_type='interaction', numeric_column='confidence',
                                      predicate='LESS_THAN', cut_off=threshold, network=self.network_suid)
        p4c.hide_edges(edge_list, network=self.network_suid)

    def set_layout(self, layout_name='force-directed'):
        """
        Sets the layout for the network visualization.

        Parameters
        ----------
        layout_name : str, optional
            Name of the layout to apply (e.g., 'force-directed'). Default is 'force-directed'.

        Returns
        -------
        None
        """
        p4c.layout_network(layout_name, network=self.network_suid)

    def export_to_png(self, file_path='network.png'):
        """
        Exports the current network view to a PNG file.

        Parameters
        ----------
        file_path : str, optional
            Path to save the PNG file. Default is 'network.png'.

        Returns
        -------
        None
        """
        p4c.export_image(filename=file_path, type='PNG', network=self.network_suid)
        print(f'Network exported to {file_path}')


def matrix_to_cyto(matrix_pd):
    """
    Converts an adjacency matrix to a Cytoscape-readable format.

    Parameters
    ----------
    matrix_pd : pd.DataFrame
        A pandas DataFrame representing an adjacency matrix.

    Returns
    -------
    cyto_edges : pd.DataFrame
        A DataFrame of edges with source, target, and interaction type for Cytoscape visualization.
    """
    matrix = matrix_pd.to_numpy()
    row_names = matrix_pd.columns.to_list()
    column_names = matrix_pd.columns.to_list()

    n_nodes = matrix.shape[0]
    n_edges = int(np.count_nonzero(matrix) / 2)

    edge_data = np.empty((n_edges, 3), dtype='object')

    c = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if matrix[i, j] != 0 and matrix[j, i] != 0:
                if matrix[i, j] == 1:
                    iToj = 'Circle'
                elif matrix[i, j] == 2:
                    iToj = 'Arrow'
                elif matrix[i, j] == 3:
                    iToj = 'Tail'
                else:
                    raise ValueError('Wrong notation on input matrix of the graph')

                if matrix[j, i] == 1:
                    jToi = 'Circle'
                elif matrix[j, i] == 2:
                    jToi = 'Arrow'
                elif matrix[j, i] == 3:
                    jToi = 'Tail'
                else:
                    raise ValueError('Wrong notation on input matrix of the graph')

                interaction = jToi + '-' + iToj
                edge_data[c] = [row_names[i], column_names[j], interaction]
                c += 1

    cyto_edges = pd.DataFrame(data=edge_data, columns=['source', 'target', 'interaction_type'])
    return cyto_edges
