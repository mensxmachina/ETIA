import itertools
import json
import os
from typing import List, Dict, Any


class PredictiveConfigurator:
    """
    Reads the available predictive learning, feature selection, and preprocessing algorithms from JSON files
    and creates the predictive configurations.

    Attributes
    ----------
    path : str
        The path to the directory containing the JSON configuration files.
    pred_algs : dict
        Dictionary containing the available predictive algorithms and their configurations.
    fs_algs : dict
        Dictionary containing the available feature selection algorithms and their configurations.
    preprocess_algs : dict
        Dictionary containing the available preprocessing algorithms and their configurations.

    Methods
    -------
    create_predictive_configs()
        Creates a list of all possible predictive configurations by combining available algorithms.
    """

    def __init__(self):
        """
        Initializes the PredictiveConfigurator by loading algorithm configurations from JSON files.
        """
        self.path = os.path.dirname(__file__)
        self.pred_algs = self._load_json('jsons/pred_algs.json')
        self.fs_algs = self._load_json('jsons/fs_algs.json')
        self.preprocess_algs = self._load_json('jsons/preprocess_algs.json')

    def _load_json(self, filename: str) -> Dict[str, Any]:
        """
        Loads the JSON file from the specified filename.

        Parameters
        ----------
        filename : str
            The name of the JSON file to load.

        Returns
        -------
        dict
            The content of the JSON file as a dictionary.
        """
        filepath = os.path.join(self.path, filename)
        with open(filepath, 'r') as file:
            return json.load(file)

    def _dict_product(self, dicts: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Generates all combinations of values from the given dictionary of lists.

        Parameters
        ----------
        dicts : dict
            A dictionary where keys are algorithm types (e.g., preprocessing, feature selection)
            and values are lists of possible configurations for each type.

        Returns
        -------
        generator
            A generator that yields dictionaries with all combinations of the input lists.
        """
        return (dict(zip(dicts.keys(), x)) for x in itertools.product(*dicts.values()))

    def create_predictive_configs(self) -> List[Dict[str, Any]]:
        """
        Creates a list of predictive configurations by combining available algorithms and their options.

        It reads configurations from the loaded JSON files for predictive models, feature selection methods,
        and preprocessing algorithms, and combines them to create all possible configurations.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries, where each dictionary is a unique combination of a predictive model,
            feature selection algorithm, and preprocessing method.
        """
        pred_configs = []
        for preprocess_name, preprocess_info in self.preprocess_algs.items():
            for p in self._dict_product(preprocess_info):
                for pred_name, pred_info in self.pred_algs.items():
                    for v in self._dict_product(pred_info):
                        for fs_name, fs_info in self.fs_algs.items():
                            for k in self._dict_product(fs_info):
                                config = {"model": pred_name}
                                config.update(v)
                                config["fs_name"] = fs_name
                                config.update(k)
                                if preprocess_name != "none":
                                    config["preprocess_method"] = preprocess_name
                                    config.update(p)
                                pred_configs.append(config)
        return pred_configs
