import json
import itertools
from typing import Optional, List, Dict

from .causal_configs import causal_configs
from .ci_tests import ci_tests
from .scores import scores

class CausalDiscoveryConfigurator:
    """
    Configurator for creating causal discovery algorithm configurations based on various parameters.

    Attributes
    ----------
    causal_algs : dict
        Dictionary of causal discovery algorithms and their parameters.
    ci_tests : dict
        Dictionary of conditional independence tests.
    scores : dict
        Dictionary of scoring functions.

    Methods
    -------
    create_causal_configs(data_type, causal_sufficiency, assume_faithfulness, time_series, time_lagged, include_algs=None, exclude_algs=None)
        Generates a list of configurations for causal discovery algorithms based on input parameters.
    """

    def __init__(self):
        self.causal_algs = causal_configs
        self.ci_tests = ci_tests
        self.scores = scores

    def _dict_product(self, dicts):
        """
        Helper method to generate the cartesian product of dictionaries.

        Parameters
        ----------
        dicts : dict
            Dictionary where the keys are parameter names and the values are lists of possible values.

        Returns
        -------
        generator
            Generator that yields the cartesian product of the parameter combinations.
        """
        return (dict(zip(dicts.keys(), x)) for x in itertools.product(*dicts.values()))

    def create_causal_configs(self, data_type: str, causal_sufficiency: bool, assume_faithfulness: bool,
                              time_series: bool, time_lagged: bool, include_algs: Optional[List[str]] = None,
                              exclude_algs: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Generates a list of causal discovery algorithm configurations based on input parameters.

        Parameters
        ----------
        data_type : str
            Type of data (e.g., 'continuous', 'discrete', 'mixed').
        causal_sufficiency : bool
            Whether causal sufficiency is assumed in the dataset.
        assume_faithfulness : bool
            Whether to assume the faithfulness condition.
        time_series : bool
            Whether the dataset is time-series.
        time_lagged : bool
            Whether time-lagged variables are included in the dataset.
        include_algs : list, optional
            List of specific algorithms to include in the configuration. Default is None.
        exclude_algs : list, optional
            List of algorithms to exclude from the configuration. Default is None.

        Returns
        -------
        dict
            Dictionary of causal discovery algorithm configurations.
        """
        # Filter conditional independence tests and scores based on data type
        ci_touse = {ci_name: ci_info for ci_name, ci_info in self.ci_tests.items() if data_type in ci_info['data_type']}
        score_touse = {sc_name: sc_info for sc_name, sc_info in self.scores.items() if data_type in sc_info['data_type']}

        causal_configs = {"CausalDiscoveryAlgorithms": {}}
        for alg_name, alg_info in self.causal_algs.items():
            # Filter algorithms based on input parameters
            if (data_type in alg_info['data_type'] and
                    causal_sufficiency in alg_info['causal_sufficiency'] and
                    assume_faithfulness in alg_info['assume_faithfulness'] and
                    time_series in alg_info['time_series'] and
                    time_lagged in alg_info['time_lagged']):

                if include_algs and alg_name not in include_algs:
                    continue
                if exclude_algs and alg_name in exclude_algs:
                    continue


                config = {}
                for param, values in alg_info['parameters'].items():
                    if param == 'ci_test':
                        ci_names = []
                        ci_params = {}
                        for test_name in values:
                            if test_name in ci_touse:
                                ci_names.append(test_name)
                                ci_params.update({k: v for k, v in ci_touse[test_name].items() if k != 'data_type'})
                        if ci_names:
                            config['ci_test'] = ci_names
                            config.update(ci_params)
                    elif param == 'score':
                        score_names = []
                        score_params = {}
                        for score_name in values:
                            if score_name in score_touse:
                                score_names.append(score_name)
                                score_params.update({k: v for k, v in score_touse[score_name].items() if k != 'data_type'})
                        if score_names:
                            config['score'] = score_names
                            config.update(score_params)
                    else:
                        config[param] = values

                # Add extra metadata
                config['causal_sufficiency'] = alg_info['causal_sufficiency']
                config['assume_faithfulness'] = alg_info['assume_faithfulness']

                causal_configs["CausalDiscoveryAlgorithms"][alg_name] = config

        return causal_configs['CausalDiscoveryAlgorithms']


# Example usage
# configurator = CausalDiscoveryConfigurator()
# print(configurator.create_causal_configs('continuous', True, False, False, False, include_algs=['notears', 'sam'], exclude_algs=['pc']))
