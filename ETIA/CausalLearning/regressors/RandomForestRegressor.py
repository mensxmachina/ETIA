from sklearn.ensemble import RandomForestRegressor


class RandomForestRegressor_:
    """
    Wrapper class for setting up a RandomForestRegressor model with custom parameters.

    Methods
    -------
    set_regressor_params(parameters)
        Configures and returns a RandomForestRegressor object with the specified parameters.
    """

    def set_regressor_params(self, parameters):
        """
        Configures and returns a RandomForestRegressor object with the specified parameters.

        Parameters
        ----------
        parameters : dict
            A dictionary containing the following keys:
                - 'n_trees': int, The number of trees in the forest.
                - 'min_samples_leaf': int or float, The minimum number of samples required to be at a leaf node.
                - 'max_depth': int, The maximum depth of the tree.

        Returns
        -------
        RandomForestRegressor
            A RandomForestRegressor object configured with the specified parameters.

        Examples
        --------
        >>> params = {'n_trees': 100, 'min_samples_leaf': 0.1, 'max_depth': 10}
        >>> regressor = RandomForestRegressor_().set_regressor_params(params)
        >>> print(regressor)
        RandomForestRegressor(max_depth=10, min_samples_leaf=0.1)
        """
        return RandomForestRegressor(
            n_estimators=parameters['n_trees'],
            min_samples_leaf=parameters['min_samples_leaf'],
            max_depth=parameters['max_depth'], n_jobs=-1
        )
