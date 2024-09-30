from sklearn.linear_model import LinearRegression


class LinearRegression_:
    """
    Wrapper class for setting up a LinearRegression model with custom parameters.

    Methods
    -------
    set_regressor_params(parameters)
        Configures and returns a LinearRegression object. Currently, LinearRegression does not require parameters in this method.
    """

    def set_regressor_params(self, parameters):
        """
        Configures and returns a LinearRegression object.

        Parameters
        ----------
        parameters : dict
            A dictionary containing the model parameters (though LinearRegression does not currently use parameters in this implementation).

        Returns
        -------
        LinearRegression
            A LinearRegression object configured with default parameters.

        Examples
        --------
        >>> params = {}
        >>> regressor = LinearRegression_().set_regressor_params(params)
        >>> print(regressor)
        LinearRegression()
        """
        return LinearRegression()
