from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from typing import Dict, Any, Optional
import logging


class PredictiveModel:
    """
    A class for creating and training predictive models.

    Methods
    -------
    random_forest(config, target_type)
        Creates a Random Forest model based on the configuration and target type.
    linear_regression()
        Creates a Linear Regression model.
    fit(config, train_X, train_y, selected_features, preprocessor, target_type)
        Fits the model to the training data using the specified configuration.
    predict(X)
        Makes predictions using the trained model.
    """

    def __init__(self):
        """
        Initializes the PredictiveModel with placeholders for the model, selected features, and preprocessor.
        """
        self.selected_features = None
        self.preprocessor = None
        self.model = None
        self.logger = logging.getLogger(__name__)

    def random_forest(self, config: Dict[str, Any], target_type: str):
        """
        Creates a Random Forest model based on the configuration and target type.

        Parameters
        ----------
        config : dict
            Configuration settings for the Random Forest model, including hyperparameters like `n_estimators`,
            `min_samples_leaf`, and `max_features`.
        target_type : str
            The type of the target variable ('categorical' for classification, 'continuous' for regression).

        Returns
        -------
        model : RandomForestClassifier or RandomForestRegressor
            The initialized Random Forest model.
        """
        n_estimators = int(config.get('n_estimators', 100))
        min_samples_leaf = config.get('min_samples_leaf', 1)
        max_features = config.get('max_features', 'auto')

        if target_type == 'categorical':
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42
            )
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=42
            )
        return model

    def linear_regression(self):
        """
        Creates a Linear Regression model.

        Returns
        -------
        model : LinearRegression
            The initialized Linear Regression model.
        """
        return LinearRegression()

    def fit(
        self,
        config: Dict[str, Any],
        train_X: Any,
        train_y: Any,
        selected_features: Any,
        preprocessor: Optional[Any],
        target_type: str
    ):
        """
        Fits the model to the training data.

        Parameters
        ----------
        config : dict
            Configuration settings for the model, including the type of model ('random_forest' or 'linear_regression').
        train_X : array-like
            Training data for the input variables.
        train_y : array-like
            Training data for the target variable.
        selected_features : any
            The features selected for model training.
        preprocessor : object, optional
            A preprocessor object that can be used to transform the input data. Default is None.
        target_type : str
            The type of the target variable ('categorical' or 'continuous').

        Raises
        ------
        ValueError
            If an unsupported model type is specified in the configuration.
        """
        self.selected_features = selected_features
        self.preprocessor = preprocessor
        model_name = config.get('model')

        if model_name == 'random_forest':
            self.model = self.random_forest(config, target_type)
        elif model_name == 'linear_regression':
            self.model = self.linear_regression()
        else:
            raise ValueError(f"Unsupported model type: {model_name}")
        if(selected_features is not None):
            train_X = train_X[:, selected_features]

        self.model.fit(train_X, train_y)

    def predict(self, X: Any) -> Any:
        """
        Makes predictions using the trained model.

        Parameters
        ----------
        X : array-like
            The input data for which predictions are to be made.

        Returns
        -------
        predictions : array-like
            The predicted values based on the input data.
        """
        if self.preprocessor:
            X = self.preprocessor.transform(X)
        return self.model.predict(X)
