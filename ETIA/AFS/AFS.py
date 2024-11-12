import os
import random
import uuid
from multiprocessing import cpu_count
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, roc_auc_score
import logging

from joblib import Parallel, delayed

from .feature_selector import FeatureSelector
from .oos import OOS
from .predictive_configurator import PredictiveConfigurator
from .predictive_model import PredictiveModel
from .preprocessor import Preprocessor


class AFS:
    """
    Automated Feature Selection (AFS) class.

    Parameters
    ----------
    depth : int, optional
        The depth of the feature selection process. Default is 1.
    verbose : bool, optional
        If True, prints detailed logs. Default is False.
    num_processors : int, optional
        Number of processors to use for parallel processing. Default is the number of CPU cores.
    oos_protocol : dict, optional
        A dictionary specifying the out-of-sample protocol. Default is a 5-fold cross-validation.
    random_seed : int, optional
        Seed for random number generator to ensure reproducibility. Default is None.

    Methods
    -------
    run_AFS(data, target_features, pred_configs=None, dataset_name='dataset')
        Runs the AFS process on the provided data and target features.
    """

    def __init__(
        self,
        depth: int = 1,
        verbose: bool = False,
        num_processors: Optional[int] = None,
        oos_protocol: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
    ):
        if depth < 0:
            raise ValueError("Depth must be a non-negative integer.")
        # Set default oos_protocol if none is provided
        self.oos_protocol = oos_protocol if oos_protocol else {
            "name": "KFoldCV",
            "folds": 5,
            "folds_to_run": 2,
        }
        self.csv_path = os.path.dirname(__file__)
        self.depth = depth
        self.verbose = verbose
        self.num_processors = num_processors if num_processors else cpu_count()
        self.random_seed = random_seed

        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO if self.verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s',
        )

        # Set random seed for reproducibility
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)

    def run_AFS(
        self,
        data: Union[str, pd.DataFrame, np.ndarray],
        target_features: Union[List[str], Dict[str, str]],
        pred_configs: Optional[Union[List[Dict[str, Any]], float]] = None,
        dataset_name: str = 'dataset',
    ) -> Dict[str, Any]:
        """
        Runs the AFS process on the provided data and target features.

        Parameters
        ----------
        data : str or pd.DataFrame or np.ndarray
            The dataset to use. Can be a filename (str), a pandas DataFrame, or a NumPy array.
        target_features : Union[Dict[str, str], List[str]]
            A dictionary mapping feature names to their types, or a list of feature names (in which case the types are inferred).
        pred_configs : Union[List[Dict[str, Any]], float], optional
            - If list, it is a list of predictive configurations provided by the user.
            - If float (between 0 and 1), it indicates the percentage of default configurations to sample and run.
            - If None, all default configurations are used.
        dataset_name : str, optional
            The name of the dataset (used for saving intermediate files). Default is 'dataset'.

        Returns
        -------
        dict
            A dictionary containing:
            - 'original_data': The original dataset
            - 'reduced_data': The dataset with only the selected features and target features
            - 'best_config': The configuration that led to the best feature selection
            - 'selected_features': The selected features for each target

        Examples
        --------
        To run feature selection on a dataset:
        >>> afs = AFS()
        >>> result = afs.run_AFS(data="data.csv", target_features=["feature1", "feature2"])
        >>> print(result["selected_features"])
        """
        if not isinstance(target_features, (list, dict)):
            raise ValueError("target_features must be a list or dictionary.")
        # Load the data based on the type
        if isinstance(data, str):
            # Assume it's a filename
            original_data = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            original_data = data.copy()
        elif isinstance(data, np.ndarray):
            # Check if data includes target feature
            if data.dtype.names is not None:
                original_data = pd.DataFrame(data)
            else:
                # Need to get feature names from elsewhere or raise an error
                raise ValueError("When data is a NumPy array, it must have named columns.")
        else:
            raise ValueError("Data must be a filename (str), pandas DataFrame, or NumPy array.")

        if(original_data.empty):
            raise ValueError("Dataframe is empty.")

        # Handle target features being either a list or a dictionary
        if isinstance(target_features, list):
            target_features = {feature: 'unknown' for feature in target_features}

        # Generate default configurations if pred_configs is None or a float
        if pred_configs is None or isinstance(pred_configs, float):
            configurator = PredictiveConfigurator()
            all_configs = configurator.create_predictive_configs()

            if isinstance(pred_configs, float):
                # pred_configs is a float indicating the percentage to sample
                sample_rate = pred_configs
                if not (0 < sample_rate <= 1):
                    raise ValueError("pred_configs as a float must be between 0 and 1.")

                total_configs = len(all_configs)
                sample_size = max(1, int(total_configs * sample_rate))
                pred_configs = random.sample(all_configs, sample_size)
                self.logger.info(f"Sampled {sample_size} out of {total_configs} configurations.")
            else:
                # Use all configurations
                pred_configs = all_configs
        elif isinstance(pred_configs, list):
            # pred_configs is a list provided by the user
            pass
        else:
            raise ValueError("pred_configs must be a list of configurations or a float between 0 and 1.")

        best_config = None
        best_score = -float('inf')
        selected_features = {}
        reduced_data = original_data.copy()

        # Prepare arguments for parallel processing
        target_items = list(target_features.items())
        args_list = [
            (
                original_data,
                target_feature,
                target_type,
                pred_configs,
                dataset_name,
                self.depth,
                None  # visited_features
            )
            for target_feature, target_type in target_items
        ]

        # Use joblib's Parallel to process targets in parallel
        results = Parallel(n_jobs=self.num_processors)(
            delayed(self._process_target)(
                data=arg[0],
                target_feature=arg[1],
                target_type=arg[2],
                pred_configs=arg[3],
                dataset_name=arg[4],
                depth=arg[5],
                visited_features=arg[6],
            ) for arg in args_list
        )

        main_target = next(iter(target_features))

        # Process results
        for (target_feature, _), result in zip(target_items, results):
            selected_features[target_feature] = result['selected_features']
            if target_feature == main_target:
                best_score = result['bbc_score']
                best_ci = result['ci']
                best_config = result['best_config']

        # Collect all selected features across all targets and depths
        all_selected_features = set()
        for features in selected_features.values():
            all_selected_features.update(features)
        # Add target features to the selected features
        all_selected_features.update(target_features.keys())
        reduced_data = reduced_data[list(all_selected_features)]
        reduced_without_target = reduced_data.loc[:, reduced_data.columns != main_target]
        if(reduced_without_target.empty):
            self.logger.info(f"No features selected for target '{target_feature}'")
            return {
                'original_data': original_data,
                'reduced_data': reduced_data,
                'best_config': best_config,
                'bbc_score': best_score,
                'ci': best_ci,
                'trained_model': None,
                'selected_features': selected_features[target_feature],
            }
        pm = PredictiveModel()
        pm.fit(
                best_config,
                reduced_without_target.values,
                reduced_data[main_target].values,
                None,
                None,
                target_features[main_target],
        )

        return {
            'original_data': original_data,
            'reduced_data': reduced_data,
            'best_config': best_config,
            'bbc_score': best_score,
            'ci': best_ci,
            'trained_model': pm,
            'selected_features': selected_features,
        }

    def _process_target(
        self,
        data: pd.DataFrame,
        target_feature: str,
        target_type: str,
        pred_configs: List[Dict[str, Any]],
        dataset_name: str,
        depth: int,
        visited_features: Optional[set] = None,
    ) -> Dict[str, Any]:
        """
        Helper function to process each target in parallel.
        """
        self.logger.info(f"Starting feature selection for target '{target_feature}'")
        return self.recursive_fs_for_target(
            data=data,
            target_feature=target_feature,
            target_type=target_type,
            pred_configs=pred_configs,
            dataset_name=dataset_name,
            depth=depth,
            visited_features=visited_features,
        )

    def recursive_fs_for_target(
        self,
        data: pd.DataFrame,
        target_feature: str,
        target_type: str,
        pred_configs: List[Dict[str, Any]],
        dataset_name: str,
        depth: int,
        visited_features: Optional[set] = None,
    ) -> Dict[str, Any]:
        """
        Recursively runs feature selection for a specific target feature up to the specified depth.
        """
        if depth == 0:
            return {
                'bbc_score': -float('inf'),
                'ci': None,
                'selected_features': [],
                'best_config': None,
            }

        if visited_features is None:
            visited_features = set()
        visited_features.add(target_feature)

        self.logger.info(f"Depth {self.depth - depth + 1}: Feature selection for '{target_feature}'")

        # Prepare data excluding the target feature and visited features
        feature_columns = [col for col in data.columns if col != target_feature and col not in visited_features]
        if not feature_columns:
            self.logger.warning(f"No features left to select for target '{target_feature}' at depth {self.depth - depth + 1}.")
            return {
                'bbc_score': -float('inf'),
                'ci': None,
                'selected_features': [],
                'best_config': None,
            }
        # Generate the folds once and use them for every configuration
        oos = OOS()
        X = data[feature_columns]
        y = data[target_feature]

        # Determine target type if unknown
        if target_type == 'unknown':
            if pd.api.types.is_numeric_dtype(y):
                target_type = 'continuous'
            else:
                target_type = 'categorical'

        train_inds, test_inds = oos.data_split(
            self.oos_protocol, X.values, y.values, target_type=target_type
        )

        # Run feature selection and model training for each configuration in parallel
        results = Parallel(n_jobs=self.num_processors)(
            delayed(self._process_config)(
                data=data,
                target_feature=target_feature,
                target_type=target_type,
                config=config,
                dataset_name=dataset_name,
                train_inds=train_inds,
                test_inds=test_inds,
                feature_columns=feature_columns,
            ) for config in pred_configs
        )

        all_scores = []
        all_fold_predictions = []
        configs_tried = []

        for result in results:
            if result is not None:
                config, mean_score, fold_predictions, selected_features_df = result
                all_scores.append(mean_score)
                all_fold_predictions.append((config, fold_predictions, selected_features_df))
                configs_tried.append(config)

        if not all_scores:
            self.logger.warning(f"No valid configurations for target '{target_feature}' at depth {self.depth - depth + 1}.")
            return {
                'bbc_score': -float('inf'),
                'ci': None,
                'selected_features': [],
                'best_config': None,
            }

        # Identify the best configuration based on average scores
        idx_best_config = np.argmax(all_scores)
        best_config = configs_tried[idx_best_config]
        best_fold_predictions = all_fold_predictions[idx_best_config][1]
        selected_features_df = all_fold_predictions[idx_best_config][2]

        # Collect predictions and true values
        best_conf_predictions = [
            (pred, true) for pred, true, _, _, _ in best_fold_predictions
        ]

        self.logger.info(f"Target: {target_feature} with optimal config: {best_config}")

        # Apply bootstrap bias correction to the best configuration
        bbc_score, ci = self.bootstrap_bias_correction(best_conf_predictions, target_type)

        # Collect selected features (assuming features may vary across folds)
        selected_features_sets = [
            fold_info[2]['sel'] for fold_info in best_fold_predictions
        ]
        # Find common selected features across all folds
        if selected_features_sets:
            selected_feature_indices = list(
                set.intersection(*map(set, selected_features_sets))
            )
            selected_feature_names = [feature_columns[idx] for idx in selected_feature_indices]
        else:
            selected_feature_names = []

        # Recursive feature selection on selected features
        all_selected_features = set()
        for feature in selected_feature_names:
            if feature not in visited_features:
                result = self.recursive_fs_for_target(
                    data,
                    target_feature=feature,
                    target_type='unknown',
                    pred_configs=pred_configs,
                    dataset_name=dataset_name,
                    depth=depth - 1,
                    visited_features=visited_features.copy(),
                )
                all_selected_features.update(result['selected_features'])
                all_selected_features.add(feature)

        return {
            'bbc_score': bbc_score,
            'ci': ci,
            'selected_features': list(all_selected_features),
            'best_config': best_config,
        }

    def _process_config(
        self,
        data: pd.DataFrame,
        target_feature: str,
        target_type: str,
        config: Dict[str, Any],
        dataset_name: str,
        train_inds: List[np.ndarray],
        test_inds: List[np.ndarray],
        feature_columns: List[str],
    ) -> Optional[Tuple[Dict[str, Any], float, List[Tuple[np.ndarray, np.ndarray, Dict[str, Any], Any, Optional[Preprocessor]]], pd.DataFrame]]:
        """
        Helper function to process each configuration in parallel.
        """
        scores, fold_predictions, selected_features_df = self.run_fs_for_config(
            data,
            target_feature,
            target_type,
            config,
            dataset_name,
            train_inds,
            test_inds,
            feature_columns,
        )
        if scores:
            mean_score = np.mean(scores)
            return (config, mean_score, fold_predictions, selected_features_df)
        else:
            return None

    def run_fs_for_config(
        self,
        data: pd.DataFrame,
        target_feature: str,
        target_type: str,
        config: Dict[str, Any],
        dataset_name: str,
        train_inds: List[np.ndarray],
        test_inds: List[np.ndarray],
        feature_columns: List[str],
    ) -> Tuple[List[float], List[Tuple[np.ndarray, np.ndarray, Dict[str, Any], Any, Optional[Preprocessor]]], pd.DataFrame]:
        """
        Runs the feature selection process for a specific configuration.
        """
        scores = []
        fold_predictions = []
        selected_features_df = None
        config_id = str(uuid.uuid4())

        for fold_num, (train_index, test_index) in enumerate(zip(train_inds, test_inds)):
            train_data = data.iloc[train_index]
            test_data = data.iloc[test_index]

            # Preprocessing
            if config.get('preprocess_method'):
                preprocessor = Preprocessor(method=config['preprocess_method'])
                train_data_preprocessed = preprocessor.fit_transform(train_data[feature_columns + [target_feature]])
                test_data_preprocessed = preprocessor.transform(test_data[feature_columns + [target_feature]])
                train_data_preprocessed_df = pd.DataFrame(
                    train_data_preprocessed, columns=feature_columns + [target_feature]
                )
                test_data_preprocessed_df = pd.DataFrame(
                    test_data_preprocessed, columns=feature_columns + [target_feature]
                )
            else:
                train_data_preprocessed_df = train_data[feature_columns + [target_feature]].copy()
                test_data_preprocessed_df = test_data[feature_columns + [target_feature]].copy()
                preprocessor = None

            # Initialize the FeatureSelector with the path to Rscript
            fs = FeatureSelector(r_path=config.get('r_path', 'Rscript'))

            # Perform feature selection
            try:
                unique_dataset_name = f"{dataset_name}_{target_feature}_{config_id}_fold{fold_num}"
                selected_features_fold_df = fs.feature_selection(
                    config=config,
                    target_name=target_feature,
                    data_pd=train_data_preprocessed_df,
                    dataset_name=unique_dataset_name,
                    verbose=self.verbose
                )
            except RuntimeError as e:
                self.logger.error(f"Feature selection failed for target '{target_feature}' with config {config}: {e}")
                continue  # Skip this fold due to error

            if selected_features_fold_df.empty:
                self.logger.warning(
                    f"No features selected for target '{target_feature}' with config {config} in fold {fold_num}. Skipping this fold."
                )
                continue

            selected_feature_indices = selected_features_fold_df['sel'].tolist()
            feature_names = [feature_columns[idx] for idx in selected_feature_indices]

            # Prepare training and testing data
            train_X = train_data_preprocessed_df[feature_names]
            train_y = train_data_preprocessed_df[target_feature]
            test_X = test_data_preprocessed_df[feature_names]
            test_y = test_data_preprocessed_df[target_feature]

            # Model Training
            pm = PredictiveModel()
            pm.fit(
                config,
                train_X.values,
                train_y.values,
                None,
                preprocessor,
                target_type,
            )
            predictions = pm.predict(test_X.values)
            fold_predictions.append(
                (predictions, test_y.values, selected_features_fold_df, pm, preprocessor)
            )

            # Scoring
            if target_type == 'categorical':
                try:
                    score = roc_auc_score(test_y.values, predictions)
                except ValueError:
                    self.logger.warning(
                        f"ROC AUC score could not be computed for fold {fold_num} due to insufficient classes."
                    )
                    continue
            else:
                score = r2_score(test_y.values, predictions)
            scores.append(score)

            # Store selected features from the first fold
            if selected_features_df is None:
                selected_features_df = selected_features_fold_df

        return scores, fold_predictions, selected_features_df

    def bootstrap_bias_correction(
        self,
        fold_predictions: List[Tuple[np.ndarray, np.ndarray]],
        target_type: str,
        B: int = 1000,
        conf_interval: float = 0.95,
    ) -> float:
        """
        Applies bootstrap bias correction to the fold predictions.
        """
        if not fold_predictions:
            return float('nan')

        all_predictions = np.concatenate([pred for pred, _ in fold_predictions])
        all_true_values = np.concatenate([true for _, true in fold_predictions])

        n_samples = len(all_predictions)
        b_scores = []

        for _ in range(B):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            sampled_preds = all_predictions[indices]
            sampled_trues = all_true_values[indices]

            if target_type == 'categorical':
                try:
                    score = roc_auc_score(sampled_trues, sampled_preds)
                except ValueError:
                    continue
            else:
                score = r2_score(sampled_trues, sampled_preds)
            b_scores.append(score)

        # Compute confidence interval
        lower_bound = (1.0 - conf_interval) / 2.0
        upper_bound = 1.0 - lower_bound
        ci = np.percentile(b_scores, [100 * lower_bound, 100 * upper_bound])
        self.logger.info(f'Confidence interval: {ci}')

        bbc_score = np.mean(b_scores)
        return bbc_score, ci
