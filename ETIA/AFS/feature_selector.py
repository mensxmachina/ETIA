import os
import subprocess
import logging
from typing import Dict, Optional, Any

import pandas as pd


class FeatureSelector:
    """
    Feature selection with the MXM R package.

    Methods
    -------
    feature_selection(config, target_name, train_idx_name=None, verbose=False)
        Runs the feature selection process based on the provided configuration.
    """

    def __init__(self, data_pd: pd.DataFrame, dataset_name: str, r_path: str):
        """
        Initializes the FeatureSelector.

        Parameters
        ----------
        data_pd : pandas.DataFrame
            The dataset to be used for feature selection.
        dataset_name : str
            The name of the dataset, used for saving intermediate files.
        r_path : str
            Path to the Rscript executable for running R-based feature selection algorithms.
        """
        self.r_path = r_path
        self.path_ = os.path.dirname(__file__)
        self.dataset_name = dataset_name + '.csv'
        self.output_file = os.path.join(self.path_, 'selected_features.csv')
        data_file_path = os.path.join(self.path_, self.dataset_name)
        data_pd.to_csv(data_file_path, index=False)

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def run_r_script(
        self,
        script_name: str,
        target_name: str,
        config: Dict[str, Any],
        train_idx_name: Optional[str] = None,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Runs the specified R script for feature selection.

        Parameters
        ----------
        script_name : str
            The name of the R script to run.
        target_name : str
            The name of the target variable in the dataset.
        config : dict
            The configuration settings for the feature selection algorithm.
        train_idx_name : str, optional
            The name of the CSV file containing the training indexes for a specific fold.
        verbose : bool, optional
            If True, prints detailed logs. Default is False.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the selected features.
        """
        script_path = os.path.join(self.path_, script_name)
        data_file_path = os.path.join(self.path_, self.dataset_name)
        args = [
            self.r_path, '--vanilla', script_path,
            data_file_path,
            target_name,
            config['ind_test_name'],
            str(config['alpha']),
            str(config['k']),
            self.output_file,
            'TRUE' if verbose else 'FALSE'
        ]
        if train_idx_name:
            train_idx_path = os.path.join(self.path_, train_idx_name)
            args.append(train_idx_path)

        result = subprocess.run(args, capture_output=True, text=True)

        if verbose:
            self.logger.info("R script stdout:")
            self.logger.info(result.stdout)
            self.logger.info("R script stderr:")
            self.logger.info(result.stderr)

        if result.returncode != 0:
            self.logger.error(f"R script {script_name} failed with return code {result.returncode}")
            self.logger.error(f"R script stderr: {result.stderr}")
            raise RuntimeError(f"R script {script_name} failed with return code {result.returncode}")

        selected_features_pd = pd.read_csv(self.output_file)
        return selected_features_pd

    def fbed(
        self,
        target_name: str,
        config: Dict[str, Any],
        train_idx_name: Optional[str] = None,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Runs the FBED feature selection algorithm.

        Parameters
        ----------
        target_name : str
            The name of the target variable.
        config : dict
            The configuration settings for the FBED algorithm.
        train_idx_name : str, optional
            The name of the CSV file with the training indexes for a specific fold.
        verbose : bool, optional
            If True, prints detailed logs. Default is False.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the selected features.
        """
        return self.run_r_script(
            'feature_selectors/fbed_with_idx.R',
            target_name,
            config,
            train_idx_name,
            verbose
        )

    def ses(
        self,
        target_name: str,
        config: Dict[str, Any],
        train_idx_name: Optional[str] = None,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Runs the SES feature selection algorithm.

        Parameters
        ----------
        target_name : str
            The name of the target variable.
        config : dict
            The configuration settings for the SES algorithm.
        train_idx_name : str, optional
            The name of the CSV file with the training indexes for a specific fold.
        verbose : bool, optional
            If True, prints detailed logs. Default is False.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the selected features.
        """
        return self.run_r_script(
            'feature_selectors/ses_with_idx.R',
            target_name,
            config,
            train_idx_name,
            verbose
        )

    def feature_selection(
        self,
        config: Dict[str, Any],
        target_name: str,
        train_idx_name: Optional[str] = None,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Runs the feature selection process based on the provided configuration.

        Parameters
        ----------
        config : dict
            The configuration settings for feature selection.
        target_name : str
            The name of the target variable.
        train_idx_name : str, optional
            The name of the CSV file with the training indexes for a specific fold.
        verbose : bool, optional
            If True, prints detailed logs. Default is False.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the selected features.
        """
        fs_name = config.get('fs_name')
        if fs_name == 'fbed':
            features = self.fbed(target_name, config, train_idx_name, verbose)
        elif fs_name == 'ses':
            features = self.ses(target_name, config, train_idx_name, verbose)
        else:
            raise ValueError(f"Unsupported feature selection algorithm: {fs_name}")

        return features
