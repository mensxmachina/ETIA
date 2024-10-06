import os
import subprocess
import logging
from typing import Dict, Optional, Any

import pandas as pd
import uuid


class FeatureSelector:
    """
    Feature selection with the MXM R package.

    Methods
    -------
    feature_selection(config, target_name, data_pd, dataset_name, train_idx_name=None, verbose=False)
        Runs the feature selection process based on the provided configuration.
    """

    def __init__(self, r_path: str):
        """
        Initializes the FeatureSelector.

        Parameters
        ----------
        r_path : str
            Path to the Rscript executable for running R-based feature selection algorithms.
        """
        self.r_path = r_path
        self.path_ = os.path.dirname(__file__)

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def run_r_script(
        self,
        script_path: str,
        data_file_path: str,
        target_name: str,
        config: Dict[str, Any],
        output_file: str,
        train_idx_name: Optional[str] = None,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Runs the specified R script for feature selection.
        """
        args = [
            self.r_path, '--vanilla', script_path,
            data_file_path,
            target_name,
            config['ind_test_name'],
            str(config['alpha']),
            str(config['k']),
            output_file,
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
            self.logger.error(f"R script {script_path} failed with return code {result.returncode}")
            self.logger.error(f"R script stderr: {result.stderr}")
            raise RuntimeError(f"R script {script_path} failed with return code {result.returncode}")

        selected_features_pd = pd.read_csv(output_file)
        return selected_features_pd

    def fbed(
        self,
        target_name: str,
        config: Dict[str, Any],
        data_file_path: str,
        output_file: str,
        train_idx_name: Optional[str] = None,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Runs the FBED feature selection algorithm.
        """
        script_path = os.path.join(self.path_, 'feature_selectors', 'fbed_with_idx.R')
        return self.run_r_script(
            script_path,
            data_file_path,
            target_name,
            config,
            output_file,
            train_idx_name,
            verbose
        )

    def ses(
        self,
        target_name: str,
        config: Dict[str, Any],
        data_file_path: str,
        output_file: str,
        train_idx_name: Optional[str] = None,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Runs the SES feature selection algorithm.
        """
        script_path = os.path.join(self.path_, 'feature_selectors', 'ses_with_idx.R')
        return self.run_r_script(
            script_path,
            data_file_path,
            target_name,
            config,
            output_file,
            train_idx_name,
            verbose
        )

    def feature_selection(
        self,
        config: Dict[str, Any],
        target_name: str,
        data_pd: pd.DataFrame,
        dataset_name: str,
        train_idx_name: Optional[str] = None,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Runs the feature selection process based on the provided configuration.
        """
        # Generate unique file names
        unique_id = str(uuid.uuid4())
        data_file_name = f"{dataset_name}_{unique_id}.csv"
        data_file_path = os.path.join(self.path_, data_file_name)
        output_file = os.path.join(self.path_, f"selected_features_{unique_id}.csv")

        data_pd.to_csv(data_file_path, index=False)
        try:
            fs_name = config.get('fs_name')
            if fs_name == 'fbed':
                features = self.fbed(target_name, config, data_file_path, output_file, train_idx_name, verbose)
            elif fs_name == 'ses':
                features = self.ses(target_name, config, data_file_path, output_file, train_idx_name, verbose)
            else:
                raise ValueError(f"Unsupported feature selection algorithm: {fs_name}")
            return features
        finally:
            # Ensure the CSV files are deleted after feature selection
            if os.path.exists(data_file_path):
                os.remove(data_file_path)
            if os.path.exists(output_file):
                os.remove(output_file)
