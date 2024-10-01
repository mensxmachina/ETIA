default_conf = {
    "Dataset":
        {
                "dataset_name": "test.csv",
                "time_lagged": False,
                "n_lags": 2
        },
    "Results_folder_path": "res/",
    "causal_sufficiency": True,
    "assume_faithfulness": False,
    "OCT":
        {
                "alpha": 0.01,
                "n_permutations": 50,
                "variables_type": "mixed",
                "out_of_sample_protocol":
                    {
                        "name": "KFoldCV",
                        "parameters":
                        {
                            "folds": 10,
                            "folds_to_run": 5
                        }
                    },
                "Regressor_parameters":
                    {
                        "name": "RandomForestRegressor",
                        "parameters":
                            {
                                "n_trees": 5,
                                "min_samples_leaf": 0.1,
                                "max_depth": 10
                            }
                    },

        }

}