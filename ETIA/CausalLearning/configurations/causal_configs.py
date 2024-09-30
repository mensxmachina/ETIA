causal_configs = {
    "pc": {
        "data_type": [
            "continuous",
            "mixed",
            "categorical"
        ],
        "causal_sufficiency": [
            True
        ],

        "assume_faithfulness": [
            False
        ],
        "missing_values": [
            False
        ],
        "time_series": [
            False
        ],
        "time_lagged": [
            False,
            True
        ],
        "parameters": {
            "ci_test": [
                "FisherZ",
                "cg_lrt",
                "dg_lrt",
                "chisquare",
                "gsquare"
            ],
            "stable": [
                True
            ]
        }
    },
    "cpc": {
        "data_type": [
            "continuous",
            "mixed",
            "categorical"
        ],
        "stable": [
            True
        ],
        "causal_sufficiency": [
            True
        ],

        "assume_faithfulness": [
            False
        ],
        "missing_values": [
            False
        ],
        "time_series": [
            False
        ],
        "time_lagged": [
            False,
            True
        ],
        "parameters": {
            "ci_test": [
                "FisherZ",
                "cg_lrt",
                "dg_lrt",
                "chisquare",
                "gsquare"
            ],
            "stable": [
                True
            ]
        }
    },
    "fges": {
        "data_type": [
            "continuous",
            "mixed",
            "categorical"
        ],
        "causal_sufficiency": [
            False#True
        ],

        "assume_faithfulness": [
            False
        ],
        "missing_values": [
            False
        ],
        "time_series": [
            False
        ],
        "time_lagged": [
            False,
            True
        ],
        "parameters": {
            "score": [
                "sem_bic_score",
                "bdeu",
                "discrete_bic",
                "cg_bic",
                "dg_bic"
            ]
        }
    },
    "fci": {
        "data_type": [
            "continuous",
            "mixed",
            "categorical"
        ],
        "causal_sufficiency": [
            False
        ],

        "assume_faithfulness": [
            True
        ],
        "missing_values": [
            False
        ],
        "time_series": [
            False
        ],
        "time_lagged": [
            False,
            True
        ],
        "parameters": {
            "ci_test": [
                "FisherZ",
                "cg_lrt",
                "dg_lrt",
                "chisquare",
                "gsquare"
            ],
            "stable": [
                True
            ]
        }
    },
    "fcimax": {
        "data_type": [
            "continuous",
            "mixed",
            "categorical"
        ],
        "causal_sufficiency": [
            False
        ],

        "assume_faithfulness": [
            True
        ],
        "missing_values": [
            False
        ],
        "time_series": [
            False
        ],
        "time_lagged": [
            False,
            True
        ],
        "parameters": {
            "ci_test": [
                "FisherZ",
                "cg_lrt",
                "dg_lrt",
                "chisquare",
                "gsquare"
            ],
            "stable": [
                True
            ]
        }
    },
    "rfci": {
        "data_type": [
            "continuous",
            "mixed",
            "categorical"
        ],
        "causal_sufficiency": [
            False
        ],

        "assume_faithfulness": [
            True
        ],
        "missing_values": [
            False
        ],
        "time_series": [
            False
        ],
        "time_lagged": [
            False,
            True
        ],
        "parameters": {
            "ci_test": [
                "FisherZ",
                "cg_lrt",
                "dg_lrt",
                "chisquare",
                "gsquare"
            ],
            "stable": [
                True
            ]
        }
    },
    "gfci": {
        "data_type": [
            "continuous",
            "mixed",
            "categorical"
        ],
        "causal_sufficiency": [
            False
        ],

        "assume_faithfulness": [
            True
        ],
        "missing_values": [
            False
        ],
        "time_series": [
            False
        ],
        "time_lagged": [
            False,
            True
        ],
        "parameters": {
            "ci_test": [
                "FisherZ",
                "cg_lrt",
                "dg_lrt",
                "chisquare",
                "gsquare"
            ],
            "stable": [
                True
            ],
            "score": [
                "sem_bic_score",
                "bdeu",
                "discrete_bic",
                "cg_bic",
                "dg_bic"
            ]
        }
    },
    "cfci": {
        "data_type": [
            "continuous",
            "mixed",
            "categorical"
        ],
        "causal_sufficiency": [
            False
        ],

        "assume_faithfulness": [
            True
        ],
        "missing_values": [
            False
        ],
        "time_series": [
            False
        ],
        "time_lagged": [
            False,
            True
        ],
        "parameters": {
            "ci_test": [
                "FisherZ",
                "cg_lrt",
                "dg_lrt",
                "chisquare",
                "gsquare"
            ],
            "stable": [
                True
            ]
        }
    },
    "svarfci": {
        "ci_test": [
            "FisherZ",
            "cg_lrt",
            "dg_lrt",
            "chisquare",
            "gsquare"
        ],
        "data_type": [
            "continuous",
            "mixed",
            "categorical"
        ],
        "causal_sufficiency": [
            False
        ],

        "assume_faithfulness": [
            True
        ],
        "missing_values": [
            False
        ],
        "time_series": [
            True
        ],
        "time_lagged": [
            False,
            True
        ],
        "parameters": {
            "ci_test": [
                "FisherZ",
                "cg_lrt",
                "dg_lrt",
                "chisquare",
                "gsquare"
            ],
            "stable": [
                True
            ]
        }
    },
    "svargfci": {
        "data_type": [
            "continuous",
            "mixed",
            "categorical"
        ],
        "causal_sufficiency": [
            False
        ],

        "assume_faithfulness": [
            True
        ],
        "missing_values": [
            False
        ],
        "time_series": [
            True
        ],
        "time_lagged": [
            False,
            True
        ],
        "parameters": {
            "ci_test": [
                "FisherZ",
                "cg_lrt",
                "dg_lrt",
                "chisquare",
                "gsquare"
            ],
            "stable": [
                True
            ],
            "score": [
                "sem_bic_score",
                "bdeu",
                "discrete_bic",
                "cg_bic",
                "dg_bic"
            ]
        }
    },
    "PCMCI": {
        "data_type": [
            "continuous",
            "mixed",
            "categorical"
        ],
        "causal_sufficiency": [
            True
        ],

        "assume_faithfulness": [
            False
        ],
        "missing_values": [
            False
        ],
        "time_series": [
            True
        ],
        "time_lagged": [
            False
        ],
        "parameters": {
            "ci_test": [
                "ParCor",
                "RobustParCor",
                "GPDC",
                "CMIknn",
                "ParCorrWLS",
                "Gsquared",
                "CMIsymb",
                "RegressionCI"
            ]
        }
    },
    "PCMCI+": {
        "data_type": [
            "continuous",
            "mixed",
            "categorical"
        ],
        "causal_sufficiency": [
            True
        ],

        "assume_faithfulness": [
            False
        ],
        "missing_values": [
            False
        ],
        "time_series": [
            True
        ],
        "time_lagged": [
            False
        ],
        "parameters": {
            "ci_test": [
                "ParCor",
                "RobustParCor",
                "GPDC",
                "CMIknn",
                "ParCorrWLS",
                "Gsquared",
                "CMIsymb",
                "RegressionCI"
            ]
        }
    },
    "LPCMCI": {
        "data_type": [
            "continuous",
            "mixed",
            "categorical"
        ],
        "causal_sufficiency": [
            False
        ],

        "assume_faithfulness": [
            False
        ],
        "missing_values": [
            False
        ],
        "time_series": [
            True
        ],
        "time_lagged": [
            False
        ],
        "parameters": {
            "ci_test": [
                "ParCor",
                "RobustParCor",
                "GPDC",
                "CMIknn",
                "ParCorrWLS",
                "Gsquared",
                "CMIsymb",
                "RegressionCI"
            ]
        }
    },
    "sam": {
        "causal_sufficiency": [
            True
        ],
        "data_type": [
            "continuous",
            "mixed"
        ],
        "admit_latent_variables": [
            False
        ],
        "assume_faithfulness": [
            True
        ],
        "missing_values": [
            False
        ],
        "time_series": [
            False
        ],
        "time_lagged": [
            False,
            True
        ],
        "parameters": {
            "lr": [
                0.001,
                0.01,
                0.1
            ],
            "dlr": [
                0.0001,
                0.001,
                0.01
            ],
            "lambda1": [
                1,
                10,
                100
            ],
            "lambda2": [
                0.0001,
                0.001,
                0.01
            ],
            "nh": [
                10,
                20,
                50
            ],
            "dnh": [
                100,
                200,
                300
            ],
            "train_epochs": [
                1000,
                3000,
                5000
            ],
            "test_epochs": [
                500,
                1000,
                1500
            ],
            "batch_size": [
                50,
                100,
                200
            ],
            "losstype": [
                "fgan",
                "gan",
                "mse"
            ]
        }
    },
    "notears": {
        "causal_sufficiency": [
            True
        ],
        "data_type": [
            "continuous",
            "mixed",
            "categorical"
        ],
        "admit_latent_variables": [
            False
        ],
        "assume_faithfulness": [
            True
        ],
        "missing_values": [
            False
        ],
        "time_series": [
            False
        ],
        "time_lagged": [
            False
        ],
        "parameters": {
            "max_iter": [
                100,
                500,
                1000
            ],
            "h_tol": [
                1e-7,
                1e-5,
                1e-3
            ],
            "threshold": [
                0.0,
                0.5,
                0.8
            ]
        }
    }
}
