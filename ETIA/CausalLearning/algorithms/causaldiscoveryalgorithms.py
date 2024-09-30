from .utils import *

cd_algorithms = { "tetrad": {'algorithms':[
    "pc",
    "cpc",
    "fges",
    "fci",
    "fcimax",
    "rfci",
    "gfci",
    "cfci",
    "svarfci",
    "svargfci"
],
    "prepare_data_function": prepare_data_tetrad,
    "Data": None},

    "tigramite" : {"algorithms":
    [
        "PCMCI",
        "PCMCI+",
        "LPCMCI"
    ],
        "prepare_data_function": prepare_data_tigramite,
        "Data": None},
    "causalnex": {"algorithms":
    [
        "notears"
    ],
        "prepare_data_function": None,
        "Data": None
                  },
    "cdt": {"algorithms":
        [
            "sam"
        ],
        "prepare_data_function": None,
        "Data": None
    }
}

