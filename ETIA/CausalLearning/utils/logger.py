import logging

def get_logger(name=None, verbose=False):
    logger = logging.getLogger(name)
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    return logger