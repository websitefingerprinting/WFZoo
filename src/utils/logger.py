import logging

LOG_FORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"


def init_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    # create formatter
    formatter = logging.Formatter(LOG_FORMAT)
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    if log_dir is not None:
        ch2 = logging.StreamHandler(open(log_dir, 'w'))
        ch2.setFormatter(formatter)
        logger.addHandler(ch2)
    return logger
