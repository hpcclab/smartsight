import logging


def get_logger(name: str) -> logging.Logger:
    """Return a SmartSight logger that does not inherit the root level.

    Importing Paddle sets the root logger to WARNING after the app has
    configured it. A NOTSET child inherits that level and drops INFO.
    An explicit INFO level still propagates to the root stderr handler.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
