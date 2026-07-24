import logging

# Configure the package-level logger
logger = logging.getLogger("experia")

# Prevent log propagation to the root logger by default to avoid duplicate logs
# if the user hasn't configured a handler.
logger.propagate = False

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
