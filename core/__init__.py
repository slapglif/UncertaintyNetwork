import os
import sys

from loguru import logger

# Set the minimum log level based on an environment variable
log_level = os.environ.get("LOG_LEVEL", "INFO")
logger.remove()
logger.add(sink=sys.stderr, level=log_level)