import os
import sys
from core.models.statespace import Mamba, MambaConfig
from typing import TypeAlias, TypeVar
from loguru import logger
from core.utils.utils import (
    check_layer,
    check_shape,
    check_shapes,
    _check_nan_inf as check_inf,
)

MambaBlock: TypeAlias = TypeVar("MambaBlock", bound=Mamba)
# Set the minimum log level based on an environment variable
log_level = os.environ.get("LOG_LEVEL", "INFO")
logger.remove()
logger.add(sink=sys.stderr, level=log_level)
