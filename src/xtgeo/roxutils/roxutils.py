"""Compatibility shim for RoxUtils.

The implementation has moved to ``xtgeo.interfaces.rms.rmsapi_utils``.
Keep this module to preserve existing import paths.
"""

import warnings
from typing import Any

from xtgeo.interfaces.rms.rmsapi_utils import RmsApiUtils


class RoxUtils(RmsApiUtils):
    """Deprecated: Use RmsApiUtils instead."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "RoxUtils is deprecated and will be removed in a future version. "
            "Use RmsApiUtils instead:\n"
            "from xtgeo.interfaces.rms import RmsApiUtils",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
