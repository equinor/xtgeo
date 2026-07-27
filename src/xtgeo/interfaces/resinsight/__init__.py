"""XTGeo interface.resinsight package."""

from ._rips_package import PropertyDataType, PropertyType, rips
from .rips_utils import RipsApiUtils

__all__ = [
    "PropertyDataType",
    "PropertyType",
    "RipsApiUtils",
    "rips",
]
