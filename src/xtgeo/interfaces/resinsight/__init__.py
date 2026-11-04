"""XTGeo interface.resinsight package."""

from typing_extensions import Final

from ._rips_package import PropertyDataType, PropertyType, rips
from .rips_utils import RipsApiUtils

SUBGRIDS_PROPERTY_NAME: Final[str] = "SUBGRIDS"


__all__ = [
    "PropertyDataType",
    "PropertyType",
    "RipsApiUtils",
    "rips",
    "SUBGRIDS_PROPERTY_NAME",
]
