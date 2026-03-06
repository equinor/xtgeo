"""XTGeo metadata package."""

from .metadata import (
    MetaDataCPGeometry,
    MetaDataCPProperty,
    MetaDataRegularCube,
    MetaDataRegularSurface,
    MetaDataTriangulatedSurface,
    MetaDataWell,
)

__all__ = [
    "MetaDataRegularCube",
    "MetaDataRegularSurface",
    "MetaDataCPGeometry",
    "MetaDataTriangulatedSurface",
    "MetaDataCPProperty",
    "MetaDataWell",
]
