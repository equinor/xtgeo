# -*- coding: utf-8 -*-
"""RESQML 2.0.1 enums and constants used across the OSDU interface."""

from __future__ import annotations

from enum import Enum

# ---------------------------------------------------------------------------
# RESQML 2.0.1 XML namespaces
# ---------------------------------------------------------------------------
NS_EPC = "http://www.energistics.org/2014/06/02/epc"
NS_RESQML20 = "http://www.energistics.org/energyml/data/resqmlv2"
NS_COMMON20 = "http://www.energistics.org/energyml/data/commonv2"
NS_XSI = "http://www.w3.org/2001/XMLSchema-instance"
NS_CONTENT_TYPES = "http://schemas.openxmlformats.org/package/2006/content-types"
NS_RELS = "http://schemas.openxmlformats.org/package/2006/relationships"

RESQML_NS_MAP = {
    "resqml": NS_RESQML20,
    "eml": NS_COMMON20,
    "xsi": NS_XSI,
}


# ---------------------------------------------------------------------------
# RESQML data object qualified types (for EPC content types and ETP URIs)
# ---------------------------------------------------------------------------
class ResqmlObjectType(str, Enum):
    """Qualified type identifiers for RESQML 2.0.1 data objects."""

    IJK_GRID_REPRESENTATION = "resqml20.obj_IjkGridRepresentation"
    GRID2D_REPRESENTATION = "resqml20.obj_Grid2dRepresentation"
    TRIANGULATED_SET_REPRESENTATION = "resqml20.obj_TriangulatedSetRepresentation"
    POINT_SET_REPRESENTATION = "resqml20.obj_PointSetRepresentation"
    POLYLINE_SET_REPRESENTATION = "resqml20.obj_PolylineSetRepresentation"
    WELLBORE_TRAJECTORY_REPRESENTATION = (
        "resqml20.obj_WellboreTrajectoryRepresentation"
    )
    WELLBORE_FRAME_REPRESENTATION = "resqml20.obj_WellboreFrameRepresentation"
    BLOCKED_WELLBORE_REPRESENTATION = "resqml20.obj_BlockedWellboreRepresentation"
    WELLBORE_FEATURE = "resqml20.obj_WellboreFeature"
    WELLBORE_INTERPRETATION = "resqml20.obj_WellboreInterpretation"
    CONTINUOUS_PROPERTY = "resqml20.obj_ContinuousProperty"
    DISCRETE_PROPERTY = "resqml20.obj_DiscreteProperty"
    CATEGORICAL_PROPERTY = "resqml20.obj_CategoricalProperty"
    LOCAL_DEPTH3D_CRS = "resqml20.obj_LocalDepth3dCrs"
    LOCAL_TIME3D_CRS = "resqml20.obj_LocalTime3dCrs"
    EPC_EXTERNAL_PART_REFERENCE = "eml20.obj_EpcExternalPartReference"


# ---------------------------------------------------------------------------
# Geometry enums
# ---------------------------------------------------------------------------
class KDirection(str, Enum):
    """K-layer direction in an IJK grid."""

    DOWN = "down"
    UP = "up"
    NOT_MONOTONIC = "not monotonic"


class CellShape(str, Enum):
    """Cell geometry shape for IJK grids."""

    HEXAHEDRAL = "hexahedral"
    TETRAHEDRAL = "tetrahedral"
    POLYHEDRAL = "polyhedral"


class Handedness(str, Enum):
    """Grid coordinate system handedness."""

    LEFT = "left"
    RIGHT = "right"


class IndexableElement(str, Enum):
    """Topological element that a property is indexed over."""

    CELLS = "cells"
    NODES = "nodes"
    FACES = "faces"
    COLUMNS = "columns"
    PILLARS = "pillars"
    LAYERS = "layers"


class PropertyKind(str, Enum):
    """Well-known RESQML property kinds."""

    POROSITY = "porosity"
    PERMEABILITY_ROCK = "permeability rock"
    NET_TO_GROSS_RATIO = "net to gross ratio"
    PRESSURE = "pressure"
    WATER_SATURATION = "water saturation"
    OIL_SATURATION = "oil saturation"
    GAS_SATURATION = "gas saturation"
    DEPTH = "depth"
    THICKNESS = "thickness"
    VOLUME = "volume"
    ROCK_VOLUME = "rock volume"
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"


# ---------------------------------------------------------------------------
# EPC content type mappings
# ---------------------------------------------------------------------------
CONTENT_TYPE_MAP = {
    ResqmlObjectType.IJK_GRID_REPRESENTATION: (
        "application/x-resqml+xml;version=2.0;type=obj_IjkGridRepresentation"
    ),
    ResqmlObjectType.GRID2D_REPRESENTATION: (
        "application/x-resqml+xml;version=2.0;type=obj_Grid2dRepresentation"
    ),
    ResqmlObjectType.TRIANGULATED_SET_REPRESENTATION: (
        "application/x-resqml+xml;version=2.0;type=obj_TriangulatedSetRepresentation"
    ),
    ResqmlObjectType.POINT_SET_REPRESENTATION: (
        "application/x-resqml+xml;version=2.0;type=obj_PointSetRepresentation"
    ),
    ResqmlObjectType.POLYLINE_SET_REPRESENTATION: (
        "application/x-resqml+xml;version=2.0;type=obj_PolylineSetRepresentation"
    ),
    ResqmlObjectType.WELLBORE_TRAJECTORY_REPRESENTATION: (
        "application/x-resqml+xml;version=2.0;type=obj_WellboreTrajectoryRepresentation"
    ),
    ResqmlObjectType.WELLBORE_FRAME_REPRESENTATION: (
        "application/x-resqml+xml;version=2.0;type=obj_WellboreFrameRepresentation"
    ),
    ResqmlObjectType.BLOCKED_WELLBORE_REPRESENTATION: (
        "application/x-resqml+xml;version=2.0;type=obj_BlockedWellboreRepresentation"
    ),
    ResqmlObjectType.WELLBORE_FEATURE: (
        "application/x-resqml+xml;version=2.0;type=obj_WellboreFeature"
    ),
    ResqmlObjectType.WELLBORE_INTERPRETATION: (
        "application/x-resqml+xml;version=2.0;type=obj_WellboreInterpretation"
    ),
    ResqmlObjectType.CONTINUOUS_PROPERTY: (
        "application/x-resqml+xml;version=2.0;type=obj_ContinuousProperty"
    ),
    ResqmlObjectType.DISCRETE_PROPERTY: (
        "application/x-resqml+xml;version=2.0;type=obj_DiscreteProperty"
    ),
    ResqmlObjectType.CATEGORICAL_PROPERTY: (
        "application/x-resqml+xml;version=2.0;type=obj_CategoricalProperty"
    ),
    ResqmlObjectType.LOCAL_DEPTH3D_CRS: (
        "application/x-resqml+xml;version=2.0;type=obj_LocalDepth3dCrs"
    ),
    ResqmlObjectType.EPC_EXTERNAL_PART_REFERENCE: (
        "application/x-eml+xml;version=2.0;type=obj_EpcExternalPartReference"
    ),
}
