# -*- coding: utf-8 -*-
"""Abstract data provider base for RESQML data access.

This defines the interface that both EPC file and ETP protocol backends implement.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import numpy as np


class ResqmlDataProvider(abc.ABC):
    """Abstract base for reading/writing RESQML data objects.

    Concrete implementations:
      - EpcFileProvider: reads/writes EPC+H5 file pairs on disk
      - EtpProvider: reads/writes via ETP 1.2 websocket protocol to RDDMS
    """

    @abc.abstractmethod
    def open(self) -> None:
        """Open/connect to the data source."""

    @abc.abstractmethod
    def close(self) -> None:
        """Close/disconnect from the data source."""

    def __enter__(self):
        """Open the provider and return it as a context manager."""
        self.open()
        return self

    def __exit__(self, *exc):
        """Close the provider on context exit."""
        self.close()

    # ---- Discovery ----

    @abc.abstractmethod
    def list_objects(self, object_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available data objects, optionally filtered by type.

        Returns list of dicts with at least 'uuid', 'title', 'type' keys.
        """

    # ---- IJK Grid ----

    @abc.abstractmethod
    def get_ijk_grid_geometry(self, uuid: str) -> Dict[str, Any]:
        """Read IJK grid geometry.

        Returns dict with keys:
          - ni, nj, nk: int dimensions
          - coord: np.ndarray shape (nj+1, ni+1, 6) pillar coordinates
          - zcorn: np.ndarray shape (nk, nj, ni, 8) corner depths
          - actnum: np.ndarray shape (nk*nj*ni,) active cell mask
          - k_direction: str ("down" or "up")
          - crs_uuid: str
        """

    @abc.abstractmethod
    def put_ijk_grid_geometry(
        self,
        uuid: str,
        title: str,
        ni: int,
        nj: int,
        nk: int,
        coord: np.ndarray,
        zcorn: np.ndarray,
        actnum: np.ndarray,
        crs_uuid: str,
        k_direction: str = "down",
    ) -> str:
        """Write IJK grid geometry. Returns UUID of created object."""

    # ---- Grid2D (Surface) ----

    @abc.abstractmethod
    def get_grid2d_geometry(self, uuid: str) -> Dict[str, Any]:
        """Read Grid2D representation (regular surface).

        Returns dict with keys:
          - ni, nj: int dimensions
          - origin_x, origin_y: float
          - di, dj: float increments
          - rotation: float (radians)
          - values: np.ndarray shape (nj, ni) z-values
          - crs_uuid: str
          - title: str
          - interpretation_uuid: str (UUID of linked HorizonInterpretation, or "")
        """

    @abc.abstractmethod
    def put_grid2d_geometry(
        self,
        uuid: str,
        title: str,
        ni: int,
        nj: int,
        origin_x: float,
        origin_y: float,
        di: float,
        dj: float,
        rotation: float,
        values: np.ndarray,
        crs_uuid: str,
        interpretation_uuid: Optional[str] = None,
    ) -> str:
        """Write Grid2D representation. Returns UUID."""

    # ---- PointSet ----

    @abc.abstractmethod
    def get_pointset(self, uuid: str) -> Dict[str, Any]:
        """Read PointSet representation.

        Returns dict with keys:
          - points: np.ndarray shape (N, 3) XYZ coordinates
          - crs_uuid: str
        """

    @abc.abstractmethod
    def put_pointset(
        self,
        uuid: str,
        title: str,
        points: np.ndarray,
        crs_uuid: str,
    ) -> str:
        """Write PointSet representation. Returns UUID."""

    # ---- PolylineSet (Polygons) ----

    @abc.abstractmethod
    def get_polylineset(self, uuid: str) -> Dict[str, Any]:
        """Read PolylineSet representation.

        Returns dict with keys:
          - polylines: list of np.ndarray, each shape (M, 3)
          - closed: list of bool, whether each polyline is closed
          - crs_uuid: str
          - title: str
          - interpretation_uuid: str (UUID of linked FaultInterpretation, or "")
        """

    @abc.abstractmethod
    def put_polylineset(
        self,
        uuid: str,
        title: str,
        polylines: List[np.ndarray],
        closed: List[bool],
        crs_uuid: str,
        interpretation_uuid: Optional[str] = None,
        line_role: Optional[str] = None,
    ) -> str:
        """Write PolylineSet representation. Returns UUID."""

    @abc.abstractmethod
    def get_fault_interpretation(self, uuid: str) -> Dict[str, Any]:
        """Read a FaultInterpretation and its referenced BoundaryFeature.

        Returns dict with keys:
          - title: str
          - feature_uuid: str
          - feature_title: str (the fault name from BoundaryFeature)
        """

    @abc.abstractmethod
    def get_horizon_interpretation(self, uuid: str) -> Dict[str, Any]:
        """Read a HorizonInterpretation and its referenced GeneticBoundaryFeature.

        Returns dict with keys:
          - title: str
          - feature_uuid: str
          - feature_title: str (the horizon name from GeneticBoundaryFeature)
        """

    # ---- Properties ----

    @abc.abstractmethod
    def get_property_values(self, uuid: str) -> Dict[str, Any]:
        """Read property values.

        Returns dict with keys:
          - values: np.ndarray (flat array of cell values)
          - title: str
          - property_kind: str
          - indexable_element: str
          - supporting_representation_uuid: str
          - is_discrete: bool
          - uom: str (unit of measure)
          - facet: optional str
        """

    @abc.abstractmethod
    def put_property_values(
        self,
        uuid: str,
        title: str,
        values: np.ndarray,
        supporting_representation_uuid: str,
        property_kind: str,
        indexable_element: str = "cells",
        is_discrete: bool = False,
        uom: str = "",
        facet: Optional[str] = None,
    ) -> str:
        """Write property values. Returns UUID."""

    # ---- CRS ----

    @abc.abstractmethod
    def get_crs(self, uuid: str) -> Dict[str, Any]:
        """Read CRS definition.

        Returns dict with keys matching LocalDepth3dCrs fields.
        """

    @abc.abstractmethod
    def put_crs(
        self,
        uuid: str,
        title: str,
        origin_x: float,
        origin_y: float,
        origin_z: float,
        areal_rotation: float,
        z_increasing_downward: bool,
        projected_crs_epsg: Optional[int] = None,
        vertical_crs_epsg: Optional[int] = None,
    ) -> str:
        """Write CRS definition. Returns UUID."""

    # ---- TriangulatedSet ----

    @abc.abstractmethod
    def get_triangulated_set(self, uuid: str) -> Dict[str, Any]:
        """Read TriangulatedSetRepresentation.

        Returns dict with keys:
          - vertices: np.ndarray shape (N, 3) XYZ coordinates
          - triangles: np.ndarray shape (M, 3) 0-based vertex indices
          - crs_uuid: str
          - title: str
        """

    @abc.abstractmethod
    def put_triangulated_set(
        self,
        uuid: str,
        title: str,
        vertices: "np.ndarray",
        triangles: "np.ndarray",
        crs_uuid: str,
    ) -> str:
        """Write TriangulatedSetRepresentation. Returns UUID."""

    # ---- WellboreTrajectory ----

    @abc.abstractmethod
    def get_wellbore_trajectory(self, uuid: str) -> Dict[str, Any]:
        """Read WellboreTrajectoryRepresentation.

        Returns dict with keys:
          - md: np.ndarray shape (N,) measured depths
          - xyz: np.ndarray shape (N, 3) XYZ coordinates
          - crs_uuid: str
          - title: str
          - frames: list of dict, each with:
              - uuid: str
              - md: np.ndarray
              - properties: list of dict with uuid, title, values, is_discrete
        """

    @abc.abstractmethod
    def put_wellbore_trajectory(
        self,
        uuid: str,
        title: str,
        md: "np.ndarray",
        xyz: "np.ndarray",
        crs_uuid: str,
    ) -> str:
        """Write WellboreTrajectoryRepresentation. Returns UUID."""

    @abc.abstractmethod
    def put_wellbore_frame(
        self,
        uuid: str,
        title: str,
        trajectory_uuid: str,
        md: "np.ndarray",
        properties: List[Dict[str, Any]],
        crs_uuid: str,
    ) -> str:
        """Write WellboreFrameRepresentation with properties. Returns UUID."""

    # ---- BlockedWellbore ----

    @abc.abstractmethod
    def get_blocked_wellbore(self, uuid: str) -> Dict[str, Any]:
        """Read BlockedWellboreRepresentation.

        Returns dict with keys:
          - md: np.ndarray shape (N,)
          - xyz: np.ndarray shape (N, 3)
          - cell_indices: np.ndarray shape (N, 3) — I, J, K
          - crs_uuid: str
          - title: str
          - grid_uuid: str
          - trajectory_uuid: str
          - properties: list of dict with uuid, title, values, is_discrete
        """

    @abc.abstractmethod
    def put_blocked_wellbore(
        self,
        uuid: str,
        title: str,
        trajectory_uuid: str,
        grid_uuid: str,
        md: "np.ndarray",
        xyz: "np.ndarray",
        cell_indices: "np.ndarray",
        properties: List[Dict[str, Any]],
        crs_uuid: str,
    ) -> str:
        """Write BlockedWellboreRepresentation. Returns UUID."""
