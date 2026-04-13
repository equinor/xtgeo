"""ResInsight API helpers for polygon data via rips."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from xtgeo.common._xyz_enum import _AttrName
from xtgeo.common.log import null_logger

from ._resinsight_base import _BaseResInsightDataRW

if TYPE_CHECKING:
    from xtgeo.xyz.polygons import Polygons

    from ._rips_package import ResInsightInstanceOrPortType

logger = null_logger(__name__)


@dataclass(frozen=True, eq=False)
class PolygonDataResInsight:
    """Immutable data container for a single ResInsight polygon.

    Each instance corresponds to one ``rips.Polygon`` object, which holds a name
    and an ordered list of ``[x, y, z]`` coordinate triples.  When an XTGeo
    :class:`~xtgeo.xyz.polygons.Polygons` contains multiple segments (POLY_ID
    groups), each segment maps to its own :class:`PolygonDataResInsight` instance.
    """

    name: str
    coordinates: list[list[float]]

    # Unhashable because coordinates is a mutable list.
    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PolygonDataResInsight):
            return NotImplemented
        return self.name == other.name and self.coordinates == other.coordinates

    def __post_init__(self) -> None:
        if not self.coordinates:
            raise ValueError("coordinates cannot be empty")
        for i, pt in enumerate(self.coordinates):
            if len(pt) != 3:
                raise ValueError(
                    f"Each coordinate must have 3 elements [x, y, z], "
                    f"but point {i} has {len(pt)} elements"
                )

    def to_xtgeo_polygons(self) -> Polygons:
        """Convert to an XTGeo :class:`~xtgeo.xyz.polygons.Polygons` object.

        The returned object contains a single polygon segment (POLY_ID = 0).

        Returns:
            A :class:`~xtgeo.xyz.polygons.Polygons` instance with ``name`` set
            to :attr:`name` and one segment with the stored coordinates.
        """
        from xtgeo.xyz.polygons import Polygons

        coords = np.asarray(self.coordinates, dtype=np.float64)
        df = pd.DataFrame(
            {
                _AttrName.XNAME.value: coords[:, 0],
                _AttrName.YNAME.value: coords[:, 1],
                _AttrName.ZNAME.value: coords[:, 2],
                _AttrName.PNAME.value: np.zeros(len(coords), dtype=np.int32),
            }
        )
        return Polygons(values=df, name=self.name)

    @classmethod
    def from_xtgeo_polygons(
        cls, polygons: Polygons, poly_id: int | None = None
    ) -> PolygonDataResInsight:
        """Create from a single segment in an XTGeo Polygons object.

        Args:
            polygons: The source XTGeo :class:`~xtgeo.xyz.polygons.Polygons`.
            poly_id: The ``POLY_ID`` value identifying which segment to extract.
                Defaults to ``None``, which selects the first segment present in
                the dataframe (by ascending ``POLY_ID``).  Pass an explicit
                integer to select a specific segment.

        Returns:
            A :class:`PolygonDataResInsight` whose :attr:`name` is taken from
            ``polygons.name`` and whose :attr:`coordinates` are the ``[x, y, z]``
            rows of the requested segment.

        Raises:
            ValueError: If the dataframe is empty or no segment with the given
                ``poly_id`` is found.
        """
        df = polygons.get_dataframe(copy=False)
        if df.empty:
            raise ValueError(f"Polygons object '{polygons.name}' contains no data")
        if poly_id is None:
            poly_id = int(df[polygons.pname].min())
        mask = df[polygons.pname] == poly_id
        segment = df[mask]
        if segment.empty:
            raise ValueError(
                f"No polygon segment with POLY_ID={poly_id} found in '{polygons.name}'"
            )
        coords = segment[[polygons.xname, polygons.yname, polygons.zname]].values
        return cls(name=polygons.name, coordinates=coords.tolist())

    @classmethod
    def from_xtgeo_polygons_all(cls, polygons: Polygons) -> list[PolygonDataResInsight]:
        """Create one instance per POLY_ID segment in an XTGeo Polygons object.

        Polygon names are always formed as ``"{polygons.name}_{poly_id}"`` to
        guarantee stable, unique names regardless of how many segments are present.
        This ensures that re-exporting a polygon object with a different number of
        segments does not leave orphaned polygons in ResInsight under stale names.

        Args:
            polygons: The source XTGeo :class:`~xtgeo.xyz.polygons.Polygons`.

        Returns:
            A list of :class:`PolygonDataResInsight` objects, one per segment,
            ordered by ascending ``POLY_ID``.
        """
        df = polygons.get_dataframe(copy=False)
        pname = polygons.pname
        xname, yname, zname = polygons.xname, polygons.yname, polygons.zname
        poly_ids = sorted(df[pname].unique())

        result: list[PolygonDataResInsight] = []
        for poly_id in poly_ids:
            segment = df[df[pname] == poly_id]
            coords = segment[[xname, yname, zname]].values.tolist()
            result.append(cls(name=f"{polygons.name}_{poly_id}", coordinates=coords))
        return result


class PolygonReader(_BaseResInsightDataRW):
    """Read polygon data from ResInsight using rips."""

    def __init__(
        self,
        instance_or_port: ResInsightInstanceOrPortType | None = None,
    ) -> None:
        super().__init__(instance_or_port)

    def load(self, polygon_name: str, find_last: bool = True) -> PolygonDataResInsight:
        """Load a polygon by name from the ResInsight project.

        Args:
            polygon_name: Name of the polygon to load.
            find_last: When multiple polygons share the same name, select the
                last match if ``True`` (default) or the first match if ``False``.

        Returns:
            A :class:`PolygonDataResInsight` populated with the polygon's name
            and coordinates.

        Raises:
            RuntimeError: If no polygon with ``polygon_name`` is found.
        """
        collection = self.get_polygon_collection()
        polygon = self.find_polygon(collection, polygon_name, find_last)
        if polygon is None:
            raise RuntimeError(f"Cannot find any polygon with name '{polygon_name}'")
        return PolygonDataResInsight(
            name=polygon.name,
            coordinates=list(polygon.coordinates),
        )

    def load_all(self) -> list[PolygonDataResInsight]:
        """Load all polygons from the ResInsight project.

        Returns:
            A list of :class:`PolygonDataResInsight` objects, one per polygon
            in the project's polygon collection.
        """
        collection = self.get_polygon_collection()
        return [
            PolygonDataResInsight(name=p.name, coordinates=list(p.coordinates))
            for p in collection.polygons()
        ]


class PolygonWriter(_BaseResInsightDataRW):
    """Write polygon data to ResInsight using rips."""

    def __init__(
        self,
        instance_or_port: ResInsightInstanceOrPortType | None = None,
    ) -> None:
        super().__init__(instance_or_port)

    def save(self, data: PolygonDataResInsight, find_last: bool = True) -> None:
        """Save a polygon to ResInsight.

        If a polygon with the same name already exists in the project's polygon
        collection, its coordinates are updated in place.  Otherwise a new
        polygon is created.

        Args:
            data: The polygon data to save.
            find_last: When multiple polygons share the same name, replace the
                last match if ``True`` (default) or the first match if ``False``.
        """
        try:
            collection = self.get_polygon_collection()
            existing = self.find_polygon(collection, data.name, find_last)
            if existing is not None:
                logger.debug(
                    "Found existing polygon named '%s'. Updating its coordinates.",
                    data.name,
                )
                existing.coordinates = data.coordinates
                existing.update()
            else:
                logger.debug(
                    "No existing polygon named '%s' found. Creating a new one.",
                    data.name,
                )
                collection.create_polygon(name=data.name, coordinates=data.coordinates)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to save ResInsight polygon data: {exc}"
            ) from exc
