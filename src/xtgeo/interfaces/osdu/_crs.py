# -*- coding: utf-8 -*-
"""CRS handling utilities for RESQML 2.0.1 local/global coordinate systems."""

from __future__ import annotations

import math
import uuid as _uuid
from dataclasses import dataclass, field
from typing import Optional, Tuple

from lxml import etree

from ._resqml_enums import NS_COMMON20, NS_RESQML20, NS_XSI, RESQML_NS_MAP


@dataclass
class LocalDepth3dCrs:
    """Represents a RESQML LocalDepth3dCrs object."""

    uuid: str = field(default_factory=lambda: str(_uuid.uuid4()))
    title: str = "Local CRS"
    origin_x: float = 0.0
    origin_y: float = 0.0
    origin_z: float = 0.0
    areal_rotation: float = 0.0  # radians
    projected_crs_epsg: Optional[int] = None
    vertical_crs_epsg: Optional[int] = None
    z_increasing_downward: bool = True
    xy_unit: str = "m"
    z_unit: str = "m"

    @property
    def rotation_degrees(self) -> float:
        """Return areal rotation in degrees."""
        return math.degrees(self.areal_rotation)

    def compute_mapaxes(
        self,
    ) -> Tuple[
        Tuple[float, float],
        Tuple[float, float],
        Tuple[float, float],
    ]:
        """Compute GRDECL MAPAXES from CRS origin and rotation."""
        ct = math.cos(self.areal_rotation)
        st = math.sin(self.areal_rotation)
        p1 = (self.origin_x, self.origin_y)
        p2 = (self.origin_x + ct, self.origin_y + st)
        p3 = (self.origin_x - st, self.origin_y + ct)
        return (p1, p2, p3)

    def local_to_global(
        self, x: float, y: float, z: float
    ) -> Tuple[float, float, float]:
        """Convert local coordinates to global."""
        ct = math.cos(self.areal_rotation)
        st = math.sin(self.areal_rotation)
        gx = self.origin_x + x * ct - y * st
        gy = self.origin_y + x * st + y * ct
        gz = self.origin_z + (z if self.z_increasing_downward else -z)
        return (gx, gy, gz)

    def global_to_local(
        self, gx: float, gy: float, gz: float
    ) -> Tuple[float, float, float]:
        """Convert global coordinates to local."""
        ct = math.cos(self.areal_rotation)
        st = math.sin(self.areal_rotation)
        dx = gx - self.origin_x
        dy = gy - self.origin_y
        x = dx * ct + dy * st
        y = -dx * st + dy * ct
        z = (
            (gz - self.origin_z)
            if self.z_increasing_downward
            else -(gz - self.origin_z)
        )
        return (x, y, z)

    def to_xml(self) -> etree._Element:
        """Serialize to RESQML 2.0.1 XML element."""
        root = etree.Element(
            f"{{{NS_RESQML20}}}LocalDepth3dCrs",
            nsmap=RESQML_NS_MAP,
        )
        root.set("uuid", self.uuid)
        root.set("schemaVersion", "2.0")
        root.set(f"{{{NS_XSI}}}type", "resqml2:obj_LocalDepth3dCrs")

        citation = etree.SubElement(root, f"{{{NS_COMMON20}}}Citation")
        title_el = etree.SubElement(citation, f"{{{NS_COMMON20}}}Title")
        title_el.text = self.title

        # UOMs (required by resqpy)
        etree.SubElement(
            root, f"{{{NS_RESQML20}}}ProjectedUom"
        ).text = self.xy_unit
        etree.SubElement(
            root, f"{{{NS_RESQML20}}}VerticalUom"
        ).text = self.z_unit
        etree.SubElement(
            root, f"{{{NS_RESQML20}}}ProjectedAxisOrder"
        ).text = "easting northing"

        # Origin
        for tag, val in [
            ("XOffset", self.origin_x),
            ("YOffset", self.origin_y),
            ("ZOffset", self.origin_z),
        ]:
            el = etree.SubElement(root, f"{{{NS_RESQML20}}}{tag}")
            el.text = str(val)

        # Rotation
        rot_el = etree.SubElement(root, f"{{{NS_RESQML20}}}ArealRotation")
        rot_el.text = str(self.areal_rotation)
        rot_el.set("uom", "rad")

        # Z direction
        z_dir = etree.SubElement(root, f"{{{NS_RESQML20}}}ZIncreasingDownward")
        z_dir.text = str(self.z_increasing_downward).lower()

        # Projected CRS
        if self.projected_crs_epsg:
            proj = etree.SubElement(root, f"{{{NS_RESQML20}}}ProjectedCrs")
            proj.set(f"{{{NS_XSI}}}type", "eml:ProjectedCrsEpsgCode")
            epsg_el = etree.SubElement(proj, f"{{{NS_COMMON20}}}EpsgCode")
            epsg_el.text = str(self.projected_crs_epsg)

        # Vertical CRS
        if self.vertical_crs_epsg:
            vert = etree.SubElement(root, f"{{{NS_RESQML20}}}VerticalCrs")
            vert.set(f"{{{NS_XSI}}}type", "eml:VerticalCrsEpsgCode")
            epsg_el = etree.SubElement(vert, f"{{{NS_COMMON20}}}EpsgCode")
            epsg_el.text = str(self.vertical_crs_epsg)

        return root

    @classmethod
    def from_xml(cls, root: etree._Element) -> "LocalDepth3dCrs":
        """Deserialize from RESQML XML element."""
        uid = root.get("uuid", str(_uuid.uuid4()))

        def _text(
            parent: etree._Element, tag: str, ns: str = NS_RESQML20
        ) -> Optional[str]:
            el = parent.find(f"{{{ns}}}{tag}")
            if el is not None:
                return el.text
            return None

        title = ""
        citation = root.find(f"{{{NS_COMMON20}}}Citation")
        if citation is not None:
            t = citation.find(f"{{{NS_COMMON20}}}Title")
            if t is not None:
                title = t.text or ""

        origin_x = float(_text(root, "XOffset") or 0.0)
        origin_y = float(_text(root, "YOffset") or 0.0)
        origin_z = float(_text(root, "ZOffset") or 0.0)

        rot_el = root.find(f"{{{NS_RESQML20}}}ArealRotation")
        areal_rotation = (
            float(rot_el.text) if rot_el is not None and rot_el.text else 0.0
        )

        z_down_el = root.find(f"{{{NS_RESQML20}}}ZIncreasingDownward")
        z_down = True
        if z_down_el is not None and z_down_el.text:
            z_down = z_down_el.text.lower() == "true"

        projected_epsg = None
        proj = root.find(f"{{{NS_RESQML20}}}ProjectedCrs")
        if proj is not None:
            epsg_el = proj.find(f"{{{NS_COMMON20}}}EpsgCode")
            if epsg_el is not None and epsg_el.text:
                projected_epsg = int(epsg_el.text)

        vertical_epsg = None
        vert = root.find(f"{{{NS_RESQML20}}}VerticalCrs")
        if vert is not None:
            epsg_el = vert.find(f"{{{NS_COMMON20}}}EpsgCode")
            if epsg_el is not None and epsg_el.text:
                vertical_epsg = int(epsg_el.text)

        return cls(
            uuid=uid,
            title=title,
            origin_x=origin_x,
            origin_y=origin_y,
            origin_z=origin_z,
            areal_rotation=areal_rotation,
            projected_crs_epsg=projected_epsg,
            vertical_crs_epsg=vertical_epsg,
            z_increasing_downward=z_down,
        )
