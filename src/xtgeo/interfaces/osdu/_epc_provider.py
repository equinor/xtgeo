# -*- coding: utf-8 -*-
"""EPC+HDF5 file container provider for RESQML 2.0.1.

The EPC format is a ZIP-based Open Packaging Convention container that holds:
  - XML parts describing RESQML data objects
  - Relationships between parts
  - Content types
  - An associated HDF5 file for bulk array data

This implementation is self-contained (no resqpy dependency) and inspired by
the resqpy Model approach for EPC structure, but reimplemented from scratch.
"""

from __future__ import annotations

import logging
import pathlib
import uuid as _uuid
import zipfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import h5py
import numpy as np
from lxml import etree

from ._crs import LocalDepth3dCrs
from ._provider_base import ResqmlDataProvider
from ._resqml_enums import (
    CONTENT_TYPE_MAP,
    NS_COMMON20,
    NS_CONTENT_TYPES,
    NS_RELS,
    NS_RESQML20,
    RESQML_NS_MAP,
    ResqmlObjectType,
)

if TYPE_CHECKING:
    import os

logger = logging.getLogger(__name__)


def _make_uuid() -> str:
    return str(_uuid.uuid4())


def _part_name(obj_type: str, uid: str) -> str:
    """Generate EPC part name from type and uuid."""
    # e.g. obj_IjkGridRepresentation_12345678-...xml
    type_short = obj_type.split(".")[-1] if "." in obj_type else obj_type
    return f"{type_short}_{uid}.xml"


def _hdf5_dataset_path(uuid: str, tag: str) -> str:
    """Standard HDF5 dataset path for a data object array."""
    return f"/RESQML/{uuid}/{tag}"


class EpcFileProvider(ResqmlDataProvider):
    """Read/write RESQML 2.0.1 data via EPC+HDF5 file pairs.

    Parameters
    ----------
    epc_path : str or Path
        Path to the .epc file. The companion .h5 file is assumed to be
        alongside with the same stem.
    mode : str
        'r' for read, 'w' for write (create new), 'a' for append.
    """

    def __init__(self, epc_path: str | os.PathLike, mode: str = "r"):
        self._epc_path = pathlib.Path(epc_path)
        self._h5_path = self._epc_path.with_suffix(".h5")
        self._mode = mode
        self._zip: Optional[zipfile.ZipFile] = None
        self._h5: Optional[h5py.File] = None
        self._parts: Dict[str, etree._Element] = {}  # part_name -> parsed XML
        self._content_types: Dict[str, str] = {}  # part_name -> content_type
        self._relationships: Dict[str, List[Dict[str, str]]] = {}  # source -> rels
        self._uuids: Dict[str, str] = {}  # uuid -> part_name

    def open(self) -> None:
        """Open the EPC archive and associated HDF5 file."""
        if self._mode == "r":
            if not self._epc_path.exists():
                raise FileNotFoundError(f"EPC file not found: {self._epc_path}")
            self._zip = zipfile.ZipFile(self._epc_path, "r")
            self._load_epc_contents()
            if self._h5_path.exists():
                self._h5 = h5py.File(self._h5_path, "r")
        elif self._mode in ("w", "a"):
            if self._mode == "a" and self._epc_path.exists():
                self._zip = zipfile.ZipFile(self._epc_path, "a")
                self._load_epc_contents()
            else:
                self._zip = zipfile.ZipFile(self._epc_path, "w")
            self._h5 = h5py.File(self._h5_path, self._mode)
        else:
            raise ValueError(f"Unsupported mode: {self._mode}")

    def close(self) -> None:
        """Flush metadata and close EPC archive and HDF5 file."""
        if self._mode in ("w", "a") and self._zip is not None:
            self._write_epc_metadata()
        if self._zip is not None:
            self._zip.close()
            self._zip = None
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def _load_epc_contents(self) -> None:
        """Parse the EPC zip contents."""
        assert self._zip is not None
        # Content types
        if "[Content_Types].xml" in self._zip.namelist():
            ct_xml = etree.fromstring(self._zip.read("[Content_Types].xml"))
            for override in ct_xml.findall(f"{{{NS_CONTENT_TYPES}}}Override"):
                part = override.get("PartName", "").lstrip("/")
                ctype = override.get("ContentType", "")
                if part:
                    self._content_types[part] = ctype

        # Load all XML parts
        for name in self._zip.namelist():
            if (
                name.endswith(".xml")
                and name != "[Content_Types].xml"
                and not name.startswith("_rels/")
            ):
                try:
                    root = etree.fromstring(self._zip.read(name))
                    self._parts[name] = root
                    uid = root.get("uuid", "")
                    if uid:
                        self._uuids[uid] = name
                except etree.XMLSyntaxError:
                    logger.warning("Skipping malformed XML part: %s", name)

    def _write_epc_metadata(self) -> None:
        """Write content types and relationships to the zip."""
        assert self._zip is not None

        # [Content_Types].xml
        types_root = etree.Element(f"{{{NS_CONTENT_TYPES}}}Types")
        for part_name, ctype in self._content_types.items():
            override = etree.SubElement(types_root, f"{{{NS_CONTENT_TYPES}}}Override")
            override.set("PartName", f"/{part_name}")
            override.set("ContentType", ctype)
        self._zip.writestr(
            "[Content_Types].xml",
            etree.tostring(types_root, xml_declaration=True, encoding="UTF-8"),
        )

        # _rels/.rels (core relationships)
        rels_root = etree.Element(f"{{{NS_RELS}}}Relationships")
        for source, rels_list in self._relationships.items():
            for rel in rels_list:
                rel_el = etree.SubElement(rels_root, f"{{{NS_RELS}}}Relationship")
                for k, v in rel.items():
                    rel_el.set(k, v)
        self._zip.writestr(
            "_rels/.rels",
            etree.tostring(rels_root, xml_declaration=True, encoding="UTF-8"),
        )

    def _add_part(
        self, obj_type: ResqmlObjectType, uid: str, xml_root: etree._Element
    ) -> str:
        """Add an XML part to the EPC container."""
        assert self._zip is not None
        part_name = _part_name(obj_type.value, uid)
        xml_bytes = etree.tostring(
            xml_root, xml_declaration=True, encoding="UTF-8", pretty_print=True
        )
        self._zip.writestr(part_name, xml_bytes)
        self._parts[part_name] = xml_root
        self._uuids[uid] = part_name
        self._content_types[part_name] = CONTENT_TYPE_MAP.get(
            obj_type, "application/xml"
        )
        return part_name

    def _write_hdf5_array(self, uuid: str, tag: str, data: np.ndarray) -> str:
        """Write an array to the companion HDF5 file."""
        assert self._h5 is not None
        path = _hdf5_dataset_path(uuid, tag)
        if path in self._h5:
            del self._h5[path]
        self._h5.create_dataset(path, data=data, compression="gzip")
        return path

    def _read_hdf5_array(self, uuid: str, tag: str) -> Optional[np.ndarray]:
        """Read an array from the companion HDF5 file."""
        if self._h5 is None:
            return None
        path = _hdf5_dataset_path(uuid, tag)
        if path in self._h5:
            return np.array(self._h5[path])
        return None

    def _find_part_by_uuid(self, uuid: str) -> Optional[etree._Element]:
        """Find a parsed XML part by UUID."""
        part_name = self._uuids.get(uuid)
        if part_name:
            return self._parts.get(part_name)
        # Fallback: scan parts
        for pname, root in self._parts.items():
            if root.get("uuid") == uuid:
                self._uuids[uuid] = pname
                return root
        return None

    def _read_hdf5_by_path(self, path: str) -> Optional[np.ndarray]:
        """Read an HDF5 dataset by its full internal path."""
        if self._h5 is None:
            return None
        if not path.startswith("/"):
            path = "/" + path
        if path in self._h5:
            return np.array(self._h5[path])
        return None

    def _read_grid_arrays_from_xml_paths(self, root, uuid: str):
        """Try to read grid arrays by parsing PathInHdfFile elements from XML."""
        coord = None
        zcorn = None
        # Find all PathInHdfFile elements
        for path_el in root.iter(f"{{{NS_COMMON20}}}PathInHdfFile"):
            path_text = path_el.text
            if not path_text:
                continue
            arr = self._read_hdf5_by_path(path_text)
            if arr is None:
                continue
            # Classify by path or shape
            if "point" in path_text.lower() or "coord" in path_text.lower():
                if arr.ndim == 4:
                    # Unified points array (nk+1, nj+1, ni+1, 3)
                    nk1, nj1, ni1, _ = arr.shape
                    coord_out = np.zeros((ni1, nj1, 6), dtype=np.float64)
                    coord_out[:, :, 0:3] = arr[0, :, :, :].transpose(1, 0, 2)
                    coord_out[:, :, 3:6] = arr[-1, :, :, :].transpose(1, 0, 2)
                    coord = coord_out.flatten()
                    z_all = arr[:, :, :, 2].transpose(2, 1, 0)
                    zcorn_out = np.zeros((ni1, nj1, nk1, 4), dtype=np.float32)
                    for c in range(4):
                        zcorn_out[:, :, :, c] = z_all
                    zcorn = zcorn_out.flatten()
                elif arr.ndim == 3:
                    coord = arr.flatten()
        return coord, zcorn

    # ---- Discovery ----

    def list_objects(self, object_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available data objects, optionally filtered by type."""
        results = []
        for part_name, root in self._parts.items():
            tag = etree.QName(root.tag).localname if root.tag else ""
            uid = root.get("uuid", "")

            # Extract title
            title = ""
            citation = root.find(f"{{{NS_COMMON20}}}Citation")
            if citation is not None:
                title_el = citation.find(f"{{{NS_COMMON20}}}Title")
                if title_el is not None:
                    title = title_el.text or ""

            obj_info = {
                "uuid": uid,
                "title": title,
                "type": tag,
                "part_name": part_name,
            }

            if object_type is None or object_type.lower() in tag.lower():
                results.append(obj_info)

        return results

    # ---- IJK Grid ----

    def get_ijk_grid_geometry(self, uuid: str) -> Dict[str, Any]:
        """Read IJK grid geometry from the data source."""
        root = self._find_part_by_uuid(uuid)
        if root is None:
            raise ValueError(f"IJK grid with UUID {uuid} not found in EPC")

        # Dimensions
        ni = int(
            root.findtext(f"{{{NS_RESQML20}}}Ni")
            or root.findtext(f".//{{{NS_RESQML20}}}Ni")
            or 0
        )
        nj = int(
            root.findtext(f"{{{NS_RESQML20}}}Nj")
            or root.findtext(f".//{{{NS_RESQML20}}}Nj")
            or 0
        )
        nk = int(
            root.findtext(f"{{{NS_RESQML20}}}Nk")
            or root.findtext(f".//{{{NS_RESQML20}}}Nk")
            or 0
        )

        # K direction
        k_dir_el = root.find(f".//{{{NS_RESQML20}}}KDirection")
        k_direction = "down"
        if k_dir_el is not None and k_dir_el.text:
            k_direction = k_dir_el.text.lower()

        # CRS reference
        crs_uuid = ""
        crs_ref = root.find(f".//{{{NS_RESQML20}}}LocalCrs")
        if crs_ref is not None:
            crs_uuid = crs_ref.get("uuid", "")
            if not crs_uuid:
                uid_el = crs_ref.find(f"{{{NS_COMMON20}}}UUID")
                if uid_el is not None:
                    crs_uuid = uid_el.text or ""

        # Arrays from HDF5
        coord = self._read_hdf5_array(uuid, "Points/Coordinates")
        zcorn = self._read_hdf5_array(uuid, "Points/ZCorners")
        actnum = self._read_hdf5_array(uuid, "CellGeometryIsDefined")

        # If arrays not found with standard paths, try alternate conventions
        if coord is None:
            coord = self._read_hdf5_array(uuid, "pillar_coordinates")
        if zcorn is None:
            zcorn = self._read_hdf5_array(uuid, "cell_corners")

        # Try resqpy-style unified Points array: shape (nk+1, nj+1, ni+1, 3)
        if coord is None and zcorn is None:
            unified = self._read_hdf5_array(uuid, "Points")
            if unified is not None and unified.ndim == 4:
                # Shape is (nk+1, nj+1, ni+1, 3) — convert to xtgeo layout
                nk1, nj1, ni1, _ = unified.shape
                # coord: (ni+1, nj+1, 6) — pillar top XYZ + bottom XYZ
                coord_out = np.zeros((ni1, nj1, 6), dtype=np.float64)
                coord_out[:, :, 0:3] = unified[0, :, :, :].transpose(
                    1, 0, 2
                )  # top layer
                coord_out[:, :, 3:6] = unified[-1, :, :, :].transpose(
                    1, 0, 2
                )  # bottom layer
                coord = coord_out.flatten()
                # zcorn: (ni+1, nj+1, nk+1, 4) — corner Z at each pillar node
                # From unified (nk+1, nj+1, ni+1, 3), extract Z values
                z_all = unified[:, :, :, 2]  # (nk+1, nj+1, ni+1)
                # Transpose to (ni+1, nj+1, nk+1) for xtgeo
                z_transposed = z_all.transpose(2, 1, 0)  # (ni+1, nj+1, nk+1)
                # Each pillar node needs 4 corner contributions — for regular grids
                # these are all the same Z value
                zcorn_out = np.zeros((ni1, nj1, nk1, 4), dtype=np.float32)
                for c in range(4):
                    zcorn_out[:, :, :, c] = z_transposed
                zcorn = zcorn_out.flatten()

        # Also try reading HDF5 path from XML (PathInHdfFile elements)
        if coord is None and zcorn is None:
            coord, zcorn = self._read_grid_arrays_from_xml_paths(root, uuid)

        if actnum is None:
            actnum = self._read_hdf5_array(uuid, "active_cells")
            if actnum is None:
                # Default: all active
                actnum = np.ones(ni * nj * nk, dtype=np.int32)

        return {
            "ni": ni,
            "nj": nj,
            "nk": nk,
            "coord": coord,
            "zcorn": zcorn,
            "actnum": actnum,
            "k_direction": k_direction,
            "crs_uuid": crs_uuid,
        }

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
        """Write IJK grid geometry to the data source."""
        if not uuid:
            uuid = _make_uuid()

        # Write HDF5 arrays
        self._write_hdf5_array(uuid, "Points/Coordinates", coord.astype(np.float64))
        self._write_hdf5_array(uuid, "Points/ZCorners", zcorn.astype(np.float32))
        self._write_hdf5_array(uuid, "CellGeometryIsDefined", actnum.astype(np.int32))

        # Build XML
        root = etree.Element(
            f"{{{NS_RESQML20}}}IjkGridRepresentation", nsmap=RESQML_NS_MAP
        )
        root.set("uuid", uuid)
        root.set("schemaVersion", "2.0")

        citation = etree.SubElement(root, f"{{{NS_COMMON20}}}Citation")
        title_el = etree.SubElement(citation, f"{{{NS_COMMON20}}}Title")
        title_el.text = title

        etree.SubElement(root, f"{{{NS_RESQML20}}}Ni").text = str(ni)
        etree.SubElement(root, f"{{{NS_RESQML20}}}Nj").text = str(nj)
        etree.SubElement(root, f"{{{NS_RESQML20}}}Nk").text = str(nk)
        etree.SubElement(root, f"{{{NS_RESQML20}}}KDirection").text = k_direction

        # CRS reference
        crs_ref = etree.SubElement(root, f"{{{NS_RESQML20}}}LocalCrs")
        crs_ref.set("uuid", crs_uuid)

        # HDF proxy reference for geometry
        geom = etree.SubElement(root, f"{{{NS_RESQML20}}}Geometry")
        points = etree.SubElement(geom, f"{{{NS_RESQML20}}}Points")
        hdf_ref = etree.SubElement(points, f"{{{NS_COMMON20}}}HdfProxy")
        hdf_ref.set("uuid", _make_uuid())
        path_el = etree.SubElement(points, f"{{{NS_COMMON20}}}PathInHdfFile")
        path_el.text = _hdf5_dataset_path(uuid, "Points/Coordinates")

        self._add_part(ResqmlObjectType.IJK_GRID_REPRESENTATION, uuid, root)
        return uuid

    # ---- Grid2D (Surface) ----

    def get_grid2d_geometry(self, uuid: str) -> Dict[str, Any]:
        """Read Grid2D representation (regular surface)."""
        root = self._find_part_by_uuid(uuid)
        if root is None:
            raise ValueError(f"Grid2D with UUID {uuid} not found in EPC")

        # Dimensions
        ni = int(root.findtext(f".//{{{NS_RESQML20}}}FastestAxisCount") or 0)
        nj = int(root.findtext(f".//{{{NS_RESQML20}}}SlowestAxisCount") or 0)

        # Origin from lattice
        origin_x = float(
            root.findtext(f".//{{{NS_RESQML20}}}Origin/{{{NS_RESQML20}}}Coordinate1")
            or 0.0
        )
        origin_y = float(
            root.findtext(f".//{{{NS_RESQML20}}}Origin/{{{NS_RESQML20}}}Coordinate2")
            or 0.0
        )

        # Extract di, dj and rotation from offset vectors
        import math

        offsets = root.findall(f".//{{{NS_RESQML20}}}Offset")
        di = 1.0
        dj = 1.0
        rotation = 0.0

        if len(offsets) >= 1:
            ox1 = float(offsets[0].findtext(f"{{{NS_RESQML20}}}Coordinate1") or 1.0)
            oy1 = float(offsets[0].findtext(f"{{{NS_RESQML20}}}Coordinate2") or 0.0)
            di = math.sqrt(ox1 * ox1 + oy1 * oy1)
            if di > 0:
                rotation = math.atan2(oy1, ox1)
            # Also try from Spacing element
            sp1 = offsets[0].findtext(
                f"{{{NS_RESQML20}}}Spacing/{{{NS_RESQML20}}}Value"
            )
            if sp1:
                di = float(sp1)

        if len(offsets) >= 2:
            sp2 = offsets[1].findtext(
                f"{{{NS_RESQML20}}}Spacing/{{{NS_RESQML20}}}Value"
            )
            if sp2:
                dj = float(sp2)
            else:
                ox2 = float(offsets[1].findtext(f"{{{NS_RESQML20}}}Coordinate1") or 0.0)
                oy2 = float(offsets[1].findtext(f"{{{NS_RESQML20}}}Coordinate2") or 1.0)
                dj = math.sqrt(ox2 * ox2 + oy2 * oy2)

        # CRS reference
        crs_uuid = ""
        crs_ref = root.find(f".//{{{NS_RESQML20}}}LocalCrs")
        if crs_ref is not None:
            crs_uuid = crs_ref.get("uuid", "")

        # Z values from HDF5
        values = self._read_hdf5_array(uuid, "ZValues")
        if values is None:
            values = self._read_hdf5_array(uuid, "z_values")
        if values is None:
            values = np.zeros((nj, ni), dtype=np.float64)

        return {
            "ni": ni,
            "nj": nj,
            "origin_x": origin_x,
            "origin_y": origin_y,
            "di": di,
            "dj": dj,
            "rotation": rotation,
            "values": values,
            "crs_uuid": crs_uuid,
        }

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
    ) -> str:
        """Write Grid2D representation (regular surface)."""
        if not uuid:
            uuid = _make_uuid()

        self._write_hdf5_array(uuid, "ZValues", values.astype(np.float64))

        root = etree.Element(
            f"{{{NS_RESQML20}}}Grid2dRepresentation", nsmap=RESQML_NS_MAP
        )
        root.set("uuid", uuid)
        root.set("schemaVersion", "2.0")

        citation = etree.SubElement(root, f"{{{NS_COMMON20}}}Citation")
        title_el = etree.SubElement(citation, f"{{{NS_COMMON20}}}Title")
        title_el.text = title

        # Grid2dPatch
        patch = etree.SubElement(root, f"{{{NS_RESQML20}}}Grid2dPatch")
        etree.SubElement(patch, f"{{{NS_RESQML20}}}FastestAxisCount").text = str(ni)
        etree.SubElement(patch, f"{{{NS_RESQML20}}}SlowestAxisCount").text = str(nj)

        # Geometry as lattice
        geom = etree.SubElement(patch, f"{{{NS_RESQML20}}}Geometry")
        origin = etree.SubElement(geom, f"{{{NS_RESQML20}}}Origin")
        etree.SubElement(origin, f"{{{NS_RESQML20}}}Coordinate1").text = str(origin_x)
        etree.SubElement(origin, f"{{{NS_RESQML20}}}Coordinate2").text = str(origin_y)
        etree.SubElement(origin, f"{{{NS_RESQML20}}}Coordinate3").text = "0.0"

        # Offset vectors (encode rotation)
        import math

        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)

        offset1 = etree.SubElement(geom, f"{{{NS_RESQML20}}}Offset")
        etree.SubElement(offset1, f"{{{NS_RESQML20}}}Coordinate1").text = str(
            di * cos_r
        )
        etree.SubElement(offset1, f"{{{NS_RESQML20}}}Coordinate2").text = str(
            di * sin_r
        )
        etree.SubElement(offset1, f"{{{NS_RESQML20}}}Coordinate3").text = "0.0"
        spacing1 = etree.SubElement(offset1, f"{{{NS_RESQML20}}}Spacing")
        etree.SubElement(spacing1, f"{{{NS_RESQML20}}}Value").text = str(di)

        offset2 = etree.SubElement(geom, f"{{{NS_RESQML20}}}Offset")
        etree.SubElement(offset2, f"{{{NS_RESQML20}}}Coordinate1").text = str(
            -dj * sin_r
        )
        etree.SubElement(offset2, f"{{{NS_RESQML20}}}Coordinate2").text = str(
            dj * cos_r
        )
        etree.SubElement(offset2, f"{{{NS_RESQML20}}}Coordinate3").text = "0.0"
        spacing2 = etree.SubElement(offset2, f"{{{NS_RESQML20}}}Spacing")
        etree.SubElement(spacing2, f"{{{NS_RESQML20}}}Value").text = str(dj)

        # Z values HDF reference
        z_vals = etree.SubElement(geom, f"{{{NS_RESQML20}}}ZValues")
        hdf_ref = etree.SubElement(z_vals, f"{{{NS_COMMON20}}}HdfProxy")
        hdf_ref.set("uuid", _make_uuid())
        path_el = etree.SubElement(z_vals, f"{{{NS_COMMON20}}}PathInHdfFile")
        path_el.text = _hdf5_dataset_path(uuid, "ZValues")

        # CRS
        crs_ref = etree.SubElement(root, f"{{{NS_RESQML20}}}LocalCrs")
        crs_ref.set("uuid", crs_uuid)

        self._add_part(ResqmlObjectType.GRID2D_REPRESENTATION, uuid, root)
        return uuid

    # ---- PointSet ----

    def get_pointset(self, uuid: str) -> Dict[str, Any]:
        """Read PointSet representation."""
        root = self._find_part_by_uuid(uuid)
        if root is None:
            raise ValueError(f"PointSet with UUID {uuid} not found in EPC")

        crs_uuid = ""
        crs_ref = root.find(f".//{{{NS_RESQML20}}}LocalCrs")
        if crs_ref is not None:
            crs_uuid = crs_ref.get("uuid", "")

        points = self._read_hdf5_array(uuid, "Points")
        if points is None:
            points = self._read_hdf5_array(uuid, "points_xyz")
        if points is None:
            points = np.zeros((0, 3), dtype=np.float64)

        return {"points": points, "crs_uuid": crs_uuid}

    def put_pointset(
        self,
        uuid: str,
        title: str,
        points: np.ndarray,
        crs_uuid: str,
    ) -> str:
        """Write PointSet representation."""
        if not uuid:
            uuid = _make_uuid()

        self._write_hdf5_array(uuid, "Points", points.astype(np.float64))

        root = etree.Element(
            f"{{{NS_RESQML20}}}PointSetRepresentation", nsmap=RESQML_NS_MAP
        )
        root.set("uuid", uuid)
        root.set("schemaVersion", "2.0")

        citation = etree.SubElement(root, f"{{{NS_COMMON20}}}Citation")
        title_el = etree.SubElement(citation, f"{{{NS_COMMON20}}}Title")
        title_el.text = title

        # NodePatch
        patch = etree.SubElement(root, f"{{{NS_RESQML20}}}NodePatch")
        etree.SubElement(patch, f"{{{NS_RESQML20}}}Count").text = str(len(points))
        geom = etree.SubElement(patch, f"{{{NS_RESQML20}}}Geometry")
        pts = etree.SubElement(geom, f"{{{NS_RESQML20}}}Points")
        hdf_ref = etree.SubElement(pts, f"{{{NS_COMMON20}}}HdfProxy")
        hdf_ref.set("uuid", _make_uuid())
        path_el = etree.SubElement(pts, f"{{{NS_COMMON20}}}PathInHdfFile")
        path_el.text = _hdf5_dataset_path(uuid, "Points")

        crs_ref = etree.SubElement(root, f"{{{NS_RESQML20}}}LocalCrs")
        crs_ref.set("uuid", crs_uuid)

        self._add_part(ResqmlObjectType.POINT_SET_REPRESENTATION, uuid, root)
        return uuid

    # ---- PolylineSet ----

    def get_polylineset(self, uuid: str) -> Dict[str, Any]:
        """Read PolylineSet representation."""
        root = self._find_part_by_uuid(uuid)
        if root is None:
            raise ValueError(f"PolylineSet with UUID {uuid} not found in EPC")

        crs_uuid = ""
        crs_ref = root.find(f".//{{{NS_RESQML20}}}LocalCrs")
        if crs_ref is not None:
            crs_uuid = crs_ref.get("uuid", "")

        # Read concatenated points and node counts
        all_points = self._read_hdf5_array(uuid, "Points")
        node_counts = self._read_hdf5_array(uuid, "NodeCountPerPolyline")
        closed_flags = self._read_hdf5_array(uuid, "ClosedPolylines")

        polylines: List[np.ndarray] = []
        closed_list: List[bool] = []

        if all_points is not None and node_counts is not None:
            offset = 0
            for i, count in enumerate(node_counts):
                c = int(count)
                polylines.append(all_points[offset : offset + c])
                offset += c
                if closed_flags is not None and i < len(closed_flags):
                    closed_list.append(bool(closed_flags[i]))
                else:
                    closed_list.append(False)
        elif all_points is not None:
            polylines.append(all_points)
            closed_list.append(False)

        return {"polylines": polylines, "closed": closed_list, "crs_uuid": crs_uuid}

    def put_polylineset(
        self,
        uuid: str,
        title: str,
        polylines: List[np.ndarray],
        closed: List[bool],
        crs_uuid: str,
    ) -> str:
        """Write PolylineSet representation."""
        if not uuid:
            uuid = _make_uuid()

        # Concatenate points
        if polylines:
            all_points = np.vstack(polylines).astype(np.float64)
            node_counts = np.array([len(p) for p in polylines], dtype=np.int32)
        else:
            all_points = np.zeros((0, 3), dtype=np.float64)
            node_counts = np.zeros(0, dtype=np.int32)

        closed_arr = np.array(closed, dtype=np.int8)

        self._write_hdf5_array(uuid, "Points", all_points)
        self._write_hdf5_array(uuid, "NodeCountPerPolyline", node_counts)
        self._write_hdf5_array(uuid, "ClosedPolylines", closed_arr)

        root = etree.Element(
            f"{{{NS_RESQML20}}}PolylineSetRepresentation", nsmap=RESQML_NS_MAP
        )
        root.set("uuid", uuid)
        root.set("schemaVersion", "2.0")

        citation = etree.SubElement(root, f"{{{NS_COMMON20}}}Citation")
        title_el = etree.SubElement(citation, f"{{{NS_COMMON20}}}Title")
        title_el.text = title

        # LinePatch
        patch = etree.SubElement(root, f"{{{NS_RESQML20}}}LinePatch")
        etree.SubElement(patch, f"{{{NS_RESQML20}}}Count").text = str(len(polylines))

        geom = etree.SubElement(patch, f"{{{NS_RESQML20}}}Geometry")
        pts = etree.SubElement(geom, f"{{{NS_RESQML20}}}Points")
        hdf_ref = etree.SubElement(pts, f"{{{NS_COMMON20}}}HdfProxy")
        hdf_ref.set("uuid", _make_uuid())
        path_el = etree.SubElement(pts, f"{{{NS_COMMON20}}}PathInHdfFile")
        path_el.text = _hdf5_dataset_path(uuid, "Points")

        crs_ref = etree.SubElement(root, f"{{{NS_RESQML20}}}LocalCrs")
        crs_ref.set("uuid", crs_uuid)

        self._add_part(ResqmlObjectType.POLYLINE_SET_REPRESENTATION, uuid, root)
        return uuid

    # ---- Properties ----

    def get_property_values(
        self, uuid: str, object_type: str = "ContinuousProperty"
    ) -> Dict[str, Any]:
        """Read property values."""
        root = self._find_part_by_uuid(uuid)
        if root is None:
            raise ValueError(f"Property with UUID {uuid} not found in EPC")

        tag = etree.QName(root.tag).localname

        # Title
        title = ""
        citation = root.find(f"{{{NS_COMMON20}}}Citation")
        if citation is not None:
            t = citation.find(f"{{{NS_COMMON20}}}Title")
            if t is not None:
                title = t.text or ""

        # Property kind
        property_kind = ""
        pk_el = root.find(f".//{{{NS_RESQML20}}}PropertyKind")
        if pk_el is not None:
            kind_title = pk_el.find(f".//{{{NS_COMMON20}}}Title")
            if kind_title is not None:
                property_kind = kind_title.text or ""
            # Fallback: resqpy StandardPropertyKind/Kind pattern
            if not property_kind:
                kind_el = pk_el.find(f"{{{NS_RESQML20}}}Kind")
                if kind_el is not None:
                    property_kind = kind_el.text or ""

        # Indexable element
        idx_el = root.find(f"{{{NS_RESQML20}}}IndexableElement")
        indexable_element = idx_el.text if idx_el is not None else "cells"

        # Supporting representation
        supp_uuid = ""
        supp_ref = root.find(f".//{{{NS_RESQML20}}}SupportingRepresentation")
        if supp_ref is not None:
            supp_uuid = supp_ref.get("uuid", "")
            # Fallback: resqpy uses <eml:UUID> child element
            if not supp_uuid:
                uid_el = supp_ref.find(f"{{{NS_COMMON20}}}UUID")
                if uid_el is not None:
                    supp_uuid = uid_el.text or ""

        # UOM
        uom = ""
        uom_el = root.find(f".//{{{NS_RESQML20}}}UOM")
        if uom_el is None:
            uom_el = root.find(f".//{{{NS_RESQML20}}}Uom")
        if uom_el is not None:
            uom = uom_el.text or ""

        # Determine if discrete
        is_discrete = "Discrete" in tag or "Categorical" in tag

        # Values from HDF5 — try multiple path conventions
        values = self._read_hdf5_array(uuid, "Values")
        if values is None:
            values = self._read_hdf5_array(uuid, "values")
        if values is None:
            values = self._read_hdf5_array(uuid, "values_patch0")
        # Try reading path from XML (resqpy convention)
        if values is None:
            for path_el in root.iter(f"{{{NS_COMMON20}}}PathInHdfFile"):
                path_text = path_el.text
                if path_text:
                    values = self._read_hdf5_by_path(path_text)
                    if values is not None:
                        break
        if values is None:
            values = np.array([], dtype=np.float64)

        # Facet
        facet = None
        facet_el = root.find(f".//{{{NS_RESQML20}}}Facet")
        if facet_el is not None:
            fval = facet_el.find(f"{{{NS_RESQML20}}}Value")
            if fval is not None:
                facet = fval.text

        return {
            "values": values,
            "title": title,
            "property_kind": property_kind,
            "indexable_element": indexable_element,
            "supporting_representation_uuid": supp_uuid,
            "is_discrete": is_discrete,
            "uom": uom,
            "facet": facet,
        }

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
        """Write property values."""
        if not uuid:
            uuid = _make_uuid()

        self._write_hdf5_array(uuid, "Values", values)

        obj_type = (
            ResqmlObjectType.DISCRETE_PROPERTY
            if is_discrete
            else ResqmlObjectType.CONTINUOUS_PROPERTY
        )
        tag_name = "DiscreteProperty" if is_discrete else "ContinuousProperty"

        root = etree.Element(f"{{{NS_RESQML20}}}{tag_name}", nsmap=RESQML_NS_MAP)
        root.set("uuid", uuid)
        root.set("schemaVersion", "2.0")

        citation = etree.SubElement(root, f"{{{NS_COMMON20}}}Citation")
        title_el = etree.SubElement(citation, f"{{{NS_COMMON20}}}Title")
        title_el.text = title

        etree.SubElement(
            root, f"{{{NS_RESQML20}}}IndexableElement"
        ).text = indexable_element

        # Supporting representation
        supp = etree.SubElement(root, f"{{{NS_RESQML20}}}SupportingRepresentation")
        supp.set("uuid", supporting_representation_uuid)

        # Property kind
        pk = etree.SubElement(root, f"{{{NS_RESQML20}}}PropertyKind")
        pk_title = etree.SubElement(pk, f"{{{NS_COMMON20}}}Title")
        pk_title.text = property_kind

        # UOM
        if uom:
            uom_el = etree.SubElement(root, f"{{{NS_RESQML20}}}UOM")
            uom_el.text = uom

        # Facet
        if facet:
            facet_el = etree.SubElement(root, f"{{{NS_RESQML20}}}Facet")
            fval = etree.SubElement(facet_el, f"{{{NS_RESQML20}}}Value")
            fval.text = facet

        # HDF reference
        patch = etree.SubElement(root, f"{{{NS_RESQML20}}}PatchOfValues")
        vals = etree.SubElement(patch, f"{{{NS_RESQML20}}}Values")
        hdf_ref = etree.SubElement(vals, f"{{{NS_COMMON20}}}HdfProxy")
        hdf_ref.set("uuid", _make_uuid())
        path_el = etree.SubElement(vals, f"{{{NS_COMMON20}}}PathInHdfFile")
        path_el.text = _hdf5_dataset_path(uuid, "Values")

        self._add_part(obj_type, uuid, root)
        return uuid

    # ---- CRS ----

    def get_crs(self, uuid: str) -> Dict[str, Any]:
        """Read CRS definition."""
        root = self._find_part_by_uuid(uuid)
        if root is None:
            raise ValueError(f"CRS with UUID {uuid} not found in EPC")
        crs = LocalDepth3dCrs.from_xml(root)
        return {
            "uuid": crs.uuid,
            "title": crs.title,
            "origin_x": crs.origin_x,
            "origin_y": crs.origin_y,
            "origin_z": crs.origin_z,
            "areal_rotation": crs.areal_rotation,
            "z_increasing_downward": crs.z_increasing_downward,
            "projected_crs_epsg": crs.projected_crs_epsg,
            "vertical_crs_epsg": crs.vertical_crs_epsg,
        }

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
        """Write CRS definition."""
        if not uuid:
            uuid = _make_uuid()

        crs = LocalDepth3dCrs(
            uuid=uuid,
            title=title,
            origin_x=origin_x,
            origin_y=origin_y,
            origin_z=origin_z,
            areal_rotation=areal_rotation,
            z_increasing_downward=z_increasing_downward,
            projected_crs_epsg=projected_crs_epsg,
            vertical_crs_epsg=vertical_crs_epsg,
        )
        xml = crs.to_xml()
        self._add_part(ResqmlObjectType.LOCAL_DEPTH3D_CRS, uuid, xml)
        return uuid
