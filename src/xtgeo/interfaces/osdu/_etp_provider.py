# -*- coding: utf-8 -*-
"""ETP 1.2 WebSocket protocol provider for RESQML 2.0.1 data access to RDDMS.

This module implements the ResqmlDataProvider interface using the ETP 1.2 protocol
to communicate with OSDU Reservoir DMS (RDDMS) endpoints.

It uses proper Avro binary encoding via the `energistics` package (pyetp)
but implements its own synchronous client tailored for
xtgeo's data types: IJK grids, Grid2D surfaces, PointSets, PolylineSets.

Connection flow:
  1. WebSocket connect with ETP subprotocol + bearer token
  2. ETP RequestSession -> OpenSession (Avro binary handshake)
  3. GetResources for discovery
  4. GetDataObjects for XML metadata
  5. GetDataArrays / PutDataArrays for bulk array data
  6. PutDataObjects for writing XML parts

Dependencies:
  - websockets (async client)
  - fastavro (Avro schema encoding/decoding, pulled in by energistics)
  - energistics (ETP 1.2 Avro schemas + message models)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid as _uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ._provider_base import ResqmlDataProvider

logger = logging.getLogger(__name__)


@dataclass
class EtpConnectionConfig:
    """Configuration for ETP WebSocket connection."""

    url: str  # ws://host:port or wss://host/api/reservoir-ddms-etp/v2/
    token: str = ""  # Bearer token (empty for local dev servers)
    dataspace: str = "eml:///dataspace('default')"
    data_partition: str = ""
    timeout_s: float = 30.0
    max_retries: int = 3


def _make_uuid() -> str:
    return str(_uuid.uuid4())


def _add_citation(parent, title: str):
    """Add a Citation element with Title, Creation, Originator, Format."""
    from datetime import datetime, timezone

    from lxml import etree

    from ._resqml_enums import NS_COMMON20

    citation = etree.SubElement(parent, f"{{{NS_COMMON20}}}Citation")
    etree.SubElement(citation, f"{{{NS_COMMON20}}}Title").text = title
    etree.SubElement(citation, f"{{{NS_COMMON20}}}Creation").text = datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    etree.SubElement(citation, f"{{{NS_COMMON20}}}Originator").text = "xtgeo"
    etree.SubElement(citation, f"{{{NS_COMMON20}}}Format").text = "xtgeo RESQML 2.0.1"
    return citation


def _uri_for_object(dataspace: str, qualified_type: str, uuid: str) -> str:
    """Build ETP URI for a data object: eml:///dataspace('x')/resqml20.Type(uuid)"""
    return f"{dataspace}/{qualified_type}({uuid})"


def _uuid_from_uri(uri: str) -> str:
    """Extract UUID from an ETP URI like ..../type(uuid)"""
    if "(" in uri and ")" in uri:
        return uri[uri.rfind("(") + 1 : uri.rfind(")")]
    return ""


def _uuid_from_ref(el) -> str:
    """Extract UUID from a RESQML DataObjectReference element.

    Handles both formats:
      - ``<LocalCrs uuid="xxx">`` (attribute)
      - ``<LocalCrs><eml:UUID>xxx</eml:UUID></LocalCrs>`` (child element)
    """
    uid = el.get("uuid", "")
    if uid:
        return uid
    # Try eml:UUID child element (OSDU RDDMS / RMS format)
    for child in el:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag in ("UUID", "UuidString"):
            return (child.text or "").strip()
    return ""


def _qualified_type_from_uri(uri: str) -> str:
    """Extract qualified type from URI."""
    parts = uri.rstrip(")").split("/")
    for p in reversed(parts):
        if "(" in p:
            return p.split("(")[0]
    return ""


def _hdf_paths_from_xml(root, ns_common: str) -> Dict[str, str]:
    """Build mapping of ancestor-element localname → HDF path from XML.

    Scans all ``PathInHdfFile`` elements and keys them by the localname
    of both their parent and grandparent elements.  This handles two
    common RESQML patterns:

    * Direct: ``<ZValues><PathInHdfFile>…`` → parent is ``ZValues``
    * Wrapped: ``<ControlPoints><Coordinates><PathInHdfFile>…``
      → parent is ``Coordinates``, grandparent is ``ControlPoints``
    """
    result: Dict[str, str] = {}
    for path_el in root.iter(f"{{{ns_common}}}PathInHdfFile"):
        if path_el.text:
            parent = path_el.getparent()
            if parent is not None:
                from lxml import etree as _et

                ptag = _et.QName(parent.tag).localname
                result[ptag] = path_el.text
                grandparent = parent.getparent()
                if grandparent is not None:
                    gptag = _et.QName(grandparent.tag).localname
                    result[gptag] = path_el.text
    return result


def _points_array_to_coord_zcorn(
    points: np.ndarray, ni: int, nj: int, nk: int
) -> tuple:
    """Convert standard RESQML Point3dHdf5Array to xtgeo coord/zcorn arrays.

    Standard RESQML stores all pillar node XYZ in a single array with shape
    (nk+1, nj+1, ni+1, 3) — K slowest, I fastest, last dim is XYZ.

    This converts to xtgeo's internal format:
      - coordsv: (ni+1, nj+1, 6) — pillar top/bottom XYZ
      - zcornsv: (ni+1, nj+1, nk+1, 4) — Z at each node (SW,SE,NW,NE)

    For unsplit grids the 4 corners at each node are identical.
    Split pillars are NOT handled here (would need PillarIndices + Columns info).
    """
    # Reshape to (nk+1, nj+1, ni+1, 3)
    expected_size = (nk + 1) * (nj + 1) * (ni + 1) * 3
    if points.size == expected_size:
        pts = points.reshape((nk + 1, nj + 1, ni + 1, 3))
    elif points.size == (nk + 1) * (nj + 1) * (ni + 1) * 3:
        # Already correct size, just try reshape
        pts = points.reshape((nk + 1, nj + 1, ni + 1, 3))
    else:
        raise ValueError(
            f"Points array size {points.size} incompatible with "
            f"(nk+1)*(nj+1)*(ni+1)*3 = {expected_size}"
        )

    # coordsv: pillar top (k=0) and bottom (k=nk) XYZ
    # Shape: (ni+1, nj+1, 6)
    # RESQML is (K, J, I, XYZ), xtgeo coordsv is
    # (I, J, 6=[xtop,ytop,ztop,xbot,ybot,zbot])
    top = pts[0, :, :, :]  # shape (nj+1, ni+1, 3)
    bot = pts[nk, :, :, :]  # shape (nj+1, ni+1, 3)

    coordsv = np.zeros((ni + 1, nj + 1, 6), dtype=np.float64)
    # Transpose from (J, I, 3) to (I, J, ...) and interleave top/bot
    coordsv[:, :, 0] = top[:, :, 0].T  # x_top
    coordsv[:, :, 1] = top[:, :, 1].T  # y_top
    coordsv[:, :, 2] = top[:, :, 2].T  # z_top
    coordsv[:, :, 3] = bot[:, :, 0].T  # x_bot
    coordsv[:, :, 4] = bot[:, :, 1].T  # y_bot
    coordsv[:, :, 5] = bot[:, :, 2].T  # z_bot

    # zcornsv: Z values at each pillar node per layer
    # For unsplit grids, all 4 corners (SW,SE,NW,NE) get the same Z value
    # Shape: (ni+1, nj+1, nk+1, 4)
    z_all = pts[:, :, :, 2]  # shape (nk+1, nj+1, ni+1) — just Z
    # Transpose to (ni+1, nj+1, nk+1)
    z_ijk = z_all.transpose((2, 1, 0))  # (ni+1, nj+1, nk+1)
    zcornsv = np.broadcast_to(
        z_ijk[..., np.newaxis], (ni + 1, nj + 1, nk + 1, 4)
    ).copy()

    return coordsv.ravel(), zcornsv.ravel()


def _parametric_to_explicit_points(
    control_points: np.ndarray,
    control_point_params: np.ndarray,
    parameters: np.ndarray,
    ni: int,
    nj: int,
    nk: int,
) -> np.ndarray:
    """Convert RESQML Point3dParametricArray (straight pillars) to explicit points.

    For straight pillars with KnotCount=2, each pillar has a top and bottom
    control point. Nodes are positioned by linear interpolation along the pillar
    based on Z-parameter values.

    Parameters
    ----------
    control_points : ndarray, shape (2, npillars, 3)
        Top (index 0) and bottom (index 1) XYZ for each pillar.
    control_point_params : ndarray, shape (2, npillars)
        Z-parameter values at top (index 0) and bottom (index 1) control points.
    parameters : ndarray, shape (nk+1, npillars)
        Z-parameter value at each layer node on each pillar.
    ni, nj, nk : int
        Grid dimensions (number of cells in each direction).

    Returns
    -------
    ndarray, shape ((nk+1)*(nj+1)*(ni+1)*3,)
        Explicit XYZ coordinates for all grid nodes, flattened.
    """
    npillars = (ni + 1) * (nj + 1)

    cp = control_points.reshape(2, npillars, 3).astype(np.float64)
    cpp = control_point_params.reshape(2, npillars).astype(np.float64)
    params = parameters.reshape(nk + 1, npillars).astype(np.float64)

    # Linear interpolation factor: t = (param - top_param) / (bot_param - top_param)
    denom = cpp[1, :] - cpp[0, :]
    # Avoid division by zero for degenerate pillars (top == bottom z-param)
    denom = np.where(np.abs(denom) < 1e-30, 1.0, denom)

    # t: shape (nk+1, npillars)
    t = (params - cpp[0, :]) / denom

    # Interpolate XYZ: shape (nk+1, npillars, 3)
    top = cp[0, :, :]  # (npillars, 3)
    bot = cp[1, :, :]  # (npillars, 3)
    points = top[np.newaxis, :, :] * (1.0 - t[:, :, np.newaxis]) + \
             bot[np.newaxis, :, :] * t[:, :, np.newaxis]

    # Reshape to standard RESQML order: (nk+1, nj+1, ni+1, 3)
    points = points.reshape(nk + 1, nj + 1, ni + 1, 3)

    return points.ravel()


def _parametric_split_to_coord_zcorn(
    control_points: np.ndarray,
    control_point_params: np.ndarray,
    parameters: np.ndarray,
    ni: int,
    nj: int,
    nk: int,
    pillar_indices: np.ndarray,
    col_elements: np.ndarray,
    col_cumlength: np.ndarray,
) -> tuple:
    """Convert parametric grid with split coordinate lines to xtgeo coord/zcorn.

    Handles faulted grids where some pillars have split Z-values for
    different adjacent cell columns (RESQML SplitCoordinateLines).

    Parameters
    ----------
    control_points : ndarray, shape (2, npillars, 3)
        Top/bottom XYZ for each original pillar.
    control_point_params : ndarray, shape (2, npillars)
        Z-parameter values at top/bottom of each pillar.
    parameters : ndarray, shape (nk+1, npillars + nsplits)
        Z-parameter at each layer node for pillars and split lines.
    ni, nj, nk : int
        Grid cell dimensions.
    pillar_indices : ndarray, shape (nsplits,)
        Original pillar index for each split coordinate line.
    col_elements : ndarray
        Flat array of column indices for each split (jagged).
    col_cumlength : ndarray, shape (nsplits,)
        Cumulative lengths into col_elements for each split.

    Returns
    -------
    tuple of (coord_flat, zcorn_flat)
        coord: shape ((ni+1)*(nj+1)*6,) — pillar top/bottom XYZ
        zcorn: shape ((ni+1)*(nj+1)*(nk+1)*4,) — Z at 4 corners per node
    """
    npillars = (ni + 1) * (nj + 1)
    nsplits = pillar_indices.shape[0]
    n_total = npillars + nsplits

    cp = control_points.reshape(2, npillars, 3).astype(np.float64)
    cpp = control_point_params.reshape(2, npillars).astype(np.float64)
    params = parameters.reshape(nk + 1, n_total).astype(np.float64)

    # --- coord from original pillars (splits don't affect pillar geometry) ---
    # Pillar index = pj*(ni+1) + pi, so reshape to (nj+1, ni+1, ...)
    cp_2d = cp.reshape(2, nj + 1, ni + 1, 3)
    coordsv = np.zeros((ni + 1, nj + 1, 6), dtype=np.float64)
    coordsv[:, :, 0:3] = cp_2d[0].transpose(1, 0, 2)  # top: (pi, pj, xyz)
    coordsv[:, :, 3:6] = cp_2d[1].transpose(1, 0, 2)  # bot: (pi, pj, xyz)

    # --- Compute Z at every (k, line) by interpolation along parent pillar ---
    parent = np.arange(n_total, dtype=np.int64)
    parent[npillars:] = pillar_indices.astype(np.int64)

    top_z_param = cpp[0, parent]  # (n_total,)
    bot_z_param = cpp[1, parent]  # (n_total,)
    denom = bot_z_param - top_z_param
    denom = np.where(np.abs(denom) < 1e-30, 1.0, denom)

    t = (params - top_z_param) / denom  # (nk+1, n_total)

    # Z coordinate at each (k, line)
    z_top = cp[0, parent, 2]  # (n_total,)
    z_bot = cp[1, parent, 2]  # (n_total,)
    z_at = z_top * (1.0 - t) + z_bot * t  # (nk+1, n_total)

    # --- Build line_map: which coordinate line each (pi, pj, corner) uses ---
    # Default: all 4 corners use the pillar's own line index
    pi_grid, pj_grid = np.meshgrid(
        np.arange(ni + 1), np.arange(nj + 1), indexing="ij"
    )
    default_line = (pj_grid * (ni + 1) + pi_grid).astype(np.int64)
    line_map = np.broadcast_to(
        default_line[..., np.newaxis], (ni + 1, nj + 1, 4)
    ).copy()

    # xtgeo zcorn corner convention (from _ecl_grid.py):
    #   c=0 (SW): cell (pi-1, pj-1) → column (pi-1) + (pj-1)*ni
    #   c=1 (SE): cell (pi,   pj-1) → column  pi    + (pj-1)*ni
    #   c=2 (NW): cell (pi-1, pj  ) → column (pi-1) +  pj   *ni
    #   c=3 (NE): cell (pi,   pj  ) → column  pi    +  pj   *ni
    _CORNER_DI = np.array([-1, 0, -1, 0])
    _CORNER_DJ = np.array([-1, -1, 0, 0])

    # Override with split coordinate lines
    start = 0
    for s in range(nsplits):
        end = int(col_cumlength[s])
        orig_pillar = int(pillar_indices[s])
        orig_pi = orig_pillar % (ni + 1)
        orig_pj = orig_pillar // (ni + 1)
        split_line = npillars + s
        for idx in range(start, end):
            col = int(col_elements[idx])
            ci = col % ni
            cj = col // ni
            # Determine which corner of pillar (orig_pi, orig_pj) this column is
            di = ci - orig_pi
            dj = cj - orig_pj
            for c in range(4):
                if _CORNER_DI[c] == di and _CORNER_DJ[c] == dj:
                    line_map[orig_pi, orig_pj, c] = split_line
                    break
        start = end

    # --- Build zcorn using vectorized fancy indexing ---
    flat_lines = line_map.ravel()  # ((ni+1)*(nj+1)*4,)
    z_selected = z_at[:, flat_lines]  # (nk+1, (ni+1)*(nj+1)*4)
    zcornsv = (
        z_selected.reshape(nk + 1, ni + 1, nj + 1, 4)
        .transpose(1, 2, 0, 3)
        .astype(np.float32)
    )

    # Duplicate boundary corners (same as _ecl_grid.duplicate_insignificant_xtgeo_zcorn)
    # South boundary: c=0,1 at pj=0 → copy from c=2,3
    zcornsv[1:ni, 0, :, 0] = zcornsv[1:ni, 0, :, 2]
    zcornsv[1:ni, 0, :, 1] = zcornsv[1:ni, 0, :, 3]
    # North boundary: c=2,3 at pj=nj → copy from c=0,1
    zcornsv[1:ni, nj, :, 2] = zcornsv[1:ni, nj, :, 0]
    zcornsv[1:ni, nj, :, 3] = zcornsv[1:ni, nj, :, 1]
    # West boundary: c=0,2 at pi=0 → copy from c=1,3
    zcornsv[0, 1:nj, :, 0] = zcornsv[0, 1:nj, :, 1]
    zcornsv[0, 1:nj, :, 2] = zcornsv[0, 1:nj, :, 3]
    # East boundary: c=1,3 at pi=ni → copy from c=0,2
    zcornsv[ni, 1:nj, :, 1] = zcornsv[ni, 1:nj, :, 0]
    zcornsv[ni, 1:nj, :, 3] = zcornsv[ni, 1:nj, :, 2]
    # SW corner: all = c=3
    zcornsv[0, 0, :, :] = zcornsv[0, 0, :, 3:4]
    # SE corner: all = c=2
    zcornsv[ni, 0, :, :] = zcornsv[ni, 0, :, 2:3]
    # NW corner: all = c=1
    zcornsv[0, nj, :, :] = zcornsv[0, nj, :, 1:2]
    # NE corner: all = c=0
    zcornsv[ni, nj, :, :] = zcornsv[ni, nj, :, 0:1]

    return coordsv.ravel(), zcornsv.ravel()


class EtpProvider(ResqmlDataProvider):
    """ETP 1.2 WebSocket provider for RESQML data access.

    Uses proper Avro binary encoding (energistics package) for protocol messages.
    Runs an async event loop internally to provide a synchronous API.

    Parameters
    ----------
    config : EtpConnectionConfig
        Connection configuration.
    """

    def __init__(self, config: EtpConnectionConfig):
        self._config = config
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws = None
        self._message_id = 2  # Start at 2 per ETP spec (1 reserved for server)
        self._session_open = False
        self._tx_uuid: Optional[str] = None  # Active transaction UUID
        self._uri_cache: Dict[str, str] = {}  # uuid → server URI cache

    def _resolve_uri(self, uuid: str, fallback_type: str = "") -> str:
        """Resolve the server-side URI for an object by UUID.

        Handles servers that use ``obj_`` prefix in qualified types (OSDU Azure)
        vs servers that strip it (local RDDMS). Caches results to avoid repeated
        ``list_objects()`` calls.

        Parameters
        ----------
        uuid : str
            Object UUID to look up.
        fallback_type : str, optional
            Qualified type to construct a URI if discovery fails (e.g.
            ``"resqml20.IjkGridRepresentation"``). Used as last resort.

        Returns
        -------
        str
            Resolved URI from the server, or constructed fallback.
        """
        if uuid in self._uri_cache:
            return self._uri_cache[uuid]

        # Populate cache from list_objects (one round-trip)
        if not self._uri_cache:
            try:
                for obj in self.list_objects():
                    self._uri_cache[obj["uuid"]] = obj["uri"]
            except Exception:
                pass

        if uuid in self._uri_cache:
            return self._uri_cache[uuid]

        # Fallback: construct URI (works for local RDDMS)
        if fallback_type:
            return _uri_for_object(self._config.dataspace, fallback_type, uuid)
        return _uri_for_object(self._config.dataspace, "resqml20.Unknown", uuid)

    def open(self) -> None:
        """Connect to the ETP server and perform handshake."""
        self._uri_cache.clear()
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._async_open())

    def close(self) -> None:
        """Send CloseSession and disconnect."""
        if self._loop and self._ws:
            with contextlib.suppress(Exception):
                self._loop.run_until_complete(self._async_close())
        if self._loop:
            self._loop.close()
            self._loop = None
        self._ws = None
        self._session_open = False

    def _run(self, coro):
        """Run an async coroutine synchronously."""
        return self._loop.run_until_complete(coro)

    # ------------------------------------------------------------------
    # Async internals (proper Avro binary protocol)
    # ------------------------------------------------------------------

    async def _async_open(self) -> None:
        import websockets

        additional_headers = {}
        if self._config.token:
            additional_headers["Authorization"] = f"Bearer {self._config.token}"
        if self._config.data_partition:
            additional_headers["data-partition-id"] = self._config.data_partition

        self._ws = await websockets.connect(
            self._config.url,
            subprotocols=[websockets.typing.Subprotocol("etp12.energistics.org")],
            additional_headers=additional_headers,
            open_timeout=self._config.timeout_s,
            max_size=2**26,  # 64 MiB (large grids can exceed 4 MiB)
        )

        await self._handshake()

    async def _async_close(self) -> None:
        from energistics.avro_handler import encode_message
        from energistics.etp.v12.datatypes.message_header import (
            MessageHeader,
            MessageHeaderFlags,
        )
        from energistics.etp.v12.protocol.core import CloseSession

        cs = CloseSession(reason="client closing")
        header = MessageHeader.from_etp_protocol_body(
            body=cs,
            message_id=self._next_message_id(),
            message_flags=MessageHeaderFlags.FIN,
        )
        with contextlib.suppress(Exception):
            await self._ws.send(encode_message(header=header, body=cs))
        await self._ws.close()

    async def _handshake(self) -> None:
        import uuid

        from energistics.avro_handler import decode_message, encode_message
        from energistics.base import Protocol, Role
        from energistics.etp.v12.datatypes import SupportedDataObject, SupportedProtocol
        from energistics.etp.v12.datatypes.message_header import (
            MessageHeader,
            MessageHeaderFlags,
        )
        from energistics.etp.v12.protocol.core import OpenSession, RequestSession

        rs = RequestSession(
            application_name="xtgeo-osdu",
            application_version="1.0.0",
            client_instance_id=uuid.uuid4(),
            requested_protocols=[
                SupportedProtocol(protocol=Protocol.DISCOVERY, role=Role.STORE),
                SupportedProtocol(protocol=Protocol.STORE, role=Role.STORE),
                SupportedProtocol(protocol=Protocol.DATA_ARRAY, role=Role.STORE),
                SupportedProtocol(protocol=Protocol.DATASPACE, role=Role.STORE),
                SupportedProtocol(protocol=Protocol.TRANSACTION, role=Role.STORE),
            ],
            supported_data_objects=[
                SupportedDataObject(qualified_type="resqml20.*"),
                SupportedDataObject(qualified_type="eml20.*"),
            ],
            supported_compression=[],
            supported_formats=["xml"],
        )

        header = MessageHeader.from_etp_protocol_body(
            body=rs,
            message_id=self._next_message_id(),
            message_flags=MessageHeaderFlags.FIN,
        )
        await self._ws.send(encode_message(header=header, body=rs))

        resp_bytes = await self._ws.recv(decode=False)
        resp_header, resp_body = decode_message(resp_bytes)

        if not isinstance(resp_body, OpenSession):
            got = type(resp_body).__name__
            raise RuntimeError(f"ETP handshake failed: expected OpenSession, got {got}")

        self._session_open = True
        logger.info(
            "ETP session opened: server=%s %s",
            resp_body.application_name,
            resp_body.application_version,
        )

    def _next_message_id(self) -> int:
        mid = self._message_id
        self._message_id += 2
        return mid

    async def _start_transaction(self) -> str:
        """Start an ETP transaction. Returns transaction UUID.

        Auto-reconnects if the server only allows one write transaction per session.
        """
        from energistics.etp.v12.protocol.transaction import StartTransaction

        msg = StartTransaction(
            readOnly=False,
            message="xtgeo write",
            dataspaceUris=[self._config.dataspace],
        )
        try:
            responses = await self._send_and_recv(msg)
        except Exception:
            # Connection may have been dropped; reconnect and retry
            await self._async_close()
            await self._async_open()
            responses = await self._send_and_recv(msg)

        for r in responses:
            if hasattr(r, "transaction_uuid"):
                self._tx_uuid = r.transaction_uuid
                return r.transaction_uuid
            if hasattr(r, "error") and r.error:
                # Server allows only one write TX per session - reconnect
                err_msg = str(getattr(r.error, "message", ""))
                err_code = getattr(r.error, "code", None)
                if "transaction" in err_msg.lower() or (
                    err_code and getattr(err_code, "value", 0) == 15
                ):
                    await self._async_close()
                    await self._async_open()
                    # Retry once after reconnect
                    responses2 = await self._send_and_recv(msg)
                    for r2 in responses2:
                        if hasattr(r2, "transaction_uuid"):
                            self._tx_uuid = r2.transaction_uuid
                            return r2.transaction_uuid
                        if hasattr(r2, "error") and r2.error:
                            raise RuntimeError(
                                f"StartTransaction failed after reconnect: {r2.error}"
                            )
                raise RuntimeError(f"StartTransaction failed: {r.error}")
        raise RuntimeError("StartTransaction: no transaction_uuid in response")

    async def _commit_transaction(self) -> None:
        """Commit the current ETP transaction."""
        from energistics.etp.v12.protocol.transaction import CommitTransaction

        if not hasattr(self, "_tx_uuid") or not self._tx_uuid:
            return
        msg = CommitTransaction(transactionUuid=self._tx_uuid)
        responses = await self._send_and_recv(msg)
        for r in responses:
            if hasattr(r, "error") and r.error:
                raise RuntimeError(f"CommitTransaction failed: {r.error}")
            if hasattr(r, "successful") and not r.successful:
                reason = getattr(r, "failure_reason", "unknown")
                raise RuntimeError(f"CommitTransaction failed: {reason}")
        self._tx_uuid = None

    def _write_in_tx(self, coro_func, *args, **kwargs):
        """Run a write coroutine wrapped in StartTransaction/CommitTransaction."""
        return self._loop.run_until_complete(
            self._async_write_in_tx(coro_func, *args, **kwargs)
        )

    async def _async_write_in_tx(self, coro_func, *args, **kwargs):
        """Start tx, run the coroutine, commit. Rollback on error."""
        from energistics.etp.v12.protocol.transaction import RollbackTransaction

        await self._start_transaction()
        try:
            result = await coro_func(*args, **kwargs)
            await self._commit_transaction()
            return result
        except Exception:
            # Rollback on failure
            if self._tx_uuid:
                try:
                    msg = RollbackTransaction(transactionUuid=self._tx_uuid)
                    await self._send_and_recv(msg)
                except Exception:
                    pass
                self._tx_uuid = None
            raise

    async def _send_and_recv(self, body):
        """Send a protocol message and receive responses (handling multi-part)."""
        from energistics.avro_handler import decode_message, encode_message
        from energistics.etp.v12.datatypes.message_header import (
            MessageHeader,
            MessageHeaderFlags,
        )

        header = MessageHeader.from_etp_protocol_body(
            body=body,
            message_id=self._next_message_id(),
            message_flags=MessageHeaderFlags.FIN,
        )
        await self._ws.send(encode_message(header=header, body=body))

        # Collect multi-part responses
        bodies = []
        while True:
            resp_bytes = await asyncio.wait_for(
                self._ws.recv(decode=False),
                timeout=self._config.timeout_s,
            )
            resp_header, resp_body = decode_message(resp_bytes)
            bodies.append(resp_body)

            # Check if this is the final message in the response
            if resp_header.is_final_message():
                break

        return bodies

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_objects(self, object_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available data objects, optionally filtered by type."""
        return self._run(self._async_list_objects(object_type))

    async def _async_list_objects(
        self, object_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        from energistics.etp.v12.datatypes.object.context_info import ContextInfo
        from energistics.etp.v12.datatypes.object.context_scope_kind import (
            ContextScopeKind,
        )
        from energistics.etp.v12.protocol.discovery import GetResources

        context = ContextInfo(
            uri=self._config.dataspace,
            depth=1,
            navigable_edges="Primary",
            include_secondary_targets=False,
            include_secondary_sources=False,
        )

        msg = GetResources(context=context, scope=ContextScopeKind.TARGETS_OR_SELF)
        responses = await self._send_and_recv(msg)

        results = []
        for resp in responses:
            resources = getattr(resp, "resources", [])
            if not resources:
                continue
            for r in resources:
                uri = r.uri
                uid = _uuid_from_uri(uri)
                qtype = _qualified_type_from_uri(uri)
                name = r.name

                obj_info = {"uuid": uid, "title": name, "type": qtype, "uri": uri}

                if (
                    object_type is None
                    or object_type.lower() in qtype.lower()
                    or object_type.lower() in name.lower()
                ):
                    results.append(obj_info)

        return results

    # ------------------------------------------------------------------
    # Deep Discovery (ETP Protocol 3 — full depth/type/edge queries)
    # ------------------------------------------------------------------

    def discover(
        self,
        *,
        uri: Optional[str] = None,
        depth: int = 1,
        scope: str = "targets",
        object_types: Optional[List[str]] = None,
        include_edges: bool = False,
        include_secondary: bool = False,
    ) -> Dict[str, Any]:
        """Deep resource discovery with configurable depth and type filtering.

        Parameters
        ----------
        uri : str, optional
            Starting URI. Defaults to current dataspace.
            Can be a specific object URI to discover its relationships.
        depth : int
            How many levels deep to traverse (1=direct children, 2=grandchildren, etc.).
            Use 0 for unlimited depth (full graph).
        scope : str
            Discovery scope: "self", "targets", "sources", "targets_or_self",
            "sources_or_self".
        object_types : list of str, optional
            Filter by qualified type. Examples:
            ``["resqml20.obj_IjkGridRepresentation"]``,
            ``["resqml20.obj_ContinuousProperty"]``.
            Use ``None`` for all types.
        include_edges : bool
            If True, also return relationship edges between objects.
        include_secondary : bool
            If True, include secondary relationship targets/sources.

        Returns
        -------
        dict with keys:
            - 'resources': list of resource dicts (uuid, title, type, uri, timestamps)
            - 'edges': list of edge dicts (if include_edges=True)

        Examples
        --------
        >>> # All objects in dataspace, unlimited depth
        >>> result = provider.discover(depth=0)

        >>> # All properties attached to a specific grid
        >>> result = provider.discover(uri=grid_uri, depth=1, scope="sources")

        >>> # Only grids, deep traversal
        >>> result = provider.discover(
        ...     depth=2,
        ...     object_types=["resqml20.obj_IjkGridRepresentation"],
        ... )
        """
        return self._run(
            self._async_discover(
                uri=uri,
                depth=depth,
                scope=scope,
                object_types=object_types,
                include_edges=include_edges,
                include_secondary=include_secondary,
            )
        )

    async def _async_discover(
        self,
        *,
        uri: Optional[str] = None,
        depth: int = 1,
        scope: str = "targets",
        object_types: Optional[List[str]] = None,
        include_edges: bool = False,
        include_secondary: bool = False,
    ) -> Dict[str, Any]:
        from energistics.etp.v12.datatypes.object.context_info import ContextInfo
        from energistics.etp.v12.datatypes.object.context_scope_kind import (
            ContextScopeKind,
        )
        from energistics.etp.v12.datatypes.object.relationship_kind import (
            RelationshipKind,
        )
        from energistics.etp.v12.protocol.discovery import GetResources

        scope_map = {
            "self": ContextScopeKind.SELF,
            "targets": ContextScopeKind.TARGETS,
            "sources": ContextScopeKind.SOURCES,
            "targets_or_self": ContextScopeKind.TARGETS_OR_SELF,
            "sources_or_self": ContextScopeKind.SOURCES_OR_SELF,
        }
        etp_scope = scope_map.get(scope.lower(), ContextScopeKind.TARGETS_OR_SELF)
        target_uri = uri or self._config.dataspace

        # ETP spec: depth must be > 0. Use large number for "unlimited".
        effective_depth = depth if depth > 0 else 2147483647

        context = ContextInfo(
            uri=target_uri,
            depth=effective_depth,
            data_object_types=object_types or [],
            navigable_edges=RelationshipKind.BOTH
            if include_secondary
            else RelationshipKind.PRIMARY,
            include_secondary_targets=include_secondary,
            include_secondary_sources=include_secondary,
        )

        msg = GetResources(
            context=context,
            scope=etp_scope,
            include_edges=include_edges,
        )
        responses = await self._send_and_recv(msg)

        resources = []
        edges = []

        for resp in responses:
            # Resources
            resp_resources = getattr(resp, "resources", None)
            if resp_resources:
                for r in resp_resources:
                    resources.append(
                        {
                            "uuid": _uuid_from_uri(r.uri),
                            "title": r.name,
                            "type": _qualified_type_from_uri(r.uri),
                            "uri": r.uri,
                            "last_changed": getattr(r, "last_changed", 0),
                            "store_created": getattr(r, "store_created", 0),
                            "active_status": str(getattr(r, "active_status", "")),
                            "source_count": getattr(r, "source_count", None),
                            "target_count": getattr(r, "target_count", None),
                        }
                    )

            # Edges
            resp_edges = getattr(resp, "edges", None)
            if resp_edges:
                for e in resp_edges:
                    edges.append(
                        {
                            "source_uri": e.source_uri,
                            "target_uri": e.target_uri,
                            "relationship_kind": str(e.relationship_kind),
                        }
                    )

        return {"resources": resources, "edges": edges}

    def get_related_objects(
        self,
        object_uuid: str,
        *,
        direction: str = "sources",
        object_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find objects related to a specific object.

        Useful for finding properties of a grid, or the grid supporting a property.

        Parameters
        ----------
        object_uuid : str
            UUID of the starting object.
        direction : str
            "sources" = who references this object (e.g. properties → grid),
            "targets" = what this object references (e.g. grid → CRS).
        object_type : str, optional
            Filter results by type.

        Returns
        -------
        list of dict
            Related objects.

        Examples
        --------
        >>> # Get all properties of a grid
        >>> props = provider.get_related_objects(grid_uuid, direction="sources")

        >>> # Get the CRS of a grid
        >>> crs = provider.get_related_objects(grid_uuid, direction="targets",
        ...                                    object_type="LocalDepth3dCrs")
        """
        # First find the URI of the object
        objects = self.list_objects()
        obj_uri = None
        for obj in objects:
            if obj["uuid"] == object_uuid:
                obj_uri = obj["uri"]
                break

        if obj_uri is None:
            raise ValueError(f"Object {object_uuid} not found in dataspace")

        type_filter = []
        if object_type:
            # Build qualified type filter
            type_filter = [
                object_type if "." in object_type else f"resqml20.{object_type}"
            ]

        result = self.discover(
            uri=obj_uri,
            depth=1,
            scope=direction,
            object_types=type_filter if type_filter else None,
        )
        return result["resources"]

    def get_deleted_resources(
        self,
        since: Optional[int] = None,
        object_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Get resources that have been deleted since a given time.

        Parameters
        ----------
        since : int, optional
            Microsecond timestamp. Only return deletions after this time.
            If None, returns all tracked deletions.
        object_types : list of str, optional
            Filter by qualified type.

        Returns
        -------
        list of dict
            Each dict has 'uri', 'deleted_time', 'uuid' keys.
        """
        return self._run(self._async_get_deleted_resources(since, object_types))

    async def _async_get_deleted_resources(
        self,
        since: Optional[int],
        object_types: Optional[List[str]],
    ) -> List[Dict[str, Any]]:
        from energistics.etp.v12.protocol.discovery import GetDeletedResources

        msg = GetDeletedResources(
            dataspace_uri=self._config.dataspace,
            delete_time_filter=since,
            data_object_types=object_types or [],
        )
        responses = await self._send_and_recv(msg)

        results = []
        for resp in responses:
            deleted = getattr(resp, "deleted_resources", [])
            for d in deleted:
                results.append(
                    {
                        "uri": d.uri,
                        "uuid": _uuid_from_uri(d.uri),
                        "deleted_time": d.deleted_time,
                    }
                )
        return results

    # ------------------------------------------------------------------
    # Store Notification (ETP Protocol 5 — change tracking)
    # ------------------------------------------------------------------

    def subscribe_notifications(
        self,
        *,
        object_types: Optional[List[str]] = None,
        uuids: Optional[List[str]] = None,
        include_object_data: bool = False,
        callback: Optional[Any] = None,
    ) -> "NotificationSubscription":
        """Subscribe to object change/delete notifications.

        Since the energistics library doesn't include Protocol 5 message classes,
        this uses a polling-based approach with GetDeletedResources and
        last_changed timestamps to detect changes.

        Parameters
        ----------
        object_types : list of str, optional
            RESQML types to watch (e.g. ["IjkGrid", "ContinuousProperty"]).
        uuids : list of str, optional
            Specific object UUIDs to watch.
        include_object_data : bool
            If True, changed objects include full data in notifications.
        callback : callable, optional
            Function called with (event_type, object_info) on changes.
            event_type is "changed", "created", or "deleted".

        Returns
        -------
        NotificationSubscription
            A subscription handle with .poll(), .stop(), and context manager support.

        Examples
        --------
        >>> sub = provider.subscribe_notifications(object_types=["IjkGrid"])
        >>> # Poll for changes (non-blocking)
        >>> events = sub.poll()
        >>> for event in events:
        ...     print(f"{event['event']}: {event['title']} ({event['uuid'][:8]})")

        >>> # Or use as context manager with callback
        >>> def on_change(event_type, info):
        ...     print(f"Object {event_type}: {info['title']}")
        >>> with provider.subscribe_notifications(callback=on_change) as sub:
        ...     # ... do work ...
        ...     sub.poll()
        """
        return NotificationSubscription(
            provider=self,
            object_types=object_types,
            uuids=uuids,
            include_object_data=include_object_data,
            callback=callback,
        )

    # ------------------------------------------------------------------
    # GetDataObjects (XML)
    # ------------------------------------------------------------------

    def _get_xml(self, uuid: str, uri: Optional[str] = None) -> Optional[str]:
        return self._run(self._async_get_xml(uuid, uri))

    async def _async_get_xml(
        self, uuid: str, uri: Optional[str] = None
    ) -> Optional[str]:
        from energistics.etp.v12.protocol.store import GetDataObjects

        if uri is None:
            objects = await self._async_list_objects()
            for obj in objects:
                if obj["uuid"] == uuid:
                    uri = obj["uri"]
                    break
            if uri is None:
                return None

        msg = GetDataObjects(uris={"0": uri}, format="xml")
        responses = await self._send_and_recv(msg)

        for resp in responses:
            data_objects = getattr(resp, "data_objects", {})
            for key, dobj in data_objects.items():
                xml_bytes = dobj.data
                if isinstance(xml_bytes, bytes):
                    return xml_bytes.decode("utf-8")
                return str(xml_bytes)
        return None

    # ------------------------------------------------------------------
    # PutDataObjects (XML)
    # ------------------------------------------------------------------

    def _put_xml(self, uri: str, xml_content: str, qualified_type: str) -> None:
        self._run(self._async_put_xml(uri, xml_content, qualified_type))

    async def _async_put_xml(
        self, uri: str, xml_content: str, qualified_type: str
    ) -> None:
        import time

        from energistics.etp.v12.datatypes.object.active_status_kind import (
            ActiveStatusKind,
        )
        from energistics.etp.v12.datatypes.object.data_object import DataObject
        from energistics.etp.v12.datatypes.object.resource import Resource
        from energistics.etp.v12.protocol.store import PutDataObjects

        now_us = int(time.time() * 1_000_000)

        resource = Resource(
            uri=uri,
            alternate_uris=[],
            name=qualified_type,
            source_count=None,
            target_count=None,
            last_changed=now_us,
            store_last_write=now_us,
            store_created=now_us,
            active_status=ActiveStatusKind.ACTIVE,
            custom_data={},
        )

        data_obj = DataObject(
            resource=resource,
            format="xml",
            blob_id=None,
            data=xml_content.encode("utf-8"),
        )

        msg = PutDataObjects(
            data_objects={"0": data_obj}, prune_contained_objects=False
        )
        await self._send_and_recv(msg)

    # ------------------------------------------------------------------
    # GetDataArrays
    # ------------------------------------------------------------------

    def _get_array(self, uri: str, path_in_resource: str) -> Optional[np.ndarray]:
        return self._run(self._async_get_array(uri, path_in_resource))

    async def _async_get_array(
        self, uri: str, path_in_resource: str
    ) -> Optional[np.ndarray]:
        from energistics.etp.v12.datatypes.data_array_types.data_array_identifier import (  # noqa: E501
            DataArrayIdentifier,
        )
        from energistics.etp.v12.protocol.data_array import GetDataArrays

        identifier = DataArrayIdentifier(uri=uri, path_in_resource=path_in_resource)
        msg = GetDataArrays(data_arrays={"0": identifier})

        try:
            responses = await self._send_and_recv(msg)
        except Exception as e:
            logger.debug("GetDataArrays failed for %s: %s", path_in_resource, e)
            return None

        for resp in responses:
            data_arrays = getattr(resp, "data_arrays", {})
            for key, arr_resp in data_arrays.items():
                etp_array = (
                    arr_resp.data_array if hasattr(arr_resp, "data_array") else arr_resp
                )
                if hasattr(etp_array, "to_numpy_array"):
                    return etp_array.to_numpy_array()
                # Fallback: manual extraction
                dims = etp_array.dimensions
                data = etp_array.data
                if hasattr(data, "item"):
                    if isinstance(data.item, bytes):
                        arr = np.frombuffer(data.item, dtype=np.float64)
                    elif hasattr(data.item, "values"):
                        arr = np.array(data.item.values)
                    else:
                        arr = np.array(data.item)
                    return arr.reshape(dims) if dims else arr
        return None

    # ------------------------------------------------------------------
    # PutDataArrays
    # ------------------------------------------------------------------

    def _put_array(self, uri: str, path_in_resource: str, data: np.ndarray) -> None:
        self._run(self._async_put_array(uri, path_in_resource, data))

    async def _async_put_array(
        self, uri: str, path_in_resource: str, data: np.ndarray
    ) -> None:
        from energistics.etp.v12.datatypes.data_array_types.data_array import (
            DataArray as EtpDataArray,
        )
        from energistics.etp.v12.datatypes.data_array_types.data_array_identifier import (  # noqa: E501
            DataArrayIdentifier,
        )
        from energistics.etp.v12.datatypes.data_array_types.put_data_arrays_type import (  # noqa: E501
            PutDataArraysType,
        )
        from energistics.etp.v12.protocol.data_array import PutDataArrays

        # Cast to an ETP-compatible numpy type and ensure contiguous ndarray
        if np.issubdtype(data.dtype, np.floating):
            arr = np.ascontiguousarray(data, dtype=np.float64)
        elif np.issubdtype(data.dtype, np.integer):
            if data.dtype.itemsize <= 4:
                arr = np.ascontiguousarray(data, dtype=np.int32)
            else:
                arr = np.ascontiguousarray(data, dtype=np.int64)
        else:
            arr = np.ascontiguousarray(data, dtype=np.float64)

        etp_array = EtpDataArray.from_numpy_array(arr)

        uid = DataArrayIdentifier(uri=uri, path_in_resource=path_in_resource)
        put_type = PutDataArraysType(uid=uid, array=etp_array, custom_data={})

        msg = PutDataArrays(data_arrays={"0": put_type})
        await self._send_and_recv(msg)

    # ------------------------------------------------------------------
    # IJK Grid
    # ------------------------------------------------------------------

    def get_ijk_grid_geometry(self, uuid: str) -> Dict[str, Any]:
        """Read IJK grid geometry from the data source."""
        from lxml import etree

        from ._resqml_enums import NS_RESQML20

        uri = self._resolve_uri(uuid, "resqml20.IjkGridRepresentation")
        xml_str = self._get_xml(uuid, uri=uri)
        if xml_str is None:
            raise ValueError(f"IJK grid {uuid} not found via ETP")

        root = etree.fromstring(
            xml_str.encode("utf-8") if isinstance(xml_str, str) else xml_str
        )

        ni = int(root.findtext(f".//{{{NS_RESQML20}}}Ni") or 0)
        nj = int(root.findtext(f".//{{{NS_RESQML20}}}Nj") or 0)
        nk = int(root.findtext(f".//{{{NS_RESQML20}}}Nk") or 0)

        k_dir_el = root.find(f".//{{{NS_RESQML20}}}KDirection")
        k_direction = "down" if k_dir_el is None else (k_dir_el.text or "down").lower()

        crs_uuid = ""
        crs_ref = root.find(f".//{{{NS_RESQML20}}}LocalCrs")
        if crs_ref is not None:
            crs_uuid = _uuid_from_ref(crs_ref)

        # Fetch arrays — try our custom paths first, then standard RESQML paths
        array_uri = self._resolve_uri(uuid, "resqml20.IjkGridRepresentation")
        coord = self._get_array(array_uri, f"/RESQML/{uuid}/Points/Coordinates")
        zcorn = self._get_array(array_uri, f"/RESQML/{uuid}/Points/ZCorners")
        actnum = self._get_array(array_uri, f"/RESQML/{uuid}/CellGeometryIsDefined")
        # Also try lowercase path (RMS exports use lowercase)
        if actnum is None:
            actnum = self._get_array(
                array_uri, f"/RESQML/{uuid}/cellGeometryIsDefined"
            )

        # If custom paths not found, try standard RESQML Point3dHdf5Array path
        if coord is None and zcorn is None:
            points = self._get_array(array_uri, f"/RESQML/{uuid}/Points")
            if points is not None:
                coord, zcorn = _points_array_to_coord_zcorn(points, ni, nj, nk)

        # If still not found, try parametric representation
        # (Point3dParametricArray — used by RMS exports)
        if coord is None and zcorn is None:
            cp = self._get_array(
                array_uri, f"/RESQML/{uuid}/controlPoints"
            )
            cpp = self._get_array(
                array_uri, f"/RESQML/{uuid}/controlPointParameters"
            )
            params = self._get_array(
                array_uri, f"/RESQML/{uuid}/parameters"
            )
            if cp is not None and cpp is not None and params is not None:
                npillars = (ni + 1) * (nj + 1)
                n_param_cols = params.reshape(nk + 1, -1).shape[1]

                if n_param_cols > npillars:
                    # Split coordinate lines present (faulted grid)
                    pil_idx = self._get_array(
                        array_uri, f"/RESQML/{uuid}/PillarIndices"
                    )
                    col_el = self._get_array(
                        array_uri,
                        f"/RESQML/{uuid}/ColumnsPerSplitCoordinateLine/elements",
                    )
                    col_cum = self._get_array(
                        array_uri,
                        f"/RESQML/{uuid}/ColumnsPerSplitCoordinateLine/cumulativeLength",
                    )
                    if pil_idx is not None and col_el is not None and col_cum is not None:
                        coord, zcorn = _parametric_split_to_coord_zcorn(
                            cp, cpp, params, ni, nj, nk,
                            pil_idx, col_el, col_cum,
                        )
                    else:
                        # Fall back to unsplit (ignore extra columns)
                        params_trimmed = params.reshape(nk + 1, -1)[:, :npillars]
                        explicit = _parametric_to_explicit_points(
                            cp, cpp, params_trimmed, ni, nj, nk
                        )
                        coord, zcorn = _points_array_to_coord_zcorn(
                            explicit, ni, nj, nk
                        )
                else:
                    # No splits — simple parametric
                    explicit = _parametric_to_explicit_points(
                        cp, cpp, params, ni, nj, nk
                    )
                    coord, zcorn = _points_array_to_coord_zcorn(
                        explicit, ni, nj, nk
                    )

        # Handle actnum shape: RESQML uses (nk, nj, ni), xtgeo needs flat (ni*nj*nk)
        # in I-fastest order (ni, nj, nk) when reshaped
        if actnum is not None and actnum.size == ni * nj * nk:
            if actnum.ndim == 3 and actnum.shape == (nk, nj, ni):
                # RESQML order (K,J,I) → transpose to xtgeo order (I,J,K) then flatten
                actnum = actnum.transpose(2, 1, 0).astype(np.int32).ravel()
            elif actnum.ndim == 1:
                actnum = actnum.astype(np.int32)
            else:
                actnum = actnum.ravel().astype(np.int32)
        elif actnum is None:
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
        from lxml import etree

        from ._resqml_enums import NS_COMMON20, NS_RESQML20, RESQML_NS_MAP

        if not uuid:
            uuid = _make_uuid()

        qualified_type = "resqml20.IjkGridRepresentation"
        uri = _uri_for_object(self._config.dataspace, qualified_type, uuid)

        # Build XML
        root = etree.Element(
            f"{{{NS_RESQML20}}}IjkGridRepresentation", nsmap=RESQML_NS_MAP
        )
        root.set("uuid", uuid)
        root.set("schemaVersion", "2.0")

        _add_citation(root, title)

        etree.SubElement(root, f"{{{NS_RESQML20}}}Ni").text = str(ni)
        etree.SubElement(root, f"{{{NS_RESQML20}}}Nj").text = str(nj)
        etree.SubElement(root, f"{{{NS_RESQML20}}}Nk").text = str(nk)
        etree.SubElement(root, f"{{{NS_RESQML20}}}KDirection").text = k_direction

        # Grid handedness (inferred from geometry)
        try:
            from xtgeo import Grid as _Grid

            _tmp = _Grid(
                coord.reshape((ni + 1, nj + 1, 6)),
                zcorn.reshape((ni + 1, nj + 1, nk + 1, 4)),
                actnum.reshape((ni, nj, nk)),
            )
            is_right = _tmp.ijk_handedness == "right"
        except Exception:
            is_right = True
        etree.SubElement(root, f"{{{NS_RESQML20}}}GridIsRighthanded").text = (
            "true" if is_right else "false"
        )

        # Pillar shape (straight for corner-point grids with linear pillars)
        etree.SubElement(root, f"{{{NS_RESQML20}}}PillarShape").text = "straight"

        crs_ref = etree.SubElement(root, f"{{{NS_RESQML20}}}LocalCrs")
        crs_ref.set("uuid", crs_uuid)

        # Geometry with external data references
        geom = etree.SubElement(root, f"{{{NS_RESQML20}}}Geometry")
        points = etree.SubElement(geom, f"{{{NS_RESQML20}}}Points")
        hdf_ref = etree.SubElement(points, f"{{{NS_COMMON20}}}HdfProxy")
        hdf_ref.set("uuid", _make_uuid())
        path_el = etree.SubElement(points, f"{{{NS_COMMON20}}}PathInHdfFile")
        path_el.text = f"/RESQML/{uuid}/Points/Coordinates"

        # ZCorners reference
        zc = etree.SubElement(geom, f"{{{NS_RESQML20}}}ZCorners")
        hdf_ref2 = etree.SubElement(zc, f"{{{NS_COMMON20}}}HdfProxy")
        hdf_ref2.set("uuid", _make_uuid())
        path_el2 = etree.SubElement(zc, f"{{{NS_COMMON20}}}PathInHdfFile")
        path_el2.text = f"/RESQML/{uuid}/Points/ZCorners"

        # CellGeometryIsDefined reference
        actref = etree.SubElement(geom, f"{{{NS_RESQML20}}}CellGeometryIsDefined")
        hdf_ref3 = etree.SubElement(actref, f"{{{NS_COMMON20}}}HdfProxy")
        hdf_ref3.set("uuid", _make_uuid())
        path_el3 = etree.SubElement(actref, f"{{{NS_COMMON20}}}PathInHdfFile")
        path_el3.text = f"/RESQML/{uuid}/CellGeometryIsDefined"

        xml_str = etree.tostring(root, encoding="unicode")

        self._run(self._start_transaction())
        self._put_xml(uri, xml_str, qualified_type)
        self._put_array(
            uri, f"/RESQML/{uuid}/Points/Coordinates", coord.astype(np.float64)
        )
        self._put_array(
            uri, f"/RESQML/{uuid}/Points/ZCorners", zcorn.astype(np.float64)
        )
        self._put_array(
            uri, f"/RESQML/{uuid}/CellGeometryIsDefined", actnum.astype(np.int32)
        )
        self._run(self._commit_transaction())

        return uuid

    # ------------------------------------------------------------------
    # Grid2D (Surface)
    # ------------------------------------------------------------------

    def get_grid2d_geometry(self, uuid: str) -> Dict[str, Any]:
        """Read Grid2D representation (regular surface)."""
        import math

        from lxml import etree

        from ._resqml_enums import NS_COMMON20, NS_RESQML20

        qualified_type = "resqml20.Grid2dRepresentation"
        uri = self._resolve_uri(uuid, qualified_type)
        xml_str = self._get_xml(uuid, uri=uri)
        if xml_str is None:
            raise ValueError(f"Grid2D {uuid} not found via ETP")

        root = etree.fromstring(
            xml_str.encode("utf-8") if isinstance(xml_str, str) else xml_str
        )

        ni = int(root.findtext(f".//{{{NS_RESQML20}}}FastestAxisCount") or 0)
        nj = int(root.findtext(f".//{{{NS_RESQML20}}}SlowestAxisCount") or 0)
        origin_x = float(
            root.findtext(f".//{{{NS_RESQML20}}}Origin/{{{NS_RESQML20}}}Coordinate1")
            or 0.0
        )
        origin_y = float(
            root.findtext(f".//{{{NS_RESQML20}}}Origin/{{{NS_RESQML20}}}Coordinate2")
            or 0.0
        )

        # Extract di, dj and rotation from offset vectors
        # RESQML spec: 1st Offset = J (slowest axis), 2nd Offset = I (fastest axis)
        offsets = root.findall(f".//{{{NS_RESQML20}}}Offset")
        di = 1.0
        dj = 1.0
        rotation = 0.0

        if len(offsets) >= 2:
            # 2nd offset = I (fastest axis) — use for di and rotation
            ox_i = float(offsets[1].findtext(f"{{{NS_RESQML20}}}Coordinate1") or 1.0)
            oy_i = float(offsets[1].findtext(f"{{{NS_RESQML20}}}Coordinate2") or 0.0)
            di = math.sqrt(ox_i * ox_i + oy_i * oy_i)
            if di > 0:
                rotation = math.atan2(oy_i, ox_i)
            sp_i = offsets[1].findtext(
                f"{{{NS_RESQML20}}}Spacing/{{{NS_RESQML20}}}Value"
            )
            if sp_i:
                di = float(sp_i)

            # 1st offset = J (slowest axis) — use for dj
            sp_j = offsets[0].findtext(
                f"{{{NS_RESQML20}}}Spacing/{{{NS_RESQML20}}}Value"
            )
            if sp_j:
                dj = float(sp_j)
            else:
                ox_j = float(
                    offsets[0].findtext(f"{{{NS_RESQML20}}}Coordinate1") or 0.0
                )
                oy_j = float(
                    offsets[0].findtext(f"{{{NS_RESQML20}}}Coordinate2") or 1.0
                )
                dj = math.sqrt(ox_j * ox_j + oy_j * oy_j)
        elif len(offsets) == 1:
            # Only one offset — treat as I direction
            ox_i = float(offsets[0].findtext(f"{{{NS_RESQML20}}}Coordinate1") or 1.0)
            oy_i = float(offsets[0].findtext(f"{{{NS_RESQML20}}}Coordinate2") or 0.0)
            di = math.sqrt(ox_i * ox_i + oy_i * oy_i)
            if di > 0:
                rotation = math.atan2(oy_i, ox_i)
            sp_i = offsets[0].findtext(
                f"{{{NS_RESQML20}}}Spacing/{{{NS_RESQML20}}}Value"
            )
            if sp_i:
                di = float(sp_i)

        crs_uuid = ""
        crs_ref = root.find(f".//{{{NS_RESQML20}}}LocalCrs")
        if crs_ref is not None:
            crs_uuid = _uuid_from_ref(crs_ref)

        uri = self._resolve_uri(uuid, "resqml20.Grid2dRepresentation")
        hdf_paths = _hdf_paths_from_xml(root, NS_COMMON20)
        zpath = hdf_paths.get("ZValues", f"/RESQML/{uuid}/ZValues")
        values = self._get_array(uri, zpath)
        if values is None and zpath != f"/RESQML/{uuid}/ZValues":
            values = self._get_array(uri, f"/RESQML/{uuid}/ZValues")
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
        import math

        from lxml import etree

        from ._resqml_enums import NS_COMMON20, NS_RESQML20, RESQML_NS_MAP

        if not uuid:
            uuid = _make_uuid()

        qualified_type = "resqml20.Grid2dRepresentation"
        uri = _uri_for_object(self._config.dataspace, qualified_type, uuid)

        root = etree.Element(
            f"{{{NS_RESQML20}}}Grid2dRepresentation", nsmap=RESQML_NS_MAP
        )
        root.set("uuid", uuid)
        root.set("schemaVersion", "2.0")

        _add_citation(root, title)

        patch = etree.SubElement(root, f"{{{NS_RESQML20}}}Grid2dPatch")
        etree.SubElement(patch, f"{{{NS_RESQML20}}}FastestAxisCount").text = str(ni)
        etree.SubElement(patch, f"{{{NS_RESQML20}}}SlowestAxisCount").text = str(nj)

        geom = etree.SubElement(patch, f"{{{NS_RESQML20}}}Geometry")
        origin_el = etree.SubElement(geom, f"{{{NS_RESQML20}}}Origin")
        etree.SubElement(origin_el, f"{{{NS_RESQML20}}}Coordinate1").text = str(
            origin_x
        )
        etree.SubElement(origin_el, f"{{{NS_RESQML20}}}Coordinate2").text = str(
            origin_y
        )
        etree.SubElement(origin_el, f"{{{NS_RESQML20}}}Coordinate3").text = "0.0"

        # Offset vectors (encode rotation)
        # RESQML spec: 1st Offset = J (slowest axis), 2nd Offset = I (fastest axis)
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)

        # 1st Offset: J direction (perpendicular to I, rotated)
        offset1 = etree.SubElement(geom, f"{{{NS_RESQML20}}}Offset")
        etree.SubElement(offset1, f"{{{NS_RESQML20}}}Coordinate1").text = str(
            -dj * sin_r
        )
        etree.SubElement(offset1, f"{{{NS_RESQML20}}}Coordinate2").text = str(
            dj * cos_r
        )
        etree.SubElement(offset1, f"{{{NS_RESQML20}}}Coordinate3").text = "0.0"
        spacing1 = etree.SubElement(offset1, f"{{{NS_RESQML20}}}Spacing")
        etree.SubElement(spacing1, f"{{{NS_RESQML20}}}Value").text = str(dj)

        # 2nd Offset: I direction (primary, carries rotation)
        offset2 = etree.SubElement(geom, f"{{{NS_RESQML20}}}Offset")
        etree.SubElement(offset2, f"{{{NS_RESQML20}}}Coordinate1").text = str(
            di * cos_r
        )
        etree.SubElement(offset2, f"{{{NS_RESQML20}}}Coordinate2").text = str(
            di * sin_r
        )
        etree.SubElement(offset2, f"{{{NS_RESQML20}}}Coordinate3").text = "0.0"
        spacing2 = etree.SubElement(offset2, f"{{{NS_RESQML20}}}Spacing")
        etree.SubElement(spacing2, f"{{{NS_RESQML20}}}Value").text = str(di)

        # Z values external data array reference
        z_vals = etree.SubElement(geom, f"{{{NS_RESQML20}}}ZValues")
        hdf_ref = etree.SubElement(z_vals, f"{{{NS_COMMON20}}}HdfProxy")
        hdf_ref.set("uuid", _make_uuid())
        path_el = etree.SubElement(z_vals, f"{{{NS_COMMON20}}}PathInHdfFile")
        path_el.text = f"/RESQML/{uuid}/ZValues"

        crs_ref = etree.SubElement(root, f"{{{NS_RESQML20}}}LocalCrs")
        crs_ref.set("uuid", crs_uuid)

        xml_str = etree.tostring(root, encoding="unicode")

        self._run(self._start_transaction())
        self._put_xml(uri, xml_str, qualified_type)
        self._put_array(uri, f"/RESQML/{uuid}/ZValues", values.astype(np.float64))
        self._run(self._commit_transaction())

        return uuid

    # ------------------------------------------------------------------
    # PointSet
    # ------------------------------------------------------------------

    def get_pointset(self, uuid: str) -> Dict[str, Any]:
        """Read PointSet representation."""
        from lxml import etree

        from ._resqml_enums import NS_COMMON20, NS_RESQML20

        qualified_type = "resqml20.PointSetRepresentation"
        uri = self._resolve_uri(uuid, qualified_type)
        xml_str = self._get_xml(uuid, uri=uri)
        if xml_str is None:
            raise ValueError(f"PointSet {uuid} not found via ETP")

        root = etree.fromstring(
            xml_str.encode("utf-8") if isinstance(xml_str, str) else xml_str
        )

        crs_uuid = ""
        crs_ref = root.find(f".//{{{NS_RESQML20}}}LocalCrs")
        if crs_ref is not None:
            crs_uuid = _uuid_from_ref(crs_ref)

        uri = self._resolve_uri(uuid, "resqml20.PointSetRepresentation")
        hdf_paths = _hdf_paths_from_xml(root, NS_COMMON20)
        pts_path = hdf_paths.get("Points", f"/RESQML/{uuid}/Points")
        points = self._get_array(uri, pts_path)
        if points is None and pts_path != f"/RESQML/{uuid}/Points":
            points = self._get_array(uri, f"/RESQML/{uuid}/Points")
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
        from lxml import etree

        from ._resqml_enums import NS_COMMON20, NS_RESQML20, RESQML_NS_MAP

        if not uuid:
            uuid = _make_uuid()

        qualified_type = "resqml20.PointSetRepresentation"
        uri = _uri_for_object(self._config.dataspace, qualified_type, uuid)

        root = etree.Element(
            f"{{{NS_RESQML20}}}PointSetRepresentation", nsmap=RESQML_NS_MAP
        )
        root.set("uuid", uuid)
        root.set("schemaVersion", "2.0")

        _add_citation(root, title)

        # NodePatch with external data reference
        patch = etree.SubElement(root, f"{{{NS_RESQML20}}}NodePatch")
        etree.SubElement(patch, f"{{{NS_RESQML20}}}Count").text = str(len(points))
        geom = etree.SubElement(patch, f"{{{NS_RESQML20}}}Geometry")
        pts = etree.SubElement(geom, f"{{{NS_RESQML20}}}Points")
        hdf_ref = etree.SubElement(pts, f"{{{NS_COMMON20}}}HdfProxy")
        hdf_ref.set("uuid", _make_uuid())
        path_el = etree.SubElement(pts, f"{{{NS_COMMON20}}}PathInHdfFile")
        path_el.text = f"/RESQML/{uuid}/Points"

        crs_ref = etree.SubElement(root, f"{{{NS_RESQML20}}}LocalCrs")
        crs_ref.set("uuid", crs_uuid)

        xml_str = etree.tostring(root, encoding="unicode")

        self._run(self._start_transaction())
        self._put_xml(uri, xml_str, qualified_type)
        self._put_array(uri, f"/RESQML/{uuid}/Points", points.astype(np.float64))
        self._run(self._commit_transaction())

        return uuid

    # ------------------------------------------------------------------
    # PolylineSet
    # ------------------------------------------------------------------

    def get_polylineset(self, uuid: str) -> Dict[str, Any]:
        """Read PolylineSet representation."""
        from lxml import etree

        from ._resqml_enums import NS_COMMON20, NS_RESQML20

        qualified_type = "resqml20.PolylineSetRepresentation"
        uri = self._resolve_uri(uuid, qualified_type)
        xml_str = self._get_xml(uuid, uri=uri)
        if xml_str is None:
            raise ValueError(f"PolylineSet {uuid} not found via ETP")

        root = etree.fromstring(
            xml_str.encode("utf-8") if isinstance(xml_str, str) else xml_str
        )

        crs_uuid = ""
        crs_ref = root.find(f".//{{{NS_RESQML20}}}LocalCrs")
        if crs_ref is not None:
            crs_uuid = _uuid_from_ref(crs_ref)

        uri = self._resolve_uri(uuid, "resqml20.PolylineSetRepresentation")
        hdf_paths = _hdf_paths_from_xml(root, NS_COMMON20)
        pts_path = hdf_paths.get("Points", f"/RESQML/{uuid}/Points")
        ncp_path = hdf_paths.get(
            "NodeCountPerPolyline", f"/RESQML/{uuid}/NodeCountPerPolyline"
        )
        closed_path = hdf_paths.get(
            "ClosedPolylines", f"/RESQML/{uuid}/ClosedPolylines"
        )
        all_points = self._get_array(uri, pts_path)
        node_counts = self._get_array(uri, ncp_path)
        closed_flags = self._get_array(uri, closed_path)

        # Handle inline BooleanConstantArray for ClosedPolylines
        if closed_flags is None:
            cp_el = root.find(f".//{{{NS_RESQML20}}}ClosedPolylines")
            if cp_el is not None:
                val_el = cp_el.find(f"{{{NS_RESQML20}}}Value")
                cnt_el = cp_el.find(f"{{{NS_RESQML20}}}Count")
                if val_el is not None and cnt_el is not None:
                    bval = val_el.text.strip().lower() in ("true", "1")
                    cnt = int(cnt_el.text or 0)
                    closed_flags = np.full(cnt, bval, dtype=bool)

        polylines: List[np.ndarray] = []
        closed_list: List[bool] = []

        if all_points is not None and node_counts is not None:
            offset = 0
            for i, count in enumerate(node_counts.flatten()):
                c = int(count)
                polylines.append(all_points.reshape(-1, 3)[offset : offset + c])
                offset += c
                if closed_flags is not None and i < closed_flags.size:
                    closed_list.append(bool(closed_flags.flatten()[i]))
                else:
                    closed_list.append(False)
        elif all_points is not None:
            polylines.append(all_points.reshape(-1, 3))
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
        from lxml import etree

        from ._resqml_enums import NS_COMMON20, NS_RESQML20, RESQML_NS_MAP

        if not uuid:
            uuid = _make_uuid()

        qualified_type = "resqml20.PolylineSetRepresentation"
        uri = _uri_for_object(self._config.dataspace, qualified_type, uuid)

        root = etree.Element(
            f"{{{NS_RESQML20}}}PolylineSetRepresentation", nsmap=RESQML_NS_MAP
        )
        root.set("uuid", uuid)
        root.set("schemaVersion", "2.0")

        _add_citation(root, title)

        # LinePatch with external data references
        patch = etree.SubElement(root, f"{{{NS_RESQML20}}}LinePatch")
        etree.SubElement(patch, f"{{{NS_RESQML20}}}Count").text = str(
            len(polylines) if polylines else 0
        )
        geom = etree.SubElement(patch, f"{{{NS_RESQML20}}}Geometry")
        pts = etree.SubElement(geom, f"{{{NS_RESQML20}}}Points")
        hdf_ref = etree.SubElement(pts, f"{{{NS_COMMON20}}}HdfProxy")
        hdf_ref.set("uuid", _make_uuid())
        path_el = etree.SubElement(pts, f"{{{NS_COMMON20}}}PathInHdfFile")
        path_el.text = f"/RESQML/{uuid}/Points"

        # NodeCountPerPolyline reference
        ncp = etree.SubElement(patch, f"{{{NS_RESQML20}}}NodeCountPerPolyline")
        hdf_ref2 = etree.SubElement(ncp, f"{{{NS_COMMON20}}}HdfProxy")
        hdf_ref2.set("uuid", _make_uuid())
        path_el2 = etree.SubElement(ncp, f"{{{NS_COMMON20}}}PathInHdfFile")
        path_el2.text = f"/RESQML/{uuid}/NodeCountPerPolyline"

        # ClosedPolylines reference
        cp = etree.SubElement(patch, f"{{{NS_RESQML20}}}ClosedPolylines")
        hdf_ref3 = etree.SubElement(cp, f"{{{NS_COMMON20}}}HdfProxy")
        hdf_ref3.set("uuid", _make_uuid())
        path_el3 = etree.SubElement(cp, f"{{{NS_COMMON20}}}PathInHdfFile")
        path_el3.text = f"/RESQML/{uuid}/ClosedPolylines"

        crs_ref = etree.SubElement(root, f"{{{NS_RESQML20}}}LocalCrs")
        crs_ref.set("uuid", crs_uuid)

        xml_str = etree.tostring(root, encoding="unicode")

        self._run(self._start_transaction())
        self._put_xml(uri, xml_str, qualified_type)

        if polylines:
            all_points = np.vstack(polylines).astype(np.float64)
            node_counts = np.array([len(p) for p in polylines], dtype=np.int64)
        else:
            all_points = np.zeros((0, 3), dtype=np.float64)
            node_counts = np.zeros(0, dtype=np.int64)

        closed_arr = np.array(closed, dtype=bool)

        self._put_array(uri, f"/RESQML/{uuid}/Points", all_points)
        self._put_array(uri, f"/RESQML/{uuid}/NodeCountPerPolyline", node_counts)
        self._put_array(uri, f"/RESQML/{uuid}/ClosedPolylines", closed_arr)
        self._run(self._commit_transaction())

        return uuid

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def get_property_values(
        self, uuid: str, object_type: str = "ContinuousProperty"
    ) -> Dict[str, Any]:
        """Read property values."""
        from lxml import etree

        from ._resqml_enums import NS_COMMON20, NS_RESQML20

        qualified_type = f"resqml20.{object_type}"
        uri = self._resolve_uri(uuid, qualified_type)
        xml_str = self._get_xml(uuid, uri=uri)
        if xml_str is None:
            raise ValueError(f"Property {uuid} not found via ETP")

        root = etree.fromstring(
            xml_str.encode("utf-8") if isinstance(xml_str, str) else xml_str
        )
        tag = etree.QName(root.tag).localname

        title = ""
        citation = root.find(f"{{{NS_COMMON20}}}Citation")
        if citation is not None:
            t = citation.find(f"{{{NS_COMMON20}}}Title")
            if t is not None:
                title = t.text or ""

        property_kind = ""
        pk_el = root.find(f".//{{{NS_RESQML20}}}PropertyKind")
        if pk_el is not None:
            kind_title = pk_el.find(f".//{{{NS_COMMON20}}}Title")
            if kind_title is not None:
                property_kind = kind_title.text or ""
            if not property_kind:
                kind_el = pk_el.find(f"{{{NS_RESQML20}}}Kind")
                if kind_el is not None:
                    property_kind = kind_el.text or ""

        idx_el = root.find(f"{{{NS_RESQML20}}}IndexableElement")
        indexable_element = idx_el.text if idx_el is not None else "cells"

        supp_uuid = ""
        supp_ref = root.find(f".//{{{NS_RESQML20}}}SupportingRepresentation")
        if supp_ref is not None:
            supp_uuid = _uuid_from_ref(supp_ref)

        uom = ""
        uom_el = root.find(f".//{{{NS_RESQML20}}}UOM")
        if uom_el is None:
            uom_el = root.find(f".//{{{NS_RESQML20}}}Uom")
        if uom_el is not None:
            uom = uom_el.text or ""

        is_discrete = "Discrete" in tag or "Categorical" in tag

        if "Discrete" in tag:
            obj_type = "resqml20.DiscreteProperty"
        elif "Categorical" in tag:
            obj_type = "resqml20.CategoricalProperty"
        else:
            obj_type = "resqml20.ContinuousProperty"

        uri = self._resolve_uri(uuid, obj_type)

        # Find the data array path from the XML PathInHdfFile element
        values = None
        path_el = root.find(f".//{{{NS_COMMON20}}}PathInHdfFile")
        if path_el is not None and path_el.text:
            values = self._get_array(uri, path_el.text)

        # Fallback: try common path conventions
        if values is None:
            for suffix in ("Values", "values_patch0", "values"):
                values = self._get_array(uri, f"/RESQML/{uuid}/{suffix}")
                if values is not None:
                    break

        if values is None:
            values = np.array([], dtype=np.float64)

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
        from lxml import etree

        from ._resqml_enums import NS_COMMON20, NS_RESQML20, RESQML_NS_MAP

        if not uuid:
            uuid = _make_uuid()

        tag_name = "DiscreteProperty" if is_discrete else "ContinuousProperty"
        qualified_type = f"resqml20.{tag_name}"
        uri = _uri_for_object(self._config.dataspace, qualified_type, uuid)

        root = etree.Element(f"{{{NS_RESQML20}}}{tag_name}", nsmap=RESQML_NS_MAP)
        root.set("uuid", uuid)
        root.set("schemaVersion", "2.0")

        _add_citation(root, title)

        etree.SubElement(
            root, f"{{{NS_RESQML20}}}IndexableElement"
        ).text = indexable_element

        supp = etree.SubElement(root, f"{{{NS_RESQML20}}}SupportingRepresentation")
        uuid_el = etree.SubElement(supp, f"{{{NS_COMMON20}}}UUID")
        uuid_el.text = supporting_representation_uuid

        pk = etree.SubElement(root, f"{{{NS_RESQML20}}}PropertyKind")
        kind_el = etree.SubElement(pk, f"{{{NS_RESQML20}}}Kind")
        kind_el.text = property_kind

        if uom:
            uom_el = etree.SubElement(root, f"{{{NS_RESQML20}}}UOM")
            uom_el.text = uom

        if facet:
            facet_el = etree.SubElement(root, f"{{{NS_RESQML20}}}Facet")
            fval = etree.SubElement(facet_el, f"{{{NS_RESQML20}}}Value")
            fval.text = facet

        # External data array reference for values
        vals = etree.SubElement(root, f"{{{NS_RESQML20}}}Values")
        hdf_ref = etree.SubElement(vals, f"{{{NS_COMMON20}}}HdfProxy")
        hdf_ref.set("uuid", _make_uuid())
        path_el = etree.SubElement(vals, f"{{{NS_COMMON20}}}PathInHdfFile")
        path_el.text = f"/RESQML/{uuid}/Values"

        xml_str = etree.tostring(root, encoding="unicode")

        self._run(self._start_transaction())
        self._put_xml(uri, xml_str, qualified_type)
        self._put_array(uri, f"/RESQML/{uuid}/Values", values)
        self._run(self._commit_transaction())

        return uuid

    # ------------------------------------------------------------------
    # CRS
    # ------------------------------------------------------------------

    def get_crs(self, uuid: str) -> Dict[str, Any]:
        """Read CRS definition."""
        from lxml import etree

        from ._crs import LocalDepth3dCrs

        qualified_type = "resqml20.LocalDepth3dCrs"
        uri = self._resolve_uri(uuid, qualified_type)
        xml_str = self._get_xml(uuid, uri=uri)
        if xml_str is None:
            raise ValueError(f"CRS {uuid} not found via ETP")

        root = etree.fromstring(
            xml_str.encode("utf-8") if isinstance(xml_str, str) else xml_str
        )
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
        from lxml import etree

        from ._crs import LocalDepth3dCrs

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
        xml_root = crs.to_xml()
        xml_str = etree.tostring(
            xml_root, xml_declaration=True, encoding="UTF-8"
        ).decode("utf-8")

        qualified_type = "resqml20.LocalDepth3dCrs"
        uri = _uri_for_object(self._config.dataspace, qualified_type, uuid)

        self._run(self._start_transaction())
        self._put_xml(uri, xml_str, qualified_type)
        self._run(self._commit_transaction())

        return uuid

    # ------------------------------------------------------------------
    # TriangulatedSet
    # ------------------------------------------------------------------

    def get_triangulated_set(self, uuid: str) -> Dict[str, Any]:
        """Read TriangulatedSetRepresentation."""
        from lxml import etree

        from ._resqml_enums import NS_COMMON20, NS_RESQML20

        qualified_type = "resqml20.TriangulatedSetRepresentation"
        uri = self._resolve_uri(uuid, qualified_type)
        xml_str = self._get_xml(uuid, uri=uri)
        if xml_str is None:
            raise ValueError(f"TriangulatedSet {uuid} not found via ETP")

        root = etree.fromstring(
            xml_str.encode("utf-8") if isinstance(xml_str, str) else xml_str
        )

        crs_uuid = ""
        crs_ref = root.find(f".//{{{NS_RESQML20}}}LocalCrs")
        if crs_ref is not None:
            crs_uuid = _uuid_from_ref(crs_ref)

        title = ""
        citation = root.find(f"{{{NS_COMMON20}}}Citation")
        if citation is not None:
            t = citation.find(f"{{{NS_COMMON20}}}Title")
            if t is not None:
                title = t.text or ""

        uri = self._resolve_uri(uuid, qualified_type)
        hdf_paths = _hdf_paths_from_xml(root, NS_COMMON20)
        # Vertices (Points): try PathInHdfFile, then resqpy convention, then RESQML
        pts_path = hdf_paths.get("Points")
        vertices = None
        if pts_path:
            vertices = self._get_array(uri, pts_path)
        if vertices is None:
            vertices = self._get_array(uri, f"/RESQML/{uuid}/points_patch0")
        if vertices is None:
            vertices = self._get_array(uri, f"/RESQML/{uuid}/Points")
        # Triangles: try PathInHdfFile, then resqpy convention, then RESQML
        tri_path = hdf_paths.get("Triangles")
        triangles = None
        if tri_path:
            triangles = self._get_array(uri, tri_path)
        if triangles is None:
            triangles = self._get_array(uri, f"/RESQML/{uuid}/triangles_patch0")
        if triangles is None:
            triangles = self._get_array(uri, f"/RESQML/{uuid}/Triangles")

        if vertices is None:
            vertices = np.zeros((0, 3), dtype=np.float64)
        if triangles is None:
            triangles = np.zeros((0, 3), dtype=np.int32)

        return {
            "vertices": vertices,
            "triangles": triangles,
            "crs_uuid": crs_uuid,
            "title": title,
        }

    def put_triangulated_set(
        self,
        uuid: str,
        title: str,
        vertices: np.ndarray,
        triangles: np.ndarray,
        crs_uuid: str,
    ) -> str:
        """Write TriangulatedSetRepresentation."""
        from lxml import etree

        from ._resqml_enums import NS_COMMON20, NS_RESQML20, RESQML_NS_MAP

        if not uuid:
            uuid = _make_uuid()

        qualified_type = "resqml20.TriangulatedSetRepresentation"
        uri = _uri_for_object(self._config.dataspace, qualified_type, uuid)

        root = etree.Element(
            f"{{{NS_RESQML20}}}TriangulatedSetRepresentation", nsmap=RESQML_NS_MAP
        )
        root.set("uuid", uuid)
        root.set("schemaVersion", "2.0")

        _add_citation(root, title)

        # SurfaceRole (required by RESQML 2.0.1 schema)
        role_el = etree.SubElement(root, f"{{{NS_RESQML20}}}SurfaceRole")
        role_el.text = "map"

        patch = etree.SubElement(root, f"{{{NS_RESQML20}}}TrianglePatch")

        patch_idx = etree.SubElement(patch, f"{{{NS_RESQML20}}}PatchIndex")
        patch_idx.text = "0"

        etree.SubElement(patch, f"{{{NS_RESQML20}}}Count").text = str(len(triangles))
        etree.SubElement(patch, f"{{{NS_RESQML20}}}NodeCount").text = str(len(vertices))

        # Triangles (IntegerHdf5Array)
        tri_el = etree.SubElement(patch, f"{{{NS_RESQML20}}}Triangles")
        hdf_ref2 = etree.SubElement(tri_el, f"{{{NS_COMMON20}}}HdfProxy")
        hdf_ref2.set("uuid", _make_uuid())
        path_el2 = etree.SubElement(tri_el, f"{{{NS_COMMON20}}}PathInHdfFile")
        path_el2.text = f"/RESQML/{uuid}/triangles_patch0"

        # Geometry with Points (Point3dHdf5Array)
        geom = etree.SubElement(patch, f"{{{NS_RESQML20}}}Geometry")
        crs_geom = etree.SubElement(geom, f"{{{NS_RESQML20}}}LocalCrs")
        crs_geom.set("uuid", crs_uuid)
        pts = etree.SubElement(geom, f"{{{NS_RESQML20}}}Points")
        hdf_ref = etree.SubElement(pts, f"{{{NS_COMMON20}}}HdfProxy")
        hdf_ref.set("uuid", _make_uuid())
        path_el = etree.SubElement(pts, f"{{{NS_COMMON20}}}PathInHdfFile")
        path_el.text = f"/RESQML/{uuid}/points_patch0"

        xml_str = etree.tostring(root, encoding="unicode")

        self._run(self._start_transaction())
        self._put_xml(uri, xml_str, qualified_type)
        self._put_array(
            uri, f"/RESQML/{uuid}/points_patch0", vertices.astype(np.float64)
        )
        self._put_array(
            uri, f"/RESQML/{uuid}/triangles_patch0", triangles.astype(np.int32)
        )
        self._run(self._commit_transaction())

        return uuid

    # ------------------------------------------------------------------
    # WellboreTrajectory
    # ------------------------------------------------------------------

    def get_wellbore_trajectory(self, uuid: str) -> Dict[str, Any]:
        """Read WellboreTrajectoryRepresentation."""
        from lxml import etree

        from ._resqml_enums import NS_COMMON20, NS_RESQML20

        qualified_type = "resqml20.WellboreTrajectoryRepresentation"
        uri = self._resolve_uri(uuid, qualified_type)
        xml_str = self._get_xml(uuid, uri=uri)
        if xml_str is None:
            raise ValueError(f"WellboreTrajectory {uuid} not found via ETP")

        root = etree.fromstring(
            xml_str.encode("utf-8") if isinstance(xml_str, str) else xml_str
        )

        crs_uuid = ""
        crs_ref = root.find(f".//{{{NS_RESQML20}}}LocalCrs")
        if crs_ref is not None:
            crs_uuid = _uuid_from_ref(crs_ref)

        title = ""
        citation = root.find(f"{{{NS_COMMON20}}}Citation")
        if citation is not None:
            t = citation.find(f"{{{NS_COMMON20}}}Title")
            if t is not None:
                title = t.text or ""

        uri = self._resolve_uri(uuid, qualified_type)
        hdf_paths = _hdf_paths_from_xml(root, NS_COMMON20)
        md_path = hdf_paths.get(
            "ControlPointParameters", f"/RESQML/{uuid}/MdValues"
        )
        pts_path = hdf_paths.get("ControlPoints", f"/RESQML/{uuid}/Points")
        md = self._get_array(uri, md_path)
        xyz = self._get_array(uri, pts_path)
        # Fallback to hardcoded paths
        if md is None and md_path != f"/RESQML/{uuid}/MdValues":
            md = self._get_array(uri, f"/RESQML/{uuid}/MdValues")
        if xyz is None and pts_path != f"/RESQML/{uuid}/Points":
            xyz = self._get_array(uri, f"/RESQML/{uuid}/Points")

        if md is None:
            md = np.zeros(0, dtype=np.float64)
        if xyz is None:
            xyz = np.zeros((0, 3), dtype=np.float64)

        # Find associated frames via discovery
        frames = []
        try:
            related = self.get_related_objects(uuid, direction="sources")
            for obj in related:
                if "WellboreFrame" in obj.get("type", ""):
                    frame_uuid = obj["uuid"]
                    frame_uri = self._resolve_uri(
                        frame_uuid,
                        "resqml20.WellboreFrameRepresentation",
                    )
                    frame_md = self._get_array(
                        frame_uri, f"/RESQML/{frame_uuid}/MdValues"
                    )
                    # Find properties on this frame
                    props = self._find_etp_frame_properties(frame_uuid)
                    frames.append(
                        {"uuid": frame_uuid, "md": frame_md, "properties": props}
                    )
        except Exception:
            pass  # Discovery may not find frames in all setups

        return {
            "md": md,
            "xyz": xyz,
            "crs_uuid": crs_uuid,
            "title": title,
            "frames": frames,
        }

    def _find_etp_frame_properties(self, frame_uuid: str) -> List[Dict[str, Any]]:
        """Find properties attached to a wellbore frame via ETP."""
        from ._resqml_enums import NS_COMMON20

        props = []
        try:
            related = self.get_related_objects(frame_uuid, direction="sources")
            for obj in related:
                if "Property" in obj.get("type", ""):
                    prop_uuid = obj["uuid"]
                    prop_type = obj.get("type", "resqml20.ContinuousProperty")
                    prop_uri = self._resolve_uri(prop_uuid, prop_type)
                    # Try to parse PathInHdfFile from XML
                    values = None
                    try:
                        from lxml import etree

                        xml_str = self._get_xml(prop_uuid, uri=prop_uri)
                        if xml_str:
                            proot = etree.fromstring(
                                xml_str.encode("utf-8")
                                if isinstance(xml_str, str)
                                else xml_str
                            )
                            hpaths = _hdf_paths_from_xml(proot, NS_COMMON20)
                            vpath = hpaths.get("Values")
                            if vpath:
                                values = self._get_array(prop_uri, vpath)
                    except Exception:
                        pass
                    if values is None:
                        for suffix in ("Values", "values_patch0", "values"):
                            values = self._get_array(
                                prop_uri, f"/RESQML/{prop_uuid}/{suffix}"
                            )
                            if values is not None:
                                break
                    is_discrete = "Discrete" in prop_type or "Categorical" in prop_type
                    props.append(
                        {
                            "uuid": prop_uuid,
                            "title": obj.get("title", ""),
                            "values": values
                            if values is not None
                            else np.array([], dtype=np.float64),
                            "is_discrete": is_discrete,
                        }
                    )
        except Exception:
            pass
        return props

    def put_wellbore_trajectory(
        self,
        uuid: str,
        title: str,
        md: np.ndarray,
        xyz: np.ndarray,
        crs_uuid: str,
    ) -> str:
        """Write WellboreTrajectoryRepresentation."""
        from lxml import etree

        from ._resqml_enums import NS_COMMON20, NS_RESQML20, RESQML_NS_MAP

        if not uuid:
            uuid = _make_uuid()

        # Create WellboreFeature → WellboreInterpretation → MdDatum chain
        feature_uuid = _make_uuid()
        interp_uuid = _make_uuid()
        md_datum_uuid = _make_uuid()
        self._put_wellbore_feature(feature_uuid, title)
        self._put_wellbore_interpretation(interp_uuid, title, feature_uuid)
        self._put_md_datum(
            md_datum_uuid, title, crs_uuid, xyz[0] if len(xyz) > 0 else np.zeros(3)
        )

        qualified_type = "resqml20.WellboreTrajectoryRepresentation"
        uri = _uri_for_object(self._config.dataspace, qualified_type, uuid)

        root = etree.Element(
            f"{{{NS_RESQML20}}}WellboreTrajectoryRepresentation", nsmap=RESQML_NS_MAP
        )
        root.set("uuid", uuid)
        root.set("schemaVersion", "2.0")

        _add_citation(root, title)

        # MdDatum reference
        md_datum_ref = etree.SubElement(root, f"{{{NS_RESQML20}}}MdDatum")
        md_datum_ref.set("uuid", md_datum_uuid)

        etree.SubElement(root, f"{{{NS_RESQML20}}}StartMd").text = str(
            float(md[0]) if len(md) > 0 else 0.0
        )
        etree.SubElement(root, f"{{{NS_RESQML20}}}FinishMd").text = str(
            float(md[-1]) if len(md) > 0 else 0.0
        )

        # RepresentedInterpretation reference
        interp_ref = etree.SubElement(
            root, f"{{{NS_RESQML20}}}RepresentedInterpretation"
        )
        interp_ref.set("uuid", interp_uuid)

        geom = etree.SubElement(root, f"{{{NS_RESQML20}}}Geometry")
        md_el = etree.SubElement(geom, f"{{{NS_RESQML20}}}ControlPointParameters")
        hdf_ref = etree.SubElement(md_el, f"{{{NS_COMMON20}}}HdfProxy")
        hdf_ref.set("uuid", _make_uuid())
        path_el = etree.SubElement(md_el, f"{{{NS_COMMON20}}}PathInHdfFile")
        path_el.text = f"/RESQML/{uuid}/MdValues"

        pts = etree.SubElement(geom, f"{{{NS_RESQML20}}}ControlPoints")
        hdf_ref2 = etree.SubElement(pts, f"{{{NS_COMMON20}}}HdfProxy")
        hdf_ref2.set("uuid", _make_uuid())
        path_el2 = etree.SubElement(pts, f"{{{NS_COMMON20}}}PathInHdfFile")
        path_el2.text = f"/RESQML/{uuid}/Points"

        crs_ref = etree.SubElement(root, f"{{{NS_RESQML20}}}LocalCrs")
        crs_ref.set("uuid", crs_uuid)

        xml_str = etree.tostring(root, encoding="unicode")

        self._run(self._start_transaction())
        self._put_xml(uri, xml_str, qualified_type)
        self._put_array(uri, f"/RESQML/{uuid}/MdValues", md.astype(np.float64))
        self._put_array(uri, f"/RESQML/{uuid}/Points", xyz.astype(np.float64))
        self._run(self._commit_transaction())

        return uuid

    def _put_wellbore_feature(self, uuid: str, title: str) -> str:
        """Write a WellboreFeature via ETP."""
        from lxml import etree

        from ._resqml_enums import NS_COMMON20, NS_RESQML20, RESQML_NS_MAP

        qualified_type = "resqml20.WellboreFeature"
        uri = _uri_for_object(self._config.dataspace, qualified_type, uuid)

        root = etree.Element(f"{{{NS_RESQML20}}}WellboreFeature", nsmap=RESQML_NS_MAP)
        root.set("uuid", uuid)
        root.set("schemaVersion", "2.0")

        _add_citation(root, title)

        xml_str = etree.tostring(root, encoding="unicode")
        self._run(self._start_transaction())
        self._put_xml(uri, xml_str, qualified_type)
        self._run(self._commit_transaction())
        return uuid

    def _put_wellbore_interpretation(
        self, uuid: str, title: str, feature_uuid: str
    ) -> str:
        """Write a WellboreInterpretation referencing a WellboreFeature via ETP."""
        from lxml import etree

        from ._resqml_enums import NS_COMMON20, NS_RESQML20, RESQML_NS_MAP

        qualified_type = "resqml20.WellboreInterpretation"
        uri = _uri_for_object(self._config.dataspace, qualified_type, uuid)

        root = etree.Element(
            f"{{{NS_RESQML20}}}WellboreInterpretation", nsmap=RESQML_NS_MAP
        )
        root.set("uuid", uuid)
        root.set("schemaVersion", "2.0")

        _add_citation(root, title)

        feat_ref = etree.SubElement(root, f"{{{NS_RESQML20}}}InterpretedFeature")
        feat_ref.set("uuid", feature_uuid)

        xml_str = etree.tostring(root, encoding="unicode")
        self._run(self._start_transaction())
        self._put_xml(uri, xml_str, qualified_type)
        self._run(self._commit_transaction())
        return uuid

    def _put_md_datum(
        self, uuid: str, title: str, crs_uuid: str, location: np.ndarray
    ) -> str:
        """Write an MdDatum object via ETP."""
        from lxml import etree

        from ._resqml_enums import NS_COMMON20, NS_RESQML20, RESQML_NS_MAP

        qualified_type = "resqml20.obj_MdDatum"
        uri = _uri_for_object(self._config.dataspace, qualified_type, uuid)

        root = etree.Element(f"{{{NS_RESQML20}}}MdDatum", nsmap=RESQML_NS_MAP)
        root.set("uuid", uuid)
        root.set("schemaVersion", "2.0")

        _add_citation(root, f"{title} MD Datum")

        loc = etree.SubElement(root, f"{{{NS_RESQML20}}}Location")
        etree.SubElement(loc, f"{{{NS_RESQML20}}}Coordinate1").text = str(
            float(location[0])
        )
        etree.SubElement(loc, f"{{{NS_RESQML20}}}Coordinate2").text = str(
            float(location[1])
        )
        etree.SubElement(loc, f"{{{NS_RESQML20}}}Coordinate3").text = str(
            float(location[2])
        )

        crs_ref = etree.SubElement(root, f"{{{NS_RESQML20}}}LocalCrs")
        crs_ref.set("uuid", crs_uuid)

        xml_str = etree.tostring(root, encoding="unicode")
        self._run(self._start_transaction())
        self._put_xml(uri, xml_str, qualified_type)
        self._run(self._commit_transaction())
        return uuid

    def put_wellbore_frame(
        self,
        uuid: str,
        title: str,
        trajectory_uuid: str,
        md: np.ndarray,
        properties: List[Dict[str, Any]],
        crs_uuid: str,
    ) -> str:
        """Write WellboreFrameRepresentation with properties."""
        from lxml import etree

        from ._resqml_enums import NS_COMMON20, NS_RESQML20, RESQML_NS_MAP

        if not uuid:
            uuid = _make_uuid()

        qualified_type = "resqml20.WellboreFrameRepresentation"
        uri = _uri_for_object(self._config.dataspace, qualified_type, uuid)

        root = etree.Element(
            f"{{{NS_RESQML20}}}WellboreFrameRepresentation", nsmap=RESQML_NS_MAP
        )
        root.set("uuid", uuid)
        root.set("schemaVersion", "2.0")

        _add_citation(root, title)

        etree.SubElement(root, f"{{{NS_RESQML20}}}NodeCount").text = str(len(md))

        md_el = etree.SubElement(root, f"{{{NS_RESQML20}}}NodeMd")
        hdf_ref = etree.SubElement(md_el, f"{{{NS_COMMON20}}}HdfProxy")
        hdf_ref.set("uuid", _make_uuid())
        path_el = etree.SubElement(md_el, f"{{{NS_COMMON20}}}PathInHdfFile")
        path_el.text = f"/RESQML/{uuid}/MdValues"

        traj_ref = etree.SubElement(root, f"{{{NS_RESQML20}}}Trajectory")
        traj_ref.set("uuid", trajectory_uuid)

        xml_str = etree.tostring(root, encoding="unicode")

        self._run(self._start_transaction())
        self._put_xml(uri, xml_str, qualified_type)
        self._put_array(uri, f"/RESQML/{uuid}/MdValues", md.astype(np.float64))
        self._run(self._commit_transaction())

        # Write properties
        for prop in properties:
            self.put_property_values(
                uuid=prop.get("uuid") or _make_uuid(),
                title=prop["title"],
                values=np.asarray(prop["values"]),
                supporting_representation_uuid=uuid,
                property_kind=prop.get("property_kind", "continuous"),
                indexable_element="nodes",
                is_discrete=prop.get("is_discrete", False),
            )

        return uuid

    # ------------------------------------------------------------------
    # BlockedWellbore
    # ------------------------------------------------------------------

    def get_blocked_wellbore(self, uuid: str) -> Dict[str, Any]:
        """Read BlockedWellboreRepresentation."""
        from lxml import etree

        from ._resqml_enums import NS_COMMON20, NS_RESQML20

        qualified_type = "resqml20.BlockedWellboreRepresentation"
        uri = self._resolve_uri(uuid, qualified_type)
        xml_str = self._get_xml(uuid, uri=uri)
        if xml_str is None:
            raise ValueError(f"BlockedWellbore {uuid} not found via ETP")

        root = etree.fromstring(
            xml_str.encode("utf-8") if isinstance(xml_str, str) else xml_str
        )

        crs_uuid = ""
        crs_ref = root.find(f".//{{{NS_RESQML20}}}LocalCrs")
        if crs_ref is not None:
            crs_uuid = _uuid_from_ref(crs_ref)

        title = ""
        citation = root.find(f"{{{NS_COMMON20}}}Citation")
        if citation is not None:
            t = citation.find(f"{{{NS_COMMON20}}}Title")
            if t is not None:
                title = t.text or ""

        grid_uuid = ""
        grid_ref = root.find(f".//{{{NS_RESQML20}}}Grid")
        if grid_ref is not None:
            grid_uuid = grid_ref.get("uuid", "")

        trajectory_uuid = ""
        traj_ref = root.find(f".//{{{NS_RESQML20}}}Trajectory")
        if traj_ref is not None:
            trajectory_uuid = traj_ref.get("uuid", "")

        uri = self._resolve_uri(uuid, qualified_type)
        md = self._get_array(uri, f"/RESQML/{uuid}/MdValues")
        xyz = self._get_array(uri, f"/RESQML/{uuid}/Points")
        cell_indices = self._get_array(uri, f"/RESQML/{uuid}/CellIndices")

        if md is None:
            md = np.zeros(0, dtype=np.float64)
        if xyz is None:
            xyz = np.zeros((0, 3), dtype=np.float64)
        if cell_indices is None:
            cell_indices = np.zeros((0, 3), dtype=np.int32)

        # Find properties
        props = self._find_etp_frame_properties(uuid)

        return {
            "md": md,
            "xyz": xyz,
            "cell_indices": cell_indices,
            "crs_uuid": crs_uuid,
            "title": title,
            "grid_uuid": grid_uuid,
            "trajectory_uuid": trajectory_uuid,
            "properties": props,
        }

    def put_blocked_wellbore(
        self,
        uuid: str,
        title: str,
        trajectory_uuid: str,
        grid_uuid: str,
        md: np.ndarray,
        xyz: np.ndarray,
        cell_indices: np.ndarray,
        properties: List[Dict[str, Any]],
        crs_uuid: str,
    ) -> str:
        """Write BlockedWellboreRepresentation."""
        from lxml import etree

        from ._resqml_enums import NS_COMMON20, NS_RESQML20, RESQML_NS_MAP

        if not uuid:
            uuid = _make_uuid()

        qualified_type = "resqml20.BlockedWellboreRepresentation"
        uri = _uri_for_object(self._config.dataspace, qualified_type, uuid)

        root = etree.Element(
            f"{{{NS_RESQML20}}}BlockedWellboreRepresentation", nsmap=RESQML_NS_MAP
        )
        root.set("uuid", uuid)
        root.set("schemaVersion", "2.0")

        _add_citation(root, title)

        etree.SubElement(root, f"{{{NS_RESQML20}}}NodeCount").text = str(len(md))
        etree.SubElement(root, f"{{{NS_RESQML20}}}CellCount").text = str(
            len(cell_indices)
        )

        traj_ref = etree.SubElement(root, f"{{{NS_RESQML20}}}Trajectory")
        traj_ref.set("uuid", trajectory_uuid)

        if grid_uuid:
            grid_ref_el = etree.SubElement(root, f"{{{NS_RESQML20}}}Grid")
            grid_ref_el.set("uuid", grid_uuid)

        md_el = etree.SubElement(root, f"{{{NS_RESQML20}}}NodeMd")
        hdf_ref = etree.SubElement(md_el, f"{{{NS_COMMON20}}}HdfProxy")
        hdf_ref.set("uuid", _make_uuid())
        path_el = etree.SubElement(md_el, f"{{{NS_COMMON20}}}PathInHdfFile")
        path_el.text = f"/RESQML/{uuid}/MdValues"

        pts = etree.SubElement(root, f"{{{NS_RESQML20}}}NodeGeometry")
        pts_geom = etree.SubElement(pts, f"{{{NS_RESQML20}}}Points")
        hdf_ref2 = etree.SubElement(pts_geom, f"{{{NS_COMMON20}}}HdfProxy")
        hdf_ref2.set("uuid", _make_uuid())
        path_el2 = etree.SubElement(pts_geom, f"{{{NS_COMMON20}}}PathInHdfFile")
        path_el2.text = f"/RESQML/{uuid}/Points"

        ci_el = etree.SubElement(root, f"{{{NS_RESQML20}}}CellIndices")
        hdf_ref3 = etree.SubElement(ci_el, f"{{{NS_COMMON20}}}HdfProxy")
        hdf_ref3.set("uuid", _make_uuid())
        path_el3 = etree.SubElement(ci_el, f"{{{NS_COMMON20}}}PathInHdfFile")
        path_el3.text = f"/RESQML/{uuid}/CellIndices"

        crs_ref_el = etree.SubElement(root, f"{{{NS_RESQML20}}}LocalCrs")
        crs_ref_el.set("uuid", crs_uuid)

        xml_str = etree.tostring(root, encoding="unicode")

        self._run(self._start_transaction())
        self._put_xml(uri, xml_str, qualified_type)
        self._put_array(uri, f"/RESQML/{uuid}/MdValues", md.astype(np.float64))
        self._put_array(uri, f"/RESQML/{uuid}/Points", xyz.astype(np.float64))
        self._put_array(
            uri, f"/RESQML/{uuid}/CellIndices", cell_indices.astype(np.int32)
        )
        self._run(self._commit_transaction())

        # Write properties
        for prop in properties:
            self.put_property_values(
                uuid=prop.get("uuid") or _make_uuid(),
                title=prop["title"],
                values=np.asarray(prop["values"]),
                supporting_representation_uuid=uuid,
                property_kind=prop.get("property_kind", "continuous"),
                indexable_element="cells",
                is_discrete=prop.get("is_discrete", False),
            )

        return uuid

    # ------------------------------------------------------------------
    # Dataspace Management (ETP Protocol 24)
    # ------------------------------------------------------------------

    def get_dataspaces(self) -> List[Dict[str, Any]]:
        """List all available dataspaces on the server.

        Returns
        -------
        list of dict
            Each dict has 'uri', 'path', 'last_changed' keys.
        """
        return self._run(self._async_get_dataspaces())

    async def _async_get_dataspaces(self) -> List[Dict[str, Any]]:
        from energistics.etp.v12.protocol.dataspace import GetDataspaces

        msg = GetDataspaces(store_last_write_filter=None)
        responses = await self._send_and_recv(msg)

        results = []
        for resp in responses:
            dataspaces = getattr(resp, "dataspaces", [])
            for ds in dataspaces:
                results.append(
                    {
                        "uri": getattr(ds, "uri", ""),
                        "path": getattr(ds, "path", ""),
                        "last_changed": getattr(ds, "last_changed", 0),
                    }
                )
        return results

    def put_dataspace(self, path: str) -> str:
        """Create a new dataspace.

        Parameters
        ----------
        path : str
            Dataspace path, e.g. "myteam/project".

        Returns
        -------
        str
            The URI of the created dataspace.
        """
        return self._run(self._async_put_dataspace(path))

    async def _async_put_dataspace(self, path: str) -> str:
        import time

        from energistics.etp.v12.datatypes.object.dataspace import Dataspace
        from energistics.etp.v12.protocol.dataspace import PutDataspaces

        uri = f"eml:///dataspace('{path}')"
        now_us = int(time.time() * 1_000_000)

        ds = Dataspace(
            uri=uri,
            path=path,
            last_changed=now_us,
            store_last_write=now_us,
            store_created=now_us,
            custom_data={},
        )

        msg = PutDataspaces(dataspaces={"0": ds})
        responses = await self._send_and_recv(msg)

        for resp in responses:
            if hasattr(resp, "error") and resp.error:
                raise RuntimeError(f"PutDataspaces failed: {resp.error}")
        return uri

    def delete_dataspace(self, path: str) -> None:
        """Delete a dataspace and all its contents.

        Parameters
        ----------
        path : str
            Dataspace path to delete, e.g. "myteam/project".
        """
        self._run(self._async_delete_dataspace(path))

    async def _async_delete_dataspace(self, path: str) -> None:
        from energistics.etp.v12.protocol.dataspace import DeleteDataspaces

        uri = f"eml:///dataspace('{path}')"
        msg = DeleteDataspaces(uris={"0": uri})
        responses = await self._send_and_recv(msg)

        for resp in responses:
            if hasattr(resp, "error") and resp.error:
                raise RuntimeError(f"DeleteDataspaces failed: {resp.error}")

    def switch_dataspace(self, path: str) -> None:
        """Switch this provider's active dataspace.

        Parameters
        ----------
        path : str
            New dataspace path, e.g. "myteam/project".
        """
        self._config.dataspace = f"eml:///dataspace('{path}')"


# ---------------------------------------------------------------------------
# Notification Subscription (polling-based change detection)
# ---------------------------------------------------------------------------


class NotificationSubscription:
    """Tracks object changes in a dataspace via polling.

    Since the energistics Python library does not include ETP Protocol 5
    (StoreNotification) message classes, this implements change detection
    by comparing resource timestamps between polls.

    Usage::

        sub = provider.subscribe_notifications(object_types=["IjkGrid"])
        events = sub.poll()  # Returns list of change events
        sub.stop()

    Or as a context manager::

        with provider.subscribe_notifications() as sub:
            events = sub.poll()
    """

    def __init__(
        self,
        provider: "EtpProvider",
        object_types: Optional[List[str]] = None,
        uuids: Optional[List[str]] = None,
        include_object_data: bool = False,
        callback: Optional[Any] = None,
    ):
        self._provider = provider
        self._object_types = object_types
        self._uuids = uuids
        self._include_object_data = include_object_data
        self._callback = callback
        self._active = True
        self._baseline: Dict[str, Dict[str, Any]] = {}
        self._last_poll_time: int = 0

        # Take initial snapshot
        self._refresh_baseline()

    def _refresh_baseline(self) -> None:
        """Capture current state as baseline for change detection."""
        import time

        type_filter = None
        if self._object_types:
            type_filter = [
                t if "." in t else f"resqml20.{t}" for t in self._object_types
            ]

        result = self._provider.discover(depth=0, object_types=type_filter)
        resources = result["resources"]

        # Filter by UUID if specified
        if self._uuids:
            uuid_set = {u.lower() for u in self._uuids}
            resources = [r for r in resources if r["uuid"].lower() in uuid_set]

        self._baseline = {r["uuid"]: r for r in resources}
        self._last_poll_time = int(time.time() * 1_000_000)

    def poll(self) -> List[Dict[str, Any]]:
        """Poll for changes since last poll (or since subscription started).

        Returns
        -------
        list of dict
            Each event dict has keys:
            - 'event': "created", "changed", or "deleted"
            - 'uuid': object UUID
            - 'title': object title
            - 'type': RESQML type
            - 'uri': object URI
            - 'timestamp': microsecond timestamp of change

        Examples
        --------
        >>> events = sub.poll()
        >>> for e in events:
        ...     print(f"{e['event']:8s} {e['type']:25s} {e['title']}")
        created  ContinuousProperty        PERMX
        changed  IjkGridRepresentation     Drogon
        """
        if not self._active:
            return []

        import time

        type_filter = None
        if self._object_types:
            type_filter = [
                t if "." in t else f"resqml20.{t}" for t in self._object_types
            ]

        result = self._provider.discover(depth=0, object_types=type_filter)
        current_resources = result["resources"]

        # Filter by UUID if specified
        if self._uuids:
            uuid_set = {u.lower() for u in self._uuids}
            current_resources = [
                r for r in current_resources if r["uuid"].lower() in uuid_set
            ]

        current_map = {r["uuid"]: r for r in current_resources}
        events: List[Dict[str, Any]] = []
        now_us = int(time.time() * 1_000_000)

        # Detect created and changed
        for uid, resource in current_map.items():
            if uid not in self._baseline:
                event = {
                    "event": "created",
                    "uuid": uid,
                    "title": resource["title"],
                    "type": resource["type"],
                    "uri": resource["uri"],
                    "timestamp": resource.get("last_changed", now_us),
                }
                events.append(event)
            else:
                old = self._baseline[uid]
                if resource.get("last_changed", 0) != old.get("last_changed", 0):
                    event = {
                        "event": "changed",
                        "uuid": uid,
                        "title": resource["title"],
                        "type": resource["type"],
                        "uri": resource["uri"],
                        "timestamp": resource.get("last_changed", now_us),
                    }
                    events.append(event)

        # Detect deleted
        for uid, resource in self._baseline.items():
            if uid not in current_map:
                event = {
                    "event": "deleted",
                    "uuid": uid,
                    "title": resource["title"],
                    "type": resource["type"],
                    "uri": resource.get("uri", ""),
                    "timestamp": now_us,
                }
                events.append(event)

        # Also check GetDeletedResources for authoritative delete tracking
        try:
            deleted = self._provider.get_deleted_resources(since=self._last_poll_time)
            {d["uuid"] for d in deleted}
            # Add any we missed
            for d in deleted:
                if d["uuid"] not in {
                    e["uuid"] for e in events if e["event"] == "deleted"
                }:
                    events.append(
                        {
                            "event": "deleted",
                            "uuid": d["uuid"],
                            "title": self._baseline.get(d["uuid"], {}).get("title", ""),
                            "type": self._baseline.get(d["uuid"], {}).get("type", ""),
                            "uri": d["uri"],
                            "timestamp": d["deleted_time"],
                        }
                    )
        except Exception:
            pass  # GetDeletedResources may not be supported

        # Update baseline
        self._baseline = current_map
        self._last_poll_time = now_us

        # Fire callbacks
        if self._callback and events:
            for event in events:
                self._callback(event["event"], event)

        return events

    def stop(self) -> None:
        """Stop this subscription (future polls return empty)."""
        self._active = False

    def __enter__(self) -> "NotificationSubscription":
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()

    def __repr__(self) -> str:
        types = self._object_types or ["all"]
        return f"NotificationSubscription(types={types}, active={self._active})"
