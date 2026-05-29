# OSDU/RESQML Interface – Developer Design Document

## Architecture Overview

The `xtgeo.interfaces.osdu` package implements reading and writing of RESQML 2.0.1
objects to/from OSDU Reservoir DDMS (RDDMS) backends, both file-based (EPC+HDF5)
and live protocol (ETP 1.2 WebSocket).

```
┌─────────────────────────────────────────────────────────────────────┐
│                       USER API LAYER                                 │
│  grid_from_osdu() · surface_from_osdu() · list_osdu_objects() ...   │
│  xtgeo.grid_from_osdu(session_or_path, name="...", uuid="...")      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                    CONVERTER LAYER                                    │
│  _ijk_grid.py  · _grid2d.py · _pointset.py · _polyline.py          │
│  _properties.py · _crs.py · _metadata.py                           │
│  Pure logic: xtgeo ↔ RESQML geometry/property transforms           │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                    PROVIDER ABSTRACTION                               │
│  ResqmlDataProvider (ABC)                                            │
│    ├─ EpcFileProvider  (EPC ZIP + HDF5 file I/O)                    │
│    └─ EtpProvider      (ETP 1.2 WebSocket binary protocol)          │
│  Common interface: get/put geometry, properties, list objects        │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
┌───────────────────────────────▼─────────────────────────────────────┐
│                    SESSION & AUTH                                     │
│  OsduSession – OAuth2, profile persistence, REST admin               │
│  EtpConnectionConfig – connection parameters                         │
└─────────────────────────────────────────────────────────────────────┘
```

## Module Inventory

| File | Purpose |
|------|---------|
| `__init__.py` | Package exports, docstring with usage examples |
| `_api.py` | **User-facing functions**: grid_from_osdu, surface_from_osdu, list/search, etc. |
| `_session.py` | OsduSession: auth, profiles, REST admin wrappers |
| `_provider_base.py` | ABC defining the provider interface |
| `_epc_provider.py` | EPC+HDF5 file provider (uses zipfile + h5py) |
| `_etp_provider.py` | ETP 1.2 WebSocket provider (uses energistics lib) |
| `_ijk_grid.py` | IJK Grid ↔ xtgeo Grid converter |
| `_grid2d.py` | Grid2D ↔ xtgeo RegularSurface converter |
| `_pointset.py` | PointSet ↔ xtgeo Points converter |
| `_polyline.py` | PolylineSet ↔ xtgeo Polygons converter |
| `_properties.py` | Grid property read/write (multi-property support) |
| `_crs.py` | LocalDepth3dCrs handling |
| `_metadata.py` | OSDU property name mapping (Eclipse ↔ OSDU) |
| `_resqml_enums.py` | RESQML enumeration types |
| `_dataspace.py` | Bulk dataspace operations (read/write/compare entire datasets) |

## Design Principles

### 1. Provider Abstraction
All data access goes through `ResqmlDataProvider`. This allows:
- Swapping between file and network backends transparently
- Testing converters against EPC files without a live server
- Future backends (e.g., REST-only OSDU v3)

### 2. Generic Type Handling
Objects are handled as XML metadata + binary arrays:
- XML is stored/retrieved as strings (provider handles serialization)
- Arrays use numpy with explicit dtype/shape contracts
- UUIDs are preserved across roundtrips when `preserve_uuids=True`

### 3. Exact Geometry Preservation
The converter layer ensures:
- Corner-point pillar coordinates: `coordsv` shape `(ni+1, nj+1, 6)` — exact float64
- Cell corners: `zcornsv` shape `(ni+1, nj+1, nk+1, 4)` — exact float64
- Activity mask: `actnumsv` shape `(ni, nj, nk)` — exact int32
- No interpolation, no grid resampling

### 4. Property Axis Convention
- **xtgeo**: properties shaped `(ni, nj, nk)` in IJK (column-major physical) order
- **RESQML/resqpy**: properties in `(nk, nj, ni)` KJI (row-major Fortran) order
- The converter transposes: `values.reshape(nk, nj, ni).transpose(2, 1, 0)`
- Roundtrip integrity verified by tests to machine precision

### 5. Cell-Split Faults
RESQML supports split-node grids for faulted geometries. The EPC provider reads:
- `SplitCoordinateLines/Count` — number of split pillars
- `SplitCoordinateLines/PillarIndices` — which pillars are split
- `ColumnsPerSplitCoordinateLine` — cell assignments per split

These map to xtgeo's internal split-node representation (subgrids).

## Provider Interface

```python
class ResqmlDataProvider(ABC):
    def open(self) -> None: ...
    def close(self) -> None: ...
    def __enter__(self): return self
    def __exit__(self, *_): self.close()

    # Discovery
    def list_objects(self, object_type: str = None) -> List[Dict[str, Any]]: ...

    # IJK Grid
    def get_ijk_grid_geometry(self, uuid: str) -> Dict[str, np.ndarray]: ...
    def put_ijk_grid(self, uuid: str, title: str, xml: str, arrays: Dict) -> None: ...

    # Grid2D (Surface)
    def get_grid2d_geometry(self, uuid: str) -> Dict[str, Any]: ...
    def put_grid2d(self, uuid: str, title: str, xml: str, arrays: Dict) -> None: ...

    # PointSet / PolylineSet
    def get_pointset_coordinates(self, uuid: str) -> np.ndarray: ...
    def put_pointset(self, uuid: str, title: str, xml: str, coords: np.ndarray) -> None: ...
    def get_polylineset_data(self, uuid: str) -> Dict[str, Any]: ...
    def put_polylineset(self, uuid: str, title: str, xml: str, data: Dict) -> None: ...

    # Properties
    def get_property_values(self, uuid: str) -> np.ndarray: ...
    def put_property(self, uuid: str, title: str, xml: str, values: np.ndarray) -> None: ...

    # CRS
    def get_crs(self, uuid: str) -> Dict[str, Any]: ...
    def put_crs(self, uuid: str, title: str, xml: str) -> None: ...
```

## ETP Protocol Details

The `EtpProvider` uses the `energistics` Python library (pyetp 1.2):

- **Connection**: Avro-encoded binary WebSocket frames
- **Discovery**: Protocol 3 (Discovery) — `GetResources` for listing
- **Store**: Protocol 4 (Store) — `GetDataObjects`, `PutDataObjects`
- **DataArray**: Protocol 9 (DataArray) — `GetDataArrays`, `PutDataArrays`
- **Dataspace**: Protocol 24 — `GetDataspaces`, `PutDataspaces`, `DeleteDataspaces`
- **Transaction**: Protocol 18 — wraps writes in atomic transactions

All async operations run in a private event loop (`asyncio.new_event_loop()`),
surfaced as synchronous methods via `_run()`.

## OsduSession REST Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/reservoir-ddms/v2/dataspaces` | Create dataspace |
| GET | `/api/reservoir-ddms/v2/dataspaces` | List dataspaces |
| GET | `/api/reservoir-ddms/v2/dataspaces/{path}` | Get dataspace info |
| DELETE | `/api/reservoir-ddms/v2/dataspaces/{path}` | Delete dataspace |
| GET | `/api/reservoir-ddms/v2/objects?dataspace=...` | List objects |
| POST | `/api/reservoir-ddms/v2/objects/search` | Search objects |
| GET | `/api/reservoir-ddms/v2/objects/{uuid}` | Get object metadata |

## Testing Strategy

Tests are in `tests/test_interfaces/test_osdu/`:

1. **EPC roundtrip tests** (`test_epc_roundtrip.py`): Pure file-based, no network.
   Test converter correctness using resqpy-generated EPC files.

2. **ETP roundtrip tests** (`test_etp_roundtrip.py`): Requires local Docker RDDMS.
   Tests full write→read→compare cycle with exact geometry verification.

3. **resqpy interop tests** (`test_resqpy_interop.py`): Verify compatibility with
   resqpy's output format (unified Points array, KJI ordering, etc.).

Run with:
```bash
cd tests/test_interfaces/test_osdu
pytest -v --noconftest
```

The `--noconftest` flag is needed because xtgeo's root conftest requires
test data that may not be available.

## Dependency Graph

```
xtgeo (core)
├── numpy
├── h5py (EPC HDF5 access)
├── lxml (RESQML XML construction/parsing)
├── packaging
└── xtgeo[osdu] (optional extras):
    ├── pyetp  ← provides the `energistics` Python module
    └── websockets
```

The OSDU module gracefully degrades:
- Without `pyetp`/`websockets`: ETP operations raise ImportError with instructions
- Without `h5py`: EPC operations raise ImportError
- Without `resqpy`: Interop tests skip

Install the optional OSDU dependencies with:

```bash
pip install xtgeo[osdu]
```

### pyetp — Role & Scope

The [`pyetp`](https://pypi.org/project/pyetp/) package
([equinor/pyetp](https://github.com/equinor/pyetp)) provides the
`energistics` Python module. It is an Equinor-maintained package that
auto-generates typed Pydantic models from the official ETP 1.2 Avro schemas.

**What pyetp provides (used as-is by xtgeo):**

- `avro_handler.encode_message()` / `decode_message()` — Avro binary
  serialization via `fastavro`
- Pydantic models for every ETP 1.2 message type (`RequestSession`,
  `GetDataObjects`, `PutDataArrays`, etc.)
- Datatype models (`MessageHeader`, `DataArrayIdentifier`, `Resource`,
  `Dataspace`, etc.)
- `Protocol` / `Role` enums
- Numpy array bridge (`DataArray.from_numpy_array()` / `.to_numpy_array()`)

**What pyetp does NOT provide:**

- No WebSocket client or connection management
- No session lifecycle handling
- No RESQML XML object models
- No domain-aware methods or high-level API

**What xtgeo builds on top:**

| Layer | Implementation |
|-------|----------------|
| WebSocket ETP client | Custom sync-over-async client in `_etp_provider.py` using `websockets` |
| RESQML 2.0.1 XML | Hand-built `lxml.etree` construction/parsing for IjkGrid, Grid2D, PointSet, PolylineSet, ContinuousProperty, DiscreteProperty, CRS |
| EPC file I/O | Full ZIP/OPC container + HDF5 handling in `_epc_provider.py` (no resqpy) |
| xtgeo ↔ RESQML converters | Bidirectional geometry transforms (pillar arrays ↔ coord/zcorn, etc.) |
| Session / auth | OAuth2 token management with profile persistence |
| High-level API | `grid_from_osdu()`, `surface_to_osdu()`, `list_osdu_objects()`, etc. |

In summary, pyetp is a **pure schema/codec layer** — xtgeo uses it for
message construction and binary serialization, and builds everything else
(client, XML models, converters, auth) from scratch.

**Known limitation:** pyetp does not include Protocol 5 (StoreNotification)
message classes. Change detection is therefore implemented via timestamp-based
polling (see Change Detection above).

## Future Considerations

- **OSDU v3 REST data API**: When REST data endpoints stabilize, add a
  `RestProvider` implementing the same interface
- **Streaming large grids**: ETP supports chunked DataArray transfers;
  currently we buffer entire arrays in memory
- **Property collections**: Group properties by realization/timestep for
  ensemble workflows
- **Well data**: RESQML WellboreTrajectory/WellboreFrame → xtgeo Well conversion
