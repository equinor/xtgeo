"""HDF5 i/o using a xtgeo variant.

The HDF5 format is a binary file format that can store large and complex datasets
efficiently. It is commonly used in scientific computing and data analysis. The xtgeo
variant of HDF5 is a specific implementation that may include additional metadata or
structure tailored for geological data.

"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Final

import h5py
import hdf5plugin
import numpy as np

from xtgeo.common._xyz_enum import _AttrType
from xtgeo.common.log import null_logger
from xtgeo.io._file import FileWrapper
from xtgeo.io._welldata._blockedwell_io import BlockedWellData
from xtgeo.io._welldata._well_io import WellData, WellLog

if TYPE_CHECKING:  # pragma: no cover
    from xtgeo.common.types import FileLike

logger = null_logger(__name__)

HDF5_FORMAT_IDCODE: Final[int] = 1401
HDF5_PROVIDER: Final[str] = "xtgeo"


def _export_wlogs_to_hdf5(well: WellData) -> dict[str, tuple[str, Any]]:
    """
    Convert WellData logs to the wlogs format used in HDF5 metadata.

    This creates a dictionary where each log name maps to a tuple of
    (log_type, code_names_or_metadata).

    Args:
        well: WellData object

    Returns:
        Dictionary with log names as keys and (type, records) tuples as values

    """
    wlogs: dict[str, tuple[str, Any]] = {}

    for log in well.logs:
        if log.is_discrete:
            log_type = _AttrType.DISC.value
            # For discrete logs, code_names is a dict mapping codes to names
            wlogs[log.name] = (log_type, log.code_names)
        else:
            # For continuous logs, always use CONT type for backward compatibility
            log_type = _AttrType.CONT.value
            wlogs[log.name] = (log_type, log.code_names)

    return wlogs


def _import_wlogs_from_hdf5(
    wlogs: dict[str, tuple[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    Convert HDF5 wlogs format to separate wlogtypes and wlogrecords dictionaries.

    This is the inverse of _export_wlogs_to_hdf5.

    Args:
        wlogs: Dictionary from HDF5 with (type, records) tuples

    Returns:
        Dictionary with "wlogtypes" and "wlogrecords" keys

    Raises:
        ValueError: If invalid log type or record format is encountered

    """
    wlogtypes: dict[str, str] = {}
    wlogrecords: dict[str, Any] = {}

    for key in wlogs:
        typ, rec = wlogs[key]

        # Validate log type - raise error for invalid types (matches legacy behavior)
        if typ in {_AttrType.DISC.value, _AttrType.CONT.value}:
            wlogtypes[key] = deepcopy(typ)
        else:
            raise ValueError(
                f"Invalid log type found in input for '{key}': {typ!r}. "
                f"Expected '{_AttrType.DISC.value}' or '{_AttrType.CONT.value}'."
            )

        # Validate and normalize log records
        if rec is None:
            wlogrecords[key] = None
        elif isinstance(rec, dict):
            # For DISC logs, convert string keys back to integers
            # (JSON serialization converts int keys to strings)
            if typ == _AttrType.DISC.value:
                wlogrecords[key] = {int(k): v for k, v in rec.items()}
            else:
                # Non-DISC logs should not have dict records, but we handle
                # this for backward compatibility with legacy files
                wlogrecords[key] = deepcopy(rec)
        elif isinstance(rec, (list, tuple)):
            wlogrecords[key] = tuple(rec)
        else:
            raise ValueError(
                f"Invalid log record found in input for '{key}': {rec!r}. "
                f"Expected None, dict, list, or tuple."
            )

    return {"wlogtypes": wlogtypes, "wlogrecords": wlogrecords}


def write_hdf5_well(
    well: WellData,
    filepath: FileLike,
    compression: str | None = "lzf",
) -> None:
    """Write well data to HDF5 file using xtgeo format.

    Args:
        well: WellData object to write
        filepath: Output HDF5 file path or file-like object
        compression: Compression method ("lzf", "blosc", or None)
            - "lzf": Fast compression (default)
            - "blosc": High compression ratio
            - None: No compression

    """
    wrapper = FileWrapper(filepath, mode="w")

    # HDF5 format requires real file paths, not in-memory streams
    if wrapper.memstream:
        raise TypeError(
            "HDF5 format does not support in-memory streams (BytesIO/StringIO). "
            "Please provide a file path instead."
        )

    logger.debug("Writing well data to HDF5: %s", wrapper.name)

    compression_filter: str | hdf5plugin.Blosc | None = None
    if compression == "blosc":
        compression_filter = hdf5plugin.Blosc(
            cname="blosclz", clevel=9, shuffle=hdf5plugin.Blosc.SHUFFLE
        )
    elif compression == "lzf":
        compression_filter = "lzf"

    # Validate compression argument: avoid silently writing uncompressed data
    if compression and compression_filter is None:
        raise ValueError(
            f"Unsupported compression '{compression}'. "
            "Expected 'lzf', 'blosc', or None."
        )

    wlogs = _export_wlogs_to_hdf5(well)

    metadata = {
        "_class_": "Well",
        "_required_": {
            "rkb": well.zpos if well.zpos is not None else 0.0,
            "xpos": well.xpos,
            "ypos": well.ypos,
            "name": well.name,
            "wlogs": wlogs,
            "mdlogname": None,  # Not used in new structure
            "zonelogname": None,  # Not used in new structure
        },
    }

    jmeta = json.dumps(metadata).encode()

    with h5py.File(wrapper.name, "w") as fh5:
        logger.debug("Creating HDF5 Well group in %s", wrapper.name)
        grp = fh5.create_group("Well")

        # Create a pseudo-dataframe structure: store index and columns
        # Index is just 0, 1, 2, ... n_records-1
        index = np.arange(well.n_records, dtype=np.int64)
        grp.create_dataset(
            "index",
            data=index,
            compression=compression_filter,
            chunks=True,
        )

        grp.create_dataset(
            "column/X_UTME",
            data=well.survey_x,
            compression=compression_filter,
            chunks=True,
        )
        grp.create_dataset(
            "column/Y_UTMN",
            data=well.survey_y,
            compression=compression_filter,
            chunks=True,
        )
        grp.create_dataset(
            "column/Z_TVDSS",
            data=well.survey_z,
            compression=compression_filter,
            chunks=True,
        )

        for log in well.logs:
            grp.create_dataset(
                f"column/{log.name}",
                data=log.values,
                compression=compression_filter,
                chunks=True,
            )

        all_columns = ["X_UTME", "Y_UTMN", "Z_TVDSS"] + list(well.log_names)
        grp.attrs["columns"] = np.array(all_columns, dtype="S")
        grp.attrs["metadata"] = jmeta
        grp.attrs["provider"] = HDF5_PROVIDER
        grp.attrs["format_idcode"] = HDF5_FORMAT_IDCODE

    logger.debug(
        "Successfully wrote well '%s' with %d records to %s",
        well.name,
        well.n_records,
        wrapper.name,
    )


def read_hdf5_well(filepath: FileLike) -> WellData:
    """Read well data from HDF5 file using xtgeo format.

    Args:
        filepath: Path to HDF5 file or file-like object

    Returns:
        WellData object

    Raises:
        ValueError: If file format is invalid or required data is missing

    """
    wrapper = FileWrapper(filepath, mode="r")

    # HDF5 format requires real file paths, not in-memory streams
    if wrapper.memstream:
        raise TypeError(
            "HDF5 format does not support in-memory streams (BytesIO/StringIO). "
            "Please provide a file path instead."
        )

    logger.debug("Reading well data from HDF5: %s", wrapper.name)

    with h5py.File(wrapper.file, "r") as fh5:
        if "Well" not in fh5:
            raise ValueError("Invalid HDF5 well file: missing 'Well' group")

        grp = fh5["Well"]

        # Read metadata
        if "metadata" not in grp.attrs:
            raise ValueError("Invalid HDF5 well file: missing metadata")

        jmeta = grp.attrs["metadata"]
        if isinstance(jmeta, bytes):
            jmeta = jmeta.decode()

        meta = json.loads(jmeta, object_pairs_hook=dict)
        req = meta.get("_required_", {})

        # Validate required metadata fields
        required_keys = ["name", "xpos", "ypos", "rkb", "wlogs"]
        missing_keys = [key for key in required_keys if key not in req]
        if missing_keys:
            raise ValueError(
                f"Invalid HDF5 well file: missing required metadata fields: "
                f"{missing_keys}"
            )

        wname = req["name"]
        xpos = req["xpos"]
        ypos = req["ypos"]
        rkb = req["rkb"]

        columns_bytes = grp.attrs.get("columns", [])
        columns = [
            col.decode() if isinstance(col, bytes) else col for col in columns_bytes
        ]

        survey_x = grp["column/X_UTME"][:].astype(np.float64)
        survey_y = grp["column/Y_UTMN"][:].astype(np.float64)
        survey_z = grp["column/Z_TVDSS"][:].astype(np.float64)

        wlogs_raw = req["wlogs"]
        wlogs_parsed = _import_wlogs_from_hdf5(wlogs_raw)
        wlogtypes = wlogs_parsed["wlogtypes"]
        wlogrecords = wlogs_parsed["wlogrecords"]

        # Identify and read log data (exclude coordinate columns)
        log_names = [
            col for col in columns if col not in ["X_UTME", "Y_UTMN", "Z_TVDSS"]
        ]

        logs = []
        for log_name in log_names:
            values = grp[f"column/{log_name}"][:].astype(np.float64)

            # Determine if discrete and get code_names
            is_discrete = (
                wlogtypes.get(log_name, _AttrType.CONT.value) == _AttrType.DISC.value
            )
            code_names = wlogrecords.get(log_name)

            log = WellLog(
                name=log_name,
                values=values,
                is_discrete=is_discrete,
                code_names=code_names,
            )
            logs.append(log)

    well = WellData(
        name=wname,
        xpos=xpos,
        ypos=ypos,
        zpos=rkb,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=tuple(logs),
    )

    logger.debug(
        "Successfully read well '%s' with %d records and %d logs",
        wname,
        well.n_records,
        len(logs),
    )

    return well


def write_hdf5_blockedwell(
    blocked_well: BlockedWellData,
    filepath: FileLike,
    compression: str | None = "lzf",
) -> None:
    """Write blocked well data to HDF5 file.

    Args:
        blocked_well: BlockedWellData object to write
        filepath: Output HDF5 file path
        compression: Compression method ("lzf", "blosc", or None)

    """
    wrapper = FileWrapper(filepath, mode="w")

    # HDF5 format requires real file paths, not in-memory streams
    if wrapper.memstream:
        raise TypeError(
            "HDF5 format does not support in-memory streams (BytesIO/StringIO). "
            "Please provide a file path instead."
        )

    logger.debug("Writing blocked well data to HDF5: %s", wrapper.name)

    compression_filter: str | hdf5plugin.Blosc | None = None

    if compression:
        if compression == "blosc":
            compression_filter = hdf5plugin.Blosc(
                cname="blosclz", clevel=9, shuffle=hdf5plugin.Blosc.SHUFFLE
            )
        elif compression == "lzf":
            compression_filter = "lzf"
        else:
            raise ValueError(
                f"Unsupported compression '{compression}'. "
                "Supported values are 'lzf', 'blosc', or None."
            )

    wlogs = _export_wlogs_to_hdf5(blocked_well)

    metadata = {
        "_class_": "BlockedWell",
        "_required_": {
            "rkb": blocked_well.zpos if blocked_well.zpos is not None else 0.0,
            "xpos": blocked_well.xpos,
            "ypos": blocked_well.ypos,
            "name": blocked_well.name,
            "wlogs": wlogs,
            "mdlogname": None,
            "zonelogname": None,
        },
    }

    jmeta = json.dumps(metadata).encode()

    with h5py.File(wrapper.name, "w") as fh5:
        logger.debug("Creating HDF5 Well group in %s", wrapper.name)
        grp = fh5.create_group("Well")

        index = np.arange(blocked_well.n_records, dtype=np.int64)
        grp.create_dataset(
            "index",
            data=index,
            compression=compression_filter,
            chunks=True,
        )

        grp.create_dataset(
            "column/X_UTME",
            data=blocked_well.survey_x,
            compression=compression_filter,
            chunks=True,
        )
        grp.create_dataset(
            "column/Y_UTMN",
            data=blocked_well.survey_y,
            compression=compression_filter,
            chunks=True,
        )
        grp.create_dataset(
            "column/Z_TVDSS",
            data=blocked_well.survey_z,
            compression=compression_filter,
            chunks=True,
        )

        # Store grid indices as columns
        grp.create_dataset(
            "column/I_INDEX",
            data=blocked_well.i_index,
            compression=compression_filter,
            chunks=True,
        )
        grp.create_dataset(
            "column/J_INDEX",
            data=blocked_well.j_index,
            compression=compression_filter,
            chunks=True,
        )
        grp.create_dataset(
            "column/K_INDEX",
            data=blocked_well.k_index,
            compression=compression_filter,
            chunks=True,
        )

        for log in blocked_well.logs:
            grp.create_dataset(
                f"column/{log.name}",
                data=log.values,
                compression=compression_filter,
                chunks=True,
            )

        all_columns = [
            "X_UTME",
            "Y_UTMN",
            "Z_TVDSS",
            "I_INDEX",
            "J_INDEX",
            "K_INDEX",
        ] + list(blocked_well.log_names)
        grp.attrs["columns"] = np.array(all_columns, dtype="S")
        grp.attrs["metadata"] = jmeta
        grp.attrs["provider"] = HDF5_PROVIDER
        grp.attrs["format_idcode"] = HDF5_FORMAT_IDCODE

    logger.debug(
        "Successfully wrote blocked well '%s' with %d blocked cells to %s",
        blocked_well.name,
        blocked_well.n_blocked_cells,
        wrapper.name,
    )


def read_hdf5_blockedwell(filepath: FileLike) -> BlockedWellData:
    """Read blocked well data from HDF5 file.

    Args:
        filepath: Path to HDF5 file

    Returns:
        BlockedWellData object

    Raises:
        ValueError: If file format is invalid or required data is missing

    """
    wrapper = FileWrapper(filepath, mode="r")

    # HDF5 format requires real file paths, not in-memory streams
    if wrapper.memstream:
        raise TypeError(
            "HDF5 format does not support in-memory streams (BytesIO/StringIO). "
            "Please provide a file path instead."
        )

    logger.debug("Reading blocked well data from HDF5: %s", wrapper.name)

    with h5py.File(wrapper.file, "r") as fh5:
        if "Well" not in fh5:
            raise ValueError("Invalid HDF5 well file: missing 'Well' group")

        grp = fh5["Well"]

        if "metadata" not in grp.attrs:
            raise ValueError("Invalid HDF5 well file: missing metadata")

        jmeta = grp.attrs["metadata"]
        if isinstance(jmeta, bytes):
            jmeta = jmeta.decode()

        meta = json.loads(jmeta, object_pairs_hook=dict)
        req = meta.get("_required_", {})

        # Validate required metadata fields
        required_keys = ["name", "xpos", "ypos", "rkb", "wlogs"]
        missing_keys = [key for key in required_keys if key not in req]
        if missing_keys:
            raise ValueError(
                f"Invalid HDF5 blocked well file: missing required metadata fields: "
                f"{missing_keys}"
            )

        wname = req["name"]
        xpos = req["xpos"]
        ypos = req["ypos"]
        rkb = req["rkb"]

        columns_bytes = grp.attrs.get("columns", [])
        columns = [
            col.decode() if isinstance(col, bytes) else col for col in columns_bytes
        ]

        survey_x = grp["column/X_UTME"][:].astype(np.float64)
        survey_y = grp["column/Y_UTMN"][:].astype(np.float64)
        survey_z = grp["column/Z_TVDSS"][:].astype(np.float64)

        if (
            "column/I_INDEX" not in grp
            or "column/J_INDEX" not in grp
            or "column/K_INDEX" not in grp
        ):
            raise ValueError(
                "File does not contain I_INDEX, J_INDEX, and K_INDEX columns "
                "required for blocked well"
            )

        i_index = grp["column/I_INDEX"][:].astype(np.float64)
        j_index = grp["column/J_INDEX"][:].astype(np.float64)
        k_index = grp["column/K_INDEX"][:].astype(np.float64)

        wlogs_raw = req["wlogs"]
        wlogs_parsed = _import_wlogs_from_hdf5(wlogs_raw)
        wlogtypes = wlogs_parsed["wlogtypes"]
        wlogrecords = wlogs_parsed["wlogrecords"]

        log_names = [
            col
            for col in columns
            if col
            not in [
                "X_UTME",
                "Y_UTMN",
                "Z_TVDSS",
                "I_INDEX",
                "J_INDEX",
                "K_INDEX",
            ]
        ]

        logs = []
        for log_name in log_names:
            values = grp[f"column/{log_name}"][:].astype(np.float64)

            is_discrete = (
                wlogtypes.get(log_name, _AttrType.CONT.value) == _AttrType.DISC.value
            )
            code_names = wlogrecords.get(log_name)

            log = WellLog(
                name=log_name,
                values=values,
                is_discrete=is_discrete,
                code_names=code_names,
            )
            logs.append(log)

    blocked_well = BlockedWellData(
        name=wname,
        xpos=xpos,
        ypos=ypos,
        zpos=rkb,
        survey_x=survey_x,
        survey_y=survey_y,
        survey_z=survey_z,
        logs=tuple(logs),
        i_index=i_index,
        j_index=j_index,
        k_index=k_index,
    )

    logger.debug(
        "Successfully read blocked well '%s' with %d records and %d blocked cells",
        blocked_well.name,
        blocked_well.n_records,
        blocked_well.n_blocked_cells,
    )

    return blocked_well
