# coding: utf-8
"""Roxar API functions for XTGeo Points/Polygons"""

from __future__ import annotations

import os
import tempfile
from enum import Enum
from typing import Any, Literal, Type, cast

import numpy as np
import pandas as pd

from xtgeo.common._xyz_enum import _AttrName
from xtgeo.common.constants import UNDEF, UNDEF_INT, UNDEF_INT_LIMIT, UNDEF_LIMIT
from xtgeo.common.log import null_logger
from xtgeo.io._file import FileWrapper
from xtgeo.roxutils import RoxUtils
from xtgeo.roxutils._roxar_loader import RoxarType, roxar, roxar_well_picks
from xtgeo.xyz import _xyz_io, points, polygons

if roxar:
    roxwp = roxar_well_picks

logger = null_logger(__name__)


class STYPE(str, Enum):
    HORIZONS = "horizons"
    ZONES = "zones"
    CLIPBOARD = "clipboard"
    GENERAL2D_DATA = "general2d_data"
    FAULTS = "faults"
    WELL_PICKS = "well_picks"

    @classmethod
    def _missing_(cls: Type[STYPE], value: object) -> None:
        valid_values = [m.value for m in cls]
        raise ValueError(f"Invalid stype {value}. Valid entries are {valid_values}")


XYZ_COLUMNS = [_AttrName.XNAME.value, _AttrName.YNAME.value, _AttrName.ZNAME.value]
REQUIRED_WELL_PICK_ATTRIBUTES = [_AttrName.M_MD_NAME.value, "WELLNAME", "TRAJECTORY"]
VALID_WELL_PICK_TYPES = ["fault", "horizon"]


def _check_input_and_version_requirement(
    roxutils: RoxUtils,
    stype: STYPE,
    category: str | list[str] | None,
    attributes: list[str] | bool | None,
    is_polygons: bool,
    mode: Literal["set", "get"],
) -> None:
    """General check of some input values."""

    if attributes is not None:
        if mode == "get" and not isinstance(attributes, (list, bool)):
            raise TypeError("'attributes' argument can only be of type list or bool")
        if mode == "set" and not isinstance(attributes, bool):
            raise TypeError("'attributes' argument can only be of type bool")

    logger.info("The stype is: %s", stype.value)

    # note: check of clipboard va Roxar API 1.2 is now removed as usage of
    # such old API versions is obsolute.
    if stype in (
        STYPE.GENERAL2D_DATA,
        STYPE.WELL_PICKS,
    ) and not roxutils.version_required("1.6"):
        raise NotImplementedError(
            f"API Support for {stype.value} is missing in this RMS version "
            f"(current API version is {roxutils.roxversion} - required is 1.6)"
        )

    if stype == STYPE.WELL_PICKS:
        if category not in VALID_WELL_PICK_TYPES:
            raise ValueError(
                f"Invalid {category=}. Valid entries are {VALID_WELL_PICK_TYPES}"
            )
        if is_polygons:
            raise ValueError(f"Polygons does not support stype={stype.value}.")


def _check_presence_in_project(
    rox: RoxUtils,
    name: str,
    category: str | list[str] | None,
    stype: STYPE,
    realisation: int,
    mode: Literal["set", "get"] = "get",
) -> None:
    """Helper to check if valid placeholder' whithin RMS."""

    logger.warning("Realisation %s not in use", realisation)

    project_attr = getattr(rox.project, stype)

    if stype in [STYPE.HORIZONS, STYPE.ZONES, STYPE.FAULTS]:
        if name not in project_attr:
            raise ValueError(f"Cannot access {name=} in {stype.value}")
        if category is None:
            raise ValueError("Need to specify category for horizons, zones and faults")
        if isinstance(category, list) or category not in project_attr.representations:
            raise ValueError(f"Cannot access {category=} in {stype.value}")
        if mode == "get" and project_attr[name][category].is_empty():
            raise RuntimeError(f"'{name}' is empty for {stype.value} {category=}")

    # only need to check presence in clipboard/general2d_data/well_picks if mode = get.
    if mode == "get":
        if stype in [STYPE.CLIPBOARD, STYPE.GENERAL2D_DATA]:
            folders = _get_rox_clipboard_folders(category)

            try:
                rox_folder = project_attr.folders[folders]
            except KeyError as keyerr:
                raise ValueError(
                    "Cannot access clipboards folder (not existing?)"
                ) from keyerr

            if name not in rox_folder:
                raise ValueError(f"Name {name} is not within Clipboard...")

        if stype == STYPE.WELL_PICKS and name not in project_attr.sets:
            raise ValueError(f"Well pick set {name} is not within Well Picks.")


def import_xyz_roxapi(
    project: RoxarType.project,
    name: str,
    category: str | list[str],
    stype: str = "horizons",
    realisation: int = 0,
    attributes: list[str] | bool | None = False,
    is_polygons: bool = False,
) -> dict[str, Any]:
    """Import a Points or Polygons item via ROXAR API spec.

    'Import' means transfer of data from Roxar API memory space to XTGeo memory space.

    ~ a part of a classmethod, and it will return the following kwargs to __init__()::

        xname
        yname
        zname
        pname # optional for Polygons
        name
        dataframe
        values=None  # since dataframe is set
        filesrc
    """

    rox = RoxUtils(project, readonly=True)
    stype = STYPE(stype.lower())

    _check_input_and_version_requirement(
        rox, stype, category, attributes, is_polygons, mode="get"
    )
    _check_presence_in_project(rox, name, category, stype, realisation, mode="get")

    if attributes and not rox.version_required("1.6"):
        result = _roxapi_import_xyz_viafile(rox, name, category, stype, is_polygons)
    elif stype == STYPE.WELL_PICKS:
        category = cast(Literal["fault", "horizon"], category)
        result = _roxapi_import_wellpicks(
            rox=rox,
            well_pick_set=name,
            wp_category=category,
            attributes=attributes,
        )
    else:
        result = _roxapi_import_xyz(
            rox, name, category, stype, realisation, is_polygons, attributes
        )

    rox.safe_close()
    return result


def _roxapi_import_xyz_viafile(
    rox: RoxUtils,
    name: str,
    category: str | list[str] | None,
    stype: STYPE,
    is_polygons: bool,
) -> dict[str, Any]:
    """Read XYZ from file due to a missing feature in Roxar API wrt attributes.

    However, attributes will be present in Roxar API from RMS version 12, and this
    routine should be replaced!
    """

    rox_xyz = _get_roxitem(
        rox,
        name,
        category,
        stype,
        mode="get",
        is_polygons=is_polygons,
    )

    kwargs = {}
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info("Made a tmp folder: %s", tmpdir)
            tfile = os.path.join(tmpdir, "generic.rmsattr")
            rox_xyz.save(tfile, roxar.FileFormat.RMS_POINTS)
            pfile = FileWrapper(tfile)
            kwargs = _xyz_io.import_rms_attr(pfile)

    except KeyError as kwe:
        logger.error(kwe)

    return kwargs


def _roxapi_import_xyz(
    rox: RoxUtils,
    name: str,
    category: str | list[str] | None,
    stype: STYPE,
    realisation: int,
    is_polygons: bool,
    attributes: bool | list[str],
) -> dict[str, str | dict | pd.DataFrame]:
    """From RMS to XTGeo"""

    kwargs: dict[str, str | dict | pd.DataFrame] = {
        "xname": _AttrName.XNAME.value,
        "yname": _AttrName.YNAME.value,
        "zname": _AttrName.ZNAME.value,
    }

    if is_polygons:
        kwargs["pname"] = "POLY_ID"

    roxxyz = _get_roxitem(
        rox,
        name,
        category,
        stype,
        mode="get",
        is_polygons=is_polygons,
    )
    values = _get_roxvalues(roxxyz, realisation=realisation)

    dfr = _roxapi_xyz_to_dataframe(values, is_polygons=is_polygons)

    # handling attributes for points, from Roxar API version 1.6
    if attributes and not is_polygons:
        attr_names = roxxyz.get_attributes_names(realisation=realisation)
        if isinstance(attributes, list):
            attr_names = [a for a in attr_names if a in attributes]
        logger.info("XYZ attribute names are: %s", attr_names)
        attr_dict = _get_rox_attrvalues(roxxyz, attr_names, realisation=realisation)
        dfr, datatypes = _add_attributes_to_dataframe(dfr, attr_dict)
        kwargs["attributes"] = datatypes

    kwargs["values"] = dfr
    return kwargs


def _roxapi_xyz_to_dataframe(
    roxxyz: list[np.ndarray] | np.ndarray, is_polygons: bool = False
) -> pd.DataFrame:
    """Transforming some XYZ from ROXAPI to a Pandas dataframe."""

    # In ROXAPI, polygons/polylines are a list of numpies, while
    # points is just a numpy array. Hence a polyg* may be identified
    # by being a list after import

    logger.info("Points/polygons/polylines from roxapi to xtgeo...")

    if is_polygons and isinstance(roxxyz, list):
        # polylines/-gons
        dfs = []
        for idx, poly in enumerate(roxxyz):
            dataset = pd.DataFrame.from_records(poly, columns=XYZ_COLUMNS)
            dataset["POLY_ID"] = idx
            dfs.append(dataset)

        dfr = pd.concat(dfs)

    elif not is_polygons and isinstance(roxxyz, np.ndarray):
        # points
        dfr = pd.DataFrame.from_records(roxxyz, columns=XYZ_COLUMNS)

    else:
        raise RuntimeError(f"Unknown error in getting data from Roxar: {type(roxxyz)}")

    dfr.reset_index(drop=True, inplace=True)
    return dfr


def _add_attributes_to_dataframe(
    dfr: pd.DataFrame, attributes: dict
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Add attributes to dataframe (points only) for Roxar API ver 1.6+"""

    logger.info("Attributes adding to dataframe...")
    newdfr = dfr.copy()

    datatypes = {}
    for name, values in attributes.items():
        dtype = str(values.dtype)
        if "int" in dtype:
            datatypes[name] = "int"
            values = np.ma.filled(values, fill_value=UNDEF_INT)
        elif "float" in dtype:
            datatypes[name] = "float"
            values = np.ma.filled(values, fill_value=UNDEF)
        else:
            datatypes[name] = "str"
            values = np.ma.filled(values, fill_value="UNDEF")

        newdfr[name] = values

    return newdfr, datatypes


def export_xyz_roxapi(
    self: points.Points | polygons.Polygons,
    project: RoxarType.Project,
    name: str,
    category: str | list[str] | None,
    stype: str,
    pfilter: dict[str, list],
    realisation: int,
    attributes: bool = False,
) -> None:
    """Export (store) a XYZ item from XTGeo to RMS via ROXAR API spec."""
    is_polygons = isinstance(self, polygons.Polygons)
    rox = RoxUtils(project, readonly=False)
    stype = STYPE(stype.lower())

    _check_input_and_version_requirement(
        rox, stype, category, attributes, is_polygons, mode="set"
    )
    _check_presence_in_project(rox, name, category, stype, realisation, mode="set")

    if attributes and not rox.version_required("1.6"):
        _roxapi_export_xyz_viafile(
            self, rox, name, category, stype, pfilter, realisation, attributes
        )
    elif stype == STYPE.WELL_PICKS:
        assert isinstance(self, points.Points)
        category = cast(Literal["fault", "horizon"], category)
        _roxapi_export_xyz_well_picks(self, rox, name, category, attributes, pfilter)
    else:
        _roxapi_export_xyz(
            self, rox, name, category, stype, pfilter, realisation, attributes
        )

    if rox._roxexternal:
        rox.project.save()

    rox.safe_close()


def _roxapi_export_xyz_viafile(
    self: points.Points | polygons.Polygons,
    rox: RoxUtils,
    name: str,
    category: str | list[str] | None,
    stype: STYPE,
    pfilter: dict[str, list],
    realisation: int,
    attributes: bool,
) -> None:
    """Set points/polys within RMS with attributes, using file workaround"""

    logger.warning("Realisation %s not in use", realisation)

    is_polygons = isinstance(self, polygons.Polygons)
    roxxyz = _get_roxitem(
        rox, name, category, stype, mode="set", is_polygons=is_polygons
    )

    # make a temporary folder and work within the with.. block
    with tempfile.TemporaryDirectory() as tmpdir:
        logger.info("Made a tmp folder: %s", tmpdir)
        ncount = self.to_file(
            os.path.join(tmpdir, "generic.rmsattr"),
            fformat="rms_attr",
            pfilter=pfilter,
            attributes=attributes,
        )

        if ncount:
            roxxyz.load(
                os.path.join(tmpdir, "generic.rmsattr"), roxar.FileFormat.RMS_POINTS
            )


def _roxapi_export_xyz(
    self: points.Points | polygons.Polygons,
    rox: RoxUtils,
    name: str,
    category: str | list[str] | None,
    stype: STYPE,
    pfilter: dict[str, list] | None,
    realisation: int,
    attributes: bool,
) -> None:
    logger.warning("Realisation %s not in use", realisation)

    df = self.get_dataframe()
    if df is None or df.empty:
        logger.warning("Empty dataframe! Skipping object update...")
        return

    is_polygons = isinstance(self, polygons.Polygons)
    xyz_columns = [self.xname, self.yname, self.zname]
    if not set(xyz_columns).issubset(df.columns):
        raise ValueError(
            f"One or all {xyz_columns=} are missing in the dataframe, "
            f"available columns are {list(df.columns)}! "
            "Rename your columns or update the corresponding 'xname', "
            "'yname' and 'zname' attributes to columns in your dataframe."
        )

    roxxyz = _get_roxitem(
        rox, name, category, stype, mode="set", is_polygons=is_polygons
    )

    df = _apply_pfilter_to_dataframe(df, pfilter)
    if df.empty:
        logger.warning("Empty dataframe after filtering! Skipping object update...")
        return

    arrxyz = (
        [polydf[xyz_columns].to_numpy() for _, polydf in df.groupby(self.pname)]
        if is_polygons
        else df[xyz_columns].to_numpy()
    )

    roxxyz.set_values(arrxyz)

    if attributes and isinstance(self, points.Points):
        for attr in _get_attribute_names_from_dataframe(df, xyz_columns):
            values = _replace_undefined_values(
                values=df[attr].values, dtype=self._attrs.get(attr), asmasked=True
            )

            logger.info("Store Point attribute %s to Roxar API", name)
            roxxyz.set_attribute_values(attr, values)


def _get_attribute_names_from_dataframe(
    df: pd.DataFrame, xyz_columns: list[str] | None = None
) -> list[str]:
    xyz_columns = xyz_columns or XYZ_COLUMNS
    return [col for col in df.columns if col not in xyz_columns]


def _get_attribute_type_from_values(values: np.ndarray) -> str:
    if "float" in str(values.dtype):
        return "float"
    if "int" in str(values.dtype):
        return "int"
    return "str"


def _replace_undefined_values(
    values: np.ndarray,
    dtype: str | None = None,
    asmasked: bool = False,
) -> np.ndarray | np.ma.MaskedArray:
    """
    Set xtgeo UNDEF values to np.nan or empty string dependent on type.
    With option to return array with masked values instead of np.nan.
    """
    values = pd.to_numeric(values, errors="ignore")

    dtype = dtype or _get_attribute_type_from_values(values)

    if dtype == "float":
        if asmasked:
            return np.ma.masked_greater(values, UNDEF_LIMIT)
        return np.where(values > UNDEF_LIMIT, np.nan, values)

    if dtype == "int":
        if asmasked:
            return np.ma.masked_greater(values, UNDEF_INT_LIMIT)
        return np.where(values > UNDEF_INT_LIMIT, np.nan, values)

    # string attributes does not support nan values
    # and requires string type array returned
    values = values.astype(str)
    return np.where(np.isin(values, ["UNDEF", "nan"]), "", values)


def _apply_pfilter_to_dataframe(
    df: pd.DataFrame, pfilter: dict[str, list] | None
) -> pd.DataFrame:
    if pfilter is not None:
        for key, val in pfilter.items():
            if key not in df:
                raise KeyError(
                    f"The requested pfilter key {key} was not found in dataframe. "
                    f"Valid keys are {list(df.columns)}"
                )
            df = df.loc[df[key].isin(val)]
    return df


def _get_rox_clipboard_folders(category: str | list[str] | None) -> list[str]:
    if category is None or category == "":
        return []

    if isinstance(category, list):
        return category

    if isinstance(category, str):
        if "|" in category:
            return category.split("|")
        if "/" in category:
            return category.split("/")
        return [category]

    raise RuntimeError(f"Cannot parse category: {category}, see documentation!")


def _get_roxitem(
    rox: RoxUtils,
    name: str,
    category: str | list[str] | None,
    stype: STYPE,
    mode: Literal["set", "get"] = "set",
    is_polygons: bool = False,
) -> Any:
    """Get the correct rox_xyz which is some pointer to a RoxarAPI structure."""

    project_attr = getattr(rox.project, stype)

    if stype not in [STYPE.CLIPBOARD, STYPE.GENERAL2D_DATA]:
        return project_attr[name][category]

    folders = _get_rox_clipboard_folders(category)
    if mode == "get":
        return project_attr.folders[folders][name]

    # clipboard folders will be created if not present, and overwritten else
    return (
        project_attr.create_polylines(name, folders)
        if is_polygons
        else project_attr.create_points(name, folders)
    )


def _get_roxvalues(rox_xyz: Any, realisation: int = 0) -> list[np.ndarray] | np.ndarray:
    """Return primary values from the Roxar API, numpy (Points) or list (Polygons)."""
    try:
        roxitem = rox_xyz.get_values(realisation)
        logger.info(roxitem)
    except KeyError as kwe:
        logger.error(kwe)

    return roxitem


def _get_rox_attrvalues(
    rox_xyz: Any, attrnames: list[str], realisation: int = 0
) -> dict[str, list[np.ndarray] | np.ndarray]:
    """Return attributes from the Roxar API, numpy (Points) or list (Polygons)."""
    roxitems = {}
    for attrname in attrnames:
        values = rox_xyz.get_attribute_values(attrname, realisation=realisation)
        roxitems[attrname] = values
    return roxitems


def _roxapi_import_wellpicks(
    rox: RoxUtils,
    well_pick_set: str,
    wp_category: Literal["fault", "horizon"] = "horizon",
    attributes: bool | list[str] = False,
) -> dict[str, str | pd.DataFrame | dict]:
    """From RMS to XTGeo"""

    rox_wp = rox.project.well_picks
    rox_wp_set = rox_wp.sets[well_pick_set]

    well_picks = [wp for wp in rox_wp_set if wp.type.name == wp_category]
    if len(well_picks) == 0:
        raise ValueError(
            f"No well picks of type '{wp_category}' found in {well_pick_set=}."
        )

    attribute_types = {}
    if attributes:
        rox_attributes = well_picks[0].attributes  # first one is valid for all
        for rox_attr in rox_attributes:
            if isinstance(attributes, list) and rox_attr.name not in attributes:
                continue
            attribute_types[rox_attr.name] = rox_attr.type.name

    dfr = _create_dataframe_from_wellpicks(well_picks, wp_category, attribute_types)

    return {
        "xname": _AttrName.XNAME.value,
        "yname": _AttrName.YNAME.value,
        "zname": _AttrName.ZNAME.value,
        "values": dfr,
        "attributes": attribute_types,
    }


def _create_dataframe_from_wellpicks(
    well_picks: list[RoxarType.well_picks.WellPick],
    wp_category: Literal["fault", "horizon"],
    attribute_types: dict[str, str],
) -> pd.DataFrame:
    """Create a dataframe from a well pick set, and selected attributes."""

    items = []
    for wp in well_picks:
        wp_attributes = {attr.name: val for attr, val in wp.get_values().items()}

        data = {
            _AttrName.XNAME.value: wp_attributes["East"],
            _AttrName.YNAME.value: wp_attributes["North"],
            _AttrName.ZNAME.value: wp_attributes["TVD_MSL"],
            _AttrName.M_MD_NAME.value: wp_attributes["MD"],
            "WELLNAME": wp.trajectory.wellbore.well.name,
            "TRAJECTORY": wp.trajectory.name,
            wp_category.upper(): wp.intersection_object.name,
        }

        for attr, dtype in attribute_types.items():
            if attr in wp_attributes:
                if wp_attributes[attr] is not None:
                    data[attr] = wp_attributes[attr]
                else:
                    if dtype == "float":
                        data[attr] = UNDEF
                    elif dtype == "int":
                        data[attr] = UNDEF_INT
                    else:
                        data[attr] = "UNDEF"

        items.append(data)

    return pd.DataFrame(items)


def _roxapi_export_xyz_well_picks(
    self: points.Points,
    rox: RoxUtils,
    well_pick_set: str,
    wp_category: Literal["horizon", "fault"],
    attributes: bool,
    pfilter: dict[str, list] | None,
) -> None:
    """
    Export/store as RMS well picks; this is only valid if points belong to wells
    """

    df = self.get_dataframe()
    if df is None or df.empty:
        logger.warning("Empty dataframe! Skipping object update...")
        return

    project_attr = getattr(rox.project, f"{wp_category}s")
    rox_wp_type = getattr(roxar.WellPickType, wp_category)

    df = _apply_pfilter_to_dataframe(df, pfilter)
    if df.empty:
        logger.warning("Empty dataframe after filtering! Skipping object update...")
        return

    required_columns = REQUIRED_WELL_PICK_ATTRIBUTES + [wp_category.upper()]
    for column in required_columns:
        if column not in df:
            raise ValueError(f"Required {column=} missing in the dataframe.")
        if df[column].isnull().any():
            raise ValueError(f"The required {column=} contains undefined values.")

    if attributes:
        attr_types = self._attrs

        attr_types = {}
        for attr in _get_attribute_names_from_dataframe(df):
            if attr not in required_columns:
                if attr in self._attrs:
                    attr_types[attr] = self._attrs[attr]
                else:
                    attr_types[attr] = _get_attribute_type_from_values(df[attr])

        rox_wp_attributes = _get_writeable_well_pick_attributes(
            rox, attr_types, rox_wp_type
        )
        for attr in rox_wp_attributes:
            df[attr] = _replace_undefined_values(
                values=df[attr].values, dtype=attr_types.get(attr), asmasked=False
            )

    mypicks = []
    for well, wp_df in df.groupby("WELLNAME"):
        rox_well_traj = rox.project.wells[well].wellbore.trajectories

        for _, wp_row in wp_df.iterrows():
            intersection_object_name = wp_row[wp_category.upper()]
            if intersection_object_name not in project_attr:
                raise ValueError(
                    f"{wp_category} '{intersection_object_name}' not in project"
                )
            traj_name = wp_row["TRAJECTORY"]
            if traj_name not in rox_well_traj:
                raise ValueError(
                    f"Trajectory name '{traj_name}' not present for {well=}"
                )
            wp = roxar.well_picks.WellPick.create(
                intersection_object=project_attr[intersection_object_name],
                trajectory=rox_well_traj[traj_name],
                md=wp_row[_AttrName.M_MD_NAME.value],
            )
            if attributes:
                for attr, rox_attr in rox_wp_attributes.items():
                    try:
                        wp.set_values({rox_attr: wp_row[attr]})
                    except ValueError as err:
                        raise ValueError(
                            f"Could not assign value '{wp_row[attr]}' to attribute "
                            f"'{attr}'. The value type {type(wp_row[attr])} might be "
                            f"incompatible with dtype of attribute '{rox_attr.type}'"
                        ) from err

            mypicks.append(wp)

    rox_wps = _get_well_pick_set(rox, well_pick_set, rox_wp_type)
    rox_wps.append(mypicks)


def _get_well_pick_set(
    rox: RoxUtils, well_pick_set: str, rox_wp_type: RoxarType.WellPickType
) -> RoxarType.well_picks.WellPickSet:
    """
    Function to retrieve a well pick set object. If the given well pick set
    name is not present, it will be created. Otherwise the current well pick
    set will be emptied for the given well pick type.
    """
    well_pick_sets = rox.project.well_picks.sets
    if well_pick_set not in well_pick_sets:
        well_pick_sets.create(well_pick_set)

    rox_wps = well_pick_sets[well_pick_set]

    rox_wps.delete_at([idx for idx, wp in enumerate(rox_wps) if wp.type == rox_wp_type])
    return rox_wps


def _get_writeable_well_pick_attributes(
    rox: RoxUtils,
    attribute_types: dict[str, str],
    rox_wp_type: RoxarType.WellPickType,
) -> dict[str, RoxarType.well_picks.WellPickAttribute]:
    """
    Function to retrive a dictionary of regular and user-defined
    roxar WellPickAttribute's. Only writable attributes are
    returned (i.e. not read_only). Attributes not present in the
    project will be created as user-defined attributes.
    """
    attributes_with_value_constraints = [
        "Structural model",
        "Lock",
        "Quality",
        "Wellpick Symbol - Horizon",
    ]
    regular_attributes = {x.name: x for x in roxwp.WellPick.get_attributes(rox_wp_type)}
    user_attributes = {
        x.name: x
        for x in rox.project.well_picks.user_attributes.get_subset(rox_wp_type)
    }

    rox_attributes = {}
    for attr, dtype in attribute_types.items():
        rox_dtype = getattr(roxar.WellPickAttributeType, dtype)

        if attr in regular_attributes:
            if (
                regular_attributes[attr].read_only
                or attr in attributes_with_value_constraints
            ):
                logger.debug("Skipping read-only attribute %s", attr)
                continue
            rox_attributes[attr] = regular_attributes[attr]

        elif attr in user_attributes:
            if user_attributes[attr].type != rox_dtype:
                raise ValueError(
                    f"Attribute type provided for '{attr}': {dtype}, is different "
                    f" from existing type in project: {user_attributes[attr].type}.\n"
                    "Either delete user defined attribute up-front, "
                    "or rename to a new unique attribute name."
                )
            rox_attributes[attr] = user_attributes[attr]

        else:
            # roxar only supports creating string or float attributes
            if dtype not in ["str", "float"]:
                raise ValueError(
                    "Only 'float' or 'str' are valid options for user-defined "
                    f"attributes. Found type {dtype} for attribute '{attr}'."
                )
            logger.info("Creating user-defined attribute %s", attr)
            rox_attributes[attr] = rox.project.well_picks.user_attributes.create(
                name=attr,
                pick_type=rox_wp_type,
                data_type=rox_dtype,
            )

    return rox_attributes
