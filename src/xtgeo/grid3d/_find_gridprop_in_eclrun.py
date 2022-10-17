import functools
import itertools
import operator
import pathlib
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union

import ecl_data_io as eclio
import numpy as np
import xtgeo
from typing_extensions import Literal

from ._ecl_inte_head import InteHead
from ._ecl_logi_head import LogiHead
from ._ecl_output_file import Phases
from ._grdecl_format import match_keyword

sat_keys = ["SOIL", "SGAS", "SWAT"]


def filter_lgr(generator):
    try:
        while True:
            entry = next(generator)
            if entry.read_keyword() == "LGR":
                warnings.warn(
                    "Found LGR in ecl run file. "
                    "LGR's are not directly supported, "
                    "instead only global values are imported."
                )
                while entry.read_keyword() != "ENDLGR":
                    entry = next(generator)
            else:
                yield entry
    except StopIteration:
        return


# Based on which phases are present some saturation values are given default
# values, e.g. if phases=Phases.OIL, the saturation of oil ("SOIL") is 1.0 and
# all other phases are 0.0.
DEFAULT_SATURATIONS = {
    Phases.OIL: {
        "SOIL": 1.0,
        "SWAT": 0.0,
        "SGAS": 0.0,
    },
    Phases.GAS: {
        "SOIL": 0.0,
        "SWAT": 0.0,
        "SGAS": 1.0,
    },
    Phases.WATER: {
        "SOIL": 0.0,
        "SWAT": 1.0,
        "SGAS": 0.0,
    },
    Phases.OIL_WATER: {
        "SGAS": 0.0,
    },
    Phases.OIL_GAS: {
        "SWAT": 0.0,
    },
    Phases.GAS_WATER: {
        "SOIL": 0.0,
    },
    Phases.OIL_WATER_GAS: dict(),
    Phases.E300_GENERIC: dict(),
}


def remainder_saturations(saturations):
    """Infers remainder saturations based on sum(saturations.values()) == 1.

    >>> remainder_saturations({'SWAT': 0.5, 'SGAS': 0.5})
    {'SOIL': 0.0}

    Args:
        saturations: Dictionary of phases, such as returned by
            :meth:`default_saturations()`.

    Returns:
        dictionary of saturation values that can be inferred.
    """
    if all(k in saturations for k in sat_keys):
        return dict()
    if any(k not in sat_keys for k in saturations):
        raise ValueError(f"Unknown saturation keys: {list(saturations.keys())}")
    rest = sum(saturations.values())
    if len(saturations) == 2 or np.allclose(rest, 1.0):
        missing = set(sat_keys).difference(set(saturations.keys()))
        return {m: 1.0 - rest for m in missing}
    return dict()


def peek_headers(generator):
    """Reads header from a ecl_data_io keyword generator without consuming keywords.

    Args:
        generator: keyword generator such as returned by ecl_data_io.lazy_read

    Returns:
        Tuple of inthead, logihead, and a modified generator which contains all original
        keywords.
    """

    def read_headers(generator):
        intehead_array = None
        logihead_array = None
        while intehead_array is None or logihead_array is None:
            entry = next(generator)
            kw = entry.read_keyword()
            if match_keyword(kw, "LOGIHEAD"):
                logihead_array = entry.read_array()
            if match_keyword(kw, "INTEHEAD"):
                intehead_array = entry.read_array()

        intehead = InteHead(intehead_array)
        logihead = LogiHead.from_file_values(logihead_array, intehead.simulator)
        return intehead, logihead

    header_generator, generator = itertools.tee(generator)
    try:
        intehead, logihead = read_headers(header_generator)
    except StopIteration as stopit:
        raise ValueError("Reached end of file without reading headers") from stopit
    return intehead, logihead, generator


def get_fetch_names(name: str) -> List[str]:
    """Given a gridproperty name, give list of supporting keyword names.

    >>> get_fetch_names('PORO')
    ['PORO']
    >>> get_fetch_names('SWAT')
    ['SOIL', 'SGAS', 'SWAT']

    Args:
        name: The name of a grid property
    Returns:
        List of grid properties that must be fetched from the file.

    """
    if any(match_keyword(name, saturation_keyword) for saturation_keyword in sat_keys):
        fetch_names = sat_keys
    else:
        fetch_names = [name]
    return fetch_names


def read_values(generator, intehead, names, lengths="all"):
    """Read the given list of parameter values from the generator.

    Reads the given list of values from the generator. Some saturation
    values may be inferred from the invariant sum(saturations.values()) == 1.0
    (see  :meth:`remainder_saturations()`.

    """

    def flatten(lst):
        return functools.reduce(operator.iconcat, lst, [])

    fetch_names = set(flatten(list(get_fetch_names(name) for name in names)))
    defaulted = list()
    if names == "all":
        values = dict()
    else:
        values = DEFAULT_SATURATIONS[intehead.phases].copy()
        defaulted = list(values.keys())

    for entry in generator:
        if all(name in values for name in fetch_names):
            break
        kw = entry.read_keyword()
        if lengths != "all":
            if entry.read_length() not in lengths:
                continue
        if names == "all":
            key = kw.rstrip()
            array = entry.read_array()
            if np.issubdtype(array.dtype, np.number) or np.issubdtype(
                array.dtype, bool
            ):
                values[key] = entry.read_array()
        else:
            matched = [name for name in fetch_names if match_keyword(kw, name)]
            if len(matched) == 1:
                if matched[0] in values and matched[0] not in defaulted:
                    raise ValueError(f"Found duplicate keyword {matched[0]}")
                values[matched[0]] = entry.read_array()
            elif len(matched) > 1:
                # This should not happen if get_fetch_names and
                # match_keyword work as intended
                raise ValueError(f"Ambiguous keywords {matched} matched vs {kw}")

    if names == "all":
        # A more consistent behavior would be to include saturations calculated
        # with remainder_saturations when names=="all" aswell, however, we do
        # not include those to keep backwards compatability.
        # TODO: deprecate this behavior
        return values
    else:
        values.update(
            **remainder_saturations({k: values[k] for k in sat_keys if k in values})
        )
        return {name: values[name] for name in names if name in values}


def check_grid_match(intehead: InteHead, logihead: LogiHead, grid):
    """Checks that the init/restart headers matches the grid

    Checks that the values given in the headers are compatible with
    the grid.
    """
    dimensions = intehead.num_x, intehead.num_y, intehead.num_z

    if logihead.dual_porosity:
        dimensions = intehead.num_x, intehead.num_y, intehead.num_z // 2

    if logihead.dual_porosity != grid.dualporo:
        raise ValueError("Grid dual poro status does not match output file")

    if dimensions != grid.dimensions:
        raise ValueError(
            "Grid dimensions do not match dimensions given in output file,"
            f" {dimensions} vs {grid.dimensions}"
        )


def expand_scalar_values(value, num_cells, dualporo: bool) -> np.ndarray:
    """Convert from scalar value to filled array of expected size.
    Args:
        value: The potentially scalar value
        num_cells: The number of cells in the grid
        dualporo: Whether the model has dual porosity
    Returns:
        If value is an array, then returns that array, otherwise
        calls np.full with the shape determined by dualporo status and
        number of cells.

    """
    if dualporo:
        return np.full(fill_value=value, shape=num_cells * 2)
    return np.full(fill_value=value, shape=num_cells)


def pick_dualporo_values(
    values: np.ndarray, actind: np.ndarray, num_cells: int, fracture: bool
) -> np.ndarray:
    """From array of values in an ecl run file, give the fracture or matrix values.

    Args:
        values: Array of values from an ecl run file.
        actind: Array of the indecies of active cells.
        num_cells: Total number of cells.
        fracture: Whether to give the fracture or matrix values.
    Returns:
        Array of either fracture or matrix values from the input values.
    """
    active_size = len(actind)
    if len(values) == 2 * num_cells:
        indsize = num_cells
    else:
        indsize = active_size
    if fracture:
        return values[-indsize:]
    return values[:indsize]


def valid_gridprop_lengths(grid):
    num_cells = np.prod(grid.dimensions)
    if grid.dualporo:
        num_fracture = len(grid.get_dualactnum_indices(fracture=True))
        num_matrix = len(grid.get_dualactnum_indices(fracture=False))
        return [2 * num_cells, num_fracture + num_matrix]
    else:
        num_active = len(grid.get_actnum_indices())
        return [num_cells, num_active]


def match_values_to_active_cells(
    values,
    actind,
    num_cells,
) -> np.ndarray:
    """Expands array of ecl run values to be one-to-one with cells.

    In the ecl run file, the values might be only those for active cells.
    This funtion expands those values to one for each cell with non-active
    indecies given the xtgeo.UNDEF/xtgeo.UNDEF_INT value.

    Args:
        values: Array of values from an ecl run file.
        actind: Array of the indecies of active cells.
        num_cells: Total number of cells.
    Returns:
        Array of input values, but guaranteed num_cells length.

    """

    if len(values) != len(actind):
        raise ValueError(
            f"Unexpected shape of values in init file: {np.asarray(values).shape}, "
            f"expected to match grid dimensions {num_cells} or "
            f"number of active cells {len(actind)}"
        )

    if np.issubdtype(values.dtype, np.integer):
        undef = xtgeo.UNDEF_INT
    else:
        undef = xtgeo.UNDEF
    result = np.full(fill_value=undef, shape=num_cells, dtype=values.dtype)
    result[actind] = values
    return result


def make_gridprop_values(values, grid, fracture):
    """Converts values given in init or restart file to one suitable for GridProperty.

    Args:
    values: The array read from the file
    grid: The grid from the ecl run
    fracture: Whether to get the fracture or matrix values

    Returns:
        Masked array of values indexed by cell
    """
    num_cells = np.prod(grid.dimensions)
    if np.isscalar(values):
        values = expand_scalar_values(values, num_cells, grid.dualporo)

    if grid.dualporo:
        actind = grid.get_dualactnum_indices(fracture=fracture, order="F")
        values = pick_dualporo_values(values, actind, num_cells, fracture)
    else:
        actind = grid.get_actnum_indices(order="F")

    if len(values) != num_cells:
        values = match_values_to_active_cells(values, actind, num_cells)

    values = values.reshape(grid.dimensions, order="F")

    if grid.dualporo:
        if fracture:
            values[grid._dualactnum.values == 1] = 0.0
        else:
            values[grid._dualactnum.values == 2] = 0.0

    return np.ma.masked_where(grid.get_actnum().values < 1, values)


def date_from_intehead(intehead: InteHead) -> Optional[int]:
    """Returns date format for use in GridProperty name given intehead."""
    if any(val is None for val in [intehead.day, intehead.month, intehead.year]):
        return None
    return intehead.day + intehead.month * 100 + intehead.year * 10000


def gridprop_params(values, name, date, grid, fracture):
    """Make dictionary of GridProperty parameters from imported values."""
    result = dict()
    result["name"] = name
    result["date"] = str(date) if date is not None else None
    result["fracture"] = fracture

    result["ncol"], result["nrow"], result["nlay"] = grid.dimensions
    result["dualporo"] = grid.dualporo
    result["dualperm"] = grid.dualperm

    result["values"] = make_gridprop_values(values, grid, fracture)

    if np.issubdtype(result["values"].dtype, np.integer):
        uniq = np.unique(values).tolist()
        codes = dict(zip(uniq, uniq))
        codes = {key: str(val) for key, val in codes.items()}
        result["codes"] = codes
        result["values"] = result["values"].astype(np.int32)
        result["discrete"] = True
    else:
        result["codes"] = dict()
        result["values"] = result["values"].astype(np.float64)
        result["discrete"] = False
    return result


def get_actnum_from_porv(init_filelike, grid):
    """Override actnum value based on the cell pore volume
    Args:
        init_filelike: The init file
        grid: The grid used by the simulator to produce the init file.
    Returns:
        None
    """
    generator = filter_lgr(eclio.lazy_read(init_filelike))
    intehead, logihead, generator = peek_headers(generator)

    check_grid_match(intehead, logihead, grid)
    porv = read_values(
        generator, intehead, ["PORV"], lengths=valid_gridprop_lengths(grid)
    )
    if porv:
        if grid.dualporo:
            num_cells = np.prod(grid.dimensions)
            actnum_matrix = np.where(
                porv["PORV"][:num_cells].reshape(grid.dimensions, order="F") > 0.0, 1, 0
            )
            actnum_fracture = np.where(
                porv["PORV"][num_cells:].reshape(grid.dimensions, order="F") > 0.0, 2, 0
            )
            grid._dualactnum.values = actnum_matrix + actnum_fracture
        else:
            acttmp = grid.get_actnum().copy()
            acttmp.values = np.where(
                porv["PORV"].reshape(grid.dimensions, order="F") > 0.0, 1, 0
            )
            grid.set_actnum(acttmp)


def find_gridprop_from_init_file(
    init_filelike,
    names: Union[List[str], Literal["all"]],
    grid,
    fracture: bool = False,
) -> List[Dict]:
    """Finds all parameters in a init matching names.

    Note: Does not check that all names are found.

    Args:
        init_filelike: The init file
        names: List of property names to be imported. Can also,
            be set to "all" to import all parameters.
        grid: The grid used by the simulator to produce the init file.
        fracture: If a dual porosity module, indicates that the fracture
            (as apposed to the matrix) grid property should be imported.
    Returns:
        List of GridProperty parameters matching the names.

    """
    init_stream = not isinstance(init_filelike, (str, Path))
    if init_stream:
        orig_pos = init_filelike.tell()
    get_actnum_from_porv(init_filelike, grid)
    if init_stream:
        init_filelike.seek(orig_pos, 0)
    generator = filter_lgr(eclio.lazy_read(init_filelike))
    intehead, logihead, generator = peek_headers(generator)

    check_grid_match(intehead, logihead, grid)

    date = date_from_intehead(intehead)
    return [
        gridprop_params(v, name, date, grid, fracture)
        for name, v in read_values(
            generator, intehead, names, lengths=valid_gridprop_lengths(grid)
        ).items()
    ]


def section_generator(generator):
    """Sections the generator as delimited by "SEQNUM" keyword.

    Unified restart files will repeat properties in sections, one for each
    date. The "SEQNUM" keyword indicates a new section. section_generator
    takes a keyword generator and returns a generator over sections.

    Note: It does so in a lighweight manner so that a section
    is emptied once the next section is requested, and the original
    generator is iterated as you progress in the sections.
    """
    try:
        first_seq = next(generator)
        if not match_keyword(first_seq.read_keyword(), "SEQNUM"):
            raise ValueError("Restart file did not start with SEQNUM.")
    except StopIteration:
        return
    while True:
        this_section = itertools.takewhile(
            lambda x: not match_keyword(x.read_keyword(), "SEQNUM"), generator
        )
        yield this_section
        # empty the section iterator
        for _ in this_section:
            pass
        # peek ahead to see if there are more elements
        # and stop if there are not
        try:
            entry = next(generator)
            generator = itertools.chain(iter([entry]), generator)
        except StopIteration:
            return


def find_gridprops_from_restart_file_sections(
    sections,
    names: Union[List[str], Literal["all"]],
    dates: Union[List[int], Literal["all", "last", "first"]],
    grid,
    fracture: bool = False,
) -> List[Dict]:
    """Finds list of parameters from an sections generator.

    See :meth:`section_generator` for suitable input generator.

    When there are multiple steps/properties matching the given
    date, the first property matching the date is selected.

    Args:
        sections: Section generator such as returned by :meth:`section_generator`.
        names: List of property names to be imported. Can also,
            be set to "all" to import all parameters.
        dates: List of xtgeo style dates (e.g. int(19990101)), can also
            be "all", "last" and "first".
        grid: The grid used by the simulator to produce the init file.
        fracture: If a dual porosity module, indicates that the fracture
            (as apposed to the matrix) grid property should be imported.
    Returns:
        List of GridProperty parameters matching the names.
    """
    first_date = None
    last_date = None
    read_properties = dict()
    for section in sections:
        intehead, logihead, section = peek_headers(section)
        check_grid_match(intehead, logihead, grid)
        date = date_from_intehead(intehead)

        if dates not in ("all", "first", "last"):
            if date not in dates:
                continue

        section_properties = {
            (name, date): gridprop_params(v, name, date, grid, fracture)
            for name, v in read_values(
                section, intehead, names, lengths=valid_gridprop_lengths(grid)
            ).items()
        }

        if dates == "first":
            if first_date is None:
                first_date = date
            elif date != first_date:
                break

        elif dates == "last":
            if date != last_date:
                last_date = date
                read_properties = dict()

        for key in section_properties:
            if key not in read_properties:
                read_properties[key] = section_properties[key]
    return list(read_properties.values())


def find_gridprops_from_restart_file(
    restart_filelike,
    names: Union[List[str], Literal["all"]],
    dates: Union[List[int], Literal["all", "first", "last"]],
    grid,
    fracture: bool = False,
    fformat: eclio.Format = eclio.Format.UNFORMATTED,
) -> List[Dict]:
    """Finds all parameters in a restart file matching the given names and dates.

    Note: Does not check that all names are found.

    Args:
        restart_filelike: The restart file.
        names: List of property names to be imported. Can also,
            be set to "all" to import all parameters.
        dates: List of xtgeo style dates (e.g. int(19990101)), can also
            be "all", "last" and "first".
        grid: The grid used by the simulator to produce the restart file.
        fracture: If a dual porosity module, indicates that the fracture
            (as apposed to the matrix) grid property should be imported.
    Returns:
        List of GridProperty parameters matching the names and dates.
    """
    close = False
    if isinstance(restart_filelike, (pathlib.Path, str)):
        if fformat == eclio.Format.UNFORMATTED:
            filehandle = open(restart_filelike, "rb")
            close = True
        elif fformat == eclio.Format.FORMATTED:
            filehandle = open(restart_filelike, "rt")
            close = True
        else:
            raise ValueError(f"Unsupported restart file format {fformat}")
    else:
        # If restart_filelike is not a filename/path we assume
        # its a stream
        filehandle = restart_filelike
        close = False
    try:
        generator = section_generator(filter_lgr(eclio.lazy_read(filehandle)))
        read_properties = find_gridprops_from_restart_file_sections(
            generator,
            names,
            dates,
            grid,
            fracture,
        )
    finally:
        if close:
            filehandle.close()
    return read_properties
