from typing import List, Union

import ecl_data_io as eclio
from typing_extensions import Literal

from ._find_gridprop_in_eclrun import (
    find_gridprop_from_init_file,
    find_gridprops_from_restart_file,
)


def decorate_name(name, dual_porosity, fracture, date=None):
    """Decorate a property name with date and matrix/fracture.

    >>> decorate_name('PORO', True, False, 19991231)
    'POROM_19991231'
    """
    decorated_name = name
    if dual_porosity:
        if fracture:
            decorated_name += "F"
        else:
            decorated_name += "M"

    if date is not None:
        decorated_name += "_" + str(date)
    return decorated_name


def import_gridprop_from_init(pfile, name, grid, fracture=False):
    """Import one parameter with the given name from an init file.

    Args:
        pfile: The init file.
        name: The name of the parmaeter
        grid: The grid used by the simulator to produce the init file.
        fracture: If a dual porosity module, indicates that the fracture
            (as apposed to the matrix) grid property should be imported.
    Raises:
        ValueError: If the parameter does not exist in the file.
    Returns:
        GridProperty parameter dictionary.
    """
    init_props = find_gridprop_from_init_file(pfile.file, [name], grid, fracture)
    if len(init_props) != 1:
        raise ValueError(f"Could not find property {name} in {pfile}")
    init_props[0]["name"] = decorate_name(
        init_props[0]["name"], grid.dualporo, fracture
    )
    return init_props[0]


def sanitize_date(
    date: Union[int, str, Literal["first", "last"]]
) -> Union[List[int], Literal["first", "last"]]:
    """
    Converts dateformats of the form 'YYYY-MM-DD', 'YYYYMMDD' or YYYYMMDD to
    list of integers of the form [YYYYMMDD] (ie. suitible for find_gridprops
    functions), but lets the special literals 'first' and 'last' remain
    unchanged.

    >>> sanitize_date('first')
    'first'
    >>> sanitize_date('last')
    'last'
    >>> sanitize_date('2020-01-01')
    [20200101]
    >>> sanitize_date('20200101')
    [20200101]
    >>> sanitize_date(20200101)
    [20200101]
    """
    if isinstance(date, int):
        return [date]
    if date not in ("first", "last"):
        try:
            if isinstance(date, str):
                if len(date) == 10 and date[4] == "-" and date[7] == "-":
                    date = date.replace("-", "")
            return [int(date)]
        except ValueError as err:
            raise ValueError(
                "valid dates are either of the "
                "form 'YYYY-MM-DD', 'YYYYMMDD' or 'first'/'last' "
                f"got {date}"
            ) from err
    return date


def sanitize_fformat(fformat: Literal["unrst", "funrst"]) -> eclio.Format:
    """Converts 'unrst' and 'funrst' to the corresponding eclio.Format.

    >>> sanitize_fformat('unrst')
    <Format.UNFORMATTED: 2>
    >>> sanitize_fformat('funrst')
    <Format.FORMATTED: 1>
    """
    if fformat == "unrst":
        return eclio.Format.UNFORMATTED
    if fformat == "funrst":
        return eclio.Format.FORMATTED
    raise ValueError(f"fformat must be either 'unrst' or 'funrst' got {fformat}")


def import_gridprop_from_restart(
    pfile,
    name: str,
    grid,
    date: Union[int, str, Literal["first", "last"]],
    fracture: bool = False,
    fformat: Literal["unrst", "funrst"] = "unrst",
):
    """Import one parameter for the given name and date in a restart file.

    Args:
        pfile: The restart file.
        name: The name of the parmaeter
        date: xtgeo style date (e.g. int(19990101) or "YYYYMMDD"), also
            accepts "YYYY-MM-DD".  "last" and "first" can be given for
            last or first date in the file
        grid: The grid used by the simulator to produce the restart file.
        fracture: If a dual porosity module, indicates that the fracture
            (as apposed to the matrix) grid property should be imported.
    Raises:
        ValueError: If the parameter does not exist in the file.
    Returns:
        GridProperty parameter dictionary.
    """
    restart_props = find_gridprops_from_restart_file(
        pfile.file,
        [name],
        sanitize_date(date),
        grid,
        fracture,
        sanitize_fformat(fformat),
    )
    if len(restart_props) == 0:
        raise ValueError(f"Could not find property {name} for {date} in {pfile.file}")
    if len(restart_props) > 1:
        raise ValueError(f"Ambiguous property {name} for {date} in {pfile.file}")
    restart_props[0]["name"] = decorate_name(
        restart_props[0]["name"], grid.dualporo, fracture, restart_props[0]["date"]
    )
    return restart_props[0]
