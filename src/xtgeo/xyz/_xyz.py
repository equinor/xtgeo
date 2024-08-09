"""XTGeo XYZ module (abstract base class)"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar
from warnings import warn

import numpy as np

from xtgeo.common.log import null_logger
from xtgeo.common.xtgeo_dialog import XTGDescription, XTGeoDialog

from . import _xyz_oper

if TYPE_CHECKING:
    import pandas as pd


xtg = XTGeoDialog()
logger = null_logger(__name__)

Polygons = TypeVar("Polygons")


class XYZ(ABC):
    """Abstract base class for XYZ objects, i.e. Points and Polygons in XTGeo.

    The XYZ base class has common methods and properties for Points and Polygons. The
    underlying data storage is a Pandas dataframe with minimal 3 (Points) or 4
    (Polygons) columns, where the two first represent X and Y coordinates.

    The third column is a number, which may represent the depth, thickness, or other
    property. For Polygons, there is a 4'th column which is an integer representing
    poly-line ID, which is handled in the Polygons class. Similarly, Points and Polygons
    can have additional columns called `attributes`.

    Note:
        You cannot use the XYZ class directly. Use the :class:`Points` or
        :class:`Polygons` classes!
    """

    def __init__(
        self,
        xname: str = "X_UTME",
        yname: str = "Y_UTMN",
        zname: str = "Z_TVDSS",
    ):
        """Concrete initialisation for base class _XYZ."""
        self._xname = xname
        self._yname = yname
        self._zname = zname

    @property
    def xname(self):
        """Returns or set the name of the X column."""
        return self._xname

    @xname.setter
    def xname(self, newname):
        self._df_column_rename(newname, self._xname)
        self._xname = newname

    @property
    def yname(self):
        """Returns or set the name of the Y column."""
        return self._yname

    @yname.setter
    def yname(self, newname):
        self._df_column_rename(newname, self._yname)
        self._yname = newname

    @property
    def zname(self):
        """Returns or set the name of the Z column."""
        return self._zname

    @zname.setter
    def zname(self, newname):
        self._df_column_rename(newname, self._zname)
        self._zname = newname

    @property
    @abstractmethod
    def dataframe(self) -> pd.DataFrame:
        """Return or set the Pandas dataframe object."""
        ...

    @property
    def nrow(self):
        """Returns the Pandas dataframe object number of rows."""
        if self.get_dataframe(copy=False) is None:
            return 0
        return len(self.get_dataframe(copy=False).index)

    def _df_column_rename(self, newname, oldname):
        if isinstance(newname, str):
            if oldname and self.get_dataframe(copy=False) is not None:
                dataframe = self.get_dataframe()
                dataframe.rename(columns={oldname: newname}, inplace=True)
                self.set_dataframe(dataframe)
        else:
            raise ValueError(f"Wrong type of input to {newname}; must be string")

    def _check_name(self, value):
        if not isinstance(value, str):
            raise ValueError(f"Wrong type of input; must be string, was {type(value)}")

        if value not in self.get_dataframe(copy=False).columns:
            raise ValueError(
                f"{value} does not exist as a column name, must be "
                f"one of: f{self.get_dataframe(copy=False).columns}"
            )

    @abstractmethod
    def copy(self):
        """Returns a deep copy of an instance"""
        ...

    def describe(self, flush=True):
        """Describe an instance by printing to stdout"""

        dsc = XTGDescription()
        dsc.title(f"Description of {self.__class__.__name__} instance")
        dsc.txt("Object ID", id(self))
        dsc.txt("xname, yname, zname", self._xname, self._yname, self._zname)

        if flush:
            dsc.flush()
            return None

        return dsc.astext()

    @abstractmethod
    def get_dataframe(self, copy=True) -> pd.DataFrame:
        """Return the Pandas dataframe object."""
        ...

    @abstractmethod
    def set_dataframe(self, dataframe: pd.DataFrame) -> None:
        """Set the Pandas dataframe object."""
        ...

    def protected_columns(self):
        """
        Returns:
            Columns not deleted by :meth:`delete_columns`, for
            instance the coordinate columns.
        """
        return [self.xname, self.yname, self.zname]

    def geometry_columns(self):
        """
        Returns:
            Columns can be deleted silently by :meth:`delete_columns`
        """
        return [self.hname, self.dhname, self.tname, self.dtname]

    def delete_columns(self, clist, strict=False):
        """Delete one or more columns by name.

        Note that the columns returned by :meth:`protected_columns(self)` (for
        instance, the coordinate columns) will not be deleted.

        Args:
            self (obj): Points or Polygons
            clist (list): Name of columns
            strict (bool): I False, will not trigger exception if a column is not
                found. Otherways a ValueError will be raised.

        Raises:
            ValueError: If strict is True and columnname not present

        Example::
            mypoly.delete_columns(["WELL_ID", mypoly.hname, mypoly.dhname])

        .. versionadded:: 2.1
        """
        for cname in clist:
            if cname in self.protected_columns():
                xtg.warnuser(
                    f"The column {cname} is protected and will not be deleted."
                )
                continue

            if cname in self.geometry_columns():
                if strict:
                    raise ValueError(f"The column {cname} is not present.")
                logger.info(
                    "The column %s is a geometry and will not be deleted.", cname
                )
                continue

            dataframe = self.get_dataframe()
            if cname not in dataframe:
                if strict:
                    raise ValueError(f"The column {cname} is not present.")
                logger.info("Trying to delete %s, but it is not present.", cname)
            else:
                dataframe.drop(cname, axis=1, inplace=True)
                self.set_dataframe(dataframe)

    def get_nwells(self, well_name_column: str = "WellName"):
        """Get number of unique wells in the instance.

        Args:
            well_name_column: Name of column with well names

        Returns:
            Number of unique wells, 0 if no well or column not present.
        """
        if well_name_column not in self.get_dataframe(copy=False).columns:
            return 0
        return len(self.get_dataframe(copy=False)[well_name_column].unique())

    def get_boundary(self):
        """Get the square XYZ window (boundaries) of the instance.

        Returns:
            (xmin, xmax, ymin, ymax, zmin, zmax)

        See also:
            The class method :func:`Polygons.boundary_from_points()`

        """
        xmin = np.nanmin(self.get_dataframe(copy=False)[self.xname].values)
        xmax = np.nanmax(self.get_dataframe(copy=False)[self.xname].values)
        ymin = np.nanmin(self.get_dataframe(copy=False)[self.yname].values)
        ymax = np.nanmax(self.get_dataframe(copy=False)[self.yname].values)
        zmin = np.nanmin(self.get_dataframe(copy=False)[self.zname].values)
        zmax = np.nanmax(self.get_dataframe(copy=False)[self.zname].values)

        return (xmin, xmax, ymin, ymax, zmin, zmax)

    def mark_in_polygons(
        self,
        poly: Polygons | list[Polygons],  # noqa: F821
        name: str = "pstatus",
        inside_value: int = 1,
        outside_value: int = 0,
    ):
        """Add a column that assign values if points are inside or outside polygons.

        This is a generic function that adds a column in the points dataframe with
        a flag for values being inside or outside polygons in a Polygons instance.

        Args:
            poly: One single xtgeo Polgons instance, or a list of Polygons instances.
            name: Name of column that flags inside or outside status
            inside_value: Flag value for being inside polygons
            outside_value: Flag value for being outside polygons

        ..versionadded:: 3.2
        """
        _xyz_oper.mark_in_polygons_mpl(self, poly, name, inside_value, outside_value)

    def operation_polygons(
        self,
        poly: Polygons | list[Polygons],  # noqa: F821
        value: float,
        opname: str = "add",
        inside: bool = True,
        version: int = 1,
    ):
        """A generic function for operations restricted to inside or outside polygon(s).

        The operations are performed on the Z values, while the 'inside' or 'outside'
        of polygons are purely based on X and Y values (typically X is East and Y in
        North coordinates).

        The operations are XYZ generic i.e. done on the points that defines the
        Polygon or the point in Points, depending on the calling instance.

        Possible ``opname`` strings:

        * ``add``: add the value
        * ``sub``: substract the value
        * ``mul``: multiply the value
        * ``div``: divide the value
        * ``set``: replace current values with value
        * ``eli``: eliminate; here value is not applied

        Args:
            poly: A single Polygons instance or a list of Polygons instances.
                The list option is only allowed when version = 2
            value: Value to add, subtract etc
            opname: Name of operation... 'add', 'sub', etc
            inside: If True do operation inside polygons; else outside. Note
                that boundary is treated as 'inside'
            version: The algorithm version, see notes below. Although version 1
                is default, version 2 is recommended as it is much faster and works
                intuitively when have multiple polygons and/or using the
                `is_inside=False` (i.e. outside)

        Note:
            ``version=1``: This function works only intuitively when using one single
            polygon in the ``poly`` instance. When having several polygons the
            operation is done sequentially per polygon which may
            lead to surprising results. For instance, using "add inside"
            into two overlapping polygons, the addition will be doubled in the
            overlapping part. Similarly, using e.g. "eli, outside" will completely
            remove all points of two non-overlapping polygons are given as input.

            ``version=2``: This is a new and recommended implementation. It works
            much faster and intuitively for both inside and outside, overlapping and
            multiple polygons within a Polygons instance.

        .. versionchanged:: 3.2 Add ``version`` option which defaults to 1.
                            Also allow that ``poly`` option can be a list of Polygons
                            when version is 2.

        """
        if version == 2:
            _xyz_oper.operation_polygons_v2(
                self, poly, value, opname=opname, inside=inside
            )
        else:
            _xyz_oper.operation_polygons_v1(
                self, poly, value, opname=opname, inside=inside
            )
            if version == 0:
                # using version 0 INTERNALLY to mark that "add_inside" etc has been
                # applied, and now raise a generic deprecation warning:
                itxt = "inside" if inside else "outside"
                warn(
                    f"You are using the method '{opname}_{itxt}()'; this will "
                    "be deprecated in future versions. Consider using "
                    f"'{opname}_{itxt}_polygons()' instead which is both faster "
                    "and works intuitively when several and/or overlapping polygons",
                    DeprecationWarning,
                )

    def add_inside(self, poly, value):
        """Add a value (scalar) to points inside polygons (old behaviour).

        Args:
            poly: A xtgeo Polygons instance
            value: Value to add to Z values inside polygons.

        See notes under :meth:`operation_polygons()` and consider instead
        :meth:`add_inside polygons()`.
        """
        self.operation_polygons(poly, value, opname="add", inside=True, version=0)

    def add_inside_polygons(
        self,
        poly: Polygons | list[Polygons],
        value: float,  # noqa: F821
    ):
        """Add a value (scalar) to points inside polygons (new behaviour).

        This is an improved implementation than :meth:`add_inside()`, and is now the
        recommended method, as it is both faster and works similar for all single
        and overlapping sub-polygons within one or more Polygons instances.

        Args:
            poly: A xtgeo Polygons instance, or a list of xtgeo Polygons instances
            value: Value to add to Z values inside polygons.

        .. versionadded:: 3.2
        """
        self.operation_polygons(poly, value, opname="add", inside=True, version=2)

    def add_outside(self, poly, value):
        """Add a value (scalar) to points outside polygons (old behaviour).

        Args:
            poly: A xtgeo Polygons instance
            value: Value to add to Z values outside polygons.

        See notes under :meth:`operation_polygons()` and consider instead
        :meth:`add_outside polygons()`.
        """
        self.operation_polygons(poly, value, opname="add", inside=False, version=0)

    def add_outside_polygons(
        self,
        poly: Polygons | list[Polygons],
        value: float,  # noqa: F821
    ):
        """Add a value (scalar) to points outside polygons (new behaviour).

        This is an improved implementation than :meth:`add_outside()`, and is now the
        recommended method, as it is both faster and works similar for all single
        and overlapping sub-polygons within one or more Polygons instances.

        Args:
            poly: A xtgeo Polygons instance, or a list of xtgeo Polygons instances
            value: Value to add to Z values outside polygons.

        .. versionadded:: 3.2
        """
        self.operation_polygons(poly, value, opname="add", inside=False, version=2)

    def sub_inside(self, poly, value):
        """Subtract a value (scalar) to points inside polygons.

        Args:
            poly: A xtgeo Polygons instance
            value: Value to subtract to Z values inside polygons.

        See notes under :meth:`operation_polygons()` and consider instead
        :meth:`sub_inside polygons()`.
        """
        self.operation_polygons(poly, value, opname="sub", inside=True, version=1)

    def sub_inside_polygons(
        self,
        poly: Polygons | list[Polygons],
        value: float,  # noqa: F821
    ):
        """Subtract a value (scalar) for points inside polygons (new behaviour).

        This is an improved implementation than :meth:`sub_inside()`, and is now the
        recommended method, as it is both faster and works similar for all single
        and overlapping sub-polygons within one or more Polygons instances.

        Args:
            poly: A xtgeo Polygons instance, or a list of xtgeo Polygons instances
            value: Value to subtract to Z values inside polygons.

        .. versionadded:: 3.2
        """
        self.operation_polygons(poly, value, opname="sub", inside=True, version=2)

    def sub_outside(self, poly, value):
        """Subtract a value (scalar) to points outside polygons.

        Args:
            poly: A xtgeo Polygons instance
            value: Value to subtract to Z values outside polygons.

        See notes under :meth:`operation_polygons()` and consider instead
        :meth:`sub_outside polygons()`.
        """
        self.operation_polygons(poly, value, opname="sub", inside=False, version=0)

    def sub_outside_polygons(
        self,
        poly: Polygons | list[Polygons],
        value: float,  # noqa: F821
    ):
        """Subtract a value (scalar) for points outside polygons (new behaviour).

        This is an improved implementation than :meth:`sub_outside()`, and is now the
        recommended method, as it is both faster and works similar for all single
        and overlapping sub-polygons within one or more Polygons instances.

        Args:
            poly: A xtgeo Polygons instance, or a list of xtgeo Polygons instances
            value: Value to subtract to Z values outside polygons.

        .. versionadded:: 3.2
        """
        self.operation_polygons(poly, value, opname="sub", inside=False, version=2)

    def mul_inside(self, poly, value):
        """Multiply a value (scalar) to points inside polygons.

        Args:
            poly: A xtgeo Polygons instance
            value: Value to multiply to Z values inside polygons.

        See notes under :meth:`operation_polygons()` and consider instead
        :meth:`mul_inside polygons()`.
        """
        self.operation_polygons(poly, value, opname="mul", inside=True, version=0)

    def mul_inside_polygons(
        self,
        poly: Polygons | list[Polygons],
        value: float,  # noqa: F821
    ):
        """Multiply a value (scalar) for points inside polygons (new behaviour).

        This is an improved implementation than :meth:`mul_inside()`, and is now the
        recommended method, as it is both faster and works similar for all single
        and overlapping sub-polygons within one or more Polygons instances.

        Args:
            poly: A xtgeo Polygons instance, or a list of xtgeo Polygons instances
            value: Value to multiply to Z values inside polygons.

        .. versionadded:: 3.2
        """
        self.operation_polygons(poly, value, opname="mul", inside=True, version=2)

    def mul_outside(self, poly, value):
        """Multiply a value (scalar) to points outside polygons.

        Args:
            poly: A xtgeo Polygons instance
            value: Value to multiply to Z values outside polygons.

        See notes under :meth:`operation_polygons()` and consider instead
        :meth:`mul_outside polygons()`.
        """
        self.operation_polygons(poly, value, opname="mul", inside=False, version=0)

    def mul_outside_polygons(
        self,
        poly: Polygons | list[Polygons],
        value: float,  # noqa: F821
    ):
        """Multiply a value (scalar) for points outside polygons (new behaviour).

        This is an improved implementation than :meth:`mul_outside()`, and is now the
        recommended method, as it is both faster and works similar for all single
        and overlapping sub-polygons within one or more Polygons instances.

        Args:
            poly: A xtgeo Polygons instance, or a list of xtgeo Polygons instances
            value: Value to multiply to Z values outside polygons.

        .. versionadded:: 3.2
        """
        self.operation_polygons(poly, value, opname="mul", inside=False, version=2)

    def div_inside(self, poly, value):
        """Divide a value (scalar) to points inside polygons.

        Args:
            poly: A xtgeo Polygons instance
            value: Value to divide Z values inside polygons.

        See notes under :meth:`operation_polygons()` and consider instead
        :meth:`div_inside polygons()`.
        """
        self.operation_polygons(poly, value, opname="div", inside=True, version=0)

    def div_inside_polygons(
        self,
        poly: Polygons | list[Polygons],
        value: float,  # noqa: F821
    ):
        """Divide a value (scalar) for points inside polygons (new behaviour).

        This is an improved implementation than :meth:`div_inside()`, and is now the
        recommended method, as it is both faster and works similar for all single
        and overlapping sub-polygons within one or more Polygons instances.

        Args:
            poly: A xtgeo Polygons instance, or a list of xtgeo Polygons instances
            value: Value to divide to Z values inside polygons.

        .. versionadded:: 3.2
        """
        self.operation_polygons(poly, value, opname="div", inside=True, version=2)

    def div_outside(self, poly, value):
        """Divide a value (scalar) outside polygons (value 0.0 will give result 0).

        Args:
            poly: A xtgeo Polygons instance
            value: Value to divide Z values outside polygons.

        See notes under :meth:`operation_polygons()` and consider instead
        :meth:`div_outside polygons()`.
        """
        self.operation_polygons(poly, value, opname="div", inside=False, version=0)

    def div_outside_polygons(
        self,
        poly: Polygons | list[Polygons],
        value: float,  # noqa: F821
    ):
        """Divide a value (scalar) for points outside polygons (new behaviour).

        Note if input value is 0.0 (division on zero), the result will be 0.0.

        This is an improved implementation than :meth:`div_outside()`, and is now the
        recommended method, as it is both faster and works similar for all single
        and overlapping sub-polygons within one or more Polygons instances.

        Args:
            poly: A xtgeo Polygons instance, or a list of xtgeo Polygons instances
            value: Value to divide to Z values outside polygons.

        .. versionadded:: 3.2
        """
        self.operation_polygons(poly, value, opname="div", inside=False, version=2)

    def set_inside(self, poly, value):
        """Set a value (scalar) to points inside polygons.

        Args:
            poly: A xtgeo Polygons instance
            value: Value to set Z values inside polygons.

        See notes under :meth:`operation_polygons()` and consider instead
        :meth:`set_inside polygons()`.
        """
        self.operation_polygons(poly, value, opname="set", inside=True, version=0)

    def set_inside_polygons(
        self,
        poly: Polygons | list[Polygons],
        value: float,  # noqa: F821
    ):
        """Set a value (scalar) for points inside polygons (new behaviour).

        This is an improved implementation than :meth:`set_inside()`, and is now the
        recommended method, as it is both faster and works similar for all single
        and overlapping sub-polygons within one or more Polygons instances.

        Args:
            poly: A xtgeo Polygons instance, or a list of xtgeo Polygons instances
            value: Value to set as Z values inside polygons.

        .. versionadded:: 3.2
        """
        self.operation_polygons(poly, value, opname="set", inside=True, version=2)

    def set_outside(self, poly, value):
        """Set a value (scalar) to points outside polygons.

        Args:
            poly: A xtgeo Polygons instance
            value: Value to set Z values outside polygons.

        See notes under :meth:`operation_polygons()` and consider instead
        :meth:`set_outside polygons()`.
        """
        self.operation_polygons(poly, value, opname="set", inside=False, version=0)

    def set_outside_polygons(
        self,
        poly: Polygons | list[Polygons],
        value: float,  # noqa: F821
    ):
        """Set a value (scalar) for points outside polygons (new behaviour).

        This is an improved implementation than :meth:`set_outside()`, and is now the
        recommended method, as it is both faster and works similar for all single
        and overlapping sub-polygons within one or more Polygons instances.

        Args:
            poly: A xtgeo Polygons instance, or a list of xtgeo Polygons instances
            value: Value to set as Z values inside polygons.

        .. versionadded:: 3.2
        """
        self.operation_polygons(poly, value, opname="set", inside=False, version=2)

    def eli_inside(self, poly):
        """Eliminate current points inside polygons (old implentation).

        Args:
            poly: A xtgeo Polygons instance

        See notes under :meth:`operation_polygons()` and consider instead
        :meth:`eli_inside polygons()`.
        """
        self.operation_polygons(poly, 0, opname="eli", inside=True, version=0)

    def eli_inside_polygons(self, poly: Polygons | list[Polygons]):  # noqa: F821
        """Remove points inside polygons.

        This is an improved implementation than :meth:`eli_inside()`, and is now the
        recommended method, as it is both faster and works similar for all single
        and overlapping sub-polygons within one or more Polygons instances.

        Args:
            poly: A xtgeo Polygons instance, or a list of xtgeo Polygons instances

        .. versionadded:: 3.2
        """
        self.operation_polygons(poly, 0, opname="eli", inside=True, version=2)

    def eli_outside(self, poly):
        """Eliminate current points outside polygons (old implentation).

        Args:
            poly: A xtgeo Polygons instance

        See notes under :meth:`operation_polygons()` and consider instead
        :meth:`eli_outside polygons()`.
        """
        self.operation_polygons(poly, 0, opname="eli", inside=False, version=0)

    def eli_outside_polygons(self, poly: Polygons | list[Polygons]):  # noqa: F821
        """Remove points outside polygons.

        This is an improved implementation than :meth:`eli_outside()`, and is now the
        recommended method, as it is both faster and works similar for all single
        and overlapping sub-polygons within one or more Polygons instances.

        Args:
            poly: A xtgeo Polygons instance, or a list of xtgeo Polygons instances

        .. versionadded:: 3.2
        """
        self.operation_polygons(poly, 0, opname="eli", inside=False, version=2)
