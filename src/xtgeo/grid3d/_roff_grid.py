from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import Optional

import numpy as np
import roffio

import xtgeo.cxtgeo._cxtgeo as _cxtgeo


@dataclass
class RoffGrid:
    """
    A RoffGrid contains a grid as represented in a roff file.  The
    grid layout is corner point geometry with increasing x,y, and z
    values. The coordinate values are stored in an local coordinate
    system, defined by the x,y,z offset and scale values which
    converts to utm/tvd values by the usual formula.

    .. code-block:: python

        x_utm = (x_local + xoffset) * xscale

    The corner lines are stored in the corner_lines array and define
    the edges of the cell. They go from the bottom layer to the
    top layer in a straight line such that the north-west  to south-west
    line of cell at index i,j,k is stored at

    .. code-block:: python

        node_index = i * (ny + 1)  + j
        bottom_x = corner_lines[node_index]
        bottom_y = corner_lines[node_index+1]
        bottom_z = corner_lines[node_index+2]
        top_x = corner_lines[node_index+3]
        top_y = corner_lines[node_index+4]
        top_z = corner_lines[node_index+5]

    The z values of the corner of a cell is stored in zvals. For any
    given corner there are 8 adjacent cells. These are usually given
    the directions

    below_nw, below_ne, below_sw, below_se,
    above_nw, above_ne, above_sw and above_se.

    where below means lower k value, above means higher k value,
    south means lower j value, north means high j value, and
    west means lower i value, east means higher i value.

    All of these cells might have different z values along the line, meaning
    there is a 'split' in the layers. How many different values there are is
    dependent on the split_enz values which for any given corner can be 1,2,4
    or 8.  1 means all cells have the same z value, 2 means the below cells
    have different z value than the above cells in that corner. 4 means split
    in north/south and east/west directions.  8 means split in all directions.

    .. code-block:: python

        node_index = i * (ny + 1) * (nz + 1) + j * (nz + 1) + k
        split_number = roff_grid.split_enz[node_index]

    The z-values are listed in c-order of i,j,k coordines, with the number
    of z-values depends on the split number. for instance, if split_enz[0]=2
    Then zvals[0] is the below z value for the corner at i=0,j=0,k=0 and
    zvals[1] is the above z value for the same corner. zvals[2] is then
    a z value for the i=0,j=0,k=1 corner. The z values are listed in
    the order of below before above, south before north, and west before east.

    The grid can optionally be divided into subgrids by layers. This is
    defined by the subgrids array which for each subgrid contains the
    number of layers to each subgrid in decreasing k-value.

    Args:
        nx (int): The number of cells in x direction.
        ny (int): The number of cells in y direction.
        nz (int): The number of cells in z direction.
        subgrids (numpy.array of numpy.int32): The number of layers to each
            subgrid by decreasing k value.
        corner_lines (numpy.array of numpy.float32): The coordinates of the top
            and bottom node for each corner line.
        split_enz (bytes): The split number for any given node.
        zvals (numpy.array of numpy.float32): Z values for each cell.
        active (numpy.array of bool): Whether a given cell is active.
        xoffset (float): translation value for utm coordinates.
        yoffset (float): translation value for utm coordinates.
        zoffset (float): translation value for tvd coordinates.
        xscale (float): scaling value for utm coordinates.
        yscale (float): scaling value for utm coordinates.
        zscale (float): scaling value for tvd coordinates.

    """

    nx: int
    ny: int
    nz: int
    corner_lines: np.ndarray
    zvals: np.ndarray

    split_enz: Optional[bytes] = None
    active: Optional[np.ndarray] = None
    subgrids: Optional[np.ndarray] = None
    xoffset: float = 0.0
    yoffset: float = 0.0
    zoffset: float = 0.0
    xscale: float = 1.0
    yscale: float = 1.0
    zscale: float = -1.0

    def __post_init__(self):
        if self.active is None:
            self.active = np.ones(self.nx * self.ny * self.nz, dtype=np.bool_)
        if self.split_enz is None:
            self.split_enz = np.ones(
                self.nx * self.ny * self.nz, dtype=np.uint8
            ).tobytes()

    def __eq__(self, other):
        if not isinstance(other, RoffGrid):
            return False
        return (
            self.nx == other.nx
            and self.ny == other.ny
            and self.nz == other.nz
            and self.xoffset == other.xoffset
            and self.yoffset == other.yoffset
            and self.zoffset == other.zoffset
            and self.xscale == other.xscale
            and self.yscale == other.yscale
            and self.zscale == other.zscale
            and np.array_equal(self.subgrids, other.subgrids)
            and np.array_equal(self.split_enz, other.split_enz)
            and np.array_equal(self.zvals, other.zvals)
            and np.array_equal(self.corner_lines, other.corner_lines)
            and np.array_equal(self.active, other.active)
        )

    @property
    def num_nodes(self):
        """
        The number of nodes in the grid, ie. the size of split_enz.
        """
        return (self.nx + 1) * (self.ny + 1) * (self.nz + 1)

    def _create_lookup(self):
        if not hasattr(self, "_lookup"):
            n = self.num_nodes
            self._lookup = np.zeros(n + 1, dtype=np.int32)
            for i in range(n):
                if self.split_enz is not None:
                    self._lookup[i + 1] = self.split_enz[i] + self._lookup[i]
                else:
                    self._lookup[i + 1] = 1 + self._lookup[i]

    def z_value(self, node):
        """
        Gives the 8 z values for any given node for
        adjacent cells in the order:

        * below_sw
        * below_se
        * below_nw
        * below_ne
        * above_sw
        * above_se
        * above_nw
        * above_ne

        Args:
            node (tuple of i,j,k index): The index of the node.

        Raises:
            ValueError if the split array contains unsupported
            split types. (must be 1,2,4 or 8)

        Returns:
            numpy array of float32 with z values for adjacent
            corners in the order given above.
        """
        i, j, k = node
        self._create_lookup()

        node_number = i * (self.ny + 1) * (self.nz + 1) + j * (self.nz + 1) + k
        pos = self._lookup[node_number]
        split = self._lookup[node_number + 1] - self._lookup[node_number]

        if split == 1:
            return np.array([self.zvals[pos]] * 8)
        elif split == 2:
            return np.array([self.zvals[pos]] * 4 + [self.zvals[pos + 1]] * 4)
        elif split == 4:
            return np.array(
                [
                    self.zvals[pos],
                    self.zvals[pos + 1],
                    self.zvals[pos + 2],
                    self.zvals[pos + 3],
                ]
                * 2
            )
        elif split == 8:
            return np.array(
                [
                    self.zvals[pos],
                    self.zvals[pos + 1],
                    self.zvals[pos + 2],
                    self.zvals[pos + 3],
                    self.zvals[pos + 4],
                    self.zvals[pos + 5],
                    self.zvals[pos + 6],
                    self.zvals[pos + 7],
                ]
            )
        else:
            raise ValueError("Only split types 1, 2, 4 and 8 are supported!")

    def xtgeo_coord(self):
        """
        Returns:
            The coordinates of nodes in the format of xtgeo.Grid.coordsv
        """
        offset = (self.xoffset, self.yoffset, self.zoffset)
        scale = (self.xscale, self.yscale, self.zscale)
        coordsv = self.corner_lines.reshape((self.nx + 1, self.ny + 1, 2, 3))
        coordsv = np.flip(coordsv, -2)
        coordsv = coordsv + offset
        coordsv *= scale
        return coordsv.reshape((self.nx + 1, self.ny + 1, 6)).astype(np.float64)

    def xtgeo_actnum(self):
        """
        Returns:
            The active field in the format of xtgeo.Grid.actnumsv
        """
        actnum = self.active.reshape((self.nx, self.ny, self.nz))
        actnum = np.flip(actnum, -1)
        return actnum.astype(np.int32)

    def xtgeo_zcorn(self):
        """
        Returns:
            The z values for nodes in the format of xtgeo.Grid.zcornsv
        """
        zcornsv = np.zeros(
            (self.nx + 1) * (self.ny + 1) * (self.nz + 1) * 4, dtype=np.float32
        )
        retval = _cxtgeo.grd3d_roff2xtgeo_splitenz(
            int(self.nz + 1),
            float(self.zoffset),
            float(self.zscale),
            self.split_enz,
            self.zvals,
            zcornsv,
        )
        if retval == 0:
            return zcornsv.reshape((self.nx + 1, self.ny + 1, self.nz + 1, 4))
        elif retval == -1:
            raise ValueError("Unsupported split type in split_enz")
        elif retval == -2:
            expected_size = (self.nx + 1) * (self.ny + 1) * (self.nz + 1)
            raise ValueError(
                "Incorrect size of splitenz,"
                f" expected {expected_size} got {len(self.split_enz)}"
            )
        elif retval == -3:
            expected_size = sum(self.split_enz)
            raise ValueError(
                "Incorrect size of zdata,"
                f" expected {expected_size} got {len(self.zvals)}"
            )
        elif retval == -4:
            raise ValueError(
                "Incorrect size of zcorn,"
                f" found {zcornsv.shape} should be multiple of {4 * self.nz}"
            )
        else:
            raise ValueError(f"Unknown error {retval} occurred")

    def xtgeo_subgrids(self):
        """
        Returns:
            The z values for nodes in the format of xtgeo.Grid.zcornsv
        """
        if self.subgrids is None:
            return None
        result = OrderedDict()
        next_ind = 1
        for i, current in enumerate(self.subgrids):
            result[f"subgrid_{i}"] = range(next_ind, current + next_ind)
            next_ind += current
        return result

    @staticmethod
    def _from_xtgeo_subgrids(xtgeo_subgrids):
        """
        Args:
            A xtgeo.Grid._subgrids dictionary
        Returns:
            The corresponding RoffGrid.subgrids
        """
        if xtgeo_subgrids is None:
            return None
        subgrids = []
        for key, value in xtgeo_subgrids.items():
            if isinstance(value, range):
                subgrids.append(value.stop - value.start)
            elif value != list(range(value[0], value[-1] + 1)):
                raise ValueError(
                    "Cannot convert non-consecutive subgrids to roff format."
                )
            else:
                subgrids.append(value[-1] + 1 - value[0])
        return np.array(subgrids, dtype=np.int32)

    @staticmethod
    def from_xtgeo_grid(xtgeo_grid):
        """
        Args:
            An xtgeo.Grid
        Returns:
            That grid geometry converted to a RoffGrid.
        """
        xtgeo_grid._xtgformat2()
        nx, ny, nz = xtgeo_grid.dimensions
        active = xtgeo_grid._actnumsv.reshape((nx, ny, nz))
        active = np.flip(active, -1).ravel().astype(np.bool_)
        corner_lines = xtgeo_grid._coordsv.reshape((nx + 1, ny + 1, 2, 3)) * np.array(
            [1, 1, -1]
        )
        corner_lines = np.flip(corner_lines, -2).ravel().astype(np.float32)
        zvals = xtgeo_grid._zcornsv.reshape((nx + 1, ny + 1, nz + 1, 4))
        zvals = np.flip(zvals, 2).ravel().view(np.float32) * -1
        split_enz = np.repeat(b"\x04", (nx + 1) * (ny + 1) * (nz + 1)).tobytes()
        subgrids = RoffGrid._from_xtgeo_subgrids(xtgeo_grid._subgrids)

        return RoffGrid(nx, ny, nz, corner_lines, zvals, split_enz, active, subgrids)

    def to_file(self, filelike, roff_format=roffio.Format.BINARY):
        """
        Writes the RoffGrid to a roff file
        Args:
            filelike (str or byte stream): The file to write to.
        """
        data = {
            "filedata": {"filetype": "grid"},
            "dimensions": {"nX": self.nx, "nY": self.ny, "nZ": self.nz},
            "translate": {
                "xoffset": np.float32(self.xoffset),
                "yoffset": np.float32(self.yoffset),
                "zoffset": np.float32(self.zoffset),
            },
            "scale": {
                "xscale": np.float32(self.xscale),
                "yscale": np.float32(self.yscale),
                "zscale": np.float32(self.zscale),
            },
            "cornerLines": {"data": self.corner_lines},
            "zvalues": {"data": self.zvals},
            "active": {"data": self.active},
        }
        if self.subgrids is not None:
            data["subgrids"] = {"nLayers": self.subgrids}
        if self.split_enz is not None:
            data["zvalues"]["splitEnz"] = self.split_enz
        roffio.write(filelike, data, roff_format=roff_format)

    @staticmethod
    def from_file(filelike):
        """
        Read a RoffGrid from a roff file
        Args:
            filelike (str or byte stream): The file to read from.
        Returns:
            The RoffGrid in the roff file.
        """
        translate_kws = {
            "dimensions": {"nX": "nx", "nY": "ny", "nZ": "nz"},
            "translate": {
                "xoffset": "xoffset",
                "yoffset": "yoffset",
                "zoffset": "zoffset",
            },
            "scale": {
                "xscale": "xscale",
                "yscale": "yscale",
                "zscale": "zscale",
            },
            "cornerLines": {"data": "corner_lines"},
            "zvalues": {"splitEnz": "split_enz", "data": "zvals"},
            "active": {"data": "active"},
            "subgrids": {"nLayers": "subgrids"},
        }
        optional_keywords = defaultdict(
            list,
            {
                "translate": ["xoffset", "yoffset", "zoffset"],
                "scale": ["xscale", "yscale", "zscale"],
                "subgrids": ["nLayers"],
                "active": ["data"],
            },
        )
        # The found dictionary contains all tags/tagkeys which we are
        # interested in with None as the initial value. We go through the
        # tag/tagkeys in the file and replace as they are found.
        found = {
            tag_name: {key_name: None for key_name in tag_keys.keys()}
            for tag_name, tag_keys in translate_kws.items()
        }
        found["filedata"] = {"filetype": None}
        with roffio.lazy_read(filelike) as tag_generator:
            for tag, keys in tag_generator:
                if tag in found:
                    # We do not destruct keys yet as this fetches the value too early.
                    # key is not a tuple but an object that fetches the value when
                    # __getitem__ is called.
                    for key in keys:
                        if key[0] in found[tag]:
                            if found[tag][key[0]] is not None:
                                raise ValueError(
                                    f"Multiple tag, tagkey pair {tag}, {key[0]}"
                                    " in {filelike}"
                                )
                            found[tag][key[0]] = key[1]

        for tag_name, keys in found.items():
            for key_name, value in keys.items():
                if value is None and key_name not in optional_keywords[tag_name]:
                    raise ValueError(
                        f"Missing non-optional keyword {tag_name}:{key_name}"
                    )

        filetype = found["filedata"]["filetype"]
        if filetype != "grid":
            raise ValueError(
                f"File {filelike} did not have filetype set to grid, found {filetype}"
            )

        return RoffGrid(
            **{
                translated: found[tag][key]
                for tag, tag_keys in translate_kws.items()
                for key, translated in tag_keys.items()
                if found[tag][key] is not None
            }
        )
