from typing import Protocol

from xtgeo.common.types import FileLike
from xtgeo.io.protocols.grid_data_io_protocol import GridDataIOProtocol


class GridFileIOProtocol(Protocol):
    """Protocol for file IO classes."""

    @staticmethod
    def from_file(file: FileLike, encoding: str) -> GridDataIOProtocol:
        """Read data from file-like object."""

    def to_file(data: GridDataIOProtocol, file: FileLike, encoding: str) -> None:
        """Write data to file-like object."""
