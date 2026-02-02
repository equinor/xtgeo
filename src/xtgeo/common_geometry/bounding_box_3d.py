from dataclasses import dataclass
from typing import Iterable

from typing_extensions import Self


@dataclass(frozen=True, init=False)
class BoundingBox3D:
    """Class representing a 3D bounding box defined by its minimum and maximum
    coordinates along each axis.

    Args:
        values: An iterable of exactly 6 floats:
            (min_x, max_x, min_y, max_y, min_z, max_z).
    """

    _values: tuple[float, float, float, float, float, float]

    def __init__(self, values: Iterable[float]) -> None:
        if values is None:
            raise ValueError("values cannot be None")

        vals = tuple(values)

        if len(vals) != 6:
            raise ValueError(
                "Expected exactly 6 values (min_x, max_x, min_y, max_y, "
                f"min_z, max_z), got {len(vals)}"
            )

        min_x, max_x, min_y, max_y, min_z, max_z = vals

        if min_x is None or max_x is None:
            raise ValueError("min_x and max_x cannot be None")
        if min_y is None or max_y is None:
            raise ValueError("min_y and max_y cannot be None")
        if min_z is None or max_z is None:
            raise ValueError("min_z and max_z cannot be None")

        if min_x > max_x:
            raise ValueError("min_x must be smaller than or equal to max_x")
        if min_y > max_y:
            raise ValueError("min_y must be smaller than or equal to max_y")
        if min_z > max_z:
            raise ValueError("min_z must be smaller than or equal to max_z")

        object.__setattr__(self, "_values", vals)

    def __eq__(self, other: object) -> bool:
        """
        Check equality with another BoundingBox3D.
        """

        if not isinstance(other, BoundingBox3D):
            return NotImplemented

        if self is other:
            return True

        return self._values == other._values

    @property
    def min_x(self) -> float:
        return self._values[0]

    @property
    def max_x(self) -> float:
        return self._values[1]

    @property
    def min_y(self) -> float:
        return self._values[2]

    @property
    def max_y(self) -> float:
        return self._values[3]

    @property
    def min_z(self) -> float:
        return self._values[4]

    @property
    def max_z(self) -> float:
        return self._values[5]

    @property
    def length_x(self) -> float:
        """Length of the bounding box along the X axis."""
        return self.max_x - self.min_x

    @property
    def length_y(self) -> float:
        """Length of the bounding box along the Y axis."""
        return self.max_y - self.min_y

    @property
    def length_z(self) -> float:
        """Length of the bounding box along the Z axis."""
        return self.max_z - self.min_z

    @property
    def volume(self) -> float:
        """Calculate the volume of the bounding box."""
        return self.length_x * self.length_y * self.length_z

    @property
    def center(self) -> Iterable[float]:
        """Center point of the bounding box."""
        return (
            (self.min_x + self.max_x) * 0.5,
            (self.min_y + self.max_y) * 0.5,
            (self.min_z + self.max_z) * 0.5,
        )

    def contains_point(self, point: Iterable[float]) -> bool:
        """Return True if a point is inside the bounding box
        (includes the boundary)."""
        x, y, z = point
        return (
            self.min_x <= x <= self.max_x
            and self.min_y <= y <= self.max_y
            and self.min_z <= z <= self.max_z
        )

    def intersects(self, other: Self) -> bool:
        """Return True if this bounding box intersects another."""

        if not isinstance(other, BoundingBox3D):
            raise ValueError("'other' must be a BoundingBox3D")

        return not (
            self.max_x < other.min_x
            or self.min_x > other.max_x
            or self.max_y < other.min_y
            or self.min_y > other.max_y
            or self.max_z < other.min_z
            or self.min_z > other.max_z
        )

    def intersection(self, other: Self) -> Self | None:
        """Return the intersection bounding box of this and another bounding box.

        If there is no intersection, return None.
        """

        if not isinstance(other, BoundingBox3D):
            raise ValueError("'other' must be a BoundingBox3D")

        if self is other:
            return self

        if not self.intersects(other):
            return None

        return type(self)(
            (
                max(self.min_x, other.min_x),
                min(self.max_x, other.max_x),
                max(self.min_y, other.min_y),
                min(self.max_y, other.max_y),
                max(self.min_z, other.min_z),
                min(self.max_z, other.max_z),
            )
        )

    def is_contained_in(self, other: Self) -> bool:
        """Return True if this bounding box is fully contained in another."""

        if not isinstance(other, BoundingBox3D):
            raise ValueError("'other' must be a BoundingBox3D")

        if self is other:
            return True

        return (
            self.min_x >= other.min_x
            and self.max_x <= other.max_x
            and self.min_y >= other.min_y
            and self.max_y <= other.max_y
            and self.min_z >= other.min_z
            and self.max_z <= other.max_z
        )

    def contains(self, other: Self) -> bool:
        """Return True if this bounding box fully contains another."""

        if not isinstance(other, BoundingBox3D):
            raise ValueError("'other' must be a BoundingBox3D")

        if self is other:
            return True

        return (
            self.min_x <= other.min_x
            and self.max_x >= other.max_x
            and self.min_y <= other.min_y
            and self.max_y >= other.max_y
            and self.min_z <= other.min_z
            and self.max_z >= other.max_z
        )

    def union(self, other: Self) -> Self:
        """Return the smallest bounding box containing both."""

        if not isinstance(other, BoundingBox3D):
            raise ValueError("'other' must be a BoundingBox3D")

        if self is other:
            return self

        return type(self)(
            (
                min(self.min_x, other.min_x),
                max(self.max_x, other.max_x),
                min(self.min_y, other.min_y),
                max(self.max_y, other.max_y),
                min(self.min_z, other.min_z),
                max(self.max_z, other.max_z),
            )
        )
