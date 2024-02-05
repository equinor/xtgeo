from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionDtype, register_extension_dtype
from pandas.core.arrays import ExtensionArray, ExtensionScalarOpsMixin

if TYPE_CHECKING:
    from numpy.typing import NDArray

UNSET = object()


@register_extension_dtype
class LazyDtype(ExtensionDtype):
    """
    A custom data type for pandas representing a lazy-loading array.

    This class inherits from pandas' ExtensionDtype and is used to define a
    custom data type that lazily loads data.

    Methods:
        type: Returns the type associated with this Dtype.
        name: Returns a string representation of the data type.
        construct_array_type: Returns the array type associated with this Dtype.

    See also:
        pandas.api.extensions.ExtensionDtype: The base class for custom
        dtypes in pandas.
    """

    name = "LazyDtype"

    @property
    def type(self):
        """
        Return the type associated with this Dtype.

        Returns:
            The LazyDtype class itself.
        """
        return LazyDtype

    @classmethod
    def construct_array_type(cls):
        """
        Return the array type associated with this Dtype.

        Returns:
            The LazyArray class.
        """
        return LazyArray


class LazyArray(ExtensionScalarOpsMixin, ExtensionArray):
    """
    An array structure that lazily loads data when accessed.

    This class inherits from pandas' ExtensionArray and provides a way to
    lazily load data into a pandas Series or DataFrame.

    Attributes:
        fn (Callable[[], Any]): A function that loads data.
        length (int): The length of the array.
        data (Any): The data stored in the array, initially unset.

    Methods:
        __getitem__: Retrieves an item from the array.
        isna: Determines if the array contains missing values.
        copy: Creates a copy of the array.
        nbytes: Returns the total bytes consumed by the elements of the array.
        take: Return elements at given positions.
        _from_sequence: Class method for creating an array from a sequence.
        _from_factorized: Class method for creating an array from factorized values.
        _concat_same_type: Class method for concatenating arrays of the same type.

    See also:
        pandas.core.arrays.ExtensionArray: Base class for custom array types in pandas.
    """

    def __init__(
        self,
        fn: Callable[[], Any],
        length: int,
        data=UNSET,
    ):
        """
        Initialize a LazyArray.

        Args:
            fn (Callable[[], Any]): A function that when called returns
            a sequence of data.
            length (int): The length of the array.
            data (Any, optional): The initial data for the array. Defaults to UNSET.
        """
        self.fn = fn
        self.length = length
        self.data = data

    def _data_loader(self) -> np.ndarray:
        if self.data is UNSET:
            self.data = self.fn()
        return self.data

    def __getitem__(self, item):
        return self._data_loader()[item]

    def __getattr__(self, item):
        return getattr(self._data_loader(), item)

    def isna(self) -> bool:
        return self._data_loader().isna()

    def __len__(self) -> int:
        return self.length

    def copy(self) -> LazyArray:
        return LazyArray(fn=self.fn, length=self.length, data=self.data)

    @property
    def dtype(self) -> LazyDtype:
        return LazyDtype()

    def nbytes(self) -> int:
        return self._data_loader().nbytes()

    def take(self, *args, **kw) -> pd.Series:
        return pd.Series(self._data_loader()).take(*args, **kw)

    @classmethod
    def _from_sequence(cls, scalars, *_, **__):
        return scalars

    @classmethod
    def _from_factorized(cls, values, *_, **__):
        return values

    @classmethod
    def _concat_same_type(cls, to_concat: Sequence[LazyArray]) -> NDArray:
        return np.concatenate([x.to_numpy() for x in to_concat])


LazyArray._add_arithmetic_ops()
LazyArray._add_comparison_ops()
