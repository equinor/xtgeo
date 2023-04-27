"""Test some critical numpy operations.

The background for this is that several tests failed after upgrade to numpy 1.24,
wrt to order="F" and masked arrays. The tests here will highlight changes in behaviour.

Cf. RegularSurface().get_val1d() method.

See numpy bug reported here: https://github.com/numpy/numpy/issues/22912
"""
import numpy as np
from packaging.version import parse as versionparse


def test_order_change_variant1():
    """Test numpy behaviour when combing order changes and masks."""

    arr = np.array([[1.0, 2.0, 3, 4], [5, 6, 7, 8]])
    arr = np.ma.masked_where(arr == 7, arr)

    val = np.ma.filled(arr, fill_value=np.nan)
    val = np.array(val, order="F")

    val = np.ma.masked_invalid(val)

    val1d = val.ravel(order="F")
    assert str(val1d) == "[1.0 5.0 2.0 6.0 3.0 -- 4.0 8.0]"

    # if versionparse(np.__version__) < versionparse("1.24"):
    #     assert str(val1d) == "[1.0 5.0 2.0 6.0 3.0 -- 4.0 8.0]"
    # else:
    #     # this may actually be a bug;
    #     assert str(val1d) == "[1.0 5.0 2.0 6.0 3.0 nan -- 8.0]"


def test_order_change_variant2():
    """Test numpy behaviour when combing order changes and masks, alternative."""

    arr = np.array([[1.0, 2.0, 3, 4], [5, 6, 7, 8]])
    arr = np.ma.masked_where(arr == 7, arr)

    arrdata = np.array(arr.data, order="F")
    arrmask = np.array(arr.mask, order="F")

    val = np.ma.array(arrdata, mask=arrmask)

    val1d = val.ravel(order="F")

    assert str(val1d) == "[1.0 5.0 2.0 6.0 3.0 -- 4.0 8.0]"
