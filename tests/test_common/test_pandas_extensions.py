from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from hypothesis import assume, given, strategies as st
from xtgeo.common.pandas_extensions import LazyArray


@dataclass
class CallableCounter:
    count: int = 0

    def __call__(self) -> list[int]:
        self.count += 1
        return [self.count]


def binery_operators() -> object:
    return st.sampled_from(
        (
            operator.add,
            operator.mul,
            operator.pow,
            operator.sub,
            operator.truediv,
        )
    )


def unary_operators() -> object:
    return st.sampled_from(
        (
            np.abs,
            np.square,
        )
    )


@given(
    op=binery_operators(),
    left=st.floats(
        allow_infinity=False,
        allow_nan=False,
        min_value=-1000,
        max_value=1000,
    ),
    right=st.floats(
        allow_infinity=False,
        allow_nan=False,
        min_value=-1000,
        max_value=1000,
    ),
    length=st.integers(
        min_value=1,
        max_value=1_000,
    ),
)
def test_lazy_array_binery_operators(
    op: Callable,
    left: float,
    right: float,
    length: int,
) -> None:
    # Avid div. by zero and blowups due to div. by close to zero.
    if op is operator.truediv:
        assume(abs(right) > 0.01)

    # Avid blow up due to power of super large.
    if op is operator.pow:
        assume(-2 <= right <= 2)
        assume(abs(left) > 0.01)

    expected = np.repeat(op(left, right), length)

    df = op(
        pd.DataFrame({"x": LazyArray(lambda: [left] * length, length=length)}),
        right,
    )
    assert np.allclose(df.x.to_numpy(), expected)


@given(
    op=unary_operators(),
    value=st.floats(
        allow_infinity=False,
        allow_nan=False,
        min_value=-1e10,
        max_value=1e10,
    ),
    length=st.integers(
        min_value=1,
        max_value=100,
    ),
)
def test_lazy_array_unary_operators(
    op: Callable,
    value: float,
    length: int,
) -> None:
    expected = np.repeat(op(value), length)
    df = op(pd.DataFrame({"x": LazyArray(lambda: [value] * length, length=length)}))
    assert np.allclose(df.x.to_numpy(), expected)


@given(calls=st.integers(min_value=0, max_value=1_000))
def test_callabole_couter(calls: int) -> None:
    cc = CallableCounter()
    assert cc.count == 0
    for i in range(1, calls + 1):
        assert cc() == [i]
    assert cc.count == calls


@given(calls=st.integers(min_value=1, max_value=1_000))
def test_lazy_array_eval_once(calls: int) -> None:
    cc = CallableCounter()
    df = pd.DataFrame({"x": LazyArray(cc, length=1)})
    x = sum(df.x.to_numpy() for _ in range(calls))  # trigger evel of lazy array
    assert cc.count == 1
    assert x == calls


def test_lazy_array_eval_accses() -> None:
    cc = CallableCounter()
    df = pd.DataFrame({"x": LazyArray(cc, length=1)})
    assert cc.count == 0
    df.x.to_numpy()  # Trigger evel of lazy array
    assert cc.count == 1
