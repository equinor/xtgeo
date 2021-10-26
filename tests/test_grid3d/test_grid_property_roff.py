import io

import hypothesis.strategies as st
import numpy as np
import pytest
import roffio
from hypothesis import HealthCheck, given, settings
from hypothesis.extra.numpy import arrays

from xtgeo.grid3d import GridProperty
from xtgeo.grid3d._roff_parameter import RoffParameter

from .grid_generator import dimensions

finites = st.floats(
    min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False, width=32
)

names = st.text(
    min_size=1, max_size=32, alphabet=st.characters(min_codepoint=60, max_codepoint=123)
)
codes = st.integers(min_value=0, max_value=100)


@st.composite
def roff_parameters(draw, dim=dimensions):
    dims = draw(dim)

    name = draw(names)
    is_discrete = draw(st.booleans())
    if is_discrete:
        num_codes = draw(st.integers(min_value=2, max_value=10))
        code_names = draw(
            st.lists(min_size=num_codes, max_size=num_codes, elements=names)
        )
        code_values = np.array(
            draw(
                st.lists(
                    unique=True, elements=codes, min_size=num_codes, max_size=num_codes
                )
            ),
            dtype=np.int32,
        )
        is_byte_values = draw(st.booleans())
        if is_byte_values:
            values = draw(
                arrays(
                    shape=dims[0] * dims[1] * dims[2],
                    dtype=np.uint8,
                    elements=st.sampled_from(code_values),
                )
            ).tobytes()
        else:
            values = draw(
                arrays(
                    shape=dims[0] * dims[1] * dims[2],
                    dtype=np.int32,
                    elements=st.sampled_from(code_values),
                )
            )
        return RoffParameter(*dims, name, values, code_names, code_values)

    else:
        values = draw(
            arrays(
                shape=dims[0] * dims[1] * dims[2], dtype=np.float32, elements=finites
            )
        )
        return RoffParameter(*dims, name, values)


@given(roff_parameters())
def test_roff_property_read_write(rpara):
    buff = io.BytesIO()
    rpara.to_file(buff)

    buff.seek(0)
    assert RoffParameter.from_file(buff, rpara.name) == rpara


@st.composite
def grid_properties(draw):
    dims = draw(dimensions)
    name = draw(names)
    is_discrete = draw(st.booleans())
    if is_discrete:
        num_codes = draw(st.integers(min_value=2, max_value=10))
        code_names = draw(
            st.lists(min_size=num_codes, max_size=num_codes, elements=names)
        )
        code_values = np.array(
            draw(
                st.lists(
                    unique=True, elements=codes, min_size=num_codes, max_size=num_codes
                )
            ),
            dtype=np.int32,
        )
        values = draw(
            arrays(
                shape=dims,
                dtype=np.int32,
                elements=st.sampled_from(code_values),
            )
        )
        gp = GridProperty(
            None,
            "guess",
            *dims,
            name,
            discrete=is_discrete,
            codes=dict(zip(code_values, code_names)),
            values=values,
        )
        gp.dtype = np.int32
        return gp
    else:
        values = draw(arrays(shape=dims, dtype=np.float64, elements=finites))
        return GridProperty(
            None, "guess", *dims, name, discrete=is_discrete, values=values
        )


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(grid_properties())
def test_roff_prop_read_xtgeo(tmp_path, xtgeo_property):
    filepath = tmp_path / "property.roff"

    xtgeo_property.to_file(filepath, name=xtgeo_property.name)

    xtgeo_property2 = GridProperty().from_file(
        filepath,
        name=xtgeo_property.name,
    )

    assert xtgeo_property.ncol == xtgeo_property2.ncol
    assert xtgeo_property.nrow == xtgeo_property2.nrow
    assert xtgeo_property.nlay == xtgeo_property2.nlay
    assert xtgeo_property.dtype == xtgeo_property2.dtype
    assert np.all(xtgeo_property.values3d == xtgeo_property2.values3d)
    assert xtgeo_property.codes == xtgeo_property2.codes


@given(roff_parameters())
def test_eq_reflexivity(roff_param):
    assert roff_param == roff_param


@given(roff_parameters(), roff_parameters())
def test_eq_symmetry(roff_param1, roff_param2):
    if roff_param1 == roff_param2:
        assert roff_param2 == roff_param1


@given(roff_parameters(), roff_parameters(), roff_parameters())
def test_eq_transitivity(roff_param1, roff_param2, roff_param3):
    if roff_param1 == roff_param2 and roff_param2 == roff_param3:
        assert roff_param1 == roff_param3


@given(roff_parameters())
def test_eq_typing(roff_param):
    assert roff_param != ""


@pytest.mark.parametrize(
    "param, expected_undef",
    [
        (RoffParameter(1, 1, 1, "", np.array([1], dtype=np.int32)), -999),
        (RoffParameter(1, 1, 1, "", np.array([1.0])), -999.0),
        (RoffParameter(1, 1, 1, "", b"\x01"), 255),
    ],
)
def test_undefined_values(param, expected_undef):
    assert param.undefined_value == expected_undef


def test_is_discrete():
    assert RoffParameter(1, 1, 1, "", np.array([1])).is_discrete
    assert not RoffParameter(1, 1, 1, "", np.array([1.0])).is_discrete
    assert RoffParameter(1, 1, 1, "", b"\x01").is_discrete


@pytest.mark.parametrize(
    "param, expected_codes",
    [
        (RoffParameter(1, 1, 1, "", np.array([1], dtype=np.int32)), {}),
        (RoffParameter(1, 1, 1, "", np.array([1.0])), {}),
        (
            RoffParameter(
                1,
                1,
                1,
                "",
                np.array([1]),
                code_names=["a", "b"],
                code_values=np.array([1, 2]),
            ),
            {1: "a", 2: "b"},
        ),
    ],
)
def test_xtgeo_codes(param, expected_codes):
    assert param.xtgeo_codes() == expected_codes


def test_to_file(tmp_path):
    roff_param = RoffParameter(1, 1, 2, "", b"\x01\xFF")
    roff_param.to_file(tmp_path / "param.roff")
    vals = roffio.read(tmp_path / "param.roff")
    assert vals["parameter"] == {"name": "", "data": b"\x01\xff"}
    assert vals["dimensions"] == {"nX": 1, "nY": 1, "nZ": 2}


def test_to_file_codes():
    buff = io.BytesIO()
    roff_param = RoffParameter(
        1,
        1,
        2,
        "a",
        b"\x01\xFF",
        code_names=["a", "b"],
        code_values=np.array([1, 2], dtype=np.int32),
    )
    roff_param.to_file(buff)
    buff.seek(0)
    vals = roffio.read(buff)
    assert np.array_equal(vals["parameter"]["codeNames"], np.array(["a", "b"]))
    assert np.array_equal(vals["parameter"]["codeValues"], np.array([1, 2]))
    assert vals["dimensions"] == {"nX": 1, "nY": 1, "nZ": 2}


@pytest.fixture()
def simple_roff_parameter_contents():
    return [
        ("filedata", {"filetype": "parameter"}),
        ("dimensions", {"nX": 1, "nY": 1, "nZ": 1}),
        ("parameter", {"name": "a", "data": np.array([1.0])}),
        ("parameter", {"name": "b", "data": np.array([2.0])}),
    ]


def test_from_file_param_first(simple_roff_parameter_contents):
    buff = io.BytesIO()
    roffio.write(buff, simple_roff_parameter_contents)
    buff.seek(0)

    roff_param = RoffParameter.from_file(buff, "a")

    assert np.array_equal(roff_param.name, "a")
    assert np.array_equal(roff_param.values, np.array([1.0]))


def test_from_file_param_last(simple_roff_parameter_contents):
    buff = io.BytesIO()
    roffio.write(buff, simple_roff_parameter_contents)
    buff.seek(0)

    roff_param = RoffParameter.from_file(buff, "b")

    assert np.array_equal(roff_param.name, "b")
    assert np.array_equal(roff_param.values, np.array([2.0]))


def test_from_file_no_filedata(simple_roff_parameter_contents):
    buff = io.BytesIO()
    del simple_roff_parameter_contents[0]
    roffio.write(buff, simple_roff_parameter_contents)
    buff.seek(0)

    with pytest.raises(ValueError, match="issing non-optional keyword"):
        RoffParameter.from_file(buff, "b")


def test_from_file_double_dimensions(simple_roff_parameter_contents):
    buff = io.BytesIO()
    simple_roff_parameter_contents.append(("dimensions", {"nX": 2, "nY": 2, "nZ": 2}))
    roffio.write(buff, simple_roff_parameter_contents)
    buff.seek(0)

    with pytest.raises(ValueError, match="Multiple tag"):
        RoffParameter.from_file(buff, "b")


def test_from_file_missing_parameter(simple_roff_parameter_contents):
    buff = io.BytesIO()
    simple_roff_parameter_contents[0][1]
    roffio.write(buff, simple_roff_parameter_contents)
    buff.seek(0)

    with pytest.raises(ValueError, match="Did not find parameter"):
        RoffParameter.from_file(buff, "c")


def test_from_file_wrong_filetype(simple_roff_parameter_contents):
    buff = io.BytesIO()
    simple_roff_parameter_contents[0][1]["filetype"] = "unknown"
    roffio.write(buff, simple_roff_parameter_contents)
    buff.seek(0)

    with pytest.raises(ValueError, match="did not have filetype"):
        RoffParameter.from_file(buff, "b")


@pytest.mark.parametrize(
    "xtgeotype, rofftype",
    [
        (np.float64, "float"),  # RMS does not accept double typed gridprops
        (np.float32, "float"),
        (np.int32, "int"),
        (np.uint8, "int"),
    ],
)
def test_from_xtgeo_dtype_cast(xtgeotype, rofftype):
    gp = GridProperty(
        ncol=1,
        nrow=1,
        nlay=1,
        discrete=np.issubdtype(xtgeotype, np.integer),
        values=np.zeros((1, 1, 1), dtype=xtgeotype),
    )
    rp = RoffParameter.from_xtgeo_grid_property(gp)

    buf = io.StringIO()
    rp.to_file(buf, roff_format=roffio.Format.ASCII)

    assert f"{rofftype} data" in buf.getvalue()


def test_from_xtgeo_mask():
    values = np.ma.zeros((2, 2, 2))
    values[0, 0, 0] = np.ma.masked
    gp = GridProperty(
        ncol=2,
        nrow=2,
        nlay=2,
        values=values,
    )
    rp = RoffParameter.from_xtgeo_grid_property(gp)
    assert rp.values[1] == -999.0
