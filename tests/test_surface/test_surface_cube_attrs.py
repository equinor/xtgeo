import pathlib

import pytest

import xtgeo

xtg = xtgeo.common.XTGeoDialog()
logger = xtg.basiclogger(__name__)

SFILE1 = pathlib.Path("cubes/etc/ib_synth_iainb.segy")
SFILE2 = pathlib.Path("cubes/reek/syntseis_20030101_seismic_depth_stack.segy")

TOP2A = pathlib.Path("surfaces/reek/2/01_topreek_rota.gri")
TOP2B = pathlib.Path("surfaces/reek/2/04_basereek_rota.gri")

# ======================================================================================
# This is a a set of tests towards a synthetic small cube made by I Bush in order to
# test all attributes in detail
# ======================================================================================


@pytest.fixture(name="loadsfile1")
def fixture_loadsfile1(testdata_path):
    """Fixture for loading a SFILE1"""
    logger.info("Load seismic file 1")
    return xtgeo.cube_from_file(testdata_path / SFILE1)


@pytest.fixture(name="loadsfile2")
def fixture_loadsfile2(testdata_path):
    """Fixture for loading a SFILE2"""
    logger.info("Load seismic file 2")
    return xtgeo.cube_from_file(testdata_path / SFILE2)


def test_single_slice_yflip_snapxy_both(loadsfile1):
    """Test cube values with exact location of maps, compared to cube samples"""
    cube1 = loadsfile1

    # checks are from spreadsheet ib_synth_iainb.odf under testdata
    chks = {
        1000: -242.064,
        1008: -226.576,
        1096: -90,
        1420: 121.104,
        2984: -211.6,
        3000: -242.064,
    }

    methods = ["nearest", "trilinear"]

    for method in methods:
        for tslice, tval in chks.items():
            surf1 = xtgeo.surface_from_cube(cube1, tslice)

            surf1.slice_cube(cube1, sampling=method, snapxy=True)

            assert surf1.values.mean() == pytest.approx(tval, abs=1e-4)


def test_single_slice_yflip_snapxy_both2(loadsfile1):
    """Test cube values with vertically interpolated location of maps"""
    cube1 = loadsfile1

    # checks are from spreadsheet ib_synth_iainb.odf under testdata
    chks = {}
    chks["nearest"] = {
        1007: -226.576,
        1097.9: -90,
        1421.9: 121.104,
    }

    chks["trilinear"] = {
        1007: -228.49601,
        1098: -87.6320,
        1422: 123.9200,
    }

    methods = ["nearest", "trilinear"]

    for method in methods:
        for tslice, tval in chks[method].items():
            surf1 = xtgeo.surface_from_cube(cube1, tslice)

            surf1.slice_cube(cube1, sampling=method, snapxy=True)

            assert surf1.values.mean() == pytest.approx(tval, abs=1e-3)


def test_single_slice_yflip_positive_snapxy(loadsfile1):
    cube1 = loadsfile1
    cube1.swapaxes()
    samplings = ["nearest", "trilinear"]

    for sampling in samplings:
        surf1 = xtgeo.surface_from_cube(cube1, 1000.0)

        surf1.slice_cube(cube1, sampling=sampling, snapxy=True, algorithm=2)

        assert surf1.values.mean() == pytest.approx(cube1.values[0, 0, 0], abs=0.000001)
        print(surf1.values.mean())
        print(cube1.values[0, 0, 0])


def test_various_attrs_algorithm2(loadsfile1):
    cube1 = loadsfile1
    surf1 = xtgeo.surface_from_cube(cube1, 2540)
    surf2 = xtgeo.surface_from_cube(cube1, 2548)

    surfx = surf1.copy()
    surfx.slice_cube_window(
        cube1,
        other=surf2,
        other_position="below",
        attribute="mean",
        sampling="trilinear",
        snapxy=True,
        ndiv=None,
        algorithm=2,
    )

    assert surfx.values.mean() == pytest.approx(176.44, abs=0.01)

    surfx = surf1.copy()
    surfx.slice_cube_window(
        cube1,
        other=surf2,
        other_position="below",
        attribute="maxneg",
        sampling="trilinear",
        snapxy=True,
        ndiv=None,
        algorithm=2,
    )

    assert surfx.values.count() == 0  # no nonmasked elements

    surfx = surf1.copy()
    surfx.slice_cube_window(
        cube1,
        other=surf2,
        other_position="below",
        attribute="sumabs",
        sampling="cube",
        snapxy=True,
        ndiv=None,
        algorithm=2,
    )

    assert surfx.values.mean() == pytest.approx(529.328, abs=0.01)


def test_various_attrs_algorithm3(loadsfile1):
    cube1 = loadsfile1
    surf1 = xtgeo.surface_from_cube(cube1, 2540)
    surf2 = xtgeo.surface_from_cube(cube1, 2548)

    surfx = surf1.copy()
    surfx.slice_cube_window(
        cube1,
        other=surf2,
        other_position="below",
        attribute="mean",
        sampling="trilinear",
        snapxy=True,
        ndiv=None,
        algorithm=3,
    )

    assert surfx.values.mean() == pytest.approx(177.34, abs=0.1)

    surfx = surf1.copy()
    surfx.slice_cube_window(
        cube1,
        other=surf2,
        other_position="below",
        attribute="maxneg",
        sampling="trilinear",
        snapxy=True,
        ndiv=None,
        algorithm=3,
    )

    assert surfx.values.count() == 0  # no nonmasked elements

    surfx = surf1.copy()
    surfx.slice_cube_window(
        cube1,
        other=surf2,
        other_position="below",
        attribute="sumabs",
        sampling="cube",
        snapxy=True,
        ndiv=None,
        algorithm=3,
    )

    assert surfx.values.mean() == pytest.approx(529.328, abs=0.01)


@pytest.mark.parametrize(
    "ndiv, expected_mean",
    (
        [2, 177.35],
        [4, 176.12],
        [12, 176.43],
    ),
)
def test_various_attrs_algorithm3_ndiv(loadsfile1, ndiv, expected_mean):
    cube1 = loadsfile1
    surf1 = xtgeo.surface_from_cube(cube1, 2540)
    surf2 = xtgeo.surface_from_cube(cube1, 2548)

    surfx = surf1.copy()
    surfx.slice_cube_window(
        cube1,
        other=surf2,
        other_position="below",
        attribute="mean",
        sampling="trilinear",
        snapxy=True,
        ndiv=ndiv,
        algorithm=3,
    )

    assert surfx.values.mean() == pytest.approx(expected_mean, abs=0.1)


@pytest.mark.bigtest
@pytest.mark.parametrize(
    "ndiv, expected_mean",
    (
        [2, 0.0013886],
        [4, 0.0013843],
        [12, 0.0013829],
    ),
)
def test_various_attrs_algorithm3_ndiv_large(loadsfile2, ndiv, expected_mean):
    cube1 = loadsfile2
    surf1 = xtgeo.surface_from_cube(cube1, 1560)
    surf2 = xtgeo.surface_from_cube(cube1, 1760)

    surfx = surf1.copy()
    surfx.slice_cube_window(
        cube1,
        other=surf2,
        other_position="below",
        attribute="mean",
        sampling="trilinear",
        snapxy=True,
        ndiv=ndiv,
        algorithm=3,
    )

    print(surfx.values.mean())
    assert surfx.values.mean() == pytest.approx(expected_mean, abs=0.00001)


def test_avg_surface(loadsfile1):
    cube1 = loadsfile1
    surf1 = xtgeo.surface_from_cube(cube1, 1100.0)
    surf2 = xtgeo.surface_from_cube(cube1, 2900.0)

    # values taken from IB spreadsheet except variance
    attributes = {
        "min": -85.2640,
        "max": 250.0,
        "rms": 196.571,
        "mean": 157.9982,
        "var": 13676.7175,
        "sumneg": -2160.80,
        "sumpos": 73418,
        "meanabs": 167.5805,
        "meanpos": 194.7427,
        "meanneg": -29.2,
        "maxneg": -85.264,
        "maxpos": 250.0,
    }

    attrs = surf1.slice_cube_window(
        cube1,
        other=surf2,
        other_position="below",
        attribute=list(attributes.keys()),
        sampling="cube",
        snapxy=True,
        ndiv=None,
        algorithm=2,
        showprogress=False,
    )

    for name in attributes:
        print(attrs[name].values.mean())
        assert attributes[name] == pytest.approx(attrs[name].values.mean(), abs=0.001)

    attrs = surf1.slice_cube_window(
        cube1,
        other=surf2,
        other_position="below",
        attribute=list(attributes.keys()),
        sampling="trilinear",
        snapxy=True,
        ndiv=None,
        algorithm=2,
    )

    # less strict abs using relative 1%:
    for name in attributes:
        assert attributes[name] == pytest.approx(attrs[name].values.mean(), rel=0.01)


def test_avg_surface2(loadsfile1):
    cube1 = loadsfile1
    surf1 = xtgeo.surface_from_cube(cube1, 2540.0)
    surf2 = xtgeo.surface_from_cube(cube1, 2548.0)

    # values taken from IB spreadsheet but not variance
    # http://www.alcula.com/calculators/statistics/variance/  --> 45.1597

    attributes = {
        "mean": 176.438,
        "var": 22.5797,
    }

    attrs = surf1.slice_cube_window(
        cube1,
        other=surf2,
        other_position="below",
        attribute=list(attributes.keys()),
        sampling="discrete",
        snapxy=True,
        ndiv=None,
        algorithm=2,
    )

    for name, value in attributes.items():
        assert value == pytest.approx(attrs[name].values.mean(), abs=0.001)


@pytest.mark.benchmark(group="cube slicing")
@pytest.mark.parametrize(
    "algorithm", [pytest.param(1, marks=pytest.mark.bigtest), 2, 3]
)
def test_avg_surface_large_cube_algorithmx(benchmark, algorithm):
    cube1 = xtgeo.Cube(ncol=120, nrow=120, nlay=100, zori=200, xinc=12, yinc=12, zinc=4)

    cube1.values[400:80, 400:80, :] = 12

    surf1 = xtgeo.surface_from_cube(cube1, 2040.0)
    surf2 = xtgeo.surface_from_cube(cube1, 2880.0)

    def run():
        _ = surf1.slice_cube_window(
            cube1,
            other=surf2,
            other_position="below",
            attribute="all",
            sampling="cube",
            snapxy=True,
            ndiv=None,
            algorithm=algorithm,
            showprogress=False,
        )

    benchmark(run)


@pytest.mark.bigtest
def test_attrs_reek(tmp_path, loadsfile2, testdata_path):
    logger.info("Make cube...")
    cube2 = loadsfile2

    t2a = xtgeo.surface_from_file(testdata_path / TOP2A)
    t2b = xtgeo.surface_from_file(testdata_path / TOP2B)

    attlist = ["maxpos", "maxneg", "mean", "rms"]

    attrs1 = t2a.slice_cube_window(
        cube2, other=t2b, sampling="trilinear", attribute=attlist, algorithm=1
    )
    attrs2 = t2a.slice_cube_window(
        cube2, other=t2b, sampling="trilinear", attribute=attlist, algorithm=2
    )
    attrs3 = t2a.slice_cube_window(
        cube2, other=t2b, sampling="trilinear", attribute=attlist, algorithm=3
    )

    for att, _ in attrs1.items():
        srf1 = attrs1[att]
        srf2 = attrs2[att]
        srf3 = attrs3[att]

        srf1.to_file(tmp_path / f"attr1_{att}.gri")
        srf1.to_file(tmp_path / f"attr2_{att}.gri")
        srf1.to_file(tmp_path / f"attr3_{att}.gri")

        assert srf1.values.mean() == pytest.approx(srf2.values.mean(), abs=0.005)
        assert srf3.values.mean() == pytest.approx(srf2.values.mean(), abs=0.005)
        print("\nok")
