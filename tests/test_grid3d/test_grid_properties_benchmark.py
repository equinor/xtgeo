import datetime
import pathlib

import numpy as np
import pytest
import resfo

import xtgeo
from xtgeo.grid3d import GridProperties

REEK_UNRST = pathlib.Path("3dgrids/reek/REEK.UNRST")
REEK_EGRID = pathlib.Path("3dgrids/reek/REEK.EGRID")
SPE9_UNRST = pathlib.Path("3dgrids/bench_spe9/BENCH_SPE9.UNRST")
SPE9_EGRID = pathlib.Path("3dgrids/bench_spe9/BENCH_SPE9.EGRID")


@pytest.fixture(scope="session")
def synthetic_unrst_case(tmp_path_factory):
    dims = (60, 60, 10)
    n_cells = int(np.prod(dims))
    path = tmp_path_factory.mktemp("synthetic_unrst") / "SYNTH.UNRST"

    def records():
        start_date = datetime.date(2000, 1, 1)
        for seqnum in range(120):
            day = start_date + datetime.timedelta(days=seqnum * 7)
            intehead = np.zeros(411, dtype=np.int32)
            intehead[8], intehead[9], intehead[10] = dims
            intehead[14] = 7  # Oil/Water/Gas
            intehead[64], intehead[65], intehead[66] = day.day, day.month, day.year
            intehead[94] = 100  # E100
            logihead = np.zeros(128, dtype=bool)

            yield ("SEQNUM  ", np.array([seqnum], dtype=np.int32))
            yield ("INTEHEAD", intehead)
            yield ("LOGIHEAD", logihead)
            yield ("PRESSURE", np.full(n_cells, seqnum, dtype=np.float32))
            yield ("SWAT    ", np.full(n_cells, seqnum / 120.0, dtype=np.float32))

    if not path.exists():
        resfo.write(path, records())
    return {
        "unrst": path,
        "grid": xtgeo.create_box_grid(dimension=dims),
        "names": ["PRESSURE", "SWAT"],
        "dates": [20000101],
    }


@pytest.mark.parametrize(
    "case",
    [
        {
            "id": "reek",
            "unrst": REEK_UNRST,
            "egrid": REEK_EGRID,
            "names": ["PRESSURE", "SWAT"],
            "dates": [19991201],
        },
        {
            "id": "bench_spe9",
            "unrst": SPE9_UNRST,
            "egrid": SPE9_EGRID,
            "names": ["PRESSURE", "SWAT"],
            "dates": [19900101],
        },
    ],
    ids=lambda case: case["id"],
)
@pytest.mark.benchmark(group="gridproperties.scan_dates")
def test_benchmark_scan_dates(benchmark, case, testdata_path):
    benchmark(GridProperties.scan_dates, pathlib.Path(testdata_path) / case["unrst"])


@pytest.mark.bigtest
@pytest.mark.benchmark(group="gridproperties.scan_dates")
def test_benchmark_scan_dates_synthetic_big(benchmark, synthetic_unrst_case):
    benchmark(GridProperties.scan_dates, synthetic_unrst_case["unrst"])


@pytest.mark.parametrize(
    "case",
    [
        {
            "id": "reek",
            "unrst": REEK_UNRST,
            "egrid": REEK_EGRID,
            "names": ["PRESSURE", "SWAT"],
            "dates": [19991201],
        },
        {
            "id": "bench_spe9",
            "unrst": SPE9_UNRST,
            "egrid": SPE9_EGRID,
            "names": ["PRESSURE", "SWAT"],
            "dates": [19900101],
        },
    ],
    ids=lambda case: case["id"],
)
@pytest.mark.benchmark(group="xtgeo.gridproperties_from_file")
def test_benchmark_gridproperties_from_unrst(benchmark, case, testdata_path):
    grid = xtgeo.grid_from_file(
        pathlib.Path(testdata_path) / case["egrid"], fformat="egrid"
    )

    def run():
        return xtgeo.gridproperties_from_file(
            pathlib.Path(testdata_path) / case["unrst"],
            fformat="unrst",
            grid=grid,
            names=case["names"],
            dates=case["dates"],
        )

    props = benchmark(run)
    assert len(props.props) == len(case["names"])


@pytest.mark.bigtest
@pytest.mark.benchmark(group="xtgeo.gridproperties_from_file")
def test_benchmark_gridproperties_from_unrst_synthetic_big(
    benchmark, synthetic_unrst_case
):
    def run():
        return xtgeo.gridproperties_from_file(
            synthetic_unrst_case["unrst"],
            fformat="unrst",
            grid=synthetic_unrst_case["grid"],
            names=synthetic_unrst_case["names"],
            dates=synthetic_unrst_case["dates"],
        )

    props = benchmark(run)
    assert len(props.props) == len(synthetic_unrst_case["names"])
