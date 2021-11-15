from os.path import join

import pytest

import xtgeo


class EclRun:
    def __init__(
        self,
        path,
        expected_dimensions,
        expected_perm,
        expected_init_props,
        expected_restart_props,
        expected_dates,
    ):
        self.path = path
        self.expected_dimensions = expected_dimensions
        self.expected_perm = expected_perm
        self.expected_restart_props = expected_restart_props
        self.expected_init_props = expected_init_props
        self.expected_dates = expected_dates

    def __repr__(self):
        return f"EclRun({self.path}, {self.expected_dimensions})"

    @property
    def grid(self):
        return xtgeo.grid_from_file(self.path + ".EGRID")

    @property
    def init_path(self):
        return self.path + ".INIT"

    @property
    def restart_path(self):
        return self.path + ".UNRST"

    def grid_with_props(self, *args, **kwargs):
        return xtgeo.grid_from_file(self.path, fformat="eclipserun", *args, **kwargs)

    def get_property_from_init(self, name, **kwargs):
        return xtgeo.gridproperty_from_file(
            self.init_path, grid=self.grid, name=name, **kwargs
        )

    def get_property_from_restart(self, name, date, **kwargs):
        return xtgeo.gridproperty_from_file(
            self.restart_path, grid=self.grid, date=date, name=name, **kwargs
        )

    def get_restart_properties(self, names, dates, *args, **kwargs):
        gps = xtgeo.GridProperties()
        gps.from_file(
            self.restart_path,
            fformat="unrst",
            grid=self.grid,
            dates=dates,
            names=names,
            **kwargs,
        )
        return gps

    def get_init_properties(self, names, *args, **kwargs):
        gps = xtgeo.GridProperties()
        gps.from_file(
            self.init_path, grid=self.grid, fformat="init", names=names, **kwargs
        )
        return gps


@pytest.fixture
def grids_etc_path(testpath):
    return join(testpath, "3dgrids", "etc")


@pytest.fixture
def single_poro_path(grids_etc_path):
    return join(grids_etc_path, "TEST_SP")


@pytest.fixture
def dual_poro_path(grids_etc_path):
    return join(grids_etc_path, "TEST_DP")


@pytest.fixture
def dual_poro_dual_perm_path(dual_poro_path):
    return dual_poro_path + "DK"


@pytest.fixture
def dual_poro_dual_perm_wg_path(grids_etc_path):
    # same as dual_poro_dual_perm but with water/gas
    # instead of oil/water
    return join(grids_etc_path, "TEST2_DPDK_WG")


@pytest.fixture
def reek_run(testpath):
    return EclRun(
        join(testpath, "3dgrids", "reek", "REEK"),
        (40, 64, 14),
        False,
        [
            "PORV",
            "DX",
            "DY",
            "DZ",
            "PERMX",
            "PERMY",
            "PERMZ",
            "MULTX",
            "MULTY",
            "MULTZ",
            "PORO",
            "NTG",
            "TOPS",
            "DEPTH",
            "TRANX",
            "TRANY",
            "TRANZ",
            "MINPVV",
            "MULTPV",
            "MULTX-",
            "MULTY-",
            "MULTZ-",
            "PVTNUM",
            "SATNUM",
            "EQLNUM",
            "FIPNUM",
            "SWCR",
            "SGCR",
            "SOWCR",
            "SOGCR",
            "SWL",
            "SWU",
            "SGL",
            "SGU",
            "KRW",
            "KRG",
            "KRO",
            "SWATINIT",
            "KRWR",
            "KRGR",
            "KRORG",
            "KRORW",
            "SWLPC",
            "SGLPC",
            "PCW",
            "PCG",
            "ENDNUM",
        ],
        [
            "PRESSURE",
            "SWAT",
            "SGAS",
            "RS",
            "WATQUIES",
            "GASQUIES",
            "PPCW",
            "FLOOILI+",
            "FLOWATI+",
            "FLOGASI+",
            "FLOOILJ+",
            "FLOWATJ+",
            "FLOGASJ+",
            "FLOOILK+",
            "FLOWATK+",
            "FLOGASK+",
        ],
        [
            19991201,
            20000101,
            20000201,
            20000301,
            20000401,
            20000501,
            20000601,
            20000701,
            20000801,
            20000901,
            20001001,
            20001101,
            20001201,
            20010101,
            20030101,
        ],
    )


@pytest.fixture
def dual_poro_run(dual_poro_path):
    return EclRun(
        dual_poro_path,
        (5, 3, 1),
        False,
        [
            "PORV",
            "DX",
            "DY",
            "DZ",
            "PERMX",
            "PERMY",
            "PERMZ",
            "MULTX",
            "MULTY",
            "MULTZ",
            "PORO",
            "NTG",
            "TOPS",
            "DEPTH",
            "TRANX",
            "TRANY",
            "TRANZ",
            "SIGMAV",
            "DZMTRXV",
            "MINPVV",
            "SIGMAGDV",
            "MULTPV",
            "MULTX-",
            "MULTY-",
            "MULTZ-",
            "BTOBALFV",
            "PVTNUM",
            "SATNUM",
            "EQLNUM",
            "FIPNUM",
            "IMBNUM",
            "KRNUMMF",
            "IMBNUMMF",
        ],
        [
            "DLYTIM",
            "PRESSURE",
            "SWAT",
            "SWMAX",
            "SWHY1",
            "SWHY2",
            "SWHY3",
            "ISTHW",
            "SOMAX",
        ],
        [
            20170101,
            20170111,
            20170121,
            20170131,
        ],
    )


@pytest.fixture
def single_poro_run(single_poro_path):
    return EclRun(
        single_poro_path,
        (5, 3, 1),
        False,
        [
            "PORV",
            "DX",
            "DY",
            "DZ",
            "PERMX",
            "PERMY",
            "PERMZ",
            "MULTX",
            "MULTY",
            "MULTZ",
            "PORO",
            "NTG",
            "TOPS",
            "DEPTH",
            "TRANX",
            "TRANY",
            "TRANZ",
            "MINPVV",
            "MULTPV",
            "MULTX-",
            "MULTY-",
            "MULTZ-",
            "PVTNUM",
            "SATNUM",
            "EQLNUM",
            "FIPNUM",
            "IMBNUM",
        ],
        [
            "PRESSURE",
            "SWAT",
            "SWMAX",
            "SWHY1",
            "SWHY2",
            "SWHY3",
            "ISTHW",
            "SOMAX",
        ],
        [
            20170101,
            20170111,
            20170121,
            20170131,
        ],
    )


@pytest.fixture
def dual_poro_dual_perm_run(dual_poro_dual_perm_path):
    return EclRun(
        dual_poro_dual_perm_path,
        (5, 3, 1),
        True,
        [
            "PORV",
            "DX",
            "DY",
            "DZ",
            "PERMX",
            "PERMY",
            "PERMZ",
            "MULTX",
            "MULTY",
            "MULTZ",
            "PORO",
            "NTG",
            "TOPS",
            "DEPTH",
            "TRANX",
            "TRANY",
            "TRANZ",
            "SIGMAV",
            "DZMTRXV",
            "MINPVV",
            "SIGMAGDV",
            "MULTPV",
            "MULTX-",
            "MULTY-",
            "MULTZ-",
            "BTOBALFV",
            "PVTNUM",
            "SATNUM",
            "EQLNUM",
            "FIPNUM",
            "IMBNUM",
            "KRNUMMF",
            "IMBNUMMF",
        ],
        [
            "DLYTIM",
            "PRESSURE",
            "SWAT",
            "SWMAX",
            "SWHY1",
            "SWHY2",
            "SWHY3",
            "ISTHW",
            "SOMAX",
        ],
        [
            20170101,
            20170111,
            20170121,
            20170131,
        ],
    )


@pytest.fixture
def dual_poro_dual_perm_wg_run(dual_poro_dual_perm_wg_path):
    return EclRun(
        dual_poro_dual_perm_wg_path,
        (5, 3, 1),
        True,
        [
            "PORV",
            "DX",
            "DY",
            "DZ",
            "PERMX",
            "PERMY",
            "PERMZ",
            "MULTX",
            "MULTY",
            "MULTZ",
            "PORO",
            "NTG",
            "TOPS",
            "DEPTH",
            "TRANX",
            "TRANY",
            "TRANZ",
            "SIGMAV",
            "DZMTRXV",
            "MINPVV",
            "SIGMAGDV",
            "MULTPV",
            "MULTX-",
            "MULTY-",
            "MULTZ-",
            "BTOBALFV",
            "PVTNUM",
            "SATNUM",
            "EQLNUM",
            "FIPNUM",
            "KRNUMMF",
        ],
        [
            "PRESSURE",
            "SWAT",
            "DLYTIM",
        ],
        [
            20170101,
            20170111,
            20170121,
            20170131,
        ],
    )


@pytest.fixture
def dual_props_run(testpath):
    return EclRun(
        join(testpath, "3dgrids/etc/DUAL"),
        (5, 3, 1),
        True,
        [
            "PORV",
            "DX",
            "DY",
            "DZ",
            "PERMX",
            "PERMY",
            "PERMZ",
            "MULTX",
            "MULTY",
            "MULTZ",
            "PORO",
            "NTG",
            "TOPS",
            "DEPTH",
            "TRANX",
            "TRANY",
            "TRANZ",
            "SIGMAV",
            "DZMTRXV",
            "MINPVV",
            "SIGMAGDV",
            "MULTPV",
            "MULTX-",
            "MULTY-",
            "MULTZ-",
            "BTOBALFV",
            "PVTNUM",
            "SATNUM",
            "EQLNUM",
            "FIPNUM",
            "IMBNUM",
            "KRNUMMF",
            "IMBNUMMF",
        ],
        [
            "DLYTIM",
            "PRESSURE",
            "SWAT",
            "SWMAX",
            "SWHY1",
            "SWHY2",
            "SWHY3",
            "ISTHW",
            "SOMAX",
            "FIPOIL",
            "FIPWAT",
            "SFIPOIL",
            "SFIPWAT",
            "DLYTIM",
            "PRESSURE",
            "SWAT",
            "SWMAX",
            "SWHY1",
            "SWHY2",
            "SWHY3",
            "ISTHW",
            "SOMAX",
            "WAT_PRES",
            "DLYTIM",
            "PRESSURE",
            "SWAT",
            "SWMAX",
            "SWHY1",
            "SWHY2",
            "SWHY3",
            "ISTHW",
            "SOMAX",
            "WAT_PRES",
        ],
        [
            20170101,
        ],
    )


@pytest.fixture(
    params=[
        "reek_run",
        "single_poro_run",
        "dual_props_run",
        "dual_poro_run",
        "dual_poro_dual_perm_run",
        "dual_poro_dual_perm_wg_run",
    ]
)
def ecl_runs(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(
    params=[
        "dual_poro_run",
        "dual_poro_dual_perm_run",
        "dual_poro_dual_perm_wg_run",
    ]
)
def dual_runs(request):
    return request.getfixturevalue(request.param)
