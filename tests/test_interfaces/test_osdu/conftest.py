"""Shared fixtures for OSDU/RESQML interface tests."""

import pytest


@pytest.fixture
def epc_path(tmp_path):
    """Return a temporary EPC file path."""
    return str(tmp_path / "test.epc")


@pytest.fixture
def etp_provider():
    """Create an ETP provider connected to local RDDMS (skip if unavailable)."""
    from xtgeo.interfaces.osdu import EtpConnectionConfig, EtpProvider

    cfg = EtpConnectionConfig(
        url="ws://localhost:9002",
        dataspace="eml:///dataspace('xtgeo/tests')",
    )
    try:
        p = EtpProvider(cfg)
        p.open()
    except Exception:
        pytest.skip("Local RDDMS not available")
    yield p
    p.close()
