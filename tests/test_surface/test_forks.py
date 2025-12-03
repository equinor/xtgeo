import logging
import subprocess
import sys

import pytest

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@pytest.mark.skipif(sys.platform == "darwin", reason="Multiprocessing issues on macOS")
def test_surface_forks() -> None:
    """Testing when surfaces are read by multiple forks"""

    process = subprocess.Popen(
        [sys.executable, "multiprocess_surfs.py"],
        cwd="examples",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate(timeout=60)
    ret_code = process.returncode

    if ret_code != 0:
        logger.info("Return code: %d", ret_code)
        logger.info("STDOUT: %s", stdout.decode())
        logger.info("STDERR: %s", stderr.decode())

    assert ret_code == 0, (
        f"Subprocess failed with exit code {ret_code}\n"
        f"STDOUT: {stdout.decode()}\n"
        f"STDERR: {stderr.decode()}"
    )
