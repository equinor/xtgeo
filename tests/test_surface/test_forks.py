import logging
import subprocess

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def test_surface_forks():
    """Testing when surfaces are read by multiple forks"""

    process = subprocess.Popen(
        ["python", "multiprocess_surfs.py"],
        cwd="examples",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = process.communicate(timeout=60)  # timeout to prevent hanging
        ret_code = process.returncode
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        ret_code = -1  # Use a custom return code for timeouts

        # Log more information for debugging
        logger.info("Return code: %d", ret_code)
        logger.info("STDOUT: %s", stdout.decode())
        logger.info("STDERR: %s", stderr.decode())

    assert ret_code == 0, f"Subprocess failed with exit code {ret_code}\n"
    f"STDOUT: {stdout.decode()}\nSTDERR: {stderr.decode()}"
