import io
import logging
from contextlib import contextmanager

import pytest
import xtgeo._internal as _internal  # type: ignore # noqa

import xtgeo


@contextmanager
def capture_logs(logger_name, level=logging.DEBUG):
    """Capture logs from a specific logger."""
    # Create a logger
    logger = logging.getLogger(logger_name)
    original_level = logger.level
    logger.setLevel(level)

    # Create a string IO handler to capture logs
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(level)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    try:
        yield log_capture
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)


# Convert to a pytest fixture for reuse across tests
@pytest.fixture
def log_capturer():
    """Fixture that returns a function to capture logs."""

    def _capture_logs(logger_name, level=logging.DEBUG):
        return capture_logs(logger_name, level)

    return _capture_logs


def test_regsurf_logging(log_capturer):
    """Test that C++ logs as expected in regular surface operations."""
    logger_name = "xtgeo.regsurf.sample_grid3d_layer"

    with log_capturer(logger_name) as log_capture:
        # Call a function that should trigger C++ logging
        grid = xtgeo.create_box_grid((10, 10, 10))
        _ = xtgeo.surface_from_grid3d(grid)

        # Get the captured logs
        log_contents = log_capture.getvalue()

        # Verify logs contain expected messages
        assert (
            "[C++: xtgeo.regsurf.sample_grid3d_layer] Sampling grid3d" in log_contents
        )


def test_logging_levels(log_capturer):
    """Test that C++ respects Python logging levels."""
    logger_name = "xtgeo.test_logger"

    # Create a custom logger for testing
    logger = logging.getLogger(logger_name)
    original_level = logger.level

    try:
        # Set the logger level to INFO - debug messages should not appear
        logger.setLevel(logging.INFO)

        with log_capturer(logger_name, level=logging.INFO) as log_capture:
            # Call a function that logs at different levels
            _internal.logging.test_logging_levels(logger_name)

            log_contents = log_capture.getvalue()

            # Debug messages should not be present
            assert "DEBUG:" not in log_contents
            # Info messages should be present
            assert "INFO:" in log_contents

    finally:
        logger.setLevel(original_level)


# Add a test for all logging levels
@pytest.mark.parametrize(
    "log_level,visible_levels,hidden_levels",
    [
        (logging.DEBUG, ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], []),
        (logging.INFO, ["INFO", "WARNING", "ERROR", "CRITICAL"], ["DEBUG"]),
        (logging.WARNING, ["WARNING", "ERROR", "CRITICAL"], ["DEBUG", "INFO"]),
        (logging.ERROR, ["ERROR", "CRITICAL"], ["DEBUG", "INFO", "WARNING"]),
        (logging.CRITICAL, ["CRITICAL"], ["DEBUG", "INFO", "WARNING", "ERROR"]),
    ],
)
def test_all_logging_levels(log_capturer, log_level, visible_levels, hidden_levels):
    """Test that all logging levels are respected correctly."""
    logger_name = f"xtgeo.test_logger.{log_level}"
    logger = logging.getLogger(logger_name)
    original_level = logger.level

    try:
        logger.setLevel(log_level)

        with log_capturer(logger_name, level=log_level) as log_capture:
            _internal.logging.test_logging_levels(logger_name)

            log_contents = log_capture.getvalue()

            # Check visible levels
            for level in visible_levels:
                assert f"{level}:" in log_contents, (
                    f"Expected {level} to be visible at log level {log_level}"
                )

            # Check hidden levels
            for level in hidden_levels:
                assert f"{level}:" not in log_contents, (
                    f"Expected {level} to be hidden at log level {log_level}"
                )

    finally:
        logger.setLevel(original_level)


# Test that thread logging works
def test_thread_logging(log_capturer):
    """Test that thread information is properly logged."""
    logger_name = "xtgeo.regsurf.sample_grid3d_layer"

    with log_capturer(logger_name) as log_capture:
        # Trigger a function that uses multiple threads
        grid = xtgeo.create_box_grid(
            (20, 20, 20)
        )  # Larger grid to ensure multi-threading
        _ = xtgeo.surface_from_grid3d(grid)

        log_contents = log_capture.getvalue()

        # Check for thread-related logging
        # This assertion might need adjustment based on your actual logging format
        assert "thread" in log_contents.lower() or "Thread" in log_contents
