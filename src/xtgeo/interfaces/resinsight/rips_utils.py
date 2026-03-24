"""Utility helpers for working with ResInsight via rips."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xtgeo.common.log import null_logger

from ._rips_package import (
    ResInsightInstanceOrPortType,
    RipsInstanceType,
    RipsProjectType,
    rips,
)

if TYPE_CHECKING:
    import pathlib

logger = null_logger(__name__)


class RipsApiUtils:
    """Utility class for connecting to a ResInsight instance and project.

    Args:
        instance_or_port: One of:
            - existing ``rips.Instance``
            - ``None`` to auto-discover a running instance
            - ``int`` as ``port``
    """

    def __init__(
        self,
        instance_or_port: ResInsightInstanceOrPortType | None = None,
    ) -> None:
        if rips is None:
            raise RuntimeError(
                "rips package is not available. Please install it to use "
                "ResInsight features."
            )

        instance_cls = getattr(rips, "Instance", None)
        if instance_cls is None:
            raise RuntimeError("rips does not expose Instance API")

        self._instance: RipsInstanceType

        if instance_or_port is None or isinstance(instance_or_port, int):
            self._instance = self.find_instance(port=instance_or_port)
        elif hasattr(instance_or_port, "project"):
            self._instance = instance_or_port
        else:
            raise TypeError(
                "instance_or_port must be None, an integer port, or a rips.Instance"
            )

    @property
    def instance(self) -> RipsInstanceType:
        """The connected ``rips.Instance``."""
        return self._instance

    @property
    def project(self) -> RipsProjectType:
        """The project object from the connected instance."""
        return self._instance.project

    def save_project(self, name: str = "") -> None:
        """Save the ResInsight project.

        Args:
            name: Path to save the project file. If empty, the project is saved
                to its current project location as determined by ResInsight.
        """
        self.project.save(name)

    def close_project(self) -> None:
        """Close the active project and open a new one."""
        self.project.close()
        logger.debug("ResInsight project closed")

    def terminate(self) -> None:
        """Terminate ResInsight instance."""
        self._instance.exit()
        logger.debug("ResInsight instance closed")

    @staticmethod
    def launch_instance(
        executable: str | pathlib.Path = "", console_mode: bool = False, port: int = -1
    ) -> RipsInstanceType:
        """Launch a new ResInsight instance using the specified executable.

        Args:
            executable: Path to the ResInsight executable. If empty, will attempt
                to launch using default system path.
            console_mode: Whether to launch ResInsight in console mode.
            port: If 0, GRPC will find an available port. If -1, use the default port
                50051 or RESINSIGHT_GRPC_PORT If anything else, ResInsight will try to
                launch with the specified portnumber.
        """
        if rips is None:
            raise RuntimeError("rips package is not available")
        instance = rips.Instance.launch(
            str(executable),
            console=console_mode,
            launch_port=port,
        )
        logger.debug("Launching ResInsight using '%s'", str(executable))
        if instance is None:
            raise RuntimeError("Failed to launch ResInsight instance")
        return instance

    @staticmethod
    def find_instance(port: int | None) -> RipsInstanceType:
        """Use available rips APIs to discover/connect to an instance.

        Note:
            Connection is port-driven.
        """
        if rips is None:
            raise RuntimeError("rips package is not available")

        if port is None:
            try:
                instance = rips.Instance.find()
            except Exception as e:
                raise RuntimeError(
                    "Unable to connect to a running ResInsight instance. "
                    "Ensure ResInsight is running and that auto-discovery can "
                    "find a single reachable instance."
                ) from e
        else:
            try:
                instance = rips.Instance.find(start_port=port, end_port=port + 1)
            except Exception as e:
                raise RuntimeError(
                    f"Unable to connect to a ResInsight instance on port {port}"
                ) from e
        return instance
