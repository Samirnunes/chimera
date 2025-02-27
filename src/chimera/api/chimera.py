from typing import Literal

from ..containers.handler import ContainersHandler
from ..nodes.masters import AggregationMaster


class Chimera:
    """
    Orchestrates the Chimera distributed system, managing workers and masters.
    """

    def __init__(self) -> None:
        """
        Initializes the Chimera system with an AggregationMaster and a ContainersHandler.
        """
        self._masters = {"aggregation": AggregationMaster()}
        self._workers_handler = ContainersHandler()

    def serve(self, master: Literal["aggregation"], port: int = 8081) -> None:
        """
        Starts the Chimera system.

        This method starts all Chimera worker containers and a specified master node.

        Args:
            master: The type of master node to start ('aggregation').
            port: The port number for the master node to listen on (default: 8081).
        """
        self._workers_handler.serve_all()
        self._masters[master.lower()].serve(port)
