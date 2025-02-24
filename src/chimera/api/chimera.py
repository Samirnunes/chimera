from typing import Literal

from ..containers.handler import ContainersHandler
from ..nodes.masters import AggregationMaster


class Chimera:
    def __init__(self) -> None:
        self._masters = {"aggregation": AggregationMaster()}
        self._workers_handler = ContainersHandler()

    def serve(self, master: Literal["aggregation"], port: int = 8081) -> None:
        self._workers_handler.serve_all()
        self._masters[master.lower()].serve(port)
