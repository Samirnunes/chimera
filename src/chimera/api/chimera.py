from typing import Literal

from ..parallelism.hybrid import DistributedAggregation
from ..workers.handler import WorkersHandler


class Chimera:
    def __init__(self) -> None:
        self._services = {"aggregation": DistributedAggregation()}
        self._workers_handler = WorkersHandler()

    def serve(self, service: Literal["aggregation"], port: int = 8081) -> None:
        self._workers_handler.serve_all()
        self._services[service.lower()].serve(port)
