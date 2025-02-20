from . import DOCKERFILE_NAME, SERVICES
from ..parallelism.data import Ensemble
from ..workers.server import WorkerServerHandler


class Chimera:
    def __init__(self) -> None:
        self._workers = WorkerServerHandler()

    def serve(self, service: str, port: int) -> None:
        self._workers.serve_all(DOCKERFILE_NAME)
        service: Ensemble = SERVICES[service.lower()]
        service.serve(port)
