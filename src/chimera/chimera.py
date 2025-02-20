from .api import DOCKERFILE_NAME
from .parallelism.data import Ensemble
from .workers.server import WorkerServerHandler


class Chimera:
    _SERVICES = {"ensemble": Ensemble()}

    def __init__(self) -> None:
        self._workers = WorkerServerHandler()

    def serve(self, service: str, port: int) -> None:
        self._workers.serve_all(DOCKERFILE_NAME)
        service: Ensemble = self._SERVICES[service.lower()]
        service.serve(port)
