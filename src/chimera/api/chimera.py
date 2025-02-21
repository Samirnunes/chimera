from ..parallelism.data import Ensemble
from ..workers.server import WorkersServersHandler


class Chimera:
    def __init__(self) -> None:
        self._services = {"ensemble": Ensemble()}
        self._workers = WorkersServersHandler()

    def serve(self, service: str, port: int = 8081) -> None:
        self._workers.serve_all()
        service: Ensemble = self._services[service.lower()]
        service.serve(port)
