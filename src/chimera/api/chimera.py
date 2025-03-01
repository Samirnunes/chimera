from ..containers.handler import ContainersHandler
from ..nodes.masters import AggregationMaster, ParameterServerMaster


def run(master: AggregationMaster | ParameterServerMaster, port: int = 8081) -> None:
    workers_handler = ContainersHandler()
    workers_handler.serve_all()
    master.serve(port)
