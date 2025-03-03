from ..containers.handler import ContainersHandler
from ..nodes.masters import AggregationMaster, ParameterServerMaster


def run(master: AggregationMaster | ParameterServerMaster, port: int = 8081) -> None:
    """
    Starts the Chimera master server and handles worker containers.

    This function initializes and starts the necessary components for a Chimera
    master node, including serving the worker containers and the master itself.

    Args:
        master: An instance of either AggregationMaster or ParameterServerMaster.
            This object represents the master node in the Chimera distributed system.
        port: The port number on which the master server will listen for connections.
            Defaults to 8081.

    Raises:
        TypeError: If the `master` argument is not an instance of
            `AggregationMaster` or `ParameterServerMaster`.

    Example:
        >>> from chimera.nodes.masters import AggregationMaster
        >>> master = AggregationMaster()
        >>> run(master, 8082) # Starts an AggregationMaster on port 8082.
    """

    ContainersHandler().serve_all()
    master.serve(port)
