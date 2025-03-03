from abc import ABC, abstractmethod


class Master(ABC):
    """
    Abstract base class for Chimera master nodes.

    This class defines the interface for all master nodes in the Chimera
    distributed system.  Subclasses (like `AggregationMaster` and `ParameterServerMaster`)
    must implement the `serve` method.
    """

    @abstractmethod
    def serve(self, port: int) -> None:
        """
        Starts the master server on the specified port.

        Args:
            port: The port number to listen on.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError
