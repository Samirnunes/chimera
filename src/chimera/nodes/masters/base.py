from abc import ABC, abstractmethod


class Master(ABC):
    @abstractmethod
    def serve(self, port: int) -> None:
        raise NotImplementedError
