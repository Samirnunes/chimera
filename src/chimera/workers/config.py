import ast
from typing import Any, List

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode
from typing_extensions import Annotated


class NetworkConfig(BaseSettings):
    CHIMERA_NETWORK_NAME: str = "chimera-network"
    CHIMERA_NETWORK_PREFIX: str = "192.168.10"
    CHIMERA_NETWORK_SUBNET_MASK: str = "23"


class WorkersConfig(BaseSettings):
    CHIMERA_WORKERS_NODES_NAMES: Annotated[
        List[str], NoDecode
    ]  # must be the filenames when nodes are defined
    CHIMERA_WORKERS_CPU_SHARES: Annotated[List[int], NoDecode] = [2]
    CHIMERA_WORKERS_HOST_PORTS: Annotated[List[int], NoDecode] = [8081]
    CHIMERA_WORKERS_CONTAINER_PORT: str = "80"
    CHIMERA_WORKERS_HOST: str = "0.0.0.0"

    @field_validator(
        "CHIMERA_WORKERS_NODES_NAMES",
        "CHIMERA_WORKERS_CPU_SHARES",
        "CHIMERA_WORKERS_HOST_PORTS",
        mode="before",
    )
    @classmethod
    def parse_lists(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return ast.literal_eval(v)
        return v
