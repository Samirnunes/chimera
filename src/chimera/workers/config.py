from typing import List

from pydantic_settings import BaseSettings


class _NetworkConfig(BaseSettings):
    CHIMERA_NETWORK_NAME: str = "chimera-network"
    CHIMERA_NETWORK_PREFIX: str = "192.168.10"
    CHIMERA_NETWORK_SUBNET_MASK: str = "23"


class _WorkersConfig(BaseSettings):
    CHIMERA_WORKERS_NODES_NAMES: List[
        str
    ]  # must be the filenames when nodes are defined
    CHIMERA_WORKERS_CPU_SHARES: List[int]
    CHIMERA_WORKERS_HOST_PORTS: List[int]
    CHIMERA_WORKERS_CONTAINER_PORT: str = "80"
    CHIMERA_WORKERS_HOST: str = "0.0.0.0"


NETWORK_CONFIG = _NetworkConfig()
WORKERS_CONFIG = _WorkersConfig()
