from pydantic_settings import BaseSettings


class NetworkConfig(BaseSettings):
    CHIMERA_NETWORK_NAME: str = "chimera-network"
    CHIMERA_NETWORK_PREFIX: str = "192.168.10"
    CHIMERA_NETWORK_SUBNET_MASK: str = "23"


class ContainerConfig(BaseSettings):
    CHIMERA_WORKERS_CONTAINER_PORT: str = "80"
