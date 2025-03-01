import ast
import ipaddress
from typing import Any, List

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode
from typing_extensions import Annotated

CHIMERA_DOCKERFILE_NAME = "Dockerfile.worker"
CHIMERA_WORKERS_FOLDER = "chimera_workers"
CHIMERA_TRAIN_DATA_FOLDER = "chimera_train_data"
CHIMERA_TRAIN_FEATURES_FILENAME = "X_train.csv"
CHIMERA_TRAIN_LABELS_FILENAME = "y_train.csv"


class NetworkConfig(BaseSettings):
    """
    Configuration settings for the Chimera network.
    """

    CHIMERA_NETWORK_NAME: str = "chimera-network"
    """Name of the Docker network."""
    CHIMERA_NETWORK_PREFIX: Annotated[str, NoDecode] = "192.168.10"
    """IP network prefix for the Docker network (e.g., '192.168.10')."""
    CHIMERA_NETWORK_SUBNET_MASK: Annotated[int, NoDecode] = 24
    """Subnet mask for the Docker network (e.g., 24)."""

    @field_validator("CHIMERA_NETWORK_PREFIX", mode="before")
    @classmethod
    def parse_network_prefix(cls, v: Any) -> str:
        """Parses the network prefix, handling string and integer inputs."""
        if isinstance(v, str):
            return v
        return str(v)

    @field_validator("CHIMERA_NETWORK_PREFIX")
    @classmethod
    def validate_network_prefix(cls, v: str) -> str:
        """Validates that the network prefix is a valid IP network."""
        try:
            ipaddress.ip_network(v + ".0", strict=False)
            return v
        except ValueError:
            raise ValueError("Invalid network prefix.")

    @field_validator("CHIMERA_NETWORK_SUBNET_MASK", mode="before")
    @classmethod
    def parse_subnet_mask(cls, v: Any) -> int:
        """Parses the subnet mask, handling string and integer inputs."""
        if isinstance(v, int):
            return v
        return int(v)

    @field_validator("CHIMERA_NETWORK_SUBNET_MASK")
    @classmethod
    def validate_subnet_mask(cls, v: int) -> int:
        """Validates that the subnet mask is between 0 and 32."""
        if not 0 <= v <= 32:
            raise ValueError("Subnet mask must be between 0 and 32.")
        return v


class WorkersConfig(BaseSettings):
    """
    Configuration settings for Chimera workers.
    """

    CHIMERA_WORKERS_NODES_NAMES: Annotated[List[str], NoDecode]
    """List of names for the Chimera worker nodes."""
    CHIMERA_WORKERS_CPU_SHARES: Annotated[List[int], NoDecode] = [2]
    """List of CPU shares for each worker node."""
    CHIMERA_WORKERS_MAPPED_PORTS: Annotated[List[int], NoDecode] = [101]
    """List of host ports to map to each worker's container port."""
    CHIMERA_WORKERS_HOST: str = "0.0.0.0"
    """Host IP address to bind the worker's port to."""
    CHIMERA_WORKERS_PORT: int = 80
    """Container port for the Chimera workers."""
    CHIMERA_WORKERS_ENDPOINTS_MAX_RETRIES: int = 0
    """Maximum number of retries for worker endpoints."""
    CHIMERA_WORKERS_ENDPOINTS_TIMEOUT: float = 100.0
    """Timeout for worker endpoints."""

    @field_validator(
        "CHIMERA_WORKERS_NODES_NAMES",
        "CHIMERA_WORKERS_CPU_SHARES",
        "CHIMERA_WORKERS_MAPPED_PORTS",
        mode="before",
    )
    @classmethod
    def parse_lists(cls, v: Any) -> List[str]:
        """Parses lists from string representations."""
        if isinstance(v, str):
            return ast.literal_eval(v)
        return v

    @field_validator("CHIMERA_WORKERS_CPU_SHARES")
    @classmethod
    def validate_cpu_shares(cls, v: List[int]) -> List[int]:
        """Validates CPU shares: must be integers >= 2."""
        if not all(isinstance(share, int) and share >= 2 for share in v):
            raise ValueError(
                "All CPU_SHARES values must be integers greater than or equal to 2."
            )
        return v

    @field_validator("CHIMERA_WORKERS_MAPPED_PORTS")
    @classmethod
    def validate_port_uniqueness(cls, v: List[int]) -> List[int]:
        """Validates that mapped ports are unique."""
        if len(v) != len(set(v)):
            raise ValueError("Mapped ports must be unique.")
        return v

    @field_validator("CHIMERA_WORKERS_NODES_NAMES")
    @classmethod
    def validate_node_name_uniqueness(cls, v: List[str]) -> List[str]:
        """Validates that node names are unique."""
        if len(v) != len(set(v)):
            raise ValueError("Node names must be unique.")
        return v
