import os

from chimera.api.container import TRAIN_FEATURES_FILENAME, TRAIN_LABELS_FILENAME

from ..api import DATA_FOLDER, DOCKERFILE_NAME
from .config import NetworkConfig, WorkersConfig


class WorkersServersHandler:
    def __init__(self) -> None:
        self._network_config = NetworkConfig()
        self._workers_config = WorkersConfig()

    def serve_all(self) -> None:
        if len(self._workers_config.CHIMERA_WORKERS_NODES_NAMES) != len(
            self._workers_config.CHIMERA_WORKERS_CPU_SHARES
        ) or len(self._workers_config.CHIMERA_WORKERS_NODES_NAMES) != len(
            self._workers_config.CHIMERA_WORKERS_HOST_PORTS
        ):
            raise ValueError(
                "Number of nodes, number of hosts names and CPU relative weights must be equal"
            )
        if any(
            [
                not (isinstance(cpu_shares, int) and cpu_shares >= 2)
                for cpu_shares in self._workers_config.CHIMERA_WORKERS_CPU_SHARES
            ]
        ):
            raise ValueError(
                "All CPU_SHARES values must be integers and greater than or equal to 2."
            )

        self._create_network()

        for i in range(len(self._workers_config.CHIMERA_WORKERS_NODES_NAMES)):
            self._build_docker_image(
                self._workers_config.CHIMERA_WORKERS_NODES_NAMES[i],
                self._workers_config.CHIMERA_WORKERS_HOST_PORTS[i],
            )
            self._run_container(
                self._workers_config.CHIMERA_WORKERS_NODES_NAMES[i],
                self._workers_config.CHIMERA_WORKERS_CPU_SHARES[i],
                self._workers_config.CHIMERA_WORKERS_HOST_PORTS[i],
                i,
            )

    def _create_network(self) -> None:
        cmd = [
            "docker",
            "network",
            "create",
            "--driver=bridge",
            f"--subnet={self._network_config.CHIMERA_NETWORK_PREFIX}.0/{self._network_config.CHIMERA_NETWORK_SUBNET_MASK}",
            f"--gateway={self._network_config.CHIMERA_NETWORK_PREFIX}.1",
            self._network_config.CHIMERA_NETWORK_NAME,
        ]
        print(os.popen(" ".join(cmd)).read())

    def _build_docker_image(self, node_name: str, host_port: int) -> None:
        image_name = node_name
        cmd = [
            "docker",
            "build",
            "--build-arg",
            f"NODE={node_name}",
            "--build-arg",
            f"PORT={host_port}",
            "--build-arg",
            f"DATA_FOLDER={DATA_FOLDER}",
            "--build-arg",
            f"TRAIN_FEATURES_FILENAME={TRAIN_FEATURES_FILENAME}",
            "--build-arg",
            f"TRAIN_LABELS_FILENAME={TRAIN_LABELS_FILENAME}",
            "--build-arg",
            f"CHIMERA_WORKERS_NODES_NAMES={self._workers_config.CHIMERA_WORKERS_NODES_NAMES}",
            "--build-arg",
            f"CHIMERA_WORKERS_CPU_SHARES={self._workers_config.CHIMERA_WORKERS_CPU_SHARES}",
            "--build-arg",
            f"CHIMERA_WORKERS_HOST_PORTS={self._workers_config.CHIMERA_WORKERS_HOST_PORTS}",
            "--build-arg",
            f"CHIMERA_WORKERS_CONTAINER_PORT={self._workers_config.CHIMERA_WORKERS_CONTAINER_PORT}",
            "--build-arg",
            f"CHIMERA_WORKERS_HOST={self._workers_config.CHIMERA_WORKERS_HOST}",
            "--network",
            "host",
            "-f",
            DOCKERFILE_NAME,
            "-t",
            image_name,
            ".",
        ]
        print(os.popen(" ".join(cmd)).read())

    def _run_container(
        self, node_name: str, cpu_shares: int, host_port: int, node_number: int
    ) -> None:
        container_name = node_name
        image_name = node_name
        container_ip = (
            f"{self._network_config.CHIMERA_NETWORK_PREFIX}.{node_number + 2}"
        )

        cmd = [
            "docker",
            "run",
            "-d",
            "-p",
            f"{host_port}:{self._workers_config.CHIMERA_WORKERS_CONTAINER_PORT}",
            "--name",
            container_name,
            "--network",
            self._network_config.CHIMERA_NETWORK_NAME,
            "--ip",
            container_ip,
            "--cpu-shares",
            str(cpu_shares),
            image_name,
        ]
        print(os.popen(" ".join(cmd)).read())
