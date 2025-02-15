import os
import subprocess
from typing import List

from .config import ContainerConfig, NetworkConfig


class WorkersRunner:
    _DOCKERFILE_NAME = "Dockerfile.worker"
    _IMAGE_NAME = "chimera-worker:latest"

    def __init__(self) -> None:
        self._network_config = NetworkConfig()
        self._container_config = ContainerConfig()

    def run(self, workers: int, cpu_shares: List[int] = []) -> None:
        if not cpu_shares:
            cpu_shares = [2] * workers

        if workers != len(cpu_shares):
            raise ValueError(
                "Number of nodes and CPU relative weights must be equal"
            )

        self._create_network()
        self._build_docker_image()

        for i in range(workers):
            self._run_container(f"node_{i}", cpu_shares[i], i)

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

    def _build_docker_image(self) -> None:
        cmd = [
            "docker",
            "build",
            "--network",
            "host",
            "-f",
            self._DOCKERFILE_NAME,
            "-t",
            self._IMAGE_NAME,
            ".",
        ]
        print(os.popen(" ".join(cmd)).read())

    def _run_container(
        self, container_name: str, cpu_shares: int, node_number: int
    ) -> None:
        """Run a Docker container using CLI"""

        host_port = 8080 + node_number
        container_ip = (
            f"{self._network_config.CHIMERA_NETWORK_PREFIX}.{node_number + 2}"
        )

        cmd = [
            "docker",
            "run",
            "-d",
            "-p",
            f"{host_port}:{self._container_config.CHIMERA_WORKERS_CONTAINER_PORT}",
            "--name",
            container_name,
            "--network",
            self._network_config.CHIMERA_NETWORK_NAME,
            "--ip",
            container_ip,
            "--cpu-shares",
            str(cpu_shares),
            self._IMAGE_NAME,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Container {container_name} started: {result.stdout.strip()}")
        else:
            print(f"Failed to start {container_name}: {result.stderr}")
