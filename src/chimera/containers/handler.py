import subprocess
from pathlib import Path

from ..utils import status_logger
from .configs import (
    CHIMERA_DOCKERFILE_NAME,
    CHIMERA_TRAIN_DATA_FOLDER,
    CHIMERA_TRAIN_FEATURES_FILENAME,
    CHIMERA_TRAIN_LABELS_FILENAME,
    CHIMERA_WORKERS_FOLDER,
    NetworkConfig,
    WorkersConfig,
)


class ContainersHandler:
    """
    Manages the creation and execution of Docker containers for Chimera workers.
    """

    def __init__(self) -> None:
        """
        Initializes the ContainersHandler with NetworkConfig and WorkersConfig.
        """
        self._network_config = NetworkConfig()
        self._workers_config = WorkersConfig()

    def serve_all(self) -> None:
        """
        Creates the Chimera network and starts all worker containers.

        This method performs the following actions:
        1. Creates the Docker network specified in NetworkConfig.
        2. Iterates through the worker configurations in WorkersConfig.
        3. Builds the Docker image for each worker.
        4. Runs the Docker container for each worker.
        5. Adds DNS entries to each worker's /etc/hosts file to allow inter-container communication.
        """
        self._create_network()

        for i in range(len(self._workers_config.CHIMERA_WORKERS_NODES_NAMES)):
            self._build_docker_image(i)
            self._run_container(i)
            self._add_dns_entries_to_container(i)

    def _create_network(self) -> None:
        """
        Creates the Docker network if it doesn't already exist.

        This method checks if a network with the name specified in NetworkConfig exists.
        If not, it creates a new bridge network with the specified subnet and gateway.
        """
        check_cmd = [
            "docker",
            "network",
            "ls",
            "--filter",
            f"name={self._network_config.CHIMERA_NETWORK_NAME}",
            "--format",
            "{{.Name}}",
        ]

        result = subprocess.run(check_cmd, capture_output=True, text=True)

        if self._network_config.CHIMERA_NETWORK_NAME in result.stdout.split():
            status_logger.info(
                f"Network {self._network_config.CHIMERA_NETWORK_NAME} already exists. Skipping creation."
            )
            return

        cmd = [
            "docker",
            "network",
            "create",
            "--driver=bridge",
            f"--subnet={self._network_config.CHIMERA_NETWORK_PREFIX}.0/{self._network_config.CHIMERA_NETWORK_SUBNET_MASK}",
            f"--gateway={self._network_config.CHIMERA_NETWORK_PREFIX}.1",
            self._network_config.CHIMERA_NETWORK_NAME,
        ]
        subprocess.run(cmd, check=True)
        status_logger.info(
            f"Successfully created '{self._network_config.CHIMERA_NETWORK_NAME}' network."
        )

    def _build_docker_image(self, i: int) -> None:
        """
        Builds the Docker image for a specific worker.

        Args:
            i: The index of the worker in the WorkersConfig list.
        """
        node_name = self._workers_config.CHIMERA_WORKERS_NODES_NAMES[i]
        image_name = node_name
        cmd = [
            "docker",
            "build",
            "--build-arg",
            f"CHIMERA_WORKERS_NODE_NAME={node_name}",
            "--build-arg",
            f"CHIMERA_WORKERS_FOLDER={CHIMERA_WORKERS_FOLDER}",
            "--build-arg",
            f"CHIMERA_DATA_FOLDER={CHIMERA_TRAIN_DATA_FOLDER}",
            "--build-arg",
            f"TRAIN_FEATURES_FILENAME={CHIMERA_TRAIN_FEATURES_FILENAME}",
            "--build-arg",
            f"TRAIN_LABELS_FILENAME={CHIMERA_TRAIN_LABELS_FILENAME}",
            "--build-arg",
            f"CHIMERA_WORKERS_NODES_NAMES={self._workers_config.CHIMERA_WORKERS_NODES_NAMES}",
            "--build-arg",
            f"CHIMERA_WORKERS_CPU_SHARES={self._workers_config.CHIMERA_WORKERS_CPU_SHARES}",
            "--build-arg",
            f"CHIMERA_WORKERS_MAPPED_PORTS={self._workers_config.CHIMERA_WORKERS_MAPPED_PORTS}",
            "--build-arg",
            f"CHIMERA_WORKERS_PORT={self._workers_config.CHIMERA_WORKERS_PORT}",
            "--build-arg",
            f"CHIMERA_WORKERS_HOST={self._workers_config.CHIMERA_WORKERS_HOST}",
            "-f",
            str(Path(__file__).resolve().parent / CHIMERA_DOCKERFILE_NAME),
            "-t",
            image_name,
            ".",
        ]
        subprocess.run(cmd, check=True)
        status_logger.info(f"Successfully built '{image_name}' docker image.")

    def _run_container(self, i: int) -> None:
        """
        Runs the Docker container for a specific worker.

        Args:
            i: The index of the worker in the WorkersConfig list.
        """
        node_name = self._workers_config.CHIMERA_WORKERS_NODES_NAMES[i]
        container_name = node_name
        image_name = node_name
        cpu_shares = self._workers_config.CHIMERA_WORKERS_CPU_SHARES[i]
        host_port = self._workers_config.CHIMERA_WORKERS_MAPPED_PORTS[i]
        container_ip = f"{self._network_config.CHIMERA_NETWORK_PREFIX}.{i + 2}"

        cmd = [
            "docker",
            "run",
            "-d",
            "-p",
            f"{self._workers_config.CHIMERA_WORKERS_HOST}:{host_port}:{self._workers_config.CHIMERA_WORKERS_PORT}/tcp",
            "--name",
            container_name,
            "--network",
            self._network_config.CHIMERA_NETWORK_NAME,
            "--ip",
            container_ip,
            "--cpu-shares",
            str(cpu_shares),
            "--add-host",
            self._workers_config.CHIMERA_WORKERS_NODES_NAMES[i] + ":" + container_ip,
            image_name,
        ]
        subprocess.run(cmd, check=True)
        status_logger.info(f"Successfully ran '{container_name}' container.")

    def _add_dns_entries_to_container(self, i: int) -> None:
        """
        Adds DNS entries for all workers to a specific worker's /etc/hosts file.

        Args:
            i: The index of the worker in the WorkersConfig list.
        """
        container_name = self._workers_config.CHIMERA_WORKERS_NODES_NAMES[i]
        for j in range(len(self._workers_config.CHIMERA_WORKERS_NODES_NAMES)):
            if i == j:
                continue
            other_container_ip = (
                f"{self._network_config.CHIMERA_NETWORK_PREFIX}.{j + 2}"
            )
            other_dns_name = self._workers_config.CHIMERA_WORKERS_NODES_NAMES[j]

            cmd = [
                "docker",
                "exec",
                container_name,
                "sh",
                "-c",
                f"echo '{other_container_ip} {other_dns_name}' >> /etc/hosts",
            ]
            subprocess.run(cmd, check=True)
            status_logger.info(
                f"Added DNS entry '{other_dns_name}:{other_container_ip}' to container '{container_name}'."
            )
