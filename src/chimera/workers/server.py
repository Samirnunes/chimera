import os
import subprocess

from .config import NETWORK_CONFIG, WORKERS_CONFIG


class WorkerServerHandler:
    def serve_all(
        self,
        dockerfile_name: str,
    ) -> None:
        if len(WORKERS_CONFIG.CHIMERA_WORKERS_NODES_NAMES) != len(
            WORKERS_CONFIG.CHIMERA_WORKERS_CPU_SHARES
        ) or len(WORKERS_CONFIG.CHIMERA_WORKERS_NODES_NAMES) != len(
            WORKERS_CONFIG.CHIMERA_WORKERS_HOST_PORTS
        ):
            raise ValueError(
                "Number of nodes, number of hosts names and CPU relative weights must be equal"
            )
        if any(
            [
                not (isinstance(cpu_shares, int) and cpu_shares >= 2)
                for cpu_shares in WORKERS_CONFIG.CHIMERA_WORKERS_CPU_SHARES
            ]
        ):
            raise ValueError(
                "All CPU_SHARES values must be integers and greater than or equal to 2."
            )

        self._create_network()

        for i in range(len(WORKERS_CONFIG.CHIMERA_WORKERS_NODES_NAMES)):
            self._build_docker_image(
                WORKERS_CONFIG.CHIMERA_WORKERS_NODES_NAMES[i],
                WORKERS_CONFIG.CHIMERA_WORKERS_HOST_PORTS[i],
                dockerfile_name,
            )
            self._run_container(
                WORKERS_CONFIG.CHIMERA_WORKERS_NODES_NAMES[i],
                WORKERS_CONFIG.CHIMERA_WORKERS_CPU_SHARES[i],
                WORKERS_CONFIG.CHIMERA_WORKERS_HOST_PORTS[i],
                i,
            )

    def _create_network(self) -> None:
        cmd = [
            "docker",
            "network",
            "create",
            "--driver=bridge",
            f"--subnet={NETWORK_CONFIG.CHIMERA_NETWORK_PREFIX}.0/{NETWORK_CONFIG.CHIMERA_NETWORK_SUBNET_MASK}",
            f"--gateway={NETWORK_CONFIG.CHIMERA_NETWORK_PREFIX}.1",
            NETWORK_CONFIG.CHIMERA_NETWORK_NAME,
        ]
        print(os.popen(" ".join(cmd)).read())

    def _build_docker_image(
        self, node_name: str, host_port: int, dockerfile_name: str
    ) -> None:
        image_name = node_name
        cmd = [
            "docker",
            "build",
            "--build-arg",
            f"NODE={node_name}",
            "--build_arg",
            f"PORT={host_port}",
            "--network",
            "host",
            "-f",
            dockerfile_name,
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
        container_ip = f"{NETWORK_CONFIG.CHIMERA_NETWORK_PREFIX}.{node_number + 2}"

        cmd = [
            "docker",
            "run",
            "-d",
            "-p",
            f"{host_port}:{WORKERS_CONFIG.CHIMERA_WORKERS_CONTAINER_PORT}",
            "--name",
            container_name,
            "--network",
            NETWORK_CONFIG.CHIMERA_NETWORK_NAME,
            "--ip",
            container_ip,
            "--cpu-shares",
            str(cpu_shares),
            image_name,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Container {container_name} started: {result.stdout.strip()}")
        else:
            print(f"Failed to start {container_name}: {result.stderr}")
