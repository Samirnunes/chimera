import os

import chimera
from chimera.nodes import ParameterServerMaster

os.environ["CHIMERA_WORKERS_NODES_NAMES"] = (
    '["sgd1", "sgd2", "sgd3", "sgd4", "sgd5", "sgd6", "sgd7"]'
)
os.environ["CHIMERA_WORKERS_CPU_SHARES"] = "[2, 2, 2, 2, 2, 2, 2]"
os.environ["CHIMERA_WORKERS_MAPPED_PORTS"] = "[81, 82, 83, 84, 85, 86, 87]"

chimera.run(ParameterServerMaster("classifier", max_iter=200), 8080)
