import os

import chimera
from chimera.nodes import ParameterServerMaster

os.environ["CHIMERA_WORKERS_NODES_NAMES"] = '["sgd1", "sgd2"]'
os.environ["CHIMERA_WORKERS_CPU_SHARES"] = "[2, 2]"
os.environ["CHIMERA_WORKERS_MAPPED_PORTS"] = "[81, 82]"

chimera.run(ParameterServerMaster("regressor", max_iter=100), 8080)
