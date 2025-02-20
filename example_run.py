import os

from chimera import Chimera

os.environ["CHIMERA_WORKERS_NODES_NAMES"] = '["example_node"]'

Chimera().serve("ensemble")
