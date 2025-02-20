from chimera import Chimera
import os

os.environ["CHIMERA_WORKERS_NODES_NAMES"] = '["example_node"]'

Chimera().serve("ensemble", 8100)
