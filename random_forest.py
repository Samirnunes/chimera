import os

from chimera import Chimera

os.environ["CHIMERA_WORKERS_NODES_NAMES"] = '["tree1", "tree2", "tree3", "tree4"]'
os.environ["CHIMERA_WORKERS_CPU_SHARES"] = "[2, 2, 2, 2]"
os.environ["CHIMERA_WORKERS_MAPPED_PORTS"] = "[81, 82, 83, 84]"


Chimera().serve("aggregation", 8082)
