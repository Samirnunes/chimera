from sklearn.tree import DecisionTreeRegressor

from chimera.parallelism.hybrid import BootstrapRegressionNode

node = BootstrapRegressionNode(DecisionTreeRegressor(max_depth=2, max_leaf_nodes=5))

if __name__ == "__main__":
    node.serve()
