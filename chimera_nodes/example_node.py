from sklearn.tree import DecisionTreeRegressor

from chimera.parallelism.hybrid import BootstrapRegressionNode

node = BootstrapRegressionNode(DecisionTreeRegressor())

if __name__ == "__main__":
    node.serve()
