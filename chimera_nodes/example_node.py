from sklearn.tree import DecisionTreeRegressor

from chimera.parallelism.data import RegressionNode

node = RegressionNode(DecisionTreeRegressor())

if __name__ == "__main__":
    node.serve()
