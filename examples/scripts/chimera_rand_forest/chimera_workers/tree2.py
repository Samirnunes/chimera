from sklearn.tree import DecisionTreeRegressor

from chimera.nodes.workers import RegressionWorker

worker = RegressionWorker(
    DecisionTreeRegressor(max_depth=2, max_leaf_nodes=5), bootstrap=True
)

if __name__ == "__main__":
    worker.serve()
