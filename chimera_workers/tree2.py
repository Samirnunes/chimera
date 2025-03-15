from sklearn.tree import DecisionTreeClassifier

from chimera.nodes.workers import ClassificationWorker

worker = ClassificationWorker(
    DecisionTreeClassifier(max_depth=5, max_leaf_nodes=15), bootstrap=True
)

if __name__ == "__main__":
    worker.serve()
