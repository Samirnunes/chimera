from chimera.nodes.workers import SGDWorker

node = SGDWorker("regressor", eta0=1e-7)

if __name__ == "__main__":
    node.serve()
