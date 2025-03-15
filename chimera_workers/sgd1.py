from chimera.nodes.workers import SGDWorker

worker = SGDWorker("regressor", eta0=1e-7)

if __name__ == "__main__":
    worker.serve()
