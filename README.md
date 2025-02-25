# Chimera
`Chimera` is a Python package for distributed machine learning (DML).

## The Package

It supports the following types of DML:

- Data Parallelism: data distributed between the workers. Each worker has a copy of the model.
- Model Parallelism: model distributed between the workers. Each worker has a copy of the dataset. This case includes Distributed SGD (Stochastic Gradient Descent) for generic neural network architectures.
- Hybrid Parallelism: data and model distributed between the workers. This case includes Distributed Ensemble Learning with generic weak learners from the `scikit-learn` package.

The implementation is Docker-based, that is, it uses docker containers to act as workers. To run the created distributed system, it will be given an standardized interface which will leverage REST API servers, on whose backend workers will run.

The communication between workers is made by message passing via the TCP protocol.

## Some Implementations Examples

![Random Forest](../images/random_forest.png)
