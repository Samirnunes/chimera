# Chimera
`Chimera` is a Python package for distributed machine learning (DML).

## The Package

It supports the following types of DML:

- Data Parallelism: data distributed between the workers. Each worker has a copy of the model.
- Model Parallelism: model distributed between the workers. Each worker has a copy of the dataset. This case includes Distributed SGD (Stochastic Gradient Descent) for generic neural network architectures.
- Hybrid Parallelism: data and model distributed between the workers. This case includes Distributed Ensemble Learning with generic weak learners from the `scikit-learn` package.

The implementation is Docker-based, that is, it uses docker containers to act as workers. To run the created distributed system, it will be given an standardized interface which will leverage REST API servers, on whose backend workers will run.

The client-master and master-workers communications are made via REST APIs. Besides, the workers-workers communications are made using message passing via the TCP protocol in the transport layer.

## Examples

### Distributed Bagging (Bootstrap Aggregating)

<p align="center">
    <img width="900" src="./images/distributed_bagging.png" alt="Distributed Bagging">
<p>

### Distributed SGD (Stochastic Gradient Descent)

<p align="center">
    <img width="900" src="./images/distributed_sgd.png" alt="Distributed SGD">
<p>
