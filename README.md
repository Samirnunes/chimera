# Chimera: A Framework for Education and Prototyping in Distributed Machine Learning

<p align="center">
    <img width="500" src="./images/logo.jpg" alt="Chimera Logo">
<p>

## Introduction

Distributed Machine Learning arises mainly in two situations: (1) when the runtime of training a model is very high, and (2) when a centralized solution is not feasible due to the size of the data, which cannot be stored on a single machine (Verbraeken et al., 2020). The first situation requires increased parallelization and higher I/O bandwidth, which can be achieved through distributed systems. The second one demands distributed storage, which is addressed by spreading data across multiple machines. These challenges are particularly relevant in large enterprises, such as big tech companies, where data is stored in different locations and processing is carried out by coordinating multiple machines in a distributed system.

Verbraeken et al. explain that, to make large datasets accessible as training data for machine learning, various algorithms must be selected and implemented to support parallel computation, data distribution, and resilience to failures. In particular, parallel computation is essential to reduce runtime, making otherwise infeasible problems solvable. As previously mentioned, data distribution addresses the large dataset issue and can be effectively handled through a distributed computing environment. Finally, resilience to failures is a critical attribute of distributed systems and can be achieved in several ways—for example, by incorporating a load balancer to ensure the system remains operational even if some workers fail.

Considering this context, `chimera` is a Python package for distributed machine learning (DML) designed for both educational and prototyping purposes. It provides a structured environment to experiment with key DML techniques, including Data Parallelism, Model Parallelism, and Hybrid Parallelism.

As a distributed computing framework, `chimera` aims to simplify the creation, in a local environment, of distributed machine learning models by streamlining the creation of a master node on the host machine and worker nodes on separate virtual machines using Docker containers. By providing a standardized API-based communication framework, `chimera` enables researchers and practitioners to test, evaluate, and optimize distributed learning algorithms with minimal configuration effort.

`chimera` supports the following types of DML techniques, whose explanations are detailed below (B. Chatterjee, 2024):

- Data Parallelism: data distributed between the workers. Each worker has a copy of the model. This case includes Distributed SGD (Stochastic Gradient Descent) for models like linear regression, logistic regression and others, depending on the loss function.

- Model Parallelism: model distributed between the workers. Each worker has a copy of the dataset. This case includes Distributed SGD (Stochastic Gradient Descent) for generic neural network architectures.

- Hybrid Parallelism: data and model distributed between the workers. This case includes Distributed Bagging (Bootstrap Aggregating) with generic weak learners from the `scikit-learn` package.

Docker containers act as Workers. To run the created distributed system, it will be given a standardized function named `run`, on which a Master type and a port must be selected for the server in the host machine. The `run` function starts the Chimera master server and handles worker containers, then initializing the necessary components for the distributed system to work.

The client-master and master-workers communications are made via REST APIs.

## Running as a Pypi Package

1. Install Poetry following the documentation: https://python-poetry.org/docs/#installing-with-the-official-installer

2. Initialize a virtual environment running the command `poetry init`

3. Install the latest version of `chimera` running the command `poetry add chimera-distributed-ml`

4. Start the Docker Daemon. You can make it either by opening Docker Desktop or by starting the Daemon via CLI (in Linux: `sudo systemctl start docker`). Docker Daemon makes Docker REST APIs available, so we can run commands like `docker build` and `docker run`, that are called internally by `chimera`.

5. Create and run distributed models with `chimera`!

## Running the Source Code

1. Install Poetry following the documentation: https://python-poetry.org/docs/#installing-with-the-official-installer

2. Clone the `chimera` project via either HTTPS or SSH:
   - HTTPS: `git clone https://github.com/Samirnunes/chimera.git`
   - SSH: `git clone git@github.com:Samirnunes/chimera.git`

3. Go to project's root directory (where `pyproject.toml` is located) and run `poetry install`. It will generate a `.venv` file in the root directory with the installed dependencies, and a `poetry.lock` file.

4. Start the Docker Daemon. You can make it either by opening Docker Desktop or by starting the Daemon via CLI (in Linux: `sudo systemctl start docker`). Docker Daemon makes Docker REST APIs available, so we can run commands like `docker build` and `docker run`, that are called internally by `chimera`.

5. Create and run distributed models with `chimera`!

## Creating and Running a Distributed Model with `chimera`

<p align="center">
    <img width="250" src="./images/chimera_files.png" alt="Chimera Files">
<p>
<p align="center"><strong>Figure:</strong> Example of Chimera files. </p>

1. After installing `chimera`, you need to create a `Master` and its `Workers`:
     - Master: create a `.py` file in your root directory. This file must specify the environment variables necessary to run the code in string format (in the case of Lists, you must follow the JSON string format for Lists) and run a `chimera` master server with `chimera.run`. For example: `chimera.run(AggregationMaster(), 8080)`. The available configuration environment variables are in the classes `NetworkConfig` and `WorkersConfig`, inside `src/chimera/containers/config.py`.

    <p align="center">
        <img width="600" src="./images/master_example.png" alt="Master Example">
    <p>
    <p align="center"><strong>Figure:</strong> Example of a master's file. </p>

     - Workers: create a folder called `chimera_workers` and create `.py` files which are going to represent your workers. Each file must initialize a `chimera` worker and call `worker.serve()` inside an `if __name__ == "__main__":` block, which will initialize the worker server when `chimera.run` is called in the master's file. Note that the environment variable `CHIMERA_WORKERS_NODES_NAMES` in the master's file must contain all the workers' file names, without the `.py` suffix.

    <p align="center">
        <img width="650" src="./images/worker_example.png" alt="Worker Example">
    <p>
    <p align="center"><strong>Figure:</strong> Example of a worker's file. </p>

2. Before running the master's file, you must specify the local training dataset for each worker. This is made by creating a folder called `chimera_train_data` containing folders with the same name as the worker's files (clearly without the `.py`). Each folder must have a `X_train.csv` file containing the features and a `y_train.csv` containing the labels. Whether `X_train.csv` and `y_train.csv` are the same or not for all the workers is up to you. Keep in mind what algorithm you want to create in the distributed environment!

3. Finally, you can run the master's file using: `poetry run python {your_master_filename.py}`. This should initialize all the worker's containers in your Docker environment and the master server in the host machine (the machine running the code).

<p align="center">
    <img width="900" src="./images/client/client_interactions.png" alt="Client interactions">
<p>
<p align="center"><strong>Figure:</strong> General Architecture for a Chimera Distributed System. It summarizes how to create a distributed model with Chimera. </p>

## Environment Variables

The following environment variables allow users to configure the `chimera` distributed machine learning system. These variables define network settings, worker configurations, and resource allocations, ensuring flexibility to different environments.

### Network Configuration

The following variables define the Docker network settings for `chimera`:

- `CHIMERA_NETWORK_NAME` (default: `"chimera-network"`)
        - The name of the Docker network where `chimera` runs.

- `CHIMERA_NETWORK_PREFIX` (default: `"192.168.10"`)
        - The IP network prefix for the Docker network.
        - Must be a valid IPv4 network prefix (e.g., `"192.168.10"`).

- `CHIMERA_NETWORK_SUBNET_MASK` (default: `24`)
        - The subnet mask for the Docker network, defining how many bits are reserved for the network.
        - Must be an integer between `0` and `32`.

### Workers Configuration

The following variables control the behavior of worker nodes in `chimera`:

- `CHIMERA_WORKERS_NODES_NAMES`
    - A list of worker node names.
    - Must be unique across all workers.
    - Example: `["worker1", "worker2", "worker3"]`.

- `CHIMERA_WORKERS_CPU_SHARES` (default: `[2]`)
    - A list of CPU shares assigned to each worker.
    - Each value must be an integer ≥ `2`.
    - Example: `[2, 4, 4]` assigns different CPU shares to three workers.

- `CHIMERA_WORKERS_MAPPED_PORTS` (default: `[101]`)
    - A list of host ports mapped to each worker’s container.
    - Must be unique across all workers.
    - Example: `[5001, 5002, 5003]` assigns distinct ports to three workers.

- `CHIMERA_WORKERS_HOST` (default: `"0.0.0.0"`)
    - The host IP address that binds worker ports.
    - `"0.0.0.0"` allows connections from any IP address.

- `CHIMERA_WORKERS_PORT` (default: `80`)
    - The internal container port that workers listen on.
    - This is the port inside the worker's container, not the exposed host port.

- `CHIMERA_WORKERS_ENDPOINTS_MAX_RETRIES` (default: `0`)
    - The maximum number of retry attempts when communicating with worker nodes.

- `CHIMERA_WORKERS_ENDPOINTS_TIMEOUT` (default: `100.0`)
    - The timeout (in seconds) for worker API endpoints.

These environment variables give users full control over how `chimera` distributes models, manages worker nodes, and configures networking in a flexible and simple manner.

## Logging

The framework uses two dedicated loggers to track system's behavior and latency metrics:

- Status Logger (`chimera_status`):
  Logs general status messages related to the system's operations, such as workflow progress, key events, and high-level actions. The logs are saved in the file `chimera_status.log`.

- Time Logger (`chimera_time`):
  Logs latency metrics, then, it's useful for monitoring and debugging time efficiency. These logs are stored in the file `chimera_time.log`.

Both loggers are configured using Python’s built-in logging module, and log messages at the INFO level. Each logger writes to its respective log file through a `FileHandler`.

## Examples

For more examples, see: https://github.com/Samirnunes/chimera-examples

### Hybrid Parallelism: Distributed Bagging (Bootstrap Aggregating)

In distributed bagging, the summarized steps are:

1. Client makes a request to Aggregation Master, which redirects it to Bootstrap Workers. Each worker contains a weak learner from the `scikit-learn` package.

2. Each Bootstrap Worker receives the request for an action:

    - fit: trains the local weak learner using the local dataset. Before fit, Worker bootstraps (samples with reposition) the local dataset. Then, it uses the collected samples to fit the local model. When the process is finished, Master sends an "ok" to the Client.

    - predict: makes inference on new data by calculating, in the Master, the mean of the predictions of each Worker's local model's predictions.

The following state machine flowchart depicts the steps in the fit action for the Aggregation Master:

<p align="center">
    <img width="600" src="./images/master/aggregation_master_fit_state_machine_flowchart.png" alt="Distributed Bagging">
<p>
<p align="center"><strong>Figure:</strong> State machine flowchart of Aggregation Master's /v1/chimera/aggregation/fit endpoint. </p>

The fitting process begins at step A, where nothing has happened yet. When the client sends a fit request to the endpoint `/v1/chimera/aggregation/fit`, at step B, the Master receives the request and broadcasts it to all workers via the endpoint `/v1/chimera/model/fit`, as shown in step C.

Then, concurrently, the Master waits for either a response, a timeout, or the maximum number of retries from each worker—this is represented by step D. If a worker times out or reaches the retry limit, it is considered failed. In the conditional step E, if at least one worker responds successfully, the flow proceeds to step G. There, the Master receives an "ok" message with status code 200 from the responsive workers, indicating that their local models have been successfully fitted using their training data. Finally, in step H, the Master sends a JSON-formatted response to the client via the endpoint `/v1/chimera/aggregation/fit`, containing an "ok" message to confirm that the distributed model is now fitted. The flow then ends at step I.

If all workers fail in step E, the flow moves to step F, where the Master returns a JSON-formatted error message to the client via the endpoint `/v1/chimera/aggregation/fit`, indicating the error. Finally, the flow ends at step I.

The following state machine flowchart depicts the steps in the predict action for the Aggregation Master:

<p align="center">
    <img width="600" src="./images/master/aggregation_master_predict_state_machine_flowchart.png" alt="Distributed Bagging">
<p>
<p align="center"><strong>Figure:</strong> State machine flowchart of Aggregation Master's /v1/chimera/aggregation/predict endpoint. </p>

The prediction process begins at step A, where nothing has happened yet. When the client sends a prediction request to the endpoint `/v1/chimera/aggregation/predict`, at step B, the Master receives the request and broadcasts it to all workers via the endpoint `/v1/chimera/model/predict`, as shown in step C.

Then, concurrently, the Master waits for either a response, a timeout, or the maximum number of retries from each worker—this is represented by step D. If a worker times out or reaches the retry limit, it is considered failed. In the conditional step E, if at least one worker has responded successfully, the flow proceeds to step G. There, the Master receives results from the responsive workers—status code 200 responses containing predictions from each local model. In step H, the Master aggregates these results by computing the arithmetic mean, and in step I, it returns a JSON-formatted response to the client via the endpoint `/v1/chimera/aggregation/predict`, containing the final predictions. The flow then ends at step J.

If all workers fail in step E, the flow goes to step F, where the Master returns a JSON-formatted error message to the client via the endpoint `/v1/chimera/aggregation/predict`, indicating the issue. Finally, the flow ends at step J.

### Data Parallelism: Distributed SGD (Stochastic Gradient Descent)

In distributed SGD, the summarized steps are:

1. Client makes a request to Parameter Server Master, which redirects it to SGD Workers.

2. Each SGD Worker receives the request for an action:

    - fit: trains the distributed model. Worker has a copy of the model on its memory. Then, for a predefined number of iterations or until convergence:
        - 1. Worker calculates the gradient considering only its local dataset;
        - 2. Worker communicates through REST API its gradient to Master, which aggregates the gradients by calculating the mean, updates the model's parameters and passes these parameters back to each Worker through REST API, so they update their local models.

    When convergence is reached, Master stops sending the parameters to Workers and stores the final model. Finally, it communicates an "ok" to Client.

    - predict: makes inference on new data using the final model available in the Master.

The following state machine flowchart depicts the steps in the fit action for the Parameter Server Master:

<p align="center">
    <img width="600" src="./images/master/parameter_server_master_fit_state_machine_flowchart.png" alt="Distributed SGD">
<p>
<p align="center"><strong>Figure:</strong> State machine flowchart of Parameter Server Master's /v1/chimera/parameter-server/fit endpoint. </p>

The fitting process begins at step A, where nothing has happened yet. When the client sends a fit request to the endpoint `/v1/chimera/parameter-server/fit`, at step B, the Master receives this request. It then performs an initial fit on its model using sample data collected from the workers, as represented in step C. This sample data is requested from any of the workers via the endpoint `/v1/chimera/sgd/request-data-sample`. The Master sends requests sequentially to the workers, and the first one to respond with a status code 200 provides the sample data. This data consists of a few rows from the worker's local training dataset. In regression, the Master learns the data columns; in classification, it also learns all the classes present in the dataset.

After the initial fit, the Master broadcasts a fit step to each worker via the endpoint `/v1/chimera/sgd/fit-step`, sending each one the current model's coefficients and intercept—this occurs in step D. From there, the Master concurrently waits for a response, a timeout, or the maximum number of retries from each worker. If all workers fail (step F)—that is, if every worker times out or exceeds the retry limit—the flow proceeds to step G and returns an error message through the endpoint `/v1/chimera/sgd/fit`. The process ends at step M.

If at least one worker responds, the Master receives results consisting of gradients for both the coefficients and the intercept. In step I, the Master aggregates these gradients by computing their arithmetic mean. Then, in step J, it checks for convergence or whether the maximum number of iterations has been reached. If either condition is met, the flow proceeds to step L, returns an "ok" JSON-formatted response via the endpoint `/v1/chimera/sgd/fit`, and concludes at step M. However, if the model has not converged and the maximum iterations have not been reached, the flow moves to step K, where the Master updates its model’s coefficients and intercept using the SGD formula. The loop continues from step D until either condition F or J is satisfied.

The following state machine flowchart depicts the steps in the predict action for the Parameter Server Master:

<p align="center">
    <img width="600" src="./images/master/parameter_server_master_predict_state_machine_flowchart.png" alt="Distributed SGD">
<p>
<p align="center"><strong>Figure:</strong> State machine flowchart of Parameter Server Master's /v1/chimera/parameter-server/predict endpoint. </p>

The prediction process begins at step A, where nothing has happened yet. When the client sends a prediction request to the endpoint `/v1/chimera/parameter-server/predict`, at step B, the Master receives the request. Then, in step C, if the model on the Master has been fitted—that is, if the endpoint `/v1/chimera/parameter-server/fit` was previously called and returned a status code 200—the Master returns the predictions to the client as a JSON-formatted response via the same prediction endpoint. Finally, the flow ends at step F.

If, in step C, the model has not been fitted, an error message is returned to the client through the endpoint `/v1/chimera/parameter-server/predict`. The flow then ends at step F, as before.

## References

### Papers

- VERBRAEKEN, Joost et al. A survey on distributed machine learning. Acm computing surveys (csur), v. 53, n. 2, p. 1-33, 2020. Disponível em: https://dl.acm.org/doi/pdf/10.1145/3377454

- B. Chatterjee, "Distributed Machine Learning," in *Proc. 25th Int. Conf. on Distributed Computing and Networking (ICDCN '24)*, Chennai, India, 2024, pp. 4–7. doi: 10.1145/3631461.3632516. Disponível em: https://dl.acm.org/doi/fullHtml/10.1145/3631461.3632516

### Websites

- https://studytrails.com/2021/02/10/distributed-machine-learning-2-architecture/

- https://www.almabetter.com/bytes/tutorials/mlops/distributed-computing-for-ml

- https://neptune.ai/blog/distributed-training

- https://learn.microsoft.com/en-us/azure/machine-learning/concept-distributed-training?view=azureml-api-2
