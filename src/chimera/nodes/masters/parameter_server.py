import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any, List, Literal, Tuple

import numpy as np
import pandas as pd
import requests  # type: ignore
import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse
from requests.adapters import HTTPAdapter  # type: ignore

from ...api.configs import (
    CHIMERA_PARAMETER_SERVER_MASTER_FIT_PATH,
    CHIMERA_PARAMETER_SERVER_MASTER_PREDICT_PATH,
    CHIMERA_SGD_WORKER_FIT_REQUEST_DATA_SAMPLE_PATH,
    CHIMERA_SGD_WORKER_FIT_STEP_PATH,
)
from ...api.dto import FitOutput, FitStepInput, PredictInput, PredictOutput
from ...api.exception import ResponseException
from ...api.response import (
    build_error_response,
    build_json_response,
    get_error_response_message,  # type: ignore
)
from ...containers.configs import WorkersConfig
from ...utils import status_logger, time_logger
from ..workers.sgd import MODEL_TYPE, MODELS_MAP
from .base import Master


class ParameterServerMaster(Master):
    """
    Implements a Parameter Server master node for Chimera.

    This class manages the model parameters and coordinates the training process
    with worker nodes using a parameter server architecture.
    """

    def __init__(
        self,
        model: Literal["regressor", "classifier"],
        epsilon: float = 10e-12,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the ParameterServerMaster.

        Args:
            model: The type of model to use ("regressor" or "classifier").
            epsilon: The convergence threshold for the training process.
            *args: Additional positional arguments passed to the model constructor.
            **kwargs: Additional keyword arguments passed to the model constructor.
        """
        kwargs.pop("eta0", None)
        self._workers_config = WorkersConfig()
        self._model_type = model
        self._model: MODEL_TYPE = MODELS_MAP[model](*args, **kwargs, eta0=1e-20)
        self._epsilon = epsilon
        self._port: int

    def serve(self, port: int = 8080) -> None:
        """
        Starts the parameter server master.

        Args:
            port: The port number to listen on.  Defaults to 8080.
        """
        self._port = port
        app = FastAPI()
        app.include_router(self._predict_router())
        app.include_router(self._fit_router())
        status_logger.info(f"Serving {self.__class__.__name__} at port {port}...")
        uvicorn.run(app, host=self._workers_config.CHIMERA_WORKERS_HOST, port=port)

    def _predict_router(self) -> APIRouter:
        """Creates the FastAPI router for the /predict endpoint."""
        router = APIRouter()

        @router.post(CHIMERA_PARAMETER_SERVER_MASTER_PREDICT_PATH)
        def predict(predict_input: PredictInput) -> JSONResponse:
            """Handles prediction requests."""
            try:
                start_master = time.time()

                if self._model_type == "regressor":
                    prediction = self._model.predict(
                        pd.DataFrame(
                            predict_input.X_pred_rows,
                            columns=predict_input.X_pred_columns,
                        )
                    )
                else:
                    prediction = [
                        probas[1]
                        for probas in self._model.predict_proba(
                            pd.DataFrame(
                                predict_input.X_pred_rows,
                                columns=predict_input.X_pred_columns,
                            )
                        )
                    ]

                response = build_json_response(
                    PredictOutput(y_pred_rows=list(prediction))
                )
                end_master = time.time()
                time_logger.info(
                    f"http://localhost:{self._port}{CHIMERA_PARAMETER_SERVER_MASTER_PREDICT_PATH} master endpoint latency = {round(end_master - start_master, 4)} s"
                )
                return response

            except Exception as e:
                status_logger.error(f"Error at {self.__class__.__name__}: {e}")
                return build_error_response(e)

        return router

    def _fit_router(self) -> APIRouter:
        """Creates the FastAPI router for the /fit endpoint."""
        router = APIRouter()

        def _fetch_fit_step_from_worker(
            port: int,
            weights_gradients: List[List[float]],
            bias_gradients: List[float],
        ) -> None:
            """Fetches a single fit step from a worker."""
            try:
                s = requests.Session()
                prefix = f"http://localhost:{port}"
                s.mount(
                    prefix,
                    HTTPAdapter(
                        max_retries=self._workers_config.CHIMERA_WORKERS_ENDPOINTS_MAX_RETRIES
                    ),
                )
                url = f"{prefix}{CHIMERA_SGD_WORKER_FIT_STEP_PATH}"

                start_worker = time.time()
                response = s.post(
                    url=url,
                    timeout=self._workers_config.CHIMERA_WORKERS_ENDPOINTS_TIMEOUT,
                    json=FitStepInput(
                        weights=list(deepcopy(self._model.coef_.flatten())),
                        bias=list(deepcopy(self._model.intercept_)),
                    ).model_dump(),
                )
                end_worker = time.time()
                time_logger.info(
                    f"{url} worker endpoint latency = {round(end_worker - start_worker, 4)} s"
                )

                response_json = response.json()

                if response.status_code == 200:
                    weights_gradients.append(response_json["weights_gradients"])
                    bias_gradients.append(response_json["bias_gradient"])
                else:
                    status_logger.error(
                        f"Error at {self.__class__.__name__}: {get_error_response_message(response)}"
                    )
                    raise ResponseException(response)
            except Exception as e:
                status_logger.error(
                    f"Error fetching fit from worker at port {port}: {e}, at {self.__class__.__name__}"
                )

        def _request_data_sample() -> Tuple:
            """Requests a data sample from a worker."""
            for port in self._workers_config.CHIMERA_WORKERS_MAPPED_PORTS:
                s = requests.Session()
                prefix = f"http://localhost:{port}"
                s.mount(
                    prefix,
                    HTTPAdapter(
                        max_retries=self._workers_config.CHIMERA_WORKERS_ENDPOINTS_MAX_RETRIES
                    ),
                )
                url = f"{prefix}{CHIMERA_SGD_WORKER_FIT_REQUEST_DATA_SAMPLE_PATH}"

                start_worker = time.time()
                response = s.get(
                    url=url,
                    timeout=self._workers_config.CHIMERA_WORKERS_ENDPOINTS_TIMEOUT,
                )
                end_worker = time.time()
                time_logger.info(
                    f"{url} worker endpoint latency = {round(end_worker - start_worker, 4)} s"
                )

                response_json = response.json()

                if response.status_code == 200:
                    return (
                        response_json["X_train_sample_columns"],
                        response_json["X_train_sample_rows"],
                        response_json["y_train_sample_columns"],
                        response_json["y_train_sample_rows"],
                    )
            status_logger.error(
                f"Error at {self.__class__.__name__}: {get_error_response_message(response)}"
            )
            raise ResponseException(response)

        @router.post(CHIMERA_PARAMETER_SERVER_MASTER_FIT_PATH)
        def fit() -> JSONResponse:
            """Handles the complete fit process."""

            def _fit_step() -> Tuple[np.ndarray, np.ndarray]:
                """Performs a single step of the iterative fitting process."""
                weights_gradients: List[List[float]] = []
                bias_gradients: List[float] = []

                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(
                            _fetch_fit_step_from_worker,
                            port,
                            weights_gradients,
                            bias_gradients,
                        )
                        for port in self._workers_config.CHIMERA_WORKERS_MAPPED_PORTS
                    ]
                    for future in futures:
                        future.result()

                if len(weights_gradients) == 0:
                    message = "All fit iterations responses from workers failed."
                    status_logger.error(
                        f"Error at {self.__class__.__name__}: {message}"
                    )
                    raise ResponseException(requests.Response(), message)

                return np.mean(np.array(weights_gradients), axis=0), np.mean(
                    np.array(bias_gradients), axis=0
                )

            try:
                start_master = time.time()
                (
                    X_train_sample_columns,
                    X_train_sample_rows,
                    _,
                    y_train_sample_rows,
                ) = _request_data_sample()

                max_iter = self._model.get_params()["max_iter"]
                y_train_samples = np.array(y_train_sample_rows).ravel()

                kwargs = {}
                if self._model_type == "classifier":
                    kwargs = {"classes": np.unique(y_train_samples)}

                self._model.partial_fit(
                    pd.DataFrame(
                        X_train_sample_rows, columns=X_train_sample_columns
                    ),
                    y_train_samples,
                    **kwargs,
                )

                mean_weights_gradients, mean_bias_gradient = _fit_step()
                current_iter = 0

                while current_iter < max_iter and any(
                    [
                        np.abs(gradient) > self._epsilon
                        for gradient in list(mean_weights_gradients)
                        + list(mean_bias_gradient)
                    ]
                ):
                    status_logger.info(
                        f"Computing SGD iteration {current_iter + 1} at {self.__class__.__name__}"
                    )
                    self._model.coef_ = self._model.coef_ - mean_weights_gradients
                    self._model.intercept_ = (
                        self._model.intercept_ - mean_bias_gradient
                    )
                    current_iter += 1
                    mean_weights_gradients, mean_bias_gradient = _fit_step()

                response = build_json_response(FitOutput(fit="ok"))
                end_master = time.time()
                time_logger.info(
                    f"http://localhost:{self._port}{CHIMERA_PARAMETER_SERVER_MASTER_FIT_PATH} master endpoint latency = {round(end_master - start_master, 4)} s"
                )
                return response
            except Exception as e:
                status_logger.error(f"Error at {self.__class__.__name__}: {e}")
                return build_error_response(e)

        return router
