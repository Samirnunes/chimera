import threading
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
    build_json_response,  # type: ignore
)
from ...containers.configs import WorkersConfig
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
        self._model: MODEL_TYPE = MODELS_MAP[model](*args, **kwargs, eta0=1e-20)
        self._epsilon = epsilon

    def serve(self, port: int = 8080) -> None:
        """
        Starts the parameter server master.

        Args:
            port: The port number to listen on.  Defaults to 8080.
        """
        app = FastAPI()
        app.include_router(self._predict_router())
        app.include_router(self._fit_router())
        uvicorn.run(app, host=self._workers_config.CHIMERA_WORKERS_HOST, port=port)

    def _predict_router(self) -> APIRouter:
        """Creates the FastAPI router for the /predict endpoint."""
        router = APIRouter()

        @router.post(CHIMERA_PARAMETER_SERVER_MASTER_PREDICT_PATH)
        def predict(predict_input: PredictInput) -> JSONResponse:
            """Handles prediction requests."""
            try:
                return build_json_response(
                    PredictOutput(
                        y_pred_rows=list(
                            self._model.predict(
                                pd.DataFrame(
                                    predict_input.X_pred_rows,
                                    columns=predict_input.X_pred_columns,
                                )
                            )
                        )
                    )
                )
            except Exception as e:
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
                response = s.post(
                    url=f"{prefix}{CHIMERA_SGD_WORKER_FIT_STEP_PATH}",
                    timeout=self._workers_config.CHIMERA_WORKERS_ENDPOINTS_TIMEOUT,
                    json=FitStepInput(
                        weights=deepcopy(list(self._model.coef_)),
                        bias=deepcopy(list(self._model.intercept_)),
                    ).model_dump(),
                )
                response_json = response.json()

                if response.status_code == 200:
                    weights_gradients.append(response_json["weights_gradients"])
                    bias_gradients.append(response_json["bias_gradient"])
                else:
                    raise ResponseException(response)
            except Exception as e:
                print(f"Error fetching fit from worker at port {port}: {e}")

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
                response = s.get(
                    url=f"{prefix}{CHIMERA_SGD_WORKER_FIT_REQUEST_DATA_SAMPLE_PATH}",
                    timeout=self._workers_config.CHIMERA_WORKERS_ENDPOINTS_TIMEOUT,
                )
                response_json = response.json()

                if response.status_code == 200:
                    return (
                        response_json["X_train_sample_columns"],
                        response_json["X_train_sample_rows"],
                        response_json["y_train_sample_columns"],
                        response_json["y_train_sample_rows"],
                    )
            raise ResponseException(response)

        @router.post(CHIMERA_PARAMETER_SERVER_MASTER_FIT_PATH)
        def fit() -> JSONResponse:
            """Handles the complete fit process."""

            def _fit_step() -> Tuple[np.ndarray, np.ndarray]:
                """Performs a single step of the iterative fitting process."""
                threads: List[threading.Thread] = []
                weights_gradients: List[List[float]] = []
                bias_gradients: List[float] = []
                for port in self._workers_config.CHIMERA_WORKERS_MAPPED_PORTS:
                    thread = threading.Thread(
                        target=_fetch_fit_step_from_worker,
                        args=(port, weights_gradients, bias_gradients),
                    )
                    threads.append(thread)
                    thread.start()

                for thread in threads:
                    thread.join()

                if len(weights_gradients) == 0:
                    message = "All fit iterations responses from workers failed."
                    raise ResponseException(requests.Response(), message)

                return np.mean(np.array(weights_gradients), axis=0), np.mean(
                    np.array(bias_gradients), axis=0
                )

            try:
                (
                    X_train_sample_columns,
                    X_train_sample_rows,
                    _,
                    y_train_sample_rows,
                ) = _request_data_sample()

                max_iter = self._model.get_params()["max_iter"]

                self._model.partial_fit(
                    pd.DataFrame(
                        X_train_sample_rows, columns=X_train_sample_columns
                    ),
                    np.array(y_train_sample_rows).ravel(),
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
                    self._model.coef_ = self._model.coef_ - mean_weights_gradients
                    self._model.intercept_ = (
                        self._model.intercept_ - mean_bias_gradient
                    )
                    current_iter += 1
                    mean_weights_gradients, mean_bias_gradient = _fit_step()

                return build_json_response(FitOutput(fit="ok"))
            except Exception as e:
                return build_error_response(e)

        return router
