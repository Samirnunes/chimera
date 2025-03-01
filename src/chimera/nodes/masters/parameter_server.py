import threading
from typing import List, Literal

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
    CHIMERA_SGD_WORKER_FIT_ITERATION_PATH,
)
from ...api.dto import FitOutput, PredictInput, PredictOutput
from ...api.exception import ResponseException
from ...api.response import (
    build_error_response,
    build_json_response,  # type: ignore
)
from ...containers.configs import WorkersConfig
from ..workers.sgd import MODEL_TYPE, MODELS_MAP
from .base import Master


class ParameterServerMaster(Master):
    def __init__(
        self, model: Literal["linear_regression", "logistic_regression"]
    ) -> None:
        self._workers_config = WorkersConfig()
        self._model: MODEL_TYPE = MODELS_MAP[model]
        self._weights: np.ndarray
        self._bias: float

    def serve(self, port: int = 8080) -> None:
        app = FastAPI()
        app.include_router(self._predict_router())
        app.include_router(self._fit_router())
        uvicorn.run(app, host=self._workers_config.CHIMERA_WORKERS_HOST, port=port)

    def _predict_router(self) -> APIRouter:
        """Creates the FastAPI router for the /predict endpoint."""
        router = APIRouter()

        @router.post(CHIMERA_PARAMETER_SERVER_MASTER_PREDICT_PATH)
        def predict(predict_input: PredictInput) -> JSONResponse:
            """Handles prediction."""
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

        def _fetch_fit_iteration_from_worker(port: int, gradients: List) -> None:
            """Fetches fit from a worker and stores the result."""
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
                    url=f"{prefix}{CHIMERA_SGD_WORKER_FIT_ITERATION_PATH}",
                    timeout=self._workers_config.CHIMERA_WORKERS_ENDPOINTS_TIMEOUT,
                )
                if response.status_code == 200:
                    gradients.append(response.json()["gradients"])
                else:
                    raise ResponseException(response)
            except Exception as e:
                print(f"Error fetching fit from worker at port {port}: {e}")

        @router.post(CHIMERA_PARAMETER_SERVER_MASTER_FIT_PATH)
        def fit(
            learning_rate: float, epochs: int, epsilon: float = 10e-8
        ) -> JSONResponse:
            """Handles fit requests by forwarding them to workers."""

            def _fit_iteration() -> np.ndarray:
                threads: List[threading.Thread] = []
                gradients: List[List] = []
                for port in self._workers_config.CHIMERA_WORKERS_MAPPED_PORTS:
                    thread = threading.Thread(
                        target=_fetch_fit_iteration_from_worker,
                        args=(port, gradients),
                    )
                    threads.append(thread)
                    thread.start()

                for thread in threads:
                    thread.join()

                if len(gradients) == 0:
                    message = "All fit iterations responses from workers failed."
                    raise ResponseException(requests.Response(), message)

                return np.mean(np.array(gradients), axis=0)

            try:
                mean_gradients = _fit_iteration()
                current_epoch = 0

                self._weights = np.zeros(len(mean_gradients) - 1)
                self._bias = 0.0

                while current_epoch < epochs and any(mean_gradients > epsilon):
                    bias_gradient: float = mean_gradients[-1]
                    weights_gradients = mean_gradients[:-1]

                    self._weights -= learning_rate * weights_gradients
                    self._bias -= learning_rate * bias_gradient

                    current_epoch += 1
                    mean_gradients = _fit_iteration()

                return build_json_response(FitOutput(fit="ok"))
            except Exception as e:
                return build_error_response(e)

        return router
