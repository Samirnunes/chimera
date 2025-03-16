import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List

import numpy as np
import requests  # type: ignore
import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse
from requests.adapters import HTTPAdapter  # type: ignore

from ...api.configs import (
    CHIMERA_AGGREGATION_MASTER_FIT_PATH,
    CHIMERA_AGGREGATION_MASTER_PREDICT_PATH,
    CHIMERA_MODEL_WORKER_FIT_PATH,
    CHIMERA_MODEL_WORKER_PREDICT_PATH,
)
from ...api.dto import FitOutput, PredictInput, PredictOutput
from ...api.exception import ResponseException
from ...api.response import (
    build_error_response,
    build_json_response,
    get_error_response_message,  # type: ignore
)
from ...containers.configs import WorkersConfig
from ...utils import status_logger, time_logger
from .base import Master


class _MeanAggregator:
    """Aggregates prediction results using mean."""

    def run(self, y_pred_list: List[float]) -> List[Any]:
        """
        Aggregates prediction results using the mean.

        Args:
            responses: List of responses from prediction workers.

        Returns:
            List of aggregated prediction results.

        Raises:
            ValueError: If no valid 'y_pred_rows' are found in responses.
            Exception: If any error occurs during response processing.
        """
        y_pred: np.ndarray = np.array(y_pred_list)
        y_pred_mean = np.mean(y_pred, axis=0)

        if isinstance(y_pred_mean, float):
            return [y_pred_mean]
        return list(y_pred_mean)


class AggregationMaster(Master):
    """Orchestrates the aggregation of predictions from workers."""

    def __init__(self) -> None:
        """Initializes the AggregationMaster."""
        self._workers_config = WorkersConfig()
        self._aggregator = _MeanAggregator()
        self._port: int

    def serve(self, port: int = 8080) -> None:
        """
        Starts the FastAPI server for the aggregation master.

        Args:
            port: Port number to listen on (default: 8080).
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

        def _fetch_predict_from_worker(
            port: int, predict_input: PredictInput, results: list
        ) -> None:
            """Fetches prediction from a worker and stores the result."""
            try:
                s = requests.Session()
                prefix = f"http://localhost:{port}"
                s.mount(
                    prefix,
                    HTTPAdapter(
                        max_retries=self._workers_config.CHIMERA_WORKERS_ENDPOINTS_MAX_RETRIES
                    ),
                )
                url = f"{prefix}{CHIMERA_MODEL_WORKER_PREDICT_PATH}"

                start_worker = time.time()
                response = requests.post(
                    url=url,
                    json=predict_input.model_dump(),
                    timeout=self._workers_config.CHIMERA_WORKERS_ENDPOINTS_TIMEOUT,
                )
                end_worker = time.time()
                time_logger.info(
                    f"{url} worker endpoint latency = {round(end_worker - start_worker, 4)} s"
                )

                if response.status_code == 200:
                    results.append(response.json()["y_pred_rows"])
                else:
                    status_logger.error(
                        f"Error at {self.__class__.__name__}: {get_error_response_message(response)}"
                    )
                    raise ResponseException(response)

            except Exception as e:
                status_logger.error(
                    f"Error fetching prediction from worker at port {port}: {e}, at {self.__class__.__name__}"
                )

        @router.post(CHIMERA_AGGREGATION_MASTER_PREDICT_PATH)
        def predict(predict_input: PredictInput) -> JSONResponse:
            """Handles prediction requests by aggregating results from workers."""
            try:
                start_master = time.time()
                results: List[Any] = []
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(
                            _fetch_predict_from_worker,
                            port,
                            predict_input,
                            results,
                        )
                        for port in self._workers_config.CHIMERA_WORKERS_MAPPED_PORTS
                    ]
                    for future in futures:
                        future.result()

                if len(results) == 0:
                    message = "All predict responses from workers failed."
                    status_logger.error(
                        f"Error at {self.__class__.__name__}: {message}"
                    )
                    raise ResponseException(requests.Response(), message)

                response = build_json_response(
                    PredictOutput(y_pred_rows=self._aggregator.run(results))
                )
                end_master = time.time()
                time_logger.info(
                    f"http://localhost:{self._port}{CHIMERA_AGGREGATION_MASTER_PREDICT_PATH} master endpoint latency = {round(end_master - start_master, 4)} s"
                )
                return response
            except Exception as e:
                status_logger.error(f"Error at {self.__class__.__name__}: {e}")
                return build_error_response(e)

        return router

    def _fit_router(self) -> APIRouter:
        """Creates the FastAPI router for the /fit endpoint."""
        router = APIRouter()

        def _fetch_fit_from_worker(port: int, results: List) -> None:
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
                url = f"{prefix}{CHIMERA_MODEL_WORKER_FIT_PATH}"

                start_worker = time.time()
                response = s.post(
                    url=url,
                    timeout=self._workers_config.CHIMERA_WORKERS_ENDPOINTS_TIMEOUT,
                )
                end_worker = time.time()
                time_logger.info(
                    f"{url} worker endpoint latency = {round(end_worker - start_worker, 4)} s"
                )

                if response.status_code == 200:
                    results.append("ok")
                else:
                    status_logger.error(
                        f"Error at {self.__class__.__name__}: {get_error_response_message(response)}"
                    )
                    raise ResponseException(response)
            except Exception as e:
                status_logger.error(
                    f"Error fetching fit from worker at port {port}: {e} at {self.__class__.__name__}"
                )

        @router.post(CHIMERA_AGGREGATION_MASTER_FIT_PATH)
        def fit() -> JSONResponse:
            """Handles fit requests by forwarding them to workers."""
            try:
                start_master = time.time()
                results: List = []

                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(
                            _fetch_fit_from_worker,
                            port,
                            results,
                        )
                        for port in self._workers_config.CHIMERA_WORKERS_MAPPED_PORTS
                    ]
                    for future in futures:
                        future.result()

                if len(results) == 0:
                    message = "All fit responses from workers failed."
                    status_logger.error(
                        f"Error at {self.__class__.__name__}: {message}"
                    )
                    raise ResponseException(requests.Response(), message)

                response = build_json_response(FitOutput(fit="ok"))
                end_master = time.time()
                time_logger.info(
                    f"http://localhost:{self._port}{CHIMERA_AGGREGATION_MASTER_FIT_PATH} master endpoint latency = {round(end_master - start_master, 4)} s"
                )
                return response
            except Exception as e:
                status_logger.error(f"Error at {self.__class__.__name__}: {e}")
                return build_error_response(e)

        return router
