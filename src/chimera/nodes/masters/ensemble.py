from typing import Any, List

import numpy as np
import requests  # type: ignore
import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse

from ...api.configs import (
    CHIMERA_ENSEMBLE_FIT_PATH,
    CHIMERA_ENSEMBLE_PREDICT_PATH,
    CHIMERA_NODE_FIT_PATH,
    CHIMERA_NODE_PREDICT_PATH,
)
from ...api.dto import FitOutput, PredictInput, PredictOutput
from ...api.exception import ResponseException
from ...api.response import (
    build_error_response,
    build_json_response,  # type: ignore
)
from ...containers.configs import WorkersConfig


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


class AggregationMaster:
    """Orchestrates the aggregation of predictions from workers."""

    def __init__(self) -> None:
        """Initializes the AggregationMaster."""
        self._workers_config = WorkersConfig()
        self._aggregator = _MeanAggregator()

    def serve(self, port: int = 8080) -> None:
        """
        Starts the FastAPI server for the aggregation master.

        Args:
            port: Port number to listen on (default: 8080).
        """
        app = FastAPI()
        app.include_router(self._predict_router())
        app.include_router(self._fit_router())
        uvicorn.run(app, host=self._workers_config.CHIMERA_WORKERS_HOST, port=port)

    def _predict_router(self) -> APIRouter:
        """Creates the FastAPI router for the /predict endpoint."""
        router = APIRouter()

        @router.post(CHIMERA_ENSEMBLE_PREDICT_PATH)
        def predict(predict_input: PredictInput) -> JSONResponse:
            """Handles prediction requests by aggregating results from workers."""
            try:
                y_pred_list = []
                for port in self._workers_config.CHIMERA_WORKERS_MAPPED_PORTS:
                    response = requests.post(
                        url=f"http://localhost:{port}{CHIMERA_NODE_PREDICT_PATH}",
                        json=predict_input.model_dump(),
                    )
                    if response.status_code != 200:
                        raise ResponseException(response)

                    y_pred_list.append(response.json()["y_pred_rows"])

                return build_json_response(
                    PredictOutput(y_pred_rows=self._aggregator.run(y_pred_list))
                )
            except Exception as e:
                return build_error_response(e)

        return router

    def _fit_router(self) -> APIRouter:
        """Creates the FastAPI router for the /fit endpoint."""
        router = APIRouter()

        @router.post(CHIMERA_ENSEMBLE_FIT_PATH)
        def fit() -> JSONResponse:
            """Handles fit requests by forwarding them to workers."""
            try:
                for port in self._workers_config.CHIMERA_WORKERS_MAPPED_PORTS:
                    response = requests.post(
                        url=f"http://localhost:{port}{CHIMERA_NODE_FIT_PATH}"
                    )
                    if response.status_code != 200:
                        raise ResponseException(response)

                return build_json_response(FitOutput(fit="ok"))
            except Exception as e:
                return build_error_response(e)

        return router
