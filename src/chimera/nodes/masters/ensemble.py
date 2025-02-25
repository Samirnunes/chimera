import json
from typing import List

import numpy as np
import requests  # type: ignore
import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse
from requests.models import Response  # type: ignore

from ...api.constants import (
    CHIMERA_ENSEMBLE_FIT_PATH,
    CHIMERA_ENSEMBLE_PREDICT_PATH,
    CHIMERA_NODE_FIT_PATH,
    CHIMERA_NODE_PREDICT_PATH,
)
from ...api.dto import FitOutput, PredictInput, PredictOutput
from ...api.response import build_error_response, build_json_response
from ...containers.configs import WorkersConfig  # type: ignore


class _MeanAggregator:
    def run(self, responses: List[Response]) -> np.ndarray:
        y_pred: np.ndarray = np.array(
            [json.loads(response.text)["y_pred"] for response in responses]
        )
        return np.mean(y_pred, axis=0)


class AggregationMaster:
    def __init__(self) -> None:
        self._workers_config = WorkersConfig()
        self._aggregator = _MeanAggregator()

    def serve(self, port: int = 8080) -> None:
        app = FastAPI()
        app.include_router(self._predict_router())
        app.include_router(self._fit_router())
        uvicorn.run(app, host=self._workers_config.CHIMERA_WORKERS_HOST, port=port)

    def _predict_router(self) -> APIRouter:
        router = APIRouter()

        @router.post(CHIMERA_ENSEMBLE_PREDICT_PATH)
        def predict(predict_input: PredictInput) -> JSONResponse:
            try:
                responses = [
                    requests.post(
                        url=f"http://localhost:{self._workers_config.CHIMERA_WORKERS_MAPPED_PORTS[i]}{CHIMERA_NODE_PREDICT_PATH}",
                        json=predict_input,
                    )
                    for i in range(
                        len(self._workers_config.CHIMERA_WORKERS_NODES_NAMES)
                    )
                ]
                return build_json_response(
                    PredictOutput(y_pred=list(self._aggregator.run(responses)))
                )
            except Exception as e:
                return build_error_response(e)

        return router

    def _fit_router(self) -> APIRouter:
        router = APIRouter()

        @router.post(CHIMERA_ENSEMBLE_FIT_PATH)
        def fit() -> JSONResponse:
            try:
                for i in range(
                    len(self._workers_config.CHIMERA_WORKERS_NODES_NAMES)
                ):
                    requests.post(
                        url=f"http://localhost:{self._workers_config.CHIMERA_WORKERS_MAPPED_PORTS[i]}{CHIMERA_NODE_FIT_PATH}"
                    )
                return build_json_response(FitOutput(fit="ok"))
            except Exception as e:
                return build_error_response(e)

        return router
