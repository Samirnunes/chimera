import json
from typing import List

import numpy as np
import requests  # type: ignore
import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse
from requests.models import Response  # type: ignore

from chimera.workers.config import WORKERS_CONFIG  # type: ignore

from ...api import (
    FitOutput,
    PredictInput,
    PredictOutput,
    build_error_response,
    build_json_response,
)
from ...api.paths import (
    ENSEMBLE_FIT_PATH,
    ENSEMBLE_PREDICT_PATH,
    NODE_FIT_PATH,
    NODE_PREDICT_PATH,
)


class _EnsembleAggregator:
    def aggregate(self, responses: List[Response]) -> np.ndarray:
        predictions: np.ndarray = np.array(
            [json.loads(response.text)["predictions"] for response in responses]
        )
        return np.mean(predictions, axis=0)


class Ensemble:
    def __init__(self) -> None:
        self._aggregator = _EnsembleAggregator()

    def serve(self, port: int = 8100) -> None:
        app = FastAPI()
        app.include_router(self._predict_router())
        app.include_router()
        uvicorn.run(app, host=WORKERS_CONFIG.CHIMERA_WORKERS_HOST, port=port)

    def _predict_router(self) -> APIRouter:
        router = APIRouter()

        @router.post(ENSEMBLE_PREDICT_PATH)
        def predict(predict_input: PredictInput) -> JSONResponse:
            try:
                responses = [
                    requests.post(
                        url=f"https://{WORKERS_CONFIG.CHIMERA_WORKERS_NODES_NAMES[i]}:{WORKERS_CONFIG.CHIMERA_WORKERS_HOST_PORTS[i]}{NODE_PREDICT_PATH}"
                    )
                    for i in range(len(WORKERS_CONFIG.CHIMERA_WORKERS_NODES_NAMES))
                ]
                return build_json_response(
                    PredictOutput(
                        predictions=list(self._aggregator.aggregate(responses))
                    )
                )
            except Exception as e:
                return build_error_response(e)

        return router

    def _fit_router(self) -> APIRouter:
        router = APIRouter()

        @router.post(ENSEMBLE_FIT_PATH)
        def fit(fit_output: FitOutput) -> JSONResponse:
            try:
                for i in range(len(WORKERS_CONFIG.CHIMERA_WORKERS_NODES_NAMES)):
                    requests.post(
                        url=f"https://{WORKERS_CONFIG.CHIMERA_WORKERS_NODES_NAMES[i]}:{WORKERS_CONFIG.CHIMERA_WORKERS_HOST_PORTS[i]}{NODE_FIT_PATH}"
                    )
                return build_json_response(FitOutput(fit="ok"))
            except Exception as e:
                return build_error_response(e)

        return router
