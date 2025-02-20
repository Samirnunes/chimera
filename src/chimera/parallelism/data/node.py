import inspect
from abc import ABC, abstractmethod
from types import FrameType

import numpy as np
import pandas as pd
import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse
from sklearn.base import ClassifierMixin, RegressorMixin

from ...api import (
    FitOutput,
    PredictInput,
    PredictOutput,
    build_error_response,
    build_json_response,
)
from ...api.paths import NODE_FIT_PATH, NODE_PREDICT_PATH
from ...workers.config import WORKERS_CONFIG


class _DataParallelismNode(ABC):
    def serve(self) -> None:
        frame: FrameType | None = inspect.currentframe()
        if frame is None:
            raise ValueError
        filename: str = inspect.getsourcefile(frame.f_back)  # type: ignore

        app = FastAPI()
        app.include_router(self._predict_router())
        app.include_router(self._fit_router())
        uvicorn.run(
            app,
            host=WORKERS_CONFIG.CHIMERA_WORKERS_HOST,
            port=WORKERS_CONFIG.CHIMERA_WORKERS_HOST_PORTS[
                WORKERS_CONFIG.CHIMERA_WORKERS_NODES_NAMES.index(filename)
            ],
        )

    @abstractmethod
    def _predict_router(self) -> APIRouter:
        raise NotImplementedError

    @abstractmethod
    def _fit_router(self) -> APIRouter:
        raise NotImplementedError


class RegressionNode(_DataParallelismNode):
    def __init__(self, regressor: RegressorMixin) -> None:
        self._regressor: RegressorMixin = regressor

    def _predict_router(self) -> APIRouter:
        router = APIRouter()

        @router.post(NODE_PREDICT_PATH)
        def predict(predict_input: PredictInput) -> JSONResponse:
            try:
                predictions: np.ndarray = self._regressor.predict(
                    pd.DataFrame.from_dict(predict_input.X)
                )
                return build_json_response(
                    PredictOutput(predictions=list(predictions))
                )
            except Exception as e:
                return build_error_response(e)

        return router

    def _fit_router(self) -> APIRouter:
        router = APIRouter()

        @router.post(NODE_FIT_PATH)
        def fit() -> RegressorMixin:
            try:
                self._regressor.fit(
                    pd.read_csv("data/X_train.csv"),
                    pd.read_csv("data/y_train.csv"),
                )
                return build_json_response(FitOutput(fit="ok"))
            except Exception as e:
                return build_error_response(e)

        return router


class ClassificationNode(_DataParallelismNode):
    def __init__(self, classifier: ClassifierMixin) -> None:
        self._classifier: ClassifierMixin = classifier

    def _predict_router(self) -> APIRouter:
        router = APIRouter()

        @router.post(NODE_PREDICT_PATH)
        def predict(node_input: PredictInput) -> JSONResponse:
            try:
                predictions: np.ndarray = self._classifier.predict_proba(
                    pd.DataFrame.from_dict(node_input.X)
                )
                return build_json_response(
                    PredictOutput(predictions=list(predictions))
                )
            except Exception as e:
                return build_error_response(e)

        return router

    def _fit_router(self) -> APIRouter:
        router = APIRouter()

        @router.post(NODE_FIT_PATH)
        def fit() -> RegressorMixin:
            try:
                self._classifier.fit(
                    pd.read_csv("data/X_train.csv"),
                    pd.read_csv("data/y_train.csv"),
                )
                return build_json_response(FitOutput(fit="ok"))
            except Exception as e:
                return build_error_response(e)

        return router
