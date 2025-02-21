from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse
from sklearn.base import ClassifierMixin, RegressorMixin

from ...api import (
    CHIMERA_DATA_FOLDER,
    CHIMERA_NODE_FIT_PATH,
    CHIMERA_NODE_PREDICT_PATH,
    FitOutput,
    PredictInput,
    PredictOutput,
    build_error_response,
    build_json_response,
)
from ...workers.config import WorkersConfig


class _DataParallelismNode(ABC):
    def __init__(self) -> None:
        self._workers_config = WorkersConfig()

    def serve(self) -> None:
        app = FastAPI()
        app.include_router(self._predict_router())
        app.include_router(self._fit_router())
        uvicorn.run(
            app,
            host=self._workers_config.CHIMERA_WORKERS_HOST,
            port=self._workers_config.CHIMERA_WORKERS_PORT,
        )

    @abstractmethod
    def _predict_router(self) -> APIRouter:
        raise NotImplementedError

    @abstractmethod
    def _fit_router(self) -> APIRouter:
        raise NotImplementedError


class RegressionNode(_DataParallelismNode):
    def __init__(self, regressor: RegressorMixin) -> None:
        super().__init__()
        self._regressor: RegressorMixin = regressor

    def _predict_router(self) -> APIRouter:
        router = APIRouter()

        @router.post(CHIMERA_NODE_PREDICT_PATH)
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

        @router.post(CHIMERA_NODE_FIT_PATH)
        def fit() -> JSONResponse:
            try:
                self._regressor.fit(
                    pd.read_csv(f"{CHIMERA_DATA_FOLDER}/X_train.csv"),
                    pd.read_csv(f"{CHIMERA_DATA_FOLDER}/y_train.csv"),
                )
                return build_json_response(FitOutput(fit="ok"))
            except Exception as e:
                return build_error_response(e)

        return router


class ClassificationNode(_DataParallelismNode):
    def __init__(self, classifier: ClassifierMixin) -> None:
        super().__init__()
        self._classifier: ClassifierMixin = classifier

    def _predict_router(self) -> APIRouter:
        router = APIRouter()

        @router.post(CHIMERA_NODE_PREDICT_PATH)
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

        @router.post(CHIMERA_NODE_FIT_PATH)
        def fit() -> JSONResponse:
            try:
                self._classifier.fit(
                    pd.read_csv("{DATA_FOLDER}/X_train.csv"),
                    pd.read_csv("{DATA_FOLDER}/y_train.csv"),
                )
                return build_json_response(FitOutput(fit="ok"))
            except Exception as e:
                return build_error_response(e)

        return router
