from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd
import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse
from sklearn.base import ClassifierMixin, RegressorMixin

from ...api.constants import CHIMERA_NODE_FIT_PATH, CHIMERA_NODE_PREDICT_PATH
from ...api.dto import FitOutput, PredictInput, PredictOutput
from ...api.response import build_error_response, build_json_response
from ...containers.configs import WorkersConfig
from ...containers.constants import (
    CHIMERA_DATA_FOLDER,
    CHIMERA_TRAIN_FEATURES_FILENAME,
    CHIMERA_TRAIN_LABELS_FILENAME,
)
from .utils import load_csv_as_fit_input


class _Bootstrapper:
    def __init__(self, random_state: int = 0) -> None:
        self.random_state = np.random.RandomState(random_state)

    def run(
        self, X: pd.DataFrame, y: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows.")

        n_rows = len(X)
        row_indices = self.random_state.choice(n_rows, size=n_rows, replace=True)
        X_bootstrap = X.iloc[row_indices].reset_index(drop=True)
        y_bootstrap = y.iloc[row_indices].reset_index(drop=True)

        return X_bootstrap, y_bootstrap


class _PredictionWorker(ABC):
    def __init__(
        self, predictor: RegressorMixin | ClassifierMixin, bootstrap: bool = False
    ) -> None:
        self._predictor = predictor
        self._bootstrap = bootstrap

        self._workers_config = WorkersConfig()
        self._bootstrapper = _Bootstrapper()

    def serve(self) -> None:
        app = FastAPI()
        app.include_router(self._fit_router())
        app.include_router(self._predict_router())
        uvicorn.run(
            app,
            host=self._workers_config.CHIMERA_WORKERS_HOST,
            port=self._workers_config.CHIMERA_WORKERS_PORT,
        )

    def _fit_router(self) -> APIRouter:
        router = APIRouter()

        @router.post(CHIMERA_NODE_FIT_PATH)
        def fit() -> JSONResponse:
            try:
                fit_input = load_csv_as_fit_input(
                    f"{CHIMERA_DATA_FOLDER}/{CHIMERA_TRAIN_FEATURES_FILENAME}",
                    f"{CHIMERA_DATA_FOLDER}/{CHIMERA_TRAIN_LABELS_FILENAME}",
                )

                X_train = pd.DataFrame(
                    fit_input.X_train_rows, columns=fit_input.X_train_columns
                )
                y_train = pd.DataFrame(
                    fit_input.y_train_rows, columns=fit_input.y_train_columns
                )

                if self._bootstrap:
                    X_train, y_train = self._bootstrapper.run(X_train, y_train)

                self._predictor.fit(X_train, y_train)

                return build_json_response(FitOutput(fit="ok"))
            except Exception as e:
                return build_error_response(e)

        return router

    @abstractmethod
    def _predict_router(self) -> APIRouter:
        raise NotImplementedError


class RegressionWorker(_PredictionWorker):
    def __init__(self, regressor: RegressorMixin, bootstrap: bool = False) -> None:
        super().__init__(regressor, bootstrap)

    def _predict_router(self) -> APIRouter:
        router = APIRouter()

        @router.post(CHIMERA_NODE_PREDICT_PATH)
        def predict(predict_input: PredictInput) -> JSONResponse:
            try:
                X_pred_rows = np.array(predict_input.X_pred_rows)
                X_pred_columns = predict_input.X_pred_columns

                y_pred: np.ndarray = self._predictor.predict(
                    pd.DataFrame(X_pred_rows, columns=X_pred_columns)
                )

                return build_json_response(
                    PredictOutput(
                        y_pred_rows=list(y_pred), y_pred_columns=X_pred_columns
                    )
                )
            except Exception as e:
                return build_error_response(e)

        return router


class ClassificationWorker(_PredictionWorker):
    def __init__(self, classifier: ClassifierMixin, bootstrap: bool = False) -> None:
        super().__init__(classifier, bootstrap)

    def _predict_router(self) -> APIRouter:
        router = APIRouter()

        @router.post(CHIMERA_NODE_PREDICT_PATH)
        def predict(predict_input: PredictInput) -> JSONResponse:
            try:
                X_pred_rows = predict_input.X_pred_rows
                X_pred_columns = predict_input.X_pred_columns
                y_pred: np.ndarray = self._predictor.predict_proba(
                    pd.DataFrame(X_pred_rows, columns=X_pred_columns)
                )
                return build_json_response(
                    PredictOutput(
                        y_pred_rows=list(y_pred), y_pred_columns=X_pred_columns
                    )
                )
            except Exception as e:
                return build_error_response(e)

        return router
