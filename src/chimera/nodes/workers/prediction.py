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
        X_bootstrap_rows = X.iloc[row_indices].reset_index(drop=True)
        y_bootstrap_rows = y.iloc[row_indices].reset_index(drop=True)

        n_cols_X = X_bootstrap_rows.shape[1]
        col_indices_X = self.random_state.choice(
            n_cols_X, size=n_cols_X, replace=True
        )
        X_bootstrap = X_bootstrap_rows.iloc[:, col_indices_X].copy()
        X_bootstrap.columns = [X.columns[i] for i in col_indices_X]

        n_cols_y = y_bootstrap_rows.shape[1]
        col_indices_y = self.random_state.choice(
            n_cols_y, size=n_cols_y, replace=True
        )
        y_bootstrap = y_bootstrap_rows.iloc[:, col_indices_y].copy()
        y_bootstrap.columns = [y.columns[i] for i in col_indices_y]

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
                X_train = pd.read_csv(
                    f"{CHIMERA_DATA_FOLDER}/{CHIMERA_TRAIN_FEATURES_FILENAME}.csv"
                )
                y_train = pd.read_csv(
                    f"{CHIMERA_DATA_FOLDER}/{CHIMERA_TRAIN_LABELS_FILENAME}.csv"
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
                y_pred: np.ndarray = self._predictor.predict(
                    pd.DataFrame.from_dict(predict_input.X)
                )
                return build_json_response(PredictOutput(y_pred=list(y_pred)))
            except Exception as e:
                return build_error_response(e)

        return router


class ClassificationWorker(_PredictionWorker):
    def __init__(self, classifier: ClassifierMixin, bootstrap: bool = False) -> None:
        super().__init__(classifier, bootstrap)

    def _predict_router(self) -> APIRouter:
        router = APIRouter()

        @router.post(CHIMERA_NODE_PREDICT_PATH)
        def predict(node_input: PredictInput) -> JSONResponse:
            try:
                y_pred: np.ndarray = self._predictor.predict_proba(
                    pd.DataFrame.from_dict(node_input.X)
                )
                return build_json_response(PredictOutput(y_pred=list(y_pred)))
            except Exception as e:
                return build_error_response(e)

        return router
