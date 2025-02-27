from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import pandas as pd
import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse
from sklearn.base import ClassifierMixin, RegressorMixin

from ...api.configs import CHIMERA_NODE_FIT_PATH, CHIMERA_NODE_PREDICT_PATH
from ...api.dto import FitOutput, PredictInput, PredictOutput, load_csv_as_fit_input
from ...api.response import build_error_response, build_json_response
from ...containers.configs import (
    CHIMERA_TRAIN_DATA_FOLDER,
    CHIMERA_TRAIN_FEATURES_FILENAME,
    CHIMERA_TRAIN_LABELS_FILENAME,
    WorkersConfig,
)


class _Bootstrapper:
    """
    Helper class for bootstrapping training data.  Creates bootstrap samples from input data.
    """

    def __init__(self, random_state: int = 0) -> None:
        """
        Initializes the _Bootstrapper with a random state for reproducibility.

        Args:
            random_state: The seed for the random number generator (default: 0).
        """
        self.random_state = np.random.RandomState(random_state)

    def run(
        self, X: pd.DataFrame, y: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates bootstrap samples from the input data.

        Args:
            X: The feature data (Pandas DataFrame).
            y: The target data (Pandas DataFrame).

        Returns:
            A tuple containing the bootstrapped feature data (X) and target data (y).

        Raises:
            ValueError: If X and y have different numbers of rows.
        """
        n_rows = len(X)
        row_indices = self.random_state.choice(n_rows, size=n_rows, replace=True)
        X_bootstrap = X.iloc[row_indices].reset_index(drop=True)
        y_bootstrap = y.iloc[row_indices].reset_index(drop=True)

        return X_bootstrap, y_bootstrap


class _PredictionWorker(ABC):
    """
    Abstract base class for Chimera prediction workers.  Handles fitting and prediction logic.
    """

    def __init__(
        self, predictor: RegressorMixin | ClassifierMixin, bootstrap: bool = False
    ) -> None:
        """
        Initializes the _PredictionWorker.

        Args:
            predictor: The scikit-learn predictor model (RegressorMixin or ClassifierMixin).
            bootstrap: Whether to use bootstrapping for model training (default: False).
        """
        self._predictor = predictor
        self._bootstrap = bootstrap

        self._workers_config = WorkersConfig()
        self._bootstrapper = _Bootstrapper()

    def serve(self) -> None:
        """
        Starts the FastAPI server for the prediction worker.
        """
        app = FastAPI()
        app.include_router(self._fit_router())
        app.include_router(self._predict_router())
        uvicorn.run(
            app,
            host=self._workers_config.CHIMERA_WORKERS_HOST,
            port=self._workers_config.CHIMERA_WORKERS_PORT,
        )

    def _fit_router(self) -> APIRouter:
        """
        Creates and returns the FastAPI router for the /fit endpoint.

        Returns:
            The FastAPI router for fitting the model.
        """
        router = APIRouter()

        @router.post(CHIMERA_NODE_FIT_PATH)
        def fit() -> JSONResponse:
            """
            Fits the prediction model using training data loaded from CSV files.
            """
            try:
                fit_input = load_csv_as_fit_input(
                    f"{CHIMERA_TRAIN_DATA_FOLDER}/{CHIMERA_TRAIN_FEATURES_FILENAME}",
                    f"{CHIMERA_TRAIN_DATA_FOLDER}/{CHIMERA_TRAIN_LABELS_FILENAME}",
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
        """
        Abstract method to create the prediction router.  Must be implemented by subclasses.

        Returns:
            The FastAPI router for making predictions.
        """
        raise NotImplementedError


class RegressionWorker(_PredictionWorker):
    """
    Chimera worker for regression tasks.
    """

    def __init__(self, regressor: RegressorMixin, bootstrap: bool = False) -> None:
        """
        Initializes the RegressionWorker.

        Args:
            regressor: The scikit-learn regressor model.
            bootstrap: Whether to use bootstrapping (default: False).
        """
        super().__init__(regressor, bootstrap)

    def _predict_router(self) -> APIRouter:
        """
        Creates and returns the FastAPI router for the /predict endpoint (regression).

        Returns:
            The FastAPI router for making regression predictions.
        """
        router = APIRouter()

        @router.post(CHIMERA_NODE_PREDICT_PATH)
        def predict(predict_input: PredictInput) -> JSONResponse:
            """
            Makes a regression prediction using the fitted model.
            """
            try:
                X_pred_rows = predict_input.X_pred_rows
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
    """
    Chimera worker for classification tasks.
    """

    def __init__(self, classifier: ClassifierMixin, bootstrap: bool = False) -> None:
        """
        Initializes the ClassificationWorker.

        Args:
            classifier: The scikit-learn classifier model.
            bootstrap: Whether to use bootstrapping (default: False).
        """
        super().__init__(classifier, bootstrap)

    def _predict_router(self) -> APIRouter:
        """
        Creates and returns the FastAPI router for the /predict endpoint (classification).

        Returns:
            The FastAPI router for making classification predictions.
        """
        router = APIRouter()

        @router.post(CHIMERA_NODE_PREDICT_PATH)
        def predict(predict_input: PredictInput) -> JSONResponse:
            """
            Makes a classification prediction using the fitted model.  Returns prediction probabilities.
            """
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
